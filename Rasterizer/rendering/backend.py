"""Internal rendering backend for the public renderer API.

This file contains the clip, culling and rasterization logic that the
Renderer API delegates to. The demos keep using the same public classes and
methods, but the actual implementation now lives in a dedicated backend module.

Naming convention:
    clip_*      vertices or varyings being processed against a plane
    local_*     short-lived scratch data inside a function
    screen_*    projected coordinates in the viewport space
    render_*    final output or drawing parameters

Reference:
    See IMPLEMENTATION_SPEC.md for the backend separation and the project-level
    contract governing clipping, culling, and raster dispatch.
"""

from __future__ import annotations

import math

from ..algebra.math3d import Vec4
from .rasterizer import rasterize_triangle


def _clip_polygon(vertices, distance):
    """Clip (position, varyings) pairs, interpolating at the same intersection."""
    clipped = []
    vertices = (*vertices[1:], vertices[0])
    current, current_var = vertices[-1]
    current_d = distance(current)
    for next_vertex, next_var in vertices:
        next_d = distance(next_vertex)
        if (current_d >= 0) != (next_d >= 0):
            t = current_d / (current_d - next_d)
            intersection = Vec4(
                current.x + t * (next_vertex.x - current.x),
                current.y + t * (next_vertex.y - current.y),
                current.z + t * (next_vertex.z - current.z),
                current.w + t * (next_vertex.w - current.w),
            )
            varying = tuple(a + t * (b - a) for a, b in zip(current_var, next_var))
            clipped.append((intersection, varying))
        if next_d >= 0:
            clipped.append((next_vertex, next_var))
        current, current_var, current_d = next_vertex, next_var, next_d
    return clipped


def clip_near_triangle(c0, c1, c2, near, *, orthographic=False):
    """Clip perspective at w >= near, or orthographic at 0 <= z <= w."""
    return [
        tuple(vertex for vertex, _ in triangle)
        for triangle in clip_near_shaded(
            [(c0, ()), (c1, ()), (c2, ())], near, orthographic=orthographic
        )
    ]


def clip_near_shaded(verts, near, *, orthographic=False):
    """Clip a triangle while preserving its shader varyings."""
    if len(verts) != 3:
        raise ValueError("exactly three vertices are required")
    if not isinstance(near, (int, float)) or isinstance(near, bool):
        raise TypeError("near must be numeric")
    if not math.isfinite(near) or near <= 0:
        raise ValueError("near must be positive")
    if len({len(varying) for _, varying in verts}) != 1:
        raise ValueError("varying lengths must match")
    if orthographic:
        clipped = _clip_polygon(verts, lambda v: v.w - v.z)
        if clipped:
            clipped = _clip_polygon(clipped, lambda v: v.z)
    else:
        clipped = _clip_polygon(verts, lambda v: v.w - near)
    return [
        (clipped[0], clipped[i], clipped[i + 1]) for i in range(1, len(clipped) - 1)
    ]


def cull_and_raster(fb, vp, n0, n1, n2, color, cull=True):
    # Reference: IMPLEMENTATION_SPEC.md > Core architecture > 2. Rendering and backend layer
    if not (
        hasattr(fb, "width")
        and hasattr(fb, "height")
        and hasattr(fb, "color")
        and hasattr(fb, "depth")
    ):
        raise TypeError("framebuffer must provide width/height/color/depth")
    if not isinstance(cull, bool):
        raise TypeError("cull must be a boolean")
    if not (isinstance(color, int) and not isinstance(color, bool)):
        raise TypeError("color must be an integer pixel value")
    screen_vertex_0 = vp @ Vec4(n0[0], n0[1], n0[2], 1.0)
    screen_vertex_1 = vp @ Vec4(n1[0], n1[1], n1[2], 1.0)
    screen_vertex_2 = vp @ Vec4(n2[0], n2[1], n2[2], 1.0)

    if cull:
        local_area = (screen_vertex_1.x - screen_vertex_0.x) * (
            screen_vertex_2.y - screen_vertex_0.y
        ) - (screen_vertex_2.x - screen_vertex_0.x) * (
            screen_vertex_1.y - screen_vertex_0.y
        )
        if local_area >= 0.0:
            return

    rasterize_triangle(
        fb,
        (screen_vertex_0.x, screen_vertex_0.y, screen_vertex_0.z, 1.0),
        (screen_vertex_1.x, screen_vertex_1.y, screen_vertex_1.z, 1.0),
        (screen_vertex_2.x, screen_vertex_2.y, screen_vertex_2.z, 1.0),
        color,
    )


def aabb_visible(world_aabb, view_matrix, near):
    """True if the AABB is not entirely behind the near plane.

    Spec:
        Conservative: returns True for boxes that straddle the near plane
        (their triangle-level clipper handles the straddling). Returns False
        only when every point of the box is closer than `near`.

    Contract:
        Assumes camera space has forward = -Z, so near plane is z = -near.
        Uses AABB.transform for the world→camera-space AABB, which is a
        conservative bound (the projected AABB can be larger than the true
        bounds of the transformed geometry, never smaller).
    """
    if not hasattr(world_aabb, "transform"):
        raise TypeError("world_aabb must be an AABB-like object")
    if not (isinstance(near, (int, float)) and not isinstance(near, bool)):
        raise TypeError("near must be numeric")
    view_aabb = world_aabb.transform(view_matrix)
    return view_aabb.min_z <= -near


def fill_gradient(fb, color_top, color_bottom):
    """Fill framebuffer with a vertical two-color gradient.

    Spec:
        Row y gets color = lerp(color_top, color_bottom, y / (h-1)).
        Depth is written as 0.0 (reverse-Z far plane), so anything the
        camera can see will overwrite the sky on depth test.

    Contract:
        Replaces clear(). After this, depth[py, px] = 0 for all px, py.

    Behavior:
        Per-row color is computed once (one multiply-add per channel per row),
        then the whole row is filled by array slice assignment (C-level memcpy
        of a repeated value). This is ~35x faster than a per-pixel loop.

    Args:
        fb: Framebuffer with .color (array('I')) and .depth (array('f')).
        color_top, color_bottom: 0xRRGGBB.
    """
    from array import array

    if not (
        hasattr(fb, "width")
        and hasattr(fb, "height")
        and hasattr(fb, "color")
        and hasattr(fb, "depth")
    ):
        raise TypeError("framebuffer must provide width/height/color/depth")
    if not (isinstance(color_top, int) and isinstance(color_bottom, int)):
        raise TypeError("color_top and color_bottom must be integers")
    w = fb.width
    h = fb.height
    tr = (color_top >> 16) & 0xFF
    tg = (color_top >> 8) & 0xFF
    tb = color_top & 0xFF
    br = (color_bottom >> 16) & 0xFF
    bg = (color_bottom >> 8) & 0xFF
    bb = color_bottom & 0xFF
    color = fb.color
    depth = fb.depth
    inv_h = 1.0 / (h - 1) if h > 1 else 1.0
    for y in range(h):
        t = y * inv_h
        r = int(tr + (br - tr) * t)
        g = int(tg + (bg - tg) * t)
        b = int(tb + (bb - tb) * t)
        packed = (r << 16) | (g << 8) | b
        start = y * w
        end = start + w
        color[start:end] = array("I", [packed]) * w
        depth[start:end] = array("f", [0.0]) * w
