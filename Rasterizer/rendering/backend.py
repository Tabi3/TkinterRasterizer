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

from ..algebra.math3d import Vec4
from .rasterizer import rasterize_triangle


def clip_near_triangle(clip_vertex_0, clip_vertex_1, clip_vertex_2, near):
    """Sutherland-Hodgman against the near plane (w >= near).

    Reference:
        IMPLEMENTATION_SPEC.md > Core architecture > 2. Rendering and backend layer
    """
    if not (
        hasattr(clip_vertex_0, "w")
        and hasattr(clip_vertex_1, "w")
        and hasattr(clip_vertex_2, "w")
    ):
        raise TypeError("clip vertices must provide a w component")
    if not (isinstance(near, (int, float)) and not isinstance(near, bool)):
        raise TypeError("near must be numeric")
    vertices = (clip_vertex_0, clip_vertex_1, clip_vertex_2)
    clipped = []
    for index in range(3):
        current = vertices[index]
        next_vertex = vertices[(index + 1) % 3]
        current_inside = current.w >= near
        next_inside = next_vertex.w >= near
        if current_inside and next_inside:
            clipped.append(next_vertex)
        elif current_inside and not next_inside:
            local_t = (near - current.w) / (next_vertex.w - current.w)
            clipped.append(
                Vec4(
                    current.x + local_t * (next_vertex.x - current.x),
                    current.y + local_t * (next_vertex.y - current.y),
                    current.z + local_t * (next_vertex.z - current.z),
                    near,
                )
            )
        elif not current_inside and next_inside:
            local_t = (near - current.w) / (next_vertex.w - current.w)
            clipped.append(
                Vec4(
                    current.x + local_t * (next_vertex.x - current.x),
                    current.y + local_t * (next_vertex.y - current.y),
                    current.z + local_t * (next_vertex.z - current.z),
                    near,
                )
            )
            clipped.append(next_vertex)
    if len(clipped) < 3:
        return []
    return [
        (clipped[0], clipped[i], clipped[i + 1]) for i in range(1, len(clipped) - 1)
    ]


def clip_near_shaded(verts, near):
    """Sutherland-Hodgman against w >= near while preserving varyings.

    Reference:
        IMPLEMENTATION_SPEC.md > Core architecture > 2. Rendering and backend layer
    """
    clipped = []
    for index in range(3):
        current_clip, current_varying = verts[index]
        next_clip, next_varying = verts[(index + 1) % 3]
        current_inside = current_clip.w >= near
        next_inside = next_clip.w >= near

        if current_inside:
            if next_inside:
                clipped.append((next_clip, next_varying))
            else:
                local_t = (near - current_clip.w) / (next_clip.w - current_clip.w)
                new_clip = Vec4(
                    current_clip.x + local_t * (next_clip.x - current_clip.x),
                    current_clip.y + local_t * (next_clip.y - current_clip.y),
                    current_clip.z + local_t * (next_clip.z - current_clip.z),
                    near,
                )
                new_varying = tuple(
                    current_varying[k]
                    + local_t * (next_varying[k] - current_varying[k])
                    for k in range(len(current_varying))
                )
                clipped.append((new_clip, new_varying))
        else:
            if next_inside:
                local_t = (near - current_clip.w) / (next_clip.w - current_clip.w)
                new_clip = Vec4(
                    current_clip.x + local_t * (next_clip.x - current_clip.x),
                    current_clip.y + local_t * (next_clip.y - current_clip.y),
                    current_clip.z + local_t * (next_clip.z - current_clip.z),
                    near,
                )
                new_varying = tuple(
                    current_varying[k]
                    + local_t * (next_varying[k] - current_varying[k])
                    for k in range(len(current_varying))
                )
                clipped.append((new_clip, new_varying))
                clipped.append((next_clip, next_varying))

    if len(clipped) < 3:
        return []
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
    return view_aabb.min_z < -near


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
        r = int(br + (tr - br) * t)
        g = int(bg + (tg - bg) * t)
        b = int(bb + (tb - bb) * t)
        packed = (r << 16) | (g << 8) | b
        start = y * w
        end = start + w
        color[start:end] = array("I", [packed]) * w
        depth[start:end] = array("f", [0.0]) * w
