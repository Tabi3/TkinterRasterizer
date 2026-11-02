"""Screen-space triangle rasterizer.

Purpose:
    Fill a triangle into a framebuffer using a cheap incrementally-updated
    edge-function algorithm with reverse-Z depth testing.

Contract:
    rasterize_triangle(fb, v0, v1, v2, color)
        fb    : Framebuffer
        v_i   : (sx, sy, z_ndc, inv_w) with reverse-Z z_ndc in [0, 1]
        color : packed 24-bit RGB value

Behavior:
    The rasterizer computes a bounding box, applies the top-left fill rule, and
    writes pixels in a stable, crack-free manner while respecting reverse-Z.

Notes:
    Depth is interpolated linearly in screen space for the standard projection
    path; perspective-correct interpolation is reserved for shader varyings.

Naming convention:
    coord_*     screen-space or vertex coordinates
    edge_*      edge-function coefficients
    local_*     scratch values scoped to a single function
    row_*       values reused across the inner scanline loop

Reference:
    See IMPLEMENTATION_SPEC.md for the render pipeline boundaries and the
    naming conventions used by the core rasterization path.
"""

import math
from collections.abc import Callable


def _triangle_bounds(vertices, width, height):
    """Return the integer bounding box of a triangle in screen space."""
    if len(vertices) != 3:
        raise ValueError("triangle must be defined by exactly 3 vertices")
    if not (isinstance(width, int) and isinstance(height, int)):
        raise TypeError("framebuffer size must be integer")
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    min_x = max(0, math.ceil(min(xs) - 0.5))
    max_x = min(width - 1, math.floor(max(xs) - 0.5))
    min_y = max(0, math.ceil(min(ys) - 0.5))
    max_y = min(height - 1, math.floor(max(ys) - 0.5))
    return min_x, max_x, min_y, max_y


def _edge_coefficients(vertex_0, vertex_1, vertex_2):
    """Compute edge coefficients for the incremental edge-function algorithm."""
    if not (len(vertex_0) >= 4 and len(vertex_1) >= 4 and len(vertex_2) >= 4):
        raise ValueError("vertices must have screen x/y/z/w fields")
    coord_x0, coord_y0, _, _ = vertex_0
    coord_x1, coord_y1, _, _ = vertex_1
    coord_x2, coord_y2, _, _ = vertex_2
    edge_a0 = coord_y1 - coord_y2
    edge_b0 = coord_x2 - coord_x1
    edge_c0 = (coord_y2 - coord_y1) * coord_x1 - (coord_x2 - coord_x1) * coord_y1
    edge_a1 = coord_y2 - coord_y0
    edge_b1 = coord_x0 - coord_x2
    edge_c1 = (coord_y0 - coord_y2) * coord_x2 - (coord_x0 - coord_x2) * coord_y2
    edge_a2 = coord_y0 - coord_y1
    edge_b2 = coord_x1 - coord_x0
    edge_c2 = (coord_y1 - coord_y0) * coord_x0 - (coord_x1 - coord_x0) * coord_y0
    return (
        (edge_a0, edge_b0, edge_c0),
        (edge_a1, edge_b1, edge_c1),
        (edge_a2, edge_b2, edge_c2),
    )


def _top_left_flags(edges):
    """Compute the top-left fill-rule flags for the three edges."""
    return tuple(
        (edge_a > 0) or (edge_a == 0 and edge_b > 0) for edge_a, edge_b, _ in edges
    )


def rasterize_triangle(fb, vertex_0, vertex_1, vertex_2, color):
    # Reference: IMPLEMENTATION_SPEC.md > Core architecture > 2. Rendering and backend layer
    if not (
        hasattr(fb, "width")
        and hasattr(fb, "height")
        and hasattr(fb, "color")
        and hasattr(fb, "depth")
    ):
        raise TypeError("framebuffer must provide width/height/color/depth")
    if not (isinstance(color, int) and not isinstance(color, bool)):
        raise TypeError("color must be an integer packed RGB value")
    # Naming scheme: coord_* = screen-space position, edge_* = edge-function
    # coefficients, local_* = scratch state scoped to this call, row_* = values
    # reused across the scanline loop.
    coord_x0, coord_y0, depth_z0, _inv_w0 = vertex_0
    coord_x1, coord_y1, depth_z1, _inv_w1 = vertex_1
    coord_x2, coord_y2, depth_z2, _inv_w2 = vertex_2

    # ----- orientation / area -------------------------------------------
    local_area = (coord_x1 - coord_x0) * (coord_y2 - coord_y0) - (
        coord_x2 - coord_x0
    ) * (coord_y1 - coord_y0)
    if local_area == 0.0:
        return
    if local_area < 0.0:
        # Force area > 0 by swapping vertex_1 <-> vertex_2. Callable code may
        # still provide either winding; the rasterizer accepts both.
        coord_x1, coord_y1, depth_z1, _inv_w1, coord_x2, coord_y2, depth_z2, _inv_w2 = (
            coord_x2,
            coord_y2,
            depth_z2,
            _inv_w2,
            coord_x1,
            coord_y1,
            depth_z1,
            _inv_w1,
        )
        local_area = -local_area
    local_inv_area = 1.0 / local_area

    # ----- bounding box, clamped to fb, sampled at pixel centers --------
    min_x, max_x, min_y, max_y = _triangle_bounds(
        (vertex_0, vertex_1, vertex_2), fb.width, fb.height
    )
    if min_x > max_x or min_y > max_y:
        return

    edge_terms = _edge_coefficients(
        (coord_x0, coord_y0, depth_z0, _inv_w0),
        (coord_x1, coord_y1, depth_z1, _inv_w1),
        (coord_x2, coord_y2, depth_z2, _inv_w2),
    )
    edge_a0, edge_b0, edge_c0 = edge_terms[0]
    edge_a1, edge_b1, edge_c1 = edge_terms[1]
    edge_a2, edge_b2, edge_c2 = edge_terms[2]
    edge_c0 += (edge_a0 + edge_b0) * 0.5
    edge_c1 += (edge_a1 + edge_b1) * 0.5
    edge_c2 += (edge_a2 + edge_b2) * 0.5
    fill_rule0, fill_rule1, fill_rule2 = _top_left_flags(
        (
            (edge_a0, edge_b0, edge_c0),
            (edge_a1, edge_b1, edge_c1),
            (edge_a2, edge_b2, edge_c2),
        )
    )

    # ----- depth gradient (linear in screen barycentrics) ---------------
    local_depth_dx = (
        edge_a0 * depth_z0 + edge_a1 * depth_z1 + edge_a2 * depth_z2
    ) * local_inv_area
    local_depth_dy = (
        edge_b0 * depth_z0 + edge_b1 * depth_z1 + edge_b2 * depth_z2
    ) * local_inv_area

    # ----- initial values at (min_x, min_y) -----------------------------
    edge_value_0 = edge_a0 * min_x + edge_b0 * min_y + edge_c0
    edge_value_1 = edge_a1 * min_x + edge_b1 * min_y + edge_c1
    edge_value_2 = edge_a2 * min_x + edge_b2 * min_y + edge_c2
    depth_value = (
        edge_value_0 * depth_z0 + edge_value_1 * depth_z1 + edge_value_2 * depth_z2
    ) * local_inv_area

    # ----- hot loop -----------------------------------------------------
    packed_color = color & 0xFFFFFF
    frame_width = fb.width
    frame_color = fb.color
    frame_depth = fb.depth

    for pixel_y in range(min_y, max_y + 1):
        row_edge_0 = edge_value_0
        row_edge_1 = edge_value_1
        row_edge_2 = edge_value_2
        row_depth = depth_value
        pixel_index = pixel_y * frame_width + min_x
        for _ in range(min_x, max_x + 1):
            inside = (
                (row_edge_0 > 0.0 or (row_edge_0 == 0.0 and fill_rule0))
                and (row_edge_1 > 0.0 or (row_edge_1 == 0.0 and fill_rule1))
                and (row_edge_2 > 0.0 or (row_edge_2 == 0.0 and fill_rule2))
            )
            if inside and row_depth > frame_depth[pixel_index]:
                frame_depth[pixel_index] = row_depth
                frame_color[pixel_index] = packed_color
            row_edge_0 += edge_a0
            row_edge_1 += edge_a1
            row_edge_2 += edge_a2
            row_depth += local_depth_dx
            pixel_index += 1
        edge_value_0 += edge_b0
        edge_value_1 += edge_b1
        edge_value_2 += edge_b2
        depth_value += local_depth_dy


def rasterize_triangle_shaded(
    fb,
    vertex_0,
    vertex_1,
    vertex_2,
    var0,
    var1,
    var2,
    fs: Callable[[tuple[float, ...]], int],
    *,
    depth_write: bool = True,
):
    """Span-based rasterizer with perspective-correct varying interpolation.

    Spec:
        For each scanline y with center y+0.5, compute integer pixel span
        [span_x0, span_x1] by inverting the three edge half-planes, then fill.

    Contract:
        - Requires w_i > 0 for all vertices (caller's responsibility).
        - Top-left rule: boundary pixels (E_i = 0) belong to the triangle
          if and only if E_i's edge is top-left for this winding,
          tl_i = (A_i > 0) or (A_i == 0 and B_i > 0). This is the exact
          same rule the edge-test version used; without it, adjacent
          triangles sharing a boundary leave a 1-pixel crack.
        - Depth test: z_ndc > depth, reverse-Z.
        - Inv_area cancels in the varying formula; not present.

    Args:
        fb: Framebuffer.
        vertex_i: (sx, sy, z_ndc, iw).
        var_i: tuple of per-vertex varying floats.
        fs: fragment shader, fs(varyings) -> 0xRRGGBB.
    """
    if not (
        hasattr(fb, "width")
        and hasattr(fb, "height")
        and hasattr(fb, "color")
        and hasattr(fb, "depth")
    ):
        raise TypeError("framebuffer must provide width/height/color/depth")
    if not callable(fs):
        raise TypeError("fragment shader must be callable")
    if not isinstance(depth_write, bool):
        raise TypeError("depth_write must be boolean")
    if len(var0) != len(var1) or len(var0) != len(var2):
        raise ValueError("varying lengths must match")
    if any(
        not math.isfinite(component)
        for v in (vertex_0, vertex_1, vertex_2)
        for component in v
    ):
        raise ValueError("screen vertices must be finite")
    if any(v[3] <= 0 for v in (vertex_0, vertex_1, vertex_2)):
        raise ValueError("inverse w must be positive")
    coord_x0, coord_y0, depth_z0, inv_w0 = vertex_0
    coord_x1, coord_y1, depth_z1, inv_w1 = vertex_1
    coord_x2, coord_y2, depth_z2, inv_w2 = vertex_2

    local_area = (coord_x1 - coord_x0) * (coord_y2 - coord_y0) - (
        coord_x2 - coord_x0
    ) * (coord_y1 - coord_y0)
    if local_area == 0.0:
        return
    if local_area < 0.0:
        coord_x1, coord_y1, depth_z1, inv_w1, coord_x2, coord_y2, depth_z2, inv_w2 = (
            coord_x2,
            coord_y2,
            depth_z2,
            inv_w2,
            coord_x1,
            coord_y1,
            depth_z1,
            inv_w1,
        )
        var1, var2 = var2, var1
        local_area = -local_area
    local_inv_area = 1.0 / local_area

    min_x, max_x, min_y, max_y = _triangle_bounds(
        (
            (coord_x0, coord_y0, depth_z0, inv_w0),
            (coord_x1, coord_y1, depth_z1, inv_w1),
            (coord_x2, coord_y2, depth_z2, inv_w2),
        ),
        fb.width,
        fb.height,
    )
    if min_x > max_x or min_y > max_y:
        return

    edge_terms = _edge_coefficients(
        (coord_x0, coord_y0, depth_z0, inv_w0),
        (coord_x1, coord_y1, depth_z1, inv_w1),
        (coord_x2, coord_y2, depth_z2, inv_w2),
    )
    edge_a0, edge_b0, edge_c0 = edge_terms[0]
    edge_a1, edge_b1, edge_c1 = edge_terms[1]
    edge_a2, edge_b2, edge_c2 = edge_terms[2]

    # Top-left rule: include E_i == 0 only for top or left edges.
    tl0 = (edge_a0 > 0.0) or (edge_a0 == 0.0 and edge_b0 > 0.0)
    tl1 = (edge_a1 > 0.0) or (edge_a1 == 0.0 and edge_b1 > 0.0)
    tl2 = (edge_a2 > 0.0) or (edge_a2 == 0.0 and edge_b2 > 0.0)

    half_a0 = edge_a0 * 0.5
    half_a1 = edge_a1 * 0.5
    half_a2 = edge_a2 * 0.5

    local_depth_dx = (
        edge_a0 * depth_z0 + edge_a1 * depth_z1 + edge_a2 * depth_z2
    ) * local_inv_area

    var_count = len(var0)
    weighted_var0 = tuple(v * inv_w0 for v in var0)
    weighted_var1 = tuple(v * inv_w1 for v in var1)
    weighted_var2 = tuple(v * inv_w2 for v in var2)

    denominator_dx = edge_a0 * inv_w0 + edge_a1 * inv_w1 + edge_a2 * inv_w2
    numerator_dx = tuple(
        edge_a0 * a + edge_a1 * b + edge_a2 * c
        for a, b, c in zip(weighted_var0, weighted_var1, weighted_var2)
    )
    frame_width = fb.width
    frame_color = fb.color
    frame_depth = fb.depth

    for pixel_y in range(min_y, max_y + 1):
        center_y = pixel_y + 0.5
        base0 = edge_b0 * center_y + edge_c0 + half_a0
        base1 = edge_b1 * center_y + edge_c1 + half_a1
        base2 = edge_b2 * center_y + edge_c2 + half_a2

        span_x0 = min_x
        span_x1 = max_x
        skip_scanline = False

        # For each edge, tighten [span_x0, span_x1] using the correct
        # integer bound formulas for the strict vs. top-left case.
        for edge_a, base, tl in (
            (edge_a0, base0, tl0),
            (edge_a1, base1, tl1),
            (edge_a2, base2, tl2),
        ):
            if edge_a > 0.0:
                bound = -base / edge_a
                # Clamp to avoid int() on huge values from near-horizontal
                # edges with a small positive A.
                if bound < min_x - 1.0:
                    bound = min_x - 1.0
                elif bound > max_x + 1.0:
                    bound = max_x + 1.0
                lo = math.ceil(bound) if tl else math.floor(bound) + 1
                span_x0 = max(span_x0, lo)
            elif edge_a < 0.0:
                bound = -base / edge_a
                if bound < min_x - 1.0:
                    bound = min_x - 1.0
                elif bound > max_x + 1.0:
                    bound = max_x + 1.0
                hi = math.floor(bound) if tl else math.ceil(bound) - 1
                span_x1 = min(span_x1, hi)
            else:
                # Horizontal edge: constrains the scanline, not px.
                if base < 0.0 or (base == 0.0 and not tl):
                    skip_scanline = True
                    break

        if skip_scanline or span_x0 > span_x1:
            continue

        cx0 = span_x0
        e0 = edge_a0 * cx0 + base0
        e1 = edge_a1 * cx0 + base1
        e2 = edge_a2 * cx0 + base2
        depth_value = (e0 * depth_z0 + e1 * depth_z1 + e2 * depth_z2) * local_inv_area

        denominator = e0 * inv_w0 + e1 * inv_w1 + e2 * inv_w2
        numerators = [
            e0 * a + e1 * b + e2 * c
            for a, b, c in zip(weighted_var0, weighted_var1, weighted_var2)
        ]
        if var_count == 2:
            n0, n1 = numerators
            dx0, dx1 = numerator_dx
        elif var_count == 3:
            n0, n1, n2 = numerators
            dx0, dx1, dx2 = numerator_dx
        elif var_count == 5:
            n0, n1, n2, n3, n4 = numerators
            dx0, dx1, dx2, dx3, dx4 = numerator_dx
        pixel_index = pixel_y * frame_width + span_x0
        for _ in range(span_x0, span_x1 + 1):
            if depth_value > frame_depth[pixel_index]:
                inv_den = 1.0 / denominator
                if var_count == 2:
                    v0 = n0 * inv_den
                    v1 = n1 * inv_den
                    fragment_color = fs((v0, v1))
                elif var_count == 3:
                    v0 = n0 * inv_den
                    v1 = n1 * inv_den
                    v2 = n2 * inv_den
                    fragment_color = fs((v0, v1, v2))
                elif var_count == 5:
                    v0 = n0 * inv_den
                    v1 = n1 * inv_den
                    v2 = n2 * inv_den
                    v3 = n3 * inv_den
                    v4 = n4 * inv_den
                    fragment_color = fs((v0, v1, v2, v3, v4))
                else:
                    fragment_color = fs(
                        tuple(numerators[k] * inv_den for k in range(var_count))
                    )
                if depth_write:
                    frame_depth[pixel_index] = depth_value
                frame_color[pixel_index] = fragment_color & 0xFFFFFF
            denominator += denominator_dx
            if var_count == 2:
                n0 += dx0
                n1 += dx1
            elif var_count == 3:
                n0 += dx0
                n1 += dx1
                n2 += dx2
            elif var_count == 5:
                n0 += dx0
                n1 += dx1
                n2 += dx2
                n3 += dx3
                n4 += dx4
            else:
                for k in range(var_count):
                    numerators[k] += numerator_dx[k]
            depth_value += local_depth_dx
            pixel_index += 1


def draw_line(fb, x0, y0, z0, x1, y1, z1, color):
    """DDA line with reverse-Z depth test.

    Spec:
        Rasterizes the segment (x0,y0,z0)-(x1,y1,z1) in screen space,
        writing (color, z) at each pixel where z > depth[pixel].

    Contract:
        Screen coords are in pixel-index space. Ints obtained by round().
        Reverse-Z test matches rasterize_triangle's convention.

    Behavior:
        steps = max(|dx|, |dy|), so dominant-axis runs exactly step.
        Screen-space z is linear in the parameter t along the projected
        segment under perspective, so linear interpolation is correct.
    """
    if not (
        hasattr(fb, "width")
        and hasattr(fb, "height")
        and hasattr(fb, "color")
        and hasattr(fb, "depth")
    ):
        raise TypeError("framebuffer must provide width/height/color/depth")
    if not (isinstance(color, int) and not isinstance(color, bool)):
        raise TypeError("color must be an integer packed RGB value")
    if not all(math.isfinite(value) for value in (x0, y0, z0, x1, y1, z1)):
        raise ValueError("line coordinates must be finite")
    # Clip before stepping so distant offscreen endpoints cannot stall a frame.
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0),
        (dx, fb.width - 1 - x0),
        (-dy, y0),
        (dy, fb.height - 1 - y0),
    ):
        if p == 0:
            if q < 0:
                return
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return
    x1, y1, z1 = x0 + dx * t1, y0 + dy * t1, z0 + dz * t1
    x0, y0, z0 = x0 + dx * t0, y0 + dy * t0, z0 + dz * t0
    x0i = round(x0)
    y0i = round(y0)
    x1i = round(x1)
    y1i = round(y1)
    steps = max(abs(x1i - x0i), abs(y1i - y0i))
    w = fb.width
    h = fb.height
    fc = fb.color
    fd = fb.depth
    packed = color & 0xFFFFFF
    if steps == 0:
        if 0 <= x0i < w and 0 <= y0i < h:
            idx = y0i * w + x0i
            if z0 > fd[idx]:
                fd[idx] = z0
                fc[idx] = packed
        return
    inv = 1.0 / steps
    dx = (x1 - x0) * inv
    dy = (y1 - y0) * inv
    dz = (z1 - z0) * inv
    x = float(x0)
    y = float(y0)
    z = z0
    for _ in range(steps + 1):
        xi = round(x)
        yi = round(y)
        if 0 <= xi < w and 0 <= yi < h:
            idx = yi * w + xi
            if z > fd[idx]:
                fd[idx] = z
                fc[idx] = packed
        x += dx
        y += dy
        z += dz
