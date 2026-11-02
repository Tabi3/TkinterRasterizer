"""Vertex and fragment shader factories.

Spec:
    Define the shader callback pairs used by the renderer for color generation,
    lighting, and texture sampling while keeping the per-frame uniforms separate
    from the compiled shader functions.

Contract:
    - `vs(vertex, u) -> (clip_pos, varyings)`
    - `fs(varyings) -> packed_rgb`
    - Uniforms are read from a `Uniforms` instance inside the vertex shader.

Behavior:
    The rasterizer handles perspective-correct interpolation across the output
    varyings, and each shader factory returns a pair of closure-based callables
    that are rebuilt only when the material definition changes.

Reference:
    See IMPLEMENTATION_SPEC.md for the public API separation and the shader
    responsibility boundary used by the render pipeline.
"""

import math


def _clamp01(x):
    return 0.0 if x < 0.0 else min(x, 1.0)


def _pack_rgb01(r, g, b):
    return (
        (int(_clamp01(r) * 255.0) << 16)
        | (int(_clamp01(g) * 255.0) << 8)
        | int(_clamp01(b) * 255.0)
    )


# ---------------------------------------------------------------------------
def make_gouraud_shader(light_color=(1.0, 0.95, 0.85), ambient=0.15):
    tint_r, tint_g, tint_b = light_color

    def vs(vertex, u):
        clip = u.transform(vertex.pos)
        m = u.model.data
        n = vertex.normal

        if n is not None:
            # World-space normal (upper-left 3x3; valid for rotation + uniform scale).
            nx = m[0] * n.x + m[1] * n.y + m[2] * n.z
            ny = m[4] * n.x + m[5] * n.y + m[6] * n.z
            nz = m[8] * n.x + m[9] * n.y + m[10] * n.z
            nl = (nx * nx + ny * ny + nz * nz) ** 0.5
            if nl > 0.0:
                inv = 1.0 / nl
                nx *= inv
                ny *= inv
                nz *= inv

            # World-space vertex position (needed for point-light distance).
            p = vertex.pos
            vx = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3]
            vy = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7]
            vz = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]

            # Accumulate lighting from all lights.
            ambient_level = max(u.ambient, ambient)
            r = ambient_level
            g = ambient_level
            b = ambient_level
            for light in u.lights:
                if not light.enabled:
                    continue

                light_x, light_y, light_z, radius, intensity = (
                    light.position.x,
                    light.position.y,
                    light.position.z,
                    light.radius,
                    light.intensity,
                )
                direction_x = light.direction.x
                direction_y = light.direction.y
                direction_z = light.direction.z

                if light.kind == "directional":
                    d = -(nx * direction_x + ny * direction_y + nz * direction_z)
                else:  # point
                    dx = light_x - vx
                    dy = light_y - vy
                    dz = light_z - vz
                    dist2 = dx * dx + dy * dy + dz * dz
                    if dist2 > radius * radius:
                        continue
                    dist = dist2**0.5
                    inv_d = 1.0 / dist
                    d = (nx * dx + ny * dy + nz * dz) * inv_d
                    d *= 1.0 - dist / radius  # linear falloff
                d = max(d, 0.0)
                r += d * intensity * light.color[0] * tint_r
                g += d * intensity * light.color[1] * tint_g
                b += d * intensity * light.color[2] * tint_b
            r = min(r, 1.0)
            g = min(g, 1.0)
            b = min(b, 1.0)
        else:
            r = g = b = 1.0

        c = vertex.color if vertex.color is not None else 0xFFFFFF
        ar = ((c >> 16) & 0xFF) / 255.0
        ag = ((c >> 8) & 0xFF) / 255.0
        ab = (c & 0xFF) / 255.0
        return clip, (ar * r, ag * g, ab * b)

    def fs(varyings):
        return _pack_rgb01(*varyings)

    return vs, fs


# ---------------------------------------------------------------------------
def make_textured_shader(texture):
    """Unlit textured. Fastest path — sample + pack."""
    tex_data = texture.data
    tw, th = texture.width, texture.height
    tw1, th1 = tw - 1, th - 1
    repeat = texture.wrap == "repeat"

    def vs(vertex, u):
        clip = u.transform(vertex.pos)
        uv = vertex.uv
        return clip, (uv.x if uv else 0.5, uv.y if uv else 0.5)

    def fs(varyings):
        u_c, v_c = varyings
        if repeat:
            if u_c < 0.0 or u_c >= 1.0:
                u_c = u_c - math.floor(u_c)
            if v_c < 0.0 or v_c >= 1.0:
                v_c = v_c - math.floor(v_c)
        else:
            if u_c < 0.0:
                u_c = 0.0
            elif u_c > 1.0:
                u_c = 1.0
            if v_c < 0.0:
                v_c = 0.0
            elif v_c > 1.0:
                v_c = 1.0
        ix = int(u_c * tw)
        ix = min(ix, tw1)
        iy = int(v_c * th)
        iy = min(iy, th1)
        return tex_data[iy * tw + ix]

    return vs, fs


# ---------------------------------------------------------------------------
def make_textured_lit_shader(texture):
    """Textured + Gouraud lighting. Lighting schema matches Uniforms:

        u.lights: list[Light]   (directional and/or point)
        u.ambient: float        (base illumination, added once)

    Varyings: (u, v, r, g, b) — three lighting channels, each pre-clamped
    to [0, 1] in the vertex shader, so the fragment stage needs no clamp.
    """
    tex_data = list(texture.data)  # list reads are ~2-3x faster than array
    tw, th = texture.width, texture.height
    tw1, th1 = tw - 1, th - 1
    repeat = texture.wrap == "repeat"

    def vs(vertex, u):
        clip = u.transform(vertex.pos)
        m = u.model.data
        n = vertex.normal
        if n is not None:
            # World-space normal (upper-left 3x3; valid for rotation + uniform scale).
            nx = m[0] * n.x + m[1] * n.y + m[2] * n.z
            ny = m[4] * n.x + m[5] * n.y + m[6] * n.z
            nz = m[8] * n.x + m[9] * n.y + m[10] * n.z
            nl = (nx * nx + ny * ny + nz * nz) ** 0.5
            if nl > 0.0:
                inv = 1.0 / nl
                nx *= inv
                ny *= inv
                nz *= inv

            # World-space vertex position (needed for point-light distance).
            p = vertex.pos
            vx = m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3]
            vy = m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7]
            vz = m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]

            amb = u.ambient
            r = amb
            g = amb
            b = amb
            for light in u.lights:
                if not light.enabled:
                    continue
                if light.kind == "directional":
                    d = -(
                        nx * light.direction.x
                        + ny * light.direction.y
                        + nz * light.direction.z
                    )
                else:  # point
                    dx = light.position.x - vx
                    dy = light.position.y - vy
                    dz = light.position.z - vz
                    dist2 = dx * dx + dy * dy + dz * dz
                    if dist2 > light.radius * light.radius:
                        continue
                    dist = dist2**0.5
                    inv_d = 1.0 / dist if dist > 0.0 else 0.0
                    d = (nx * dx + ny * dy + nz * dz) * inv_d
                    d *= 1.0 - dist / light.radius  # linear falloff
                d = max(d, 0.0)
                intensity = light.intensity
                r += d * intensity * light.color[0]
                g += d * intensity * light.color[1]
                b += d * intensity * light.color[2]
            r = min(r, 1.0)
            g = min(g, 1.0)
            b = min(b, 1.0)
        else:
            r = g = b = 1.0

        uv = vertex.uv
        return clip, (uv.x if uv else 0.5, uv.y if uv else 0.5, r, g, b)

    # Clamp/repeat specialized at factory time (one fewer branch per fragment).
    if repeat:

        def fs(varyings):
            u_c, v_c, lit_r, lit_g, lit_b = varyings
            if u_c < 0.0 or u_c >= 1.0:
                u_c -= math.floor(u_c)
            if v_c < 0.0 or v_c >= 1.0:
                v_c -= math.floor(v_c)
            ix = int(u_c * tw)
            ix = min(ix, tw1)
            iy = int(v_c * th)
            iy = min(iy, th1)
            tex = tex_data[iy * tw + ix]
            # lit_* <= 1 guaranteed by vertex-side clamp, so no fragment clamp.
            return (
                (int((tex >> 16 & 0xFF) * lit_r) << 16)
                | (int((tex >> 8 & 0xFF) * lit_g) << 8)
                | int((tex & 0xFF) * lit_b)
            )
    else:

        def fs(varyings):
            u_c, v_c, lit_r, lit_g, lit_b = varyings
            if u_c < 0.0:
                u_c = 0.0
            elif u_c > 1.0:
                u_c = 1.0
            if v_c < 0.0:
                v_c = 0.0
            elif v_c > 1.0:
                v_c = 1.0
            ix = int(u_c * tw)
            ix = min(ix, tw1)
            iy = int(v_c * th)
            iy = min(iy, th1)
            tex = tex_data[iy * tw + ix]
            return (
                (int((tex >> 16 & 0xFF) * lit_r) << 16)
                | (int((tex >> 8 & 0xFF) * lit_g) << 8)
                | int((tex & 0xFF) * lit_b)
            )

    return vs, fs
