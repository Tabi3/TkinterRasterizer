"""Matrix and camera transform helpers.

Spec:
    Build the view, projection, and viewport transforms used by the renderer and
    camera system while maintaining a stable right-handed Y-up convention.

Contract:
    - Camera space uses the common forward = -Z convention.
    - Matrices remain row-major flat arrays for consistent math operations.
    - Transform builders are small and pure so they can be reused in demos and
      tests without hidden global state.

Behavior:
    This module is responsible for the shape of the coordinate system and for the
    conversion between world, camera, clip, NDC, and screen space.

Reference:
    See IMPLEMENTATION_SPEC.md for the transform contract and the coordinate
    conventions used across the renderer and geometry pipeline.
"""

from __future__ import annotations

import math
from math import cos, sin, sqrt

from .math3d import Mat3, Mat4, cross, dot, normalize


# ---------------------------------------------------------------------------
# Quaternion — rotation authoring only. Bake to Mat4 before the render loop.
# ---------------------------------------------------------------------------


class Quat:
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):

        if not (isinstance(x, (int, float)) and not isinstance(x, bool)):
            raise TypeError("x must be a real numeric value")

        if not (isinstance(y, (int, float)) and not isinstance(y, bool)):
            raise TypeError("y must be a real numeric value")

        if not (isinstance(z, (int, float)) and not isinstance(z, bool)):
            raise TypeError("z must be a real numeric value")

        if not (isinstance(w, (int, float)) and not isinstance(w, bool)):
            raise TypeError("w must be a real numeric value")

        self.x = float(x)

        self.y = float(y)

        self.z = float(z)

        self.w = float(w)

    # ----- constructors ---------------------------------------------------
    @classmethod
    def identity(cls):

        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_axis_angle(cls, axis, angle_rad):

        a = normalize(axis)

        half = angle_rad * 0.5

        s = sin(half)

        return cls(a.x * s, a.y * s, a.z * s, cos(half))

    @classmethod
    def from_euler(cls, rx=0.0, ry=0.0, rz=0.0):
        """Intrinsic X then Y then Z (matches Blender's default XYZ Euler)."""

        hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5

        cx, sx = cos(hx), sin(hx)

        cy, sy = cos(hy), sin(hy)

        cz, sz = cos(hz), sin(hz)

        return cls(
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )

    # ----- algebra --------------------------------------------------------

    def __mul__(self, o):

        if isinstance(o, Quat):
            a, b = self, o

            return Quat(
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            )

        return NotImplemented

    def conjugate(self):

        return Quat(-self.x, -self.y, -self.z, self.w)

    def normalize(self):

        l = sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

        if l == 0.0:
            return Quat.identity()

        inv = 1.0 / l

        return Quat(self.x * inv, self.y * inv, self.z * inv, self.w * inv)

    def to_mat4(self):

        x, y, z, w = self.x, self.y, self.z, self.w

        xx, yy, zz = x * x, y * y, z * z

        xy, xz, yz = x * y, x * z, y * z

        wx, wy, wz = w * x, w * y, w * z

        return Mat4(
            (
                1 - 2 * (yy + zz),
                2 * (xy - wz),
                2 * (xz + wy),
                0,
                2 * (xy + wz),
                1 - 2 * (xx + zz),
                2 * (yz - wx),
                0,
                2 * (xz - wy),
                2 * (yz + wx),
                1 - 2 * (xx + yy),
                0,
                0,
                0,
                0,
                1,
            )
        )

    def to_mat3(self):
        """Upper-left 3x3 — normal matrix before inverse-transpose."""

        m = self.to_mat4().data

        return Mat3((m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]))

    def slerp(self, o, t):

        d = self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w

        if d < 0.0:
            o = Quat(-o.x, -o.y, -o.z, -o.w)

            d = -d

        if d > 0.9995:
            return Quat(
                self.x + (o.x - self.x) * t,
                self.y + (o.y - self.y) * t,
                self.z + (o.z - self.z) * t,
                self.w + (o.w - self.w) * t,
            ).normalize()

        theta = math.acos(d)

        st = sin(theta)

        a = sin((1.0 - t) * theta) / st

        b = sin(t * theta) / st

        return Quat(
            self.x * a + o.x * b,
            self.y * a + o.y * b,
            self.z * a + o.z * b,
            self.w * a + o.w * b,
        ).normalize()

    def __repr__(self):

        return f"Quat({self.x!r}, {self.y!r}, {self.z!r}, {self.w!r})"


# ---------------------------------------------------------------------------
# Basic transforms
# ---------------------------------------------------------------------------


def identity():

    return Mat4.identity()


def translate(v):

    return Mat4.from_rows(
        (1, 0, 0, v.x),
        (0, 1, 0, v.y),
        (0, 0, 1, v.z),
        (0, 0, 0, 1),
    )


def scale(v):

    return Mat4.from_rows(
        (v.x, 0, 0, 0),
        (0, v.y, 0, 0),
        (0, 0, v.z, 0),
        (0, 0, 0, 1),
    )


def rotate_x(rad):

    c, s = cos(rad), sin(rad)

    return Mat4.from_rows(
        (1, 0, 0, 0),
        (0, c, -s, 0),
        (0, s, c, 0),
        (0, 0, 0, 1),
    )


def rotate_y(rad):

    c, s = cos(rad), sin(rad)

    return Mat4.from_rows(
        (c, 0, s, 0),
        (0, 1, 0, 0),
        (-s, 0, c, 0),
        (0, 0, 0, 1),
    )


def rotate_z(rad):

    c, s = cos(rad), sin(rad)

    return Mat4.from_rows(
        (c, -s, 0, 0),
        (s, c, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )


def rotate_axis(axis, rad):

    return Quat.from_axis_angle(axis, rad).to_mat4()


def compose_trs(pos, rot, scale_vec):
    """T * R * S. rot may be a Quat or a Mat4."""

    rotation_matrix = rot.to_mat4() if isinstance(rot, Quat) else rot

    return translate(pos) @ rotation_matrix @ scale(scale_vec)


# ---------------------------------------------------------------------------
# Camera / view
# ---------------------------------------------------------------------------


def look_at(eye, target, up):
    """Right-handed view matrix. Camera looks down -Z in view space.

    Matches Blender's camera convention (forward = -Z) and Godot's
    Node3D.look_at() convention (right-handed, up defaults to +Y).
    """

    f = normalize(target - eye)  # world-space forward

    r = normalize(cross(f, up))  # right

    u = cross(r, f)  # corrected up

    # View space basis:  x=right, y=up, z=-forward  (camera looks down -Z)
    # Row i is the i-th view-space axis expressed in world coordinates.
    return Mat4.from_rows(
        (r.x, r.y, r.z, -dot(r, eye)),
        (u.x, u.y, u.z, -dot(u, eye)),
        (-f.x, -f.y, -f.z, dot(f, eye)),
        (0, 0, 0, 1),
    )


def perspective_reverse_z(fov_y_rad, aspect, near, far=None):
    """Reverse-Z perspective. far=None -> infinite-far form.

    NDC z in [0,1]: near plane -> z=1, infinity -> z=0.
    Clear depth to 0.0; test `z > depth` (reverse-Z comparison).

    Infinite-far form (far=None):
        clip.z is constant (= near).  All depth information is in clip.w.
        The only clipping constraint on z is `clip.z <= clip.w`
        (i.e. `near <= view_depth`), which is exactly the near-plane test.

    Finite-far form (far given):
        clip.z varies linearly with view depth.  Use this if downstream
        code (e.g. a frustum clipper) needs |clip.z| <= clip.w to be a
        meaningful distance test rather than a near-plane-only test.

    Args:
        fov_y_rad: vertical field of view, radians, in (0, pi).
        aspect:    width / height, > 0.
        near:      near-plane distance, > 0.
        far:       optional; far-plane distance, > near.

    Raises:
        ValueError on any nonsensical argument.
    """

    if not (near > 0.0):
        raise ValueError(f"near must be > 0 (got {near})")

    if far is not None and not (far > near):
        raise ValueError(f"far ({far}) must be > near ({near})")

    if not (aspect > 0.0):
        raise ValueError(f"aspect must be > 0 (got {aspect})")

    if not (0.0 < fov_y_rad < math.pi):
        raise ValueError(f"fov_y_rad ({fov_y_rad}) must be in (0, pi)")

    f = 1.0 / math.tan(fov_y_rad * 0.5)

    if far is None:
        # Infinite-far reverse-Z.

        return Mat4.from_rows(
            (f / aspect, 0, 0, 0),
            (0, f, 0, 0),
            (0, 0, 0, near),  # z_clip = near * w_clip
            (0, 0, -1, 0),  # w_clip = -z_view
        )

    # Finite-far reverse-Z.
    a = near / (far - near)  # -> 0 as far -> inf

    b = near * far / (far - near)  # -> near as far -> inf

    return Mat4.from_rows(
        (f / aspect, 0, 0, 0),
        (0, f, 0, 0),
        (0, 0, a, b),  # z_clip = a*z_view + b
        (0, 0, -1, 0),
    )


def ndc_perspective(x, y, z_view, f, aspect):
    """Standard perspective."""

    w = -z_view

    if w <= 1e-9:
        return (0.0, 0.0)

    return (f * x / (aspect * w), f * y / w)


def ndc_orthographic(x, y, z_view, half_h, aspect):
    """Orthographic. half_h = half vertical extent in world units."""

    return (x / (half_h * aspect), y / half_h)


def ndc_panini(x, y, z_view, f, aspect):
    """Sharpless panini projection (2014).

    Cylinder-wrapped horizontally, sphere-corrected vertically.
    Peripheral content is compressed; the center matches perspective.
    Verticals stay straight; horizontals curve inward at the edges.

    References: T. Sharpless, "The Panini Projection" (2014).
    """

    w = -z_view

    if w <= 1e-9:
        return (0.0, 0.0)

    r = math.sqrt(x * x + w * w)

    s = math.sqrt(x * x + y * y + w * w)

    if r < 1e-9 or s < 1e-9:
        return (0.0, 0.0)

    phi = math.atan2(x, w)  # horizontal angle

    theta = math.asin(min(1.0, r / s))  # radial angle

    norm = (2.0 / math.pi) * theta  # 0..1; keeps center perspective

    return (f * math.sin(phi) * norm / aspect, f * (y / r) * norm)


def ndc_ellipsoid(x, y, z_view, f, aspect, strength=0.5):
    """Blend between perspective (strength=0) and panini (strength=1).

    Any value in (0, 1) gives a smooth ellipsoid-flavoured projection
    with tunable peripheral compression.
    """

    w = -z_view

    if w <= 1e-9:
        return (0.0, 0.0)

    px = f * x / (aspect * w)

    py = f * y / w

    r = math.sqrt(x * x + w * w)

    s = math.sqrt(x * x + y * y + w * w)

    if r < 1e-9 or s < 1e-9:
        return (px, py)

    phi = math.atan2(x, w)

    theta = math.asin(min(1.0, r / s))

    norm = (2.0 / math.pi) * theta

    nx = f * math.sin(phi) * norm / aspect

    ny = f * (y / r) * norm

    k = 1.0 - strength

    return (px * k + nx * strength, py * k + ny * strength)


def viewport(x, y, w, h):
    """NDC [-1,1]^2 -> screen pixels. Flips Y: screen Y grows downward.

    Z passes through unchanged (reverse-Z: NDC z is already [0,1];
    near=1, far=0 — no remapping).
    """

    return Mat4.from_rows(
        (w * 0.5, 0, 0, x + w * 0.5),
        (0, -h * 0.5, 0, y + h * 0.5),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    )


def blender_zup_to_yup():
    """Convert Blender/3ds-Max Z-up world coordinates into our Y-up world.

    Mapping:  (x, y, z)_blender  ->  (x, z, -y)_yup
    Equivalent to -90 degrees about X.  This is the same convention
    Blender's own glTF exporter uses.

    What it preserves:
        Blender +Z (world up)      ->  +Y  (our up)
        Blender +X (right)         ->  +X  (our right)

    What it does *not* preserve:
        Blender's typical camera *aim* direction is -Y in world space;
        after this rotation that becomes +Z.  Our cameras (Godot-style)
        look down -Z.  If you need to *aim* a camera at something after
        conversion, either build the aim with look_at() in Y-up space
        directly, or apply rotate_y(math.pi) afterwards (which also
        flips X — usually undesirable, hence the caveat).
    """

    return Mat4.from_rows(
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    )


def yup_to_blender_zup():
    """Inverse of blender_zup_to_yup().  Rotation matrix -> transpose."""

    return blender_zup_to_yup().transpose()
