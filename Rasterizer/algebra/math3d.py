"""Pure-Python 3D linear algebra primitives.

The core math layer is intentionally dependency-free and designed to mirror the
minimum algebra needed by a software rasterizer: positions, normals, UVs,
transforms, and the projection math used by the camera stack. The object model
is value-oriented and side-effect free so the same vectors and matrices can be
reused across mesh generation, scene traversal, and raster output without hidden
state.

Spec:
    The engine uses a right-handed Cartesian convention with camera forward set
    to -Z in view space. Matrix storage is row-major, and operations are written
    to match the conventions used by the render pipeline and the implementation
    specification.

Contract:
    - Keep the types lightweight and allocation-friendly.
    - Preserve row-major flat storage for matrix data.
    - Treat vectors as simple values, not mutable scene objects.
    - Avoid scientific Python dependencies so the project remains portable.

Reference:
    See IMPLEMENTATION_SPEC.md for the full coordinate-system discussion,
    transform contract, and rendering-math derivations used across the project.
"""

from __future__ import annotations

from math import sqrt
from types import NotImplementedType
from typing import overload


def _assert_real(value, name):
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise TypeError(
            f"{name} must be a real numeric value, got {type(value).__name__}"
        )


def _assert_vec2_like(value, name):
    if not (hasattr(value, "x") and hasattr(value, "y")):
        raise TypeError(f"{name} must provide x/y components")
    _assert_real(value.x, f"{name}.x")
    _assert_real(value.y, f"{name}.y")


def _assert_vec3_like(value, name):
    if not (hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z")):
        raise TypeError(f"{name} must provide x/y/z components")
    _assert_real(value.x, f"{name}.x")
    _assert_real(value.y, f"{name}.y")
    _assert_real(value.z, f"{name}.z")


def _assert_vec4_like(value, name):
    if not (
        hasattr(value, "x")
        and hasattr(value, "y")
        and hasattr(value, "z")
        and hasattr(value, "w")
    ):
        raise TypeError(f"{name} must provide x/y/z/w components")
    _assert_real(value.x, f"{name}.x")
    _assert_real(value.y, f"{name}.y")
    _assert_real(value.z, f"{name}.z")
    _assert_real(value.w, f"{name}.w")


def _assert_matrix_data(data, rows, cols, name):
    if not isinstance(data, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of length {rows * cols}")
    if len(data) != rows * cols:
        raise ValueError(f"{name} must contain exactly {rows * cols} values")
    for i, item in enumerate(data):
        _assert_real(item, f"{name}[{i}]")


# ---------------------------------------------------------------------------
# Vec2
# ---------------------------------------------------------------------------
class Vec2:
    """Two-component vector for UVs, screen offsets, and 2D interpolation.

    The type is intentionally tiny and mirrors the 3D and 4D vector APIs so the
    renderer can share common operations such as normalization, dot products, and
    linear interpolation without branching on dimensionality.
    """

    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        _assert_real(x, "x")
        _assert_real(y, "y")
        self.x = float(x)
        self.y = float(y)

    def __add__(self, o):
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Vec2(self.x - o.x, self.y - o.y)

    def __neg__(self):
        return Vec2(-self.x, -self.y)

    def __mul__(self, s):
        if isinstance(s, (int, float)):
            return Vec2(self.x * s, self.y * s)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, s):
        return Vec2(self.x / s, self.y / s)

    def __eq__(self, o):
        return isinstance(o, Vec2) and self.x == o.x and self.y == o.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Vec2({self.x!r}, {self.y!r})"

    def dot(self, o):
        return self.x * o.x + self.y * o.y

    def length_sq(self):
        return self.x * self.x + self.y * self.y

    def length(self):
        return sqrt(self.length_sq())

    def normalize(self):
        l = self.length()
        return Vec2(self.x / l, self.y / l) if l > 0.0 else Vec2()

    def lerp(self, o, t):
        return Vec2(self.x + (o.x - self.x) * t, self.y + (o.y - self.y) * t)

    @classmethod
    def zero(cls):
        return cls(0.0, 0.0)

    @classmethod
    def one(cls):
        return cls(1.0, 1.0)


# ---------------------------------------------------------------------------
# Vec3
# ---------------------------------------------------------------------------
class Vec3:
    """Three-component vector used for positions, directions, and normals.

    The renderer treats Vec3 as the canonical geometric primitive for object
    space, world space, camera space, and shading calculations. Its operations
    stay deliberately simple so the math layer remains easy to reason about while
    avoiding costly abstractions in the hot path.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        _assert_real(x, "x")
        _assert_real(y, "y")
        _assert_real(z, "z")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, s):
        if isinstance(s, (int, float)):
            return Vec3(self.x * s, self.y * s, self.z * s)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, s):
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __eq__(self, o):
        return isinstance(o, Vec3) and self.x == o.x and self.y == o.y and self.z == o.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"Vec3({self.x!r}, {self.y!r}, {self.z!r})"

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def length_sq(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    def length(self):
        return sqrt(self.length_sq())

    def normalize(self):
        l = self.length()
        return Vec3(self.x / l, self.y / l, self.z / l) if l > 0.0 else Vec3()

    def lerp(self, o, t):
        return Vec3(
            self.x + (o.x - self.x) * t,
            self.y + (o.y - self.y) * t,
            self.z + (o.z - self.z) * t,
        )

    def to_vec4(self, w=1.0):
        return Vec4(self.x, self.y, self.z, w)

    @classmethod
    def zero(cls):
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def one(cls):
        return cls(1.0, 1.0, 1.0)

    @classmethod
    def unit_x(cls):
        return cls(1.0, 0.0, 0.0)

    @classmethod
    def unit_y(cls):
        return cls(0.0, 1.0, 0.0)

    @classmethod
    def unit_z(cls):
        return cls(0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Vec4
# ---------------------------------------------------------------------------
class Vec4:
    """Homogeneous 4D vector for affine transforms and clip-space coordinates.

    This type is used for projective math: transformed points carry a w component,
    and perspective divide occurs only when the render pipeline needs to convert
    from clip space to normalized device coordinates. The w term is therefore a
    crucial part of the camera and rasterizer contract.
    """

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        _assert_real(x, "x")
        _assert_real(y, "y")
        _assert_real(z, "z")
        _assert_real(w, "w")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def __add__(self, o):
        return Vec4(self.x + o.x, self.y + o.y, self.z + o.z, self.w + o.w)

    def __sub__(self, o):
        return Vec4(self.x - o.x, self.y - o.y, self.z - o.z, self.w - o.w)

    def __neg__(self):
        return Vec4(-self.x, -self.y, -self.z, -self.w)

    def __mul__(self, s):
        if isinstance(s, (int, float)):
            return Vec4(self.x * s, self.y * s, self.z * s, self.w * s)
        return NotImplemented

    __rmul__ = __mul__

    def __truediv__(self, s):
        return Vec4(self.x / s, self.y / s, self.z / s, self.w / s)

    def __eq__(self, o):
        return (
            isinstance(o, Vec4)
            and self.x == o.x
            and self.y == o.y
            and self.z == o.z
            and self.w == o.w
        )

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.w))

    def __repr__(self):
        return f"Vec4({self.x!r}, {self.y!r}, {self.z!r}, {self.w!r})"

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z + self.w * o.w

    def to_vec3(self):
        """Perspective divide — caller should clip so w > 0 first."""
        inv = 1.0 / self.w
        return Vec3(self.x * inv, self.y * inv, self.z * inv)

    def to_xyz(self):
        """Drop w without dividing — for directions."""
        return Vec3(self.x, self.y, self.z)

    @classmethod
    def from_vec3(cls, v, w=1.0):
        return cls(v.x, v.y, v.z, w)


# ---------------------------------------------------------------------------
# Mat3 — 3x3, flat 9 floats, row-major
# ---------------------------------------------------------------------------
class Mat3:
    __slots__ = ("data",)

    def __init__(self, data):
        _assert_matrix_data(data, 3, 3, "Mat3")
        self.data = tuple(data)

    @classmethod
    def identity(cls):
        return cls((1, 0, 0, 0, 1, 0, 0, 0, 1))

    @classmethod
    def from_rows(cls, r0, r1, r2):
        return cls(tuple(r0) + tuple(r1) + tuple(r2))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self.data[i * 3 + j]
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __eq__(self, o):
        return isinstance(o, Mat3) and self.data == o.data

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        return f"Mat3({self.data!r})"

    @overload
    def __matmul__(self, o: Vec3, /) -> Vec3: ...

    @overload
    def __matmul__(self, o: Mat3, /) -> Mat3: ...

    def __matmul__(self, o: object) -> Vec3 | Mat3 | NotImplementedType:
        a = self.data
        if isinstance(o, Mat3):
            b = o.data
            return Mat3(
                (
                    a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
                    a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
                    a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
                    a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
                    a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
                    a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
                    a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
                    a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
                    a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
                )
            )
        if isinstance(o, Vec3):
            return Vec3(
                a[0] * o.x + a[1] * o.y + a[2] * o.z,
                a[3] * o.x + a[4] * o.y + a[5] * o.z,
                a[6] * o.x + a[7] * o.y + a[8] * o.z,
            )
        return NotImplemented

    def transpose(self):
        d = self.data
        return Mat3((d[0], d[3], d[6], d[1], d[4], d[7], d[2], d[5], d[8]))

    def determinant(self):
        a = self.data
        return (
            a[0] * (a[4] * a[8] - a[5] * a[7])
            - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6])
        )

    def inverse(self):
        a = self.data
        det = self.determinant()
        if det == 0.0:
            raise ValueError("Mat3 is singular")
        inv = 1.0 / det
        return Mat3(
            (
                (a[4] * a[8] - a[5] * a[7]) * inv,
                (a[2] * a[7] - a[1] * a[8]) * inv,
                (a[1] * a[5] - a[2] * a[4]) * inv,
                (a[5] * a[6] - a[3] * a[8]) * inv,
                (a[0] * a[8] - a[2] * a[6]) * inv,
                (a[2] * a[3] - a[0] * a[5]) * inv,
                (a[3] * a[7] - a[4] * a[6]) * inv,
                (a[1] * a[6] - a[0] * a[7]) * inv,
                (a[0] * a[4] - a[1] * a[3]) * inv,
            )
        )


# ---------------------------------------------------------------------------
# Mat4 — 4x4, flat 16 floats, row-major
# ---------------------------------------------------------------------------
class Mat4:
    __slots__ = ("data",)

    def __init__(self, data):
        _assert_matrix_data(data, 4, 4, "Mat4")
        self.data = tuple(data)

    def mul_vec4_into(self, x: float, y: float, z: float, w: float, out: Vec4) -> None:
        d = self.data
        out.x = d[0] * x + d[1] * y + d[2] * z + d[3] * w
        out.y = d[4] * x + d[5] * y + d[6] * z + d[7] * w
        out.z = d[8] * x + d[9] * y + d[10] * z + d[11] * w
        out.w = d[12] * x + d[13] * y + d[14] * z + d[15] * w

    @classmethod
    def identity(cls):
        return cls((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))

    @classmethod
    def from_rows(cls, r0, r1, r2, r3):
        return cls(tuple(r0) + tuple(r1) + tuple(r2) + tuple(r3))

    @classmethod
    def from_cols(cls, c0, c1, c2, c3):
        return cls(
            (
                c0[0],
                c1[0],
                c2[0],
                c3[0],
                c0[1],
                c1[1],
                c2[1],
                c3[1],
                c0[2],
                c1[2],
                c2[2],
                c3[2],
                c0[3],
                c1[3],
                c2[3],
                c3[3],
            )
        )

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self.data[i * 4 + j]
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __eq__(self, o):
        return isinstance(o, Mat4) and self.data == o.data

    def __hash__(self):
        return hash(self.data)

    def __repr__(self):
        return f"Mat4({self.data!r})"

    def row(self, i):
        d = self.data
        return (d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3])

    def col(self, j):
        d = self.data
        return (d[j], d[j + 4], d[j + 8], d[j + 12])

    @overload
    def __matmul__(self, o: Vec4, /) -> Vec4: ...

    @overload
    def __matmul__(self, o: Mat4, /) -> Mat4: ...

    def __matmul__(self, o: object) -> Vec4 | Mat4 | NotImplementedType:
        a = self.data
        if isinstance(o, Mat4):
            b = o.data
            return Mat4(
                (
                    a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
                    a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
                    a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
                    a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],
                    a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
                    a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
                    a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
                    a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],
                    a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
                    a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
                    a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
                    a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],
                    a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
                    a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
                    a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
                    a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15],
                )
            )
        if isinstance(o, Vec4):
            x, y, z, w = o.x, o.y, o.z, o.w
            return Vec4(
                a[0] * x + a[1] * y + a[2] * z + a[3] * w,
                a[4] * x + a[5] * y + a[6] * z + a[7] * w,
                a[8] * x + a[9] * y + a[10] * z + a[11] * w,
                a[12] * x + a[13] * y + a[14] * z + a[15] * w,
            )
        return NotImplemented

    def __mul__(self, s):
        if isinstance(s, (int, float)):
            return Mat4(tuple(v * s for v in self.data))
        return NotImplemented

    __rmul__ = __mul__

    def transpose(self):
        d = self.data
        return Mat4(
            (
                d[0],
                d[4],
                d[8],
                d[12],
                d[1],
                d[5],
                d[9],
                d[13],
                d[2],
                d[6],
                d[10],
                d[14],
                d[3],
                d[7],
                d[11],
                d[15],
            )
        )

    def inverse(self):
        """Closed-form 4x4 inverse (Lengyel, Mathematics for 3D Game Programming)."""
        m = self.data

        s0 = m[0] * m[5] - m[1] * m[4]
        s1 = m[0] * m[6] - m[2] * m[4]
        s2 = m[0] * m[7] - m[3] * m[4]
        s3 = m[1] * m[6] - m[2] * m[5]
        s4 = m[1] * m[7] - m[3] * m[5]
        s5 = m[2] * m[7] - m[3] * m[6]

        c5 = m[10] * m[15] - m[11] * m[14]
        c4 = m[9] * m[15] - m[11] * m[13]
        c3 = m[9] * m[14] - m[10] * m[13]
        c2 = m[8] * m[15] - m[11] * m[12]
        c1 = m[8] * m[14] - m[10] * m[12]
        c0 = m[8] * m[13] - m[9] * m[12]

        det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
        if det == 0.0:
            raise ValueError("Mat4 is singular")
        inv = 1.0 / det

        return Mat4(
            (
                (m[5] * c5 - m[6] * c4 + m[7] * c3) * inv,
                (-m[1] * c5 + m[2] * c4 - m[3] * c3) * inv,
                (m[13] * s5 - m[14] * s4 + m[15] * s3) * inv,
                (-m[9] * s5 + m[10] * s4 - m[11] * s3) * inv,
                (-m[4] * c5 + m[6] * c2 - m[7] * c1) * inv,
                (m[0] * c5 - m[2] * c2 + m[3] * c1) * inv,
                (-m[12] * s5 + m[14] * s2 - m[15] * s1) * inv,
                (m[8] * s5 - m[10] * s2 + m[11] * s1) * inv,
                (m[4] * c4 - m[5] * c2 + m[7] * c0) * inv,
                (-m[0] * c4 + m[1] * c2 - m[3] * c0) * inv,
                (m[12] * s4 - m[13] * s2 + m[15] * s0) * inv,
                (-m[8] * s4 + m[9] * s2 - m[11] * s0) * inv,
                (-m[4] * c3 + m[5] * c1 - m[6] * c0) * inv,
                (m[0] * c3 - m[1] * c1 + m[2] * c0) * inv,
                (-m[12] * s3 + m[13] * s1 - m[14] * s0) * inv,
                (m[8] * s3 - m[9] * s1 + m[10] * s0) * inv,
            )
        )


# ---------------------------------------------------------------------------
# Free-function aliases — faster in tight code, and easier to pass around.
# ---------------------------------------------------------------------------
def dot(a, b):
    return a.dot(b)


def cross(a, b):
    return a.cross(b)


def normalize(v):
    return v.normalize()


def length(v):
    return v.length()


def length_sq(v):
    return v.length_sq()


def lerp(a, b, t):
    return a.lerp(b, t)


def clamp(x, lo, hi):
    return lo if x < lo else min(x, hi)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _mat_almost_eq(a, b, eps=1e-12):
    return all(abs(x - y) < eps for x, y in zip(a.data, b.data))
