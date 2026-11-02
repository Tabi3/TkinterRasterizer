"""Vector, matrix, and transform primitives with explicit package exports."""

from .affine import write_trs
from .math3d import Mat3, Mat4, Vec2, Vec3, Vec4


__all__ = ["Mat3", "Mat4", "Vec2", "Vec3", "Vec4", "write_trs"]
