"""Explicit mutable transform output; ordinary Mat4 operators remain value-based.

Spec: IMPLEMENTATION_SPEC.md#allocation-and-mutable-scratch.
"""

from math import cos, sin

from .math3d import Mat4


def write_trs(
    output: Mat4,
    *,
    x: float = 0,
    y: float = 0,
    z: float = 0,
    yaw: float = 0,
    pitch: float = 0,
    roll: float = 0,
    sx: float = 1,
    sy: float = 1,
    sz: float = 1,
) -> Mat4:
    """Overwrite output with T * Ry * Rx * Rz * S and return the same object.

    Callers must own output exclusively while it is being updated. Data is
    replaced with a tuple snapshot, preserving normal-matrix cache invalidation
    and the immutable data contract of ordinary matrix values.
    """

    # Spec: IMPLEMENTATION_SPEC.md#allocation-and-mutable-scratch.
    # Compute each sine/cosine once, avoiding temporary rotation matrices.
    cy, syaw = cos(yaw), sin(yaw)

    cx, sp = cos(pitch), sin(pitch)

    cr, sr = cos(roll), sin(roll)

    r00 = cy * cr + syaw * sp * sr

    r01 = -cy * sr + syaw * sp * cr

    r02 = syaw * cx

    r10 = cx * sr

    r11 = cx * cr

    r12 = -sp

    r20 = -syaw * cr + cy * sp * sr

    r21 = syaw * sr + cy * sp * cr

    r22 = cy * cx

    # Column scaling and translation complete the affine transform directly.
    output.data = (
        r00 * sx,
        r01 * sy,
        r02 * sz,
        x,
        r10 * sx,
        r11 * sy,
        r12 * sz,
        y,
        r20 * sx,
        r21 * sy,
        r22 * sz,
        z,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    return output
