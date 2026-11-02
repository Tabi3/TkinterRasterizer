"""Reusable draw state. Spec: IMPLEMENTATION_SPEC.md#api-and-organization."""

from __future__ import annotations

from collections.abc import Callable

from ..algebra.math3d import Mat3, Mat4, Vec3, Vec4
from ..scene.scene import Light


class Uniforms:
    """Per-draw uniform state shared by the vertex and fragment shaders.

    The renderer fills these values each frame or each draw call: model/view
    matrices, camera position, and aspect ratio. Callers supply lights and
    timing (`time` and `delta_time`, both seconds). Keeping
    them separate from the Material itself ensures that a single material can be
    reused for many objects with different transforms and lighting states.
    """

    __slots__ = (
        "_normal_key",
        "_normal_matrix",
        "ambient",
        "aspect",
        "camera_pos",
        "delta_time",
        "lights",
        "model",
        "project",
        "time",
        "transform",
        "view",
    )

    def __init__(self) -> None:

        self._normal_key = None

        self._normal_matrix = Mat3.identity()

        self.model: Mat4 = Mat4.identity()

        self.view: Mat4 = Mat4.identity()

        self.project: Callable[[Vec4], Vec4] = _default_project

        self.transform: Callable[[Vec4], Vec4] = _default_transform

        self.aspect: float = 1.0

        self.camera_pos: Vec3 = Vec3.zero()

        # Timing is owned by the caller; draws never advance simulation time.
        self.delta_time: float = 0.0

        self.time: float = 0.0

        self.lights: list[Light] = []

        self.ambient: float = 0.1

    @property
    def normal_matrix(self) -> Mat3:
        """Inverse transpose, cached per model; singular models yield zero normals."""

        data = self.model.data

        if self._normal_key != data:
            linear = Mat3(
                (
                    data[0],
                    data[1],
                    data[2],
                    data[4],
                    data[5],
                    data[6],
                    data[8],
                    data[9],
                    data[10],
                )
            )

            self._normal_matrix = linear.inverse().transpose() if linear.determinant() != 0.0 else Mat3((0.0,) * 9)

            self._normal_key = data

        return self._normal_matrix


def _default_project(_world_pos: Vec4) -> Vec4:

    return Vec4(0, 0, 1, 1)


def _default_transform(_view_pos: Vec4) -> Vec4:

    return Vec4(0, 0, 1, 1)
