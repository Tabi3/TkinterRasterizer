"""Camera and projection policy. Spec: IMPLEMENTATION_SPEC.md#numerical-conventions."""

from __future__ import annotations

import math

from ..algebra.math3d import Mat4, Vec3, Vec4
from ..algebra.transforms import look_at, ndc_ellipsoid, ndc_orthographic, ndc_panini
from ..data.conventions import PROJECTIONS


class Camera:
    """Camera abstraction used by the public rendering API.

    The camera owns the view transform, the projection mode, and the near plane
    convention used across the renderer. While the view matrix is linear, the
    nonlinear projection modes (panini and ellipsoid) are handled in the
    per-vertex camera.project method rather than a single baked matrix.
    """

    __slots__ = (
        "_view_cache",
        "_view_key",
        "ellipsoid_strength",
        "far",
        "fov_y",
        "near",
        "ortho_scale",
        "position",
        "projection",
        "target",
        "up",
    )

    def __init__(
        self,
        position: Vec3,
        target: Vec3,
        up: Vec3 | None = None,
        fov_y_deg: float = 60.0,
        near: float = 0.1,
        projection: str = "perspective",
        ellipsoid_strength: float = 0.5,
        ortho_scale: float = 3.0,
        far: float = 1000.0,
    ) -> None:

        if not isinstance(position, Vec3):
            raise TypeError("position must be a Vec3")

        if not isinstance(target, Vec3):
            raise TypeError("target must be a Vec3")

        if not (up is None or isinstance(up, Vec3)):
            raise TypeError("up must be a Vec3 or None")

        if not (isinstance(fov_y_deg, (int, float)) and not isinstance(fov_y_deg, bool)):
            raise TypeError("fov_y_deg must be a real number")

        if not (isinstance(near, (int, float)) and not isinstance(near, bool)):
            raise TypeError("near must be a real number")

        if projection not in PROJECTIONS:
            raise ValueError("projection must be one of 'perspective', 'orthographic', 'panini', 'ellipsoid'")

        for name, value in (
            ("near", near),
            ("far", far),
            ("fov_y_deg", fov_y_deg),
            ("ortho_scale", ortho_scale),
            ("ellipsoid_strength", ellipsoid_strength),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")

            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if near <= 0 or far <= near:
            raise ValueError("require 0 < near < far")

        if not 0 < fov_y_deg < 180 or ortho_scale <= 0:
            raise ValueError("require 0 < fov_y_deg < 180 and ortho_scale > 0")

        if not 0 <= ellipsoid_strength <= 1:
            raise ValueError("ellipsoid_strength must be in [0, 1]")

        self.far    = far
        self.near   = near
        self.fov_y  = math.radians(fov_y_deg)

        self.position   = position
        self.target     = target
        self.up         = up if up is not None else Vec3.unit_y()

        self.projection         = projection
        self.ellipsoid_strength = ellipsoid_strength
        self.ortho_scale        = ortho_scale

        self._view_cache: Mat4 | None               = None
        self._view_key: tuple[float, ...] | None    = None

    def view_matrix(self) -> Mat4:

        key = self._view_key

        if (
            key is None
            or self._view_cache is None
            or key[0] != self.position.x
            or key[1] != self.position.y
            or key[2] != self.position.z
            or key[3] != self.target.x
            or key[4] != self.target.y
            or key[5] != self.target.z
            or key[6] != self.up.x
            or key[7] != self.up.y
            or key[8] != self.up.z
        ):
            self._view_key = (
                self.position.x,
                self.position.y,
                self.position.z,
                self.target.x,
                self.target.y,
                self.target.z,
                self.up.x,
                self.up.y,
                self.up.z,
            )

            self._view_cache = look_at(self.position, self.target, self.up)

        return self._view_cache

    def project(self, view_pos: Vec4, aspect: float) -> Vec4:
        """Map view space to clip coordinates.

        Perspective and nonlinear modes use w=-Z and depth=near/w.
        Orthographic uses w=1 and linear reverse depth over [near, far],
        ensuring affine attribute interpolation. Nonlinear filled triangles
        remain straight-chord approximations; see IMPLEMENTATION_SPEC.md.
        """

        x, y, z = view_pos.x, view_pos.y, view_pos.z

        if not math.isfinite(aspect) or aspect <= 0:
            raise ValueError("aspect must be finite and positive")

        if self.projection == "orthographic":
            nx, ny = ndc_orthographic(x, y, z, self.ortho_scale, aspect)

            return Vec4(nx, ny, (self.far + z) / (self.far - self.near), 1.0)

        w = -z

        f = 1.0 / math.tan(self.fov_y * 0.5)

        if self.projection == "perspective":
            # Preserve behind-eye coordinates for homogeneous near clipping.

            return Vec4(f * x / aspect, f * y, self.near, w)

        if self.projection == "panini":
            nx, ny = ndc_panini(x, y, z, f, aspect)

        elif self.projection == "ellipsoid":
            nx, ny = ndc_ellipsoid(x, y, z, f, aspect, self.ellipsoid_strength)

        else:
            raise ValueError(f"unknown projection: {self.projection!r}")

        return Vec4(nx * w, ny * w, self.near, w)

    def mvp_matrix(self, model: Mat4, aspect: float) -> Mat4 | None:
        """Return the baked perspective MVP, or None for other modes.

        Orthographic currently uses project() for its finite linear depth.
        Panini and ellipsoid are nonlinear and cannot be baked into a matrix.
        """

        if self.projection != "perspective":
            return None

        from ..algebra.transforms import perspective_reverse_z

        proj = perspective_reverse_z(self.fov_y, aspect, self.near)

        return proj @ self.view_matrix() @ model
