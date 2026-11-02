"""Public rendering API for the Rasterizer library.

Purpose:
    Expose the stable camera, object, and renderer surface used by the demos and
    library consumers while keeping the implementation details in dedicated math
    and backend modules.

Contract:
    - Camera and object transforms define the public scene API.
    - Renderer.draw_* methods accept a framebuffer, camera, and mesh data.
    - Clipping, culling, and rasterization are delegated to the backend layer.

Behavior:
    The render pipeline is model -> view -> projection -> clip -> divide ->
    viewport -> cull -> rasterize, and remains consistent with the reverse-Z
    framebuffer conventions used throughout the project.

Reference:
    See IMPLEMENTATION_SPEC.md for the public API boundary and the renderer's
    role within the refactored architecture.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from ..algebra.math3d import Mat3, Mat4, Vec3, Vec4
from ..algebra.transforms import (
    look_at,
    ndc_ellipsoid,
    ndc_orthographic,
    ndc_panini,
    viewport,
)
from ..scene.scene import Light
from .backend import (
    aabb_visible,
    clip_near_shaded,
    clip_near_triangle,
    cull_and_raster,
    fill_gradient,
)
from .rasterizer import draw_line, rasterize_triangle_shaded

# ---------------------------------------------------------------------------


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
            try:
                self._normal_matrix = linear.inverse().transpose()
            except ValueError:
                self._normal_matrix = Mat3((0.0,) * 9)
            self._normal_key = data
        return self._normal_matrix


def _default_project(_world_pos: Vec4) -> Vec4:
    return Vec4(0, 0, 1, 1)


def _default_transform(_view_pos: Vec4) -> Vec4:
    return Vec4(0, 0, 1, 1)


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
        if not (
            isinstance(fov_y_deg, (int, float)) and not isinstance(fov_y_deg, bool)
        ):
            raise TypeError("fov_y_deg must be a real number")
        if not (isinstance(near, (int, float)) and not isinstance(near, bool)):
            raise TypeError("near must be a real number")
        if projection not in {"perspective", "orthographic", "panini", "ellipsoid"}:
            raise ValueError(
                "projection must be one of 'perspective', 'orthographic', "
                "'panini', 'ellipsoid'"
            )
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
        self.far = far
        self.position = position
        self.target = target
        self.up = up if up is not None else Vec3.unit_y()
        self.fov_y = math.radians(fov_y_deg)
        self.near = near
        self.projection = projection
        self.ellipsoid_strength = ellipsoid_strength
        self.ortho_scale = ortho_scale

        self._view_cache: Mat4 | None = None
        self._view_key: tuple[float, ...] | None = None

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


class Object3D:
    """Simple object wrapper for easy draw calls from the public API.

    This is a convenience layer around a mesh and a transform matrix. It keeps
    the library user-facing surface compact without making the lower-level render
    path more complicated.
    """

    __slots__ = ("color", "mesh", "transform")

    def __init__(
        self,
        mesh: Any,
        transform: Mat4 | None = None,
        color: int = 0xFFFFFF,
    ) -> None:
        self.mesh = mesh
        self.transform = transform if transform is not None else Mat4.identity()
        self.color = color


# ---------------------------------------------------------------------------
class Renderer:
    """Public renderer surface that orchestrates scene drawing.

    The API keeps the high-level operations exposed to application code small and
    consistent, while the actual clipping, culling, and rasterization logic is
    delegated to the backend modules. This preserves the repository's layering
    goals without exposing the internal implementation details to users.
    """

    def __init__(self, width: int, height: int) -> None:
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
        ):
            raise TypeError("renderer dimensions must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("renderer dimensions must be positive")
        self.width = width
        self.height = height
        self._aspect = width / height
        self._viewport = viewport(0, 0, width, height)

    def _validate_target(self, fb: Any) -> None:
        if (fb.width, fb.height) != (self.width, self.height):
            raise ValueError("framebuffer dimensions must match the renderer")

    def clear(self, fb: Any, color: int) -> None:
        self._validate_target(fb)
        fb.clear(color)

    # ----- full pipeline -------------------------------------------------
    def draw_object(self, fb: Any, obj: Object3D, camera: Camera) -> None:
        self.draw_mesh(fb, obj.mesh, obj.transform, obj.color, camera)

    def draw_mesh(
        self, fb: Any, mesh: Any, model_matrix: Mat4, color: int, camera: Camera
    ) -> None:
        """Unshaded draw. Same clip convention as draw_mesh_shaded:
        per-vertex camera-space position -> camera.project() -> clip.

        Reference:
            IMPLEMENTATION_SPEC.md > Public API boundary > Renderer and > Core architecture > 2. Rendering and backend layer
        """
        self._validate_target(fb)
        aspect = self._aspect
        view = camera.view_matrix()
        near = camera.near
        vp = self._viewport
        clip: list[Vec4] = []
        mvp = camera.mvp_matrix(model_matrix, aspect)
        if mvp is not None:
            for vertex in mesh.vertices:
                pos = vertex.pos
                clip_vertex = Vec4(0, 0, 0, 0)
                mvp.mul_vec4_into(pos.x, pos.y, pos.z, 1.0, clip_vertex)
                clip.append(clip_vertex)
        else:
            for vertex in mesh.vertices:
                world = model_matrix @ vertex.pos.to_vec4(1.0)
                clip.append(camera.project(view @ world, aspect))

        indices = iter(mesh.indices)
        for i0, i1, i2 in zip(indices, indices, indices):
            c0, c1, c2 = clip[i0], clip[i1], clip[i2]

            if all(_inside_depth(c, camera) for c in (c0, c1, c2)):
                n0 = (c0.x / c0.w, c0.y / c0.w, c0.z / c0.w)
                n1 = (c1.x / c1.w, c1.y / c1.w, c1.z / c1.w)
                n2 = (c2.x / c2.w, c2.y / c2.w, c2.z / c2.w)
                self._cull_and_raster(fb, vp, n0, n1, n2, color)
                continue

            for a, b, cc in clip_near_triangle(
                c0, c1, c2, near, orthographic=camera.projection == "orthographic"
            ):
                n0 = (a.x / a.w, a.y / a.w, a.z / a.w)
                n1 = (b.x / b.w, b.y / b.w, b.z / b.w)
                n2 = (cc.x / cc.w, cc.y / cc.w, cc.z / cc.w)
                self._cull_and_raster(fb, vp, n0, n1, n2, color)

    def draw_mesh_shaded(
        self,
        fb: Any,
        mesh: Any,
        vs: Callable[[Any, Uniforms], tuple[Vec4, tuple[float, ...]]],
        fs: Callable[[tuple[float, ...]], int],
        uniforms: Uniforms,
        camera: Camera,
        model: Mat4,
        cull: bool = True,
        *,
        depth_write: bool = True,
    ) -> None:
        self._validate_target(fb)
        aspect = self._aspect
        u = uniforms
        u.aspect = aspect
        u.model = model
        u.view = camera.view_matrix()
        u.camera_pos = camera.position
        view = u.view

        mvp = camera.mvp_matrix(model, aspect)
        if mvp is not None:
            d = mvp.data

            def transform(vertex_pos):
                x, y, z = vertex_pos.x, vertex_pos.y, vertex_pos.z
                return Vec4(
                    d[0] * x + d[1] * y + d[2] * z + d[3],
                    d[4] * x + d[5] * y + d[6] * z + d[7],
                    d[8] * x + d[9] * y + d[10] * z + d[11],
                    d[12] * x + d[13] * y + d[14] * z + d[15],
                )

            u.transform = transform
        else:

            def transform(vertex_pos):
                world = model @ vertex_pos.to_vec4(1.0)
                return camera.project(view @ world, aspect)

            u.transform = transform

        def project_world(world_pos):
            return camera.project(view @ world_pos, aspect)

        u.project = project_world

        near = camera.near
        vp = self._viewport
        screen_vertex = Vec4(0, 0, 0, 0)

        vertices = mesh.vertices
        if not vertices:
            return
        clip, varyings = zip(*[vs(vert, u) for vert in vertices])

        indices = iter(mesh.indices)
        for i0, i1, i2 in zip(indices, indices, indices):
            c0, c1, c2 = clip[i0], clip[i1], clip[i2]
            v0, v1, v2 = varyings[i0], varyings[i1], varyings[i2]

            if all(_inside_depth(c, camera) for c in (c0, c1, c2)):
                clipped = [((c0, v0), (c1, v1), (c2, v2))]
            else:
                clipped = clip_near_shaded(
                    [(c0, v0), (c1, v1), (c2, v2)],
                    near,
                    orthographic=camera.projection == "orthographic",
                )

            for (tc0, tv0), (tc1, tv1), (tc2, tv2) in clipped:
                vp.mul_vec4_into(
                    tc0.x / tc0.w,
                    tc0.y / tc0.w,
                    tc0.z / tc0.w,
                    1.0,
                    screen_vertex,
                )
                screen_0_x, screen_0_y, screen_0_z = (
                    screen_vertex.x,
                    screen_vertex.y,
                    screen_vertex.z,
                )
                vp.mul_vec4_into(
                    tc1.x / tc1.w,
                    tc1.y / tc1.w,
                    tc1.z / tc1.w,
                    1.0,
                    screen_vertex,
                )
                screen_1_x, screen_1_y, screen_1_z = (
                    screen_vertex.x,
                    screen_vertex.y,
                    screen_vertex.z,
                )
                vp.mul_vec4_into(
                    tc2.x / tc2.w,
                    tc2.y / tc2.w,
                    tc2.z / tc2.w,
                    1.0,
                    screen_vertex,
                )
                screen_2_x, screen_2_y, screen_2_z = (
                    screen_vertex.x,
                    screen_vertex.y,
                    screen_vertex.z,
                )

                if cull and (
                    (screen_1_x - screen_0_x) * (screen_2_y - screen_0_y)
                    - (screen_2_x - screen_0_x) * (screen_1_y - screen_0_y)
                    >= 0.0
                ):
                    continue

                rasterize_triangle_shaded(
                    fb,
                    (screen_0_x, screen_0_y, screen_0_z, 1.0 / tc0.w),
                    (screen_1_x, screen_1_y, screen_1_z, 1.0 / tc1.w),
                    (screen_2_x, screen_2_y, screen_2_z, 1.0 / tc2.w),
                    tv0,
                    tv1,
                    tv2,
                    fs,
                    depth_write=depth_write,
                )

    def draw_scene(
        self, fb: Any, root: Any, camera: Camera, uniforms: Uniforms
    ) -> None:
        """Traverse, cull, and draw every renderable node."""
        self._validate_target(fb)

        def draw_node(node):
            if self._is_visible(node, camera):
                self.draw_mesh_shaded(
                    fb,
                    node.mesh,
                    node.material.vs,
                    node.material.fs,
                    uniforms,
                    camera,
                    node.world_transform,
                    cull=node.material.cull,
                    depth_write=node.material.depth_write,
                )

        root.traverse(Mat4.identity(), draw_node)

    def draw_instanced(
        self,
        fb: Any,
        mesh: Any,
        material: Any,
        uniforms: Uniforms,
        camera: Camera,
        transforms: Sequence[Mat4],
    ) -> None:
        """Draw `mesh` once per model matrix in `transforms`.

        Convenience wrapper — no measurable performance gain over calling
        draw_mesh_shaded in a loop. The GPU-level win for instancing depends
        on hardware vertex-shader caching, which does not apply to a Python
        rasterizer.
        """
        self._validate_target(fb)
        for xform in transforms:
            self.draw_mesh_shaded(
                fb,
                mesh,
                material.vs,
                material.fs,
                uniforms,
                camera,
                xform,
                cull=material.cull,
                depth_write=material.depth_write,
            )

    def draw_skybox(
        self, fb: Any, color_top: int = 0x5070A0, color_bottom: int = 0x101828
    ) -> None:
        """Fill the framebuffer with a vertical sky gradient.

        Spec:
            Replacement for clear(). Depth is reset to 0 (reverse-Z far), so
            any subsequently drawn geometry (z > 0) wins the depth test.

        Contract:
            For perspective, the gradient matches the visual horizon. For
            panini/ellipsoid, it's an approximation (gradient ignores the
            horizontal warp); acceptable because the sky has no straight-line
            reference features.

        Args:
            color_top: color at the top of the screen.
            color_bottom: color at the bottom.
        """
        self._validate_target(fb)
        fill_gradient(fb, color_top, color_bottom)

    def draw_mesh_wireframe(
        self,
        fb: Any,
        mesh: Any,
        model: Mat4,
        color: int,
        camera: Camera,
        curve_segments: int = 1,
    ) -> None:
        """Draw mesh edges as lines.

        Spec:
            Per-edge 3D-to-screen projection, then DDA line fill.

        Contract:
            curve_segments=1 gives straight-chord edges (correct for perspective
            and orthographic, which are projective). curve_segments > 1 samples
            the 3D edge in N segments before projecting — needed for panini and
            ellipsoid, where straight 3D edges project to curves.

        Args:
            curve_segments: 1 for perspective/orthographic, 12-16 for
                panini/ellipsoid at high fov.
        """
        self._validate_target(fb)
        if isinstance(curve_segments, bool) or not isinstance(curve_segments, int):
            raise TypeError("curve_segments must be an integer")
        if curve_segments <= 0:
            raise ValueError("curve_segments must be positive")
        view = camera.view_matrix()
        aspect = self._aspect
        vp = self._viewport
        verts = mesh.vertices

        for i0, i1 in mesh.edges():
            p0 = verts[i0].pos
            p1 = verts[i1].pos
            prev = None
            for k in range(curve_segments + 1):
                t = k / curve_segments
                wx = p0.x + (p1.x - p0.x) * t
                wy = p0.y + (p1.y - p0.y) * t
                wz = p0.z + (p1.z - p0.z) * t
                world = model @ Vec4(wx, wy, wz, 1.0)
                current = view @ world
                if prev is not None:
                    segment = _clip_view_segment(prev, current, camera)
                    if segment is not None:
                        screen = []
                        for endpoint in segment:
                            clip = camera.project(endpoint, aspect)
                            inv_w = 1.0 / clip.w
                            point = vp @ Vec4(
                                clip.x * inv_w, clip.y * inv_w, clip.z * inv_w, 1.0
                            )
                            screen.append(point)
                        a, b = screen
                        draw_line(fb, a.x, a.y, a.z, b.x, b.y, b.z, color)
                prev = current

    def draw_scene_wireframe(
        self,
        fb: Any,
        root: Any,
        camera: Camera,
        uniforms: Uniforms,
        color: int = 0xFFFFFF,
        curve_segments: int = 1,
    ) -> None:
        """Wireframe render of a scene graph.

        Spec:
            Same traversal + cull as draw_scene; draws edges instead of filled
            triangles. Call after draw_scene to overlay, or instead of it for
            wireframe-only mode.

        Contract:
            Uses reverse-Z depth test, so wireframe pixels are only written
            where the line's depth is greater than existing depth. Overlaying
            on top of a filled scene therefore shows interior edges only if
            the wireframe lines are closer than the surfaces they sit on.
            For a "see-all-edges" overlay, clear depth first.
        """
        view = camera.view_matrix()

        def visit(node):
            if node.mesh is None:
                return
            if not aabb_visible(node.world_aabb(), view, camera.near):
                return
            self.draw_mesh_wireframe(
                fb,
                node.mesh,
                node.world_transform,
                color,
                camera,
                curve_segments=curve_segments,
            )

        root.traverse(Mat4.identity(), visit)

    # ----- NDC-direct (Vulkan triangle path) -----------------------------
    def draw_ndc_triangle(
        self,
        fb: Any,
        n0: tuple[float, float, float],
        n1: tuple[float, float, float],
        n2: tuple[float, float, float],
        color: int,
        cull: bool = True,
    ) -> None:
        """Draw an NDC triangle. cull=True (default) rejects back-facing triangles.

        For tests that only exercise rasterization or depth, pass cull=False
        to bypass the winding check.
        """
        self._validate_target(fb)
        self._cull_and_raster(fb, self._viewport, n0, n1, n2, color, cull=cull)

    def _cull_and_raster(
        self,
        fb: Any,
        vp: Any,
        n0: tuple[float, float, float],
        n1: tuple[float, float, float],
        n2: tuple[float, float, float],
        color: int,
        cull: bool = True,
    ) -> None:
        cull_and_raster(fb, vp, n0, n1, n2, color, cull=cull)

    def _is_visible(self, node: Any, camera: Camera) -> bool:
        """True if `node` (requires world_transform current) can be drawn."""
        if node.mesh is None or node.material is None:
            return False
        return aabb_visible(node.world_aabb(), camera.view_matrix(), camera.near)


def _clip_near_triangle(c0, c1, c2, near):
    """Compatibility wrapper around the backend clip implementation."""
    return clip_near_triangle(c0, c1, c2, near)


def _clip_near_shaded(verts, near):
    """Compatibility wrapper around the backend shaded-clip implementation."""
    return clip_near_shaded(verts, near)


def _inside_depth(vertex: Vec4, camera: Camera) -> bool:
    if camera.projection == "orthographic":
        return 0 <= vertex.z <= vertex.w
    return vertex.w >= camera.near


def _clip_view_segment(a: Vec4, b: Vec4, camera: Camera):
    """Clip a view-space segment before any nonlinear projection."""
    start, end = 0.0, 1.0
    planes = [(lambda v: -v.z - camera.near)]
    if camera.projection == "orthographic":
        planes.append(lambda v: camera.far + v.z)
    for distance in planes:
        da, db = distance(a), distance(b)
        if da < 0 and db < 0:
            return None
        if (da < 0) != (db < 0):
            t = da / (da - db)
            if da < 0:
                start = max(start, t)
            else:
                end = min(end, t)
    if start > end:
        return None
    return tuple(
        Vec4(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t, 1.0)
        for t in (start, end)
    )
