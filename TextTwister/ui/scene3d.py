"""Lit, texture-labelled boxes and stage projections for the game UI."""

import math
from collections import OrderedDict
from dataclasses import dataclass

from Rasterizer.algebra.affine import write_trs
from Rasterizer.algebra.math3d import Mat4, Vec2, Vec3
from Rasterizer.algebra.transforms import rotate_x, rotate_y, scale
from Rasterizer.data.box_faces import BOX_FACES
from Rasterizer.data.conventions import PROJECTIONS
from Rasterizer.geometry.mesh import Mesh, MeshBuilder
from Rasterizer.materials.material import Material
from Rasterizer.materials.texture import Texture
from Rasterizer.rendering.drawing2d import draw_text, fill_rect
from Rasterizer.rendering.framebuffer import Framebuffer
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms
from Rasterizer.scene.scene import Light

from ..data.theme import BACKGROUND, BOX_CACHE, MATERIAL_CACHE, TRAY


Point = tuple[float, float]

Triangle = tuple[Point, Point, Point]


def projection_for_stage(stage: int) -> str:

    return PROJECTIONS[(max(1, stage) - 1) % len(PROJECTIONS)]


def contains(triangle: Triangle, x: float, y: float) -> bool:

    a, b, c = triangle

    if (b[0] - a[0]) * (c[1] - a[1]) == (b[1] - a[1]) * (c[0] - a[0]):
        return False

    edges = tuple((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) for a, b in zip(triangle, (triangle[1], triangle[2], triangle[0])))

    return min(edges) >= 0 or max(edges) <= 0


def labelled_box(subdivisions: int) -> Mesh:
    """Only the front is tessellated/textured; the other five faces sample the rim."""

    builder = MeshBuilder()

    for normal_data, right, up, origin in BOX_FACES:
        normal = Vec3(*normal_data)

        front = normal.z > 0

        count = subdivisions if front else 1

        grid = []

        for row in range(count + 1):
            indices = []

            for column in range(count + 1):
                u, v = column / count, row / count

                indices.append(
                    builder.vertex(
                        Vec3(
                            origin[0] + u * right[0] + v * up[0],
                            origin[1] + u * right[1] + v * up[1],
                            origin[2] + u * right[2] + v * up[2],
                        ),
                        normal,
                        Vec2(u, 1 - v) if front else Vec2(0, 0),
                    )
                )

            grid.append(indices)

        for row in range(count):
            for column in range(count):
                builder.quad(
                    grid[row][column],
                    grid[row][column + 1],
                    grid[row + 1][column + 1],
                    grid[row + 1][column],
                )

    return builder.build()


@dataclass(frozen=True)
class BoxImage:
    framebuffer: Framebuffer

    triangles: tuple[Triangle, ...]


class Scene3D:
    def __init__(self) -> None:

        self.meshes = {1: labelled_box(1), 3: labelled_box(3)}

        self.uniforms = Uniforms()

        self.uniforms.ambient = 0.38

        self.uniforms.lights = [Light(direction=Vec3(-0.45, -0.7, -1), color=(1.0, 1.0, 1.0), intensity=0.85)]

        self.materials: OrderedDict[tuple[str, int, int, int], Material] = OrderedDict()

        self.boxes: OrderedDict[tuple[str, int, int, int, str, float, float], BoxImage] = OrderedDict()

        self.renderers: dict[tuple[int, int], Renderer] = {}

        self.rack_fb: Framebuffer | None = None

        self._cameras: dict[tuple[str, float, bool], Camera] = {}

        self._rack_models = [Mat4.identity() for _ in range(8)]

        self._tray_model = Mat4.identity()

        self._tray_length = -1

    def renderer(self, width: int, height: int) -> Renderer:

        key = (width, height)

        if key not in self.renderers:
            self.renderers[key] = Renderer(width, height)

        return self.renderers[key]

    def material(self, text: str, width: int, height: int, color: int) -> Material:

        key = (text, width, height, color)

        if key not in self.materials:
            fb = Framebuffer(width, height)

            fb.clear(color)

            fill_rect(fb, 3, 3, width - 6, height - 6, self._tint(color, 0.86))

            fill_rect(fb, 5, 5, width - 10, height - 10, color)

            font_scale = max(1, min(5, (height - 12) // 7, (width - 12) // max(1, len(text) * 6)))

            text_width = (len(text) * 6 - 1) * font_scale

            draw_text(
                fb,
                text,
                (width - text_width) // 2,
                (height - 7 * font_scale) // 2,
                BACKGROUND,
                font_scale,
            )

            texture = Texture.from_framebuffer(fb)

            self.materials[key] = Material.textured(texture, lit=True)

            if len(self.materials) > MATERIAL_CACHE:
                self.materials.popitem(last=False)

        self.materials.move_to_end(key)

        return self.materials[key]

    @staticmethod
    def _tint(color: int, factor: float) -> int:

        return sum(int(((color >> shift) & 255) * factor) << shift for shift in (16, 8, 0))

    def camera(self, projection: str, aspect: float, *, rack: bool = False) -> Camera:
        """Reuse stationary cameras per viewport/projection; poses do not change."""

        key = (projection, aspect, rack)

        if key not in self._cameras:
            distance = 6.0 if rack else 6.0 * max(1.0, aspect / 2)

            self._cameras[key] = Camera(
                Vec3(0, 0, distance),
                Vec3.zero(),
                projection=projection,
                fov_y_deg=math.degrees(2 * math.atan(1.05 / distance)),
                ortho_scale=1.05,
                ellipsoid_strength=0.7,
                far=distance * 4 + 50,
            )

        return self._cameras[key]

    def _draw(
        self,
        fb: Framebuffer,
        mesh: Mesh,
        material: Material,
        model: Mat4,
        camera: Camera,
    ) -> None:

        self.renderer(fb.width, fb.height).draw(fb, mesh, camera, material=material, transform=model, uniforms=self.uniforms)

    @staticmethod
    def _mesh(projection: str) -> int:

        return 3 if projection in ("panini", "ellipsoid") else 1

    def box(
        self,
        text: str,
        width: int,
        height: int,
        color: int,
        projection: str,
        *,
        yaw: float = 0.13,
        pitch: float = 0.18,
    ) -> BoxImage:

        key = (text, width, height, color, projection, yaw, pitch)

        if key not in self.boxes:
            fb = Framebuffer(width, height)

            fb.clear(BACKGROUND)

            aspect = width / height

            mesh = self.meshes[self._mesh(projection)]

            camera = self.camera(projection, aspect)

            model = rotate_y(yaw) @ rotate_x(pitch) @ scale(Vec3(aspect * 1.75, 1.65, 0.42))

            self._draw(
                fb,
                mesh,
                self.material(text, max(64, width), max(48, height), color),
                model,
                camera,
            )

            triangles = self.projected_triangles(mesh, model, camera, width, height)

            self.boxes[key] = BoxImage(fb, triangles)

            if len(self.boxes) > BOX_CACHE:
                self.boxes.popitem(last=False)

        self.boxes.move_to_end(key)

        return self.boxes[key]

    @staticmethod
    def projected_triangles(mesh: Mesh, model: Mat4, camera: Camera, width: int, height: int) -> tuple[Triangle, ...]:

        view_model = camera.view_matrix() @ model

        points: list[Point] = []

        for vertex in mesh.vertices:
            clip = camera.project(view_model @ vertex.pos.to_vec4(1), width / height)

            points.append(((clip.x / clip.w + 1) * width / 2, (1 - clip.y / clip.w) * height / 2))

        triangles = []

        for index in range(mesh.triangle_count()):
            a, b, c = (points[i] for i in mesh.triangle(index))

            area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

            if area < 0:
                triangles.append((a, b, c))

        return tuple(triangles)

    def rack(
        self,
        rack: str,
        width: int,
        height: int,
        projection: str,
        elapsed: float,
        color: int,
    ) -> Framebuffer:

        if self.rack_fb is None or (self.rack_fb.width, self.rack_fb.height) != (
            width,
            height,
        ):
            self.rack_fb = Framebuffer(width, height)

        fb = self.rack_fb

        fb.clear(BACKGROUND)

        camera = self.camera(projection, width / height, rack=True)

        mesh = self.meshes[self._mesh(projection)]

        self.uniforms.time = elapsed

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#allocation-and-rendering.
        # The stationary tray changes only when the rack length changes.
        if len(rack) != self._tray_length:
            write_trs(
                self._tray_model,
                y=-0.57,
                z=-0.25,
                pitch=-0.1,
                sx=len(rack) * 1.32 + 0.35,
                sy=0.14,
                sz=0.8,
            )

            self._tray_length = len(rack)

        self._draw(fb, mesh, self.material("", 64, 48, TRAY), self._tray_model, camera)

        for index, letter in enumerate(rack):
            phase = elapsed * 0.7 + index * 0.8

            x = (index - (len(rack) - 1) / 2) * 1.32

            # Reuse one owned matrix per letter; no temporary rotations/vectors.
            model = write_trs(
                self._rack_models[index],
                x=x,
                y=0.03 + math.sin(phase) * 0.045,
                z=math.sin(phase) * 0.12,
                yaw=0.08 + math.sin(phase) * 0.23,
                pitch=0.18 + math.sin(phase * 0.7) * 0.06,
                roll=math.sin(phase * 0.6) * 0.035,
                sx=1.08,
                sy=1.18,
                sz=0.72,
            )

            self._draw(fb, mesh, self.material(letter, 64, 64, color), model, camera)

        return fb


def blit(source: Framebuffer, target: Framebuffer, x: int, y: int) -> None:
    """Compatibility helper for Framebuffer.blit; copies clipped color/depth."""

    target.blit(source, x, y)
