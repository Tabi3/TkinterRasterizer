"""Scene graph light rig demo.

Spec:
    This demo exercises the public renderer API with a multi-light Gouraud scene
    and a skybox background while keeping the example code separate from the
    engine implementation.

Reference:
    IMPLEMENTATION_SPEC.md §3.3, "Demo layer" and §2.2, "Rendering and backend
    layer" describe the intended separation between the public API and the
    backend pipeline used by this script.
"""

import math
import sys

try:
    from .common import advance_time, parse_resolution
except ImportError:  # pragma: no cover - package import fallback
    try:
        from common import advance_time, parse_resolution
    except ImportError:  # pragma: no cover - direct repo launch fallback
        from Rasterizer.demos.common import advance_time, parse_resolution

from Rasterizer.algebra.math3d import Vec3
from Rasterizer.algebra.transforms import rotate_y, translate
from Rasterizer.display.tk_backend import Viewer
from Rasterizer.geometry.mesh import make_cube
from Rasterizer.materials.material import Material
from Rasterizer.materials.shaders import make_gouraud_shader
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms
from Rasterizer.scene.scene import Light, Node

_state = {}


def _build_scene():
    mat = Material(*make_gouraud_shader(), name="gouraud")
    root = Node(name="root")

    center = make_cube(size=1.4)
    for v in center.vertices:
        v.color = 0xE8D8C8
    root.add_child(Node(mesh=center, material=mat, name="center"))

    palette = [0xFF6060, 0x60FF60, 0x6060FF, 0xFFD060]
    for i in range(4):
        angle = (i / 4.0) * 2.0 * math.pi
        cx = 3.0 * math.cos(angle)
        cz = 3.0 * math.sin(angle)
        small = make_cube(size=0.8)
        for v in small.vertices:
            v.color = palette[i]
        root.add_child(
            Node(
                mesh=small,
                material=mat,
                local_transform=translate(Vec3(cx, 0.8, cz)),
                name=f"cube_{i}",
            )
        )
    return root


def draw(fb, dt):
    t = advance_time(_state, dt)
    if "renderer" not in _state:
        _state["renderer"] = Renderer(fb.width, fb.height)
        _state["root"] = _build_scene()
        _state["camera"] = Camera(
            Vec3(0, 2.5, 7), Vec3(0, 0.5, 0), fov_y_deg=55, near=0.1
        )
        u = Uniforms()
        u.ambient = 0.15
        u.lights = [
            Light(
                kind="directional",
                direction=Vec3(-0.5, -1.0, -0.3),
                color=(1.0, 0.95, 0.85),
                intensity=1.0,
            ),
            Light(
                kind="point",
                position=Vec3(2.5, 2.5, 1.5),
                color=(1.0, 0.6, 0.3),
                intensity=1.5,
                radius=9.0,
            ),
        ]
        _state["uniforms"] = u

    _state["root"].local_transform = rotate_y(t * 0.3)
    _state["uniforms"].time = t
    _state["uniforms"].delta_time = dt

    renderer = _state["renderer"]
    renderer.draw_skybox(fb, 0x5070A0, 0x00AAFF)
    renderer.draw_scene(fb, _state["root"], _state["camera"], _state["uniforms"])


if __name__ == "__main__":
    w, h, scale = parse_resolution(sys.argv)
    _viewer = Viewer(w, h, title="...", render_scale=scale)
    _viewer.set_frame_callback(draw)
    _viewer.run()
