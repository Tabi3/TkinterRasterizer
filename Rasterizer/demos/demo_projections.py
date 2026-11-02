"""Projection comparison demo.

Spec:
    This example swaps the active camera projection mode while keeping the same
    scene graph, shader setup, and public draw API.

Keys:
    1 / 2 / 3 / 4 = perspective / orthographic / panini / ellipsoid
    W = toggle wireframe
    Escape = quit

Reference:
    IMPLEMENTATION_SPEC.md §2.1, "Math and geometry layer" and §3.3, "Demo
    layer" explain the public API and projection conventions used here.
"""

import sys
import time
from array import array

try:
    from .common import parse_resolution
except ImportError:  # pragma: no cover - package import fallback
    try:
        from common import parse_resolution
    except ImportError:  # pragma: no cover - direct repo launch fallback
        from Rasterizer.demos.common import parse_resolution

from Rasterizer.algebra.math3d import Vec3
from Rasterizer.algebra.transforms import rotate_x, rotate_y
from Rasterizer.display.tk_backend import Viewer
from Rasterizer.geometry.mesh import make_cube
from Rasterizer.materials.material import Material
from Rasterizer.materials.shaders import make_gouraud_shader
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms
from Rasterizer.scene.scene import Light, Node

_proj_names = ["perspective", "orthographic", "panini", "ellipsoid"]
_state = {}
_viewer = None


def draw(fb, dt):
    if "renderer" not in _state:
        _state["renderer"] = Renderer(fb.width, fb.height)
        cube = make_cube(size=2.0, subdivisions=3)
        for v in cube.vertices:
            v.color = 0xE0D0B0
        mat = Material(*make_gouraud_shader(), name="gouraud")
        root = Node(name="root")
        root.add_child(Node(mesh=cube, material=mat, name="cube"))
        _state["root"] = root
        _state["camera"] = Camera(
            Vec3(0, 1.5, 5),
            Vec3(0, 0, 0),
            fov_y_deg=90,
            near=0.1,
            projection="perspective",
        )
        u = Uniforms()
        u.ambient = 0.2
        u.lights = [Light(kind="directional", direction=Vec3(-0.5, -1.0, -0.5))]
        _state["uniforms"] = u
        _state["wireframe"] = False

    if _viewer is not None:
        for i, name in enumerate(_proj_names):
            if _viewer.key_was_pressed(str(i + 1)):
                _state["camera"].projection = name
                _viewer.root.title(f"Rasterizer — {name} ({fb.width}x{fb.height})")
                print(f"projection: {name}")
        if _viewer.key_was_pressed("w"):
            _state["wireframe"] = not _state["wireframe"]
            print(f"wireframe: {_state['wireframe']}")

    t = time.perf_counter()
    _state["root"].children[0].local_transform = rotate_y(t * 0.5) @ rotate_x(t * 0.3)
    _state["uniforms"].time = t

    renderer = _state["renderer"]
    renderer.draw_skybox(fb, 0x5070A0, 0x101828)

    if _state["wireframe"]:
        fb.depth[:] = array("f", [0.0]) * fb.size
        cs = 12 if _state["camera"].projection in ("panini", "ellipsoid") else 1
        renderer.draw_scene_wireframe(
            fb,
            _state["root"],
            _state["camera"],
            _state["uniforms"],
            color=0x00FF88,
            curve_segments=cs,
        )
    else:
        renderer.draw_scene(fb, _state["root"], _state["camera"], _state["uniforms"])


if __name__ == "__main__":
    w, h, scale = parse_resolution(sys.argv)
    _viewer = Viewer(w, h, title="...", render_scale=scale)
    _viewer.set_frame_callback(draw)
    _viewer.run()
