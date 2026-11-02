"""Textured lit cube demo.

Spec:
    This demo covers the textured-lighting path that consumes the renderer's
    public API and shader callbacks while staying outside the engine internals.

Reference:
    IMPLEMENTATION_SPEC.md §2.2, "Rendering and backend layer" and §3.3,
    "Demo layer" define the role of this script in the repo structure.
"""

import sys
import time

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
from Rasterizer.materials.shaders import make_textured_lit_shader
from Rasterizer.materials.texture import Texture
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms
from Rasterizer.scene.scene import Light, Node

_state = {}


def draw(fb, dt):
    if "renderer" not in _state:
        _state["renderer"] = Renderer(fb.width, fb.height)
        tex = Texture.checkerboard(
            size=128, cells=8, color_a=0xFF4040, color_b=0x4040FF
        )
        cube = make_cube(size=2.0)
        mat = Material(*make_textured_lit_shader(tex), name="textured_lit")
        root = Node(name="root")
        root.add_child(Node(mesh=cube, material=mat, name="cube"))
        _state["root"] = root
        _state["camera"] = Camera(
            Vec3(0, 1.5, 5), Vec3(0, 0, 0), fov_y_deg=55, near=0.1
        )
        u = Uniforms()
        u.ambient = 0.2
        u.lights = [
            Light(kind="directional", direction=Vec3(-0.5, -1.0, -0.5), intensity=1.0)
        ]
        _state["uniforms"] = u

    t = time.perf_counter()
    _state["root"].children[0].local_transform = rotate_y(t * 0.6) @ rotate_x(t * 0.3)
    _state["uniforms"].time = t

    renderer = _state["renderer"]
    renderer.draw_skybox(fb, 0x5060A0, 0x203040)
    renderer.draw_scene(fb, _state["root"], _state["camera"], _state["uniforms"])


if __name__ == "__main__":
    w, h, scale = parse_resolution(sys.argv)
    _viewer = Viewer(w, h, title=f"Rasterizer — Textured ({w}x{h})", render_scale=scale)
    _viewer.set_frame_callback(draw)
    _viewer.run()
