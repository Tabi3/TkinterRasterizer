"""Textured cube. Spec: IMPLEMENTATION_SPEC.md#api-and-organization."""

import sys
from pathlib import Path


if __package__ in (None, ""):  # Support direct file execution from any directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from Rasterizer import (
    Camera,
    Framebuffer,
    Light,
    Material,
    Node,
    Renderer,
    Texture,
    Vec3,
    make_cube,
)
from Rasterizer.algebra.affine import write_trs
from Rasterizer.demos.common import advance_time, demo_viewer


_state = {}


def draw(fb: Framebuffer, dt: float) -> None:
    """Update owned pose state from elapsed seconds and draw the demo."""

    # Spec: IMPLEMENTATION_SPEC.md#api-and-organization.
    # Initialize scene resources once; each frame only updates pose/time.
    t = advance_time(_state, dt)
    if "renderer" not in _state:
        _state["renderer"] = Renderer(fb.width, fb.height)
        tex = Texture.checkerboard(size=128, cells=8, color_a=0xFF4040, color_b=0x4040FF)
        cube = make_cube(size=2.0)
        mat = Material.textured(tex, lit=True)
        root = Node(cube, mat, name="cube")
        _state["root"] = root
        _state["camera"] = Camera(Vec3(0, 1.5, 5), Vec3(0, 0, 0), fov_y_deg=55, near=0.1)
        u = _state["renderer"].uniforms
        u.ambient = 0.2
        u.lights = [Light(kind="directional", direction=Vec3(-0.5, -1.0, -0.5), intensity=1.0)]
        _state["uniforms"] = u
    write_trs(_state["root"].local_transform, yaw=t * 0.6, pitch=t * 0.3)
    _state["uniforms"].time = t
    _state["uniforms"].delta_time = dt
    renderer = _state["renderer"]
    renderer.render(fb, _state["root"], _state["camera"], background=(0x5060A0, 0x203040))


if __name__ == "__main__":
    demo_viewer(draw, "Textured cube").run()
