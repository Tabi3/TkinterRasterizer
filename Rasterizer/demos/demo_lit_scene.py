"""Multiple lights. Spec: IMPLEMENTATION_SPEC.md#api-and-organization."""

import math
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
    Vec3,
    make_cube,
)
from Rasterizer.algebra.affine import write_trs
from Rasterizer.algebra.transforms import translate
from Rasterizer.demos.common import advance_time, demo_viewer


_state = {}


def _build_scene() -> Node:

    mat     = Material.gouraud()
    root    = Node(name="root")
    center  = make_cube(size=1.4)
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


def draw(fb: Framebuffer, dt: float) -> None:
    """Update owned pose state from elapsed seconds and draw the demo."""

    # Spec: IMPLEMENTATION_SPEC.md#api-and-organization.
    # Initialize scene resources once; each frame only updates pose/time.
    t = advance_time(_state, dt)
    if "renderer" not in _state:
        _state["renderer"] = Renderer(fb.width, fb.height)
        _state["root"] = _build_scene()
        _state["camera"] = Camera(Vec3(0, 2.5, 7), Vec3(0, 0.5, 0), fov_y_deg=55, near=0.1)
        u = _state["renderer"].uniforms
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
    write_trs(_state["root"].local_transform, yaw=t * 0.3)
    _state["uniforms"].time = t
    _state["uniforms"].delta_time = dt
    renderer = _state["renderer"]
    renderer.render(fb, _state["root"], _state["camera"], background=(0x5070A0, 0x00AAFF))


if __name__ == "__main__":
    demo_viewer(draw, "Light rig").run()
