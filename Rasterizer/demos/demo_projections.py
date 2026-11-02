"""Projection comparison. Spec: IMPLEMENTATION_SPEC.md#api-and-organization."""

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
from Rasterizer.data.conventions import PROJECTIONS
from Rasterizer.demos.common import advance_time, demo_viewer


_proj_names = PROJECTIONS

_state = {}

_viewer = None


def draw(fb: Framebuffer, dt: float) -> None:
    """Update owned pose state from elapsed seconds and draw the demo."""

    # Spec: IMPLEMENTATION_SPEC.md#api-and-organization.
    # Initialize scene resources once; each frame only updates pose/time.
    t = advance_time(_state, dt)
    if "renderer" not in _state:
        _state["renderer"] = Renderer(fb.width, fb.height)
        cube = make_cube(size=2.0, subdivisions=3)
        for v in cube.vertices:
            v.color = 0xE0D0B0
        mat = Material.gouraud()
        root = Node(cube, mat, name="cube")
        _state["root"] = root
        _state["camera"] = Camera(
            Vec3(0, 1.5, 5),
            Vec3(0, 0, 0),
            fov_y_deg=90,
            near=0.1,
            projection="perspective",
        )
        u = _state["renderer"].uniforms
        u.ambient = 0.2
        u.lights = [Light(kind="directional", direction=Vec3(-0.5, -1.0, -0.5))]
        _state["uniforms"] = u
        _state["wireframe"] = False
    if _viewer is not None:
        for key in _viewer.key_presses():
            if key in ("1", "2", "3", "4"):
                name = _proj_names[int(key) - 1]
                _state["camera"].projection = name
                _viewer.root.title(f"Rasterizer - {name} ({fb.width}x{fb.height})")
                print(f"projection: {name}")
            elif key == "w":
                _state["wireframe"] = not _state["wireframe"]
                print(f"wireframe: {_state['wireframe']}")
    write_trs(_state["root"].local_transform, yaw=t * 0.5, pitch=t * 0.3)
    _state["uniforms"].time = t
    _state["uniforms"].delta_time = dt
    renderer = _state["renderer"]
    renderer.render(
        fb,
        _state["root"],
        _state["camera"],
        background=(0x5070A0, 0x101828),
        wireframe=_state["wireframe"],
    )


if __name__ == "__main__":
    _viewer = demo_viewer(draw, "Projection comparison")

    _viewer.run()
