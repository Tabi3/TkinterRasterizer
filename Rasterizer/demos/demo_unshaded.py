"""Flat-color cube. Spec: IMPLEMENTATION_SPEC.md#api-and-organization."""

import sys
from pathlib import Path


if __package__ in (None, ""):  # Support direct file execution from any directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from Rasterizer import Camera, Framebuffer, Mat4, Renderer, Vec3, make_cube
from Rasterizer.algebra.affine import write_trs
from Rasterizer.demos.common import advance_time, demo_viewer


_state = {}


def draw(framebuffer: Framebuffer, dt: float) -> None:
    """Update owned pose state from elapsed seconds and draw the demo."""

    # Spec: IMPLEMENTATION_SPEC.md#api-and-organization.
    # Initialize scene resources once; each frame only updates pose/time.
    t = advance_time(_state, dt)
    if "renderer" not in _state:
        _state["renderer"]  = Renderer(framebuffer.width, framebuffer.height)
        _state["mesh"]      = make_cube()
        _state["transform"] = Mat4.identity()
        _state["camera"]    = Camera(Vec3(0, 0, 4), Vec3.zero())

    # Spec: IMPLEMENTATION_SPEC.md#allocation-and-mutable-scratch.
    transform   = write_trs(_state["transform"], yaw=t * 0.4)
    renderer    = _state["renderer"]
    renderer.clear(framebuffer, 0x101018)
    renderer.draw(
        framebuffer,
        _state["mesh"],
        _state["camera"],
        transform=transform,
        color=0xA0D8FF,
    )


if __name__ == "__main__":
    demo_viewer(draw, "Unshaded cube").run()
