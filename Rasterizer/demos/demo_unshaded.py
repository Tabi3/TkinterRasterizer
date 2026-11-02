"""Minimal unshaded cube demo using the direct mesh draw path."""

import sys
import time

try:
    from .common import parse_resolution
except ImportError:  # pragma: no cover - direct script execution
    from common import parse_resolution

from Rasterizer.algebra.math3d import Vec3
from Rasterizer.algebra.transforms import rotate_y
from Rasterizer.display.tk_backend import Viewer
from Rasterizer.geometry.mesh import make_cube
from Rasterizer.rendering.renderer import Camera, Renderer

_state = {}


def draw(framebuffer, _dt):
    if "renderer" not in _state:
        _state["renderer"] = Renderer(framebuffer.width, framebuffer.height)
        _state["mesh"] = make_cube()
        _state["camera"] = Camera(Vec3(0, 0, 4), Vec3.zero())

    _state["renderer"].clear(framebuffer, 0x101018)
    _state["renderer"].draw_mesh(
        framebuffer,
        _state["mesh"],
        rotate_y(time.perf_counter() * 0.4),
        0xA0D8FF,
        _state["camera"],
    )


if __name__ == "__main__":
    width, height, scale = parse_resolution(sys.argv)
    viewer = Viewer(width, height, title="Unshaded rasterizer", render_scale=scale)
    viewer.set_frame_callback(draw)
    viewer.run()
