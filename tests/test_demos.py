"""Headless smoke tests for runnable demos."""

from Rasterizer.demos import demo_unshaded
from Rasterizer.rendering.framebuffer import Framebuffer


def test_unshaded_demo_renders_without_opening_a_window():
    demo_unshaded._state.clear()
    framebuffer = Framebuffer(64, 64)

    demo_unshaded.draw(framebuffer, 0.0)

    assert any(framebuffer.color)
