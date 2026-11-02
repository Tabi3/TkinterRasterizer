"""Rasterizer coverage tests for triangle fill, depth behavior, and line drawing.

The hot path is intentionally exercised here because a large share of the engine's
correctness depends on pixel coverage, winding conventions, and reverse-Z depth
comparisons.
"""

from Rasterizer.rendering.framebuffer import Framebuffer
from Rasterizer.rendering.rasterizer import (
    draw_line,
    rasterize_triangle,
    rasterize_triangle_shaded,
)


def _count_nonzero(fb):
    return sum(1 for c in fb.color if c != 0)


def test_rasterize_basic_triangle():
    fb = Framebuffer(20, 20)
    fb.clear(0)
    rasterize_triangle(
        fb,
        (5.0, 0.0, 0.5, 1.0),
        (15.0, 15.0, 0.5, 1.0),
        (0.0, 15.0, 0.5, 1.0),
        0xFF0000,
    )
    assert _count_nonzero(fb) > 30


def test_rasterize_degenerate():
    fb = Framebuffer(10, 10)
    fb.clear(0)
    rasterize_triangle(
        fb, (5.0, 5.0, 0.5, 1.0), (5.0, 5.0, 0.5, 1.0), (5.0, 5.0, 0.5, 1.0), 0xFF0000
    )
    assert _count_nonzero(fb) == 0


def test_rasterize_offscreen():
    fb = Framebuffer(10, 10)
    fb.clear(0)
    rasterize_triangle(
        fb,
        (-100.0, -100.0, 0.5, 1.0),
        (-50.0, -100.0, 0.5, 1.0),
        (-100.0, -50.0, 0.5, 1.0),
        0xFF0000,
    )
    assert _count_nonzero(fb) == 0


def test_rasterize_shaded_triangle_coverage():
    fb = Framebuffer(64, 64)
    fb.clear(0)
    rasterize_triangle_shaded(
        fb,
        (10.0, 10.0, 0.5, 1.0),
        (54.0, 10.0, 0.5, 1.0),
        (10.0, 54.0, 0.5, 1.0),
        (0.0,),
        (0.0,),
        (0.0,),
        lambda v: 0xFFFFFF,
    )
    count = _count_nonzero(fb)

    # Expected coverage for a right triangle with legs of 44 units, vertices
    # at integer coords, rasterized at pixel centers with the top-left rule:
    # strictly-interior pixels satisfy px + py < 63 in [10, 53]^2.
    # The diagonal edge (px + py == 63) is a bottom/right edge under this
    # winding, so the top-left rule excludes all 44 of its pixels.
    expected = sum(1 for py in range(10, 54) for px in range(10, 54) if px + py < 63)
    assert count == expected, f"expected {expected}, got {count}"


def test_rasterize_shaded_shared_edge():
    fb = Framebuffer(64, 64)
    fb.clear(0)
    fs = lambda v: 0xFFFFFF
    rasterize_triangle_shaded(
        fb,
        (10.0, 10.0, 0.5, 1.0),
        (54.0, 10.0, 0.5, 1.0),
        (54.0, 54.0, 0.5, 1.0),
        (0.0,),
        (0.0,),
        (0.0,),
        fs,
    )
    rasterize_triangle_shaded(
        fb,
        (10.0, 10.0, 0.5, 1.0),
        (54.0, 54.0, 0.5, 1.0),
        (10.0, 54.0, 0.5, 1.0),
        (0.0,),
        (0.0,),
        (0.0,),
        fs,
    )
    for py in range(10, 54):
        for px in range(10, 54):
            assert fb.get(px, py) != 0, f"crack at ({px},{py})"


def test_rasterize_shaded_depth_test():
    fb = Framebuffer(16, 16)
    fb.clear(0)
    fs = lambda v: 0xFFFFFF
    rasterize_triangle_shaded(
        fb,
        (0.0, 0.0, 0.9, 1.0),
        (16.0, 0.0, 0.9, 1.0),
        (0.0, 16.0, 0.9, 1.0),
        (0.0,),
        (0.0,),
        (0.0,),
        fs,
    )
    assert fb.get(2, 2) == 0xFFFFFF
    rasterize_triangle_shaded(
        fb,
        (0.0, 0.0, 0.1, 1.0),
        (16.0, 0.0, 0.1, 1.0),
        (0.0, 16.0, 0.1, 1.0),
        (0.0,),
        (0.0,),
        (0.0,),
        lambda v: 0xFF0000,
    )
    assert fb.get(2, 2) == 0xFFFFFF  # far loses


def test_rasterize_shaded_varying_interpolation():
    fb = Framebuffer(16, 16)
    fb.clear(0)
    captured = []

    def fs(v):
        captured.append(v)
        return 0xFFFFFF

    rasterize_triangle_shaded(
        fb,
        (0.0, 0.0, 0.5, 1.0),
        (15.0, 0.0, 0.5, 1.0),
        (0.0, 15.0, 0.5, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        fs,
    )
    assert len(captured) > 50
    # u+v should be roughly in [0, 1] for interior pixels
    for u, v in captured:
        assert -0.01 <= u <= 1.01 and -0.01 <= v <= 1.01


def test_draw_line_basic():
    fb = Framebuffer(16, 16)
    fb.clear(0)
    draw_line(fb, 0.0, 0.0, 0.9, 15.0, 15.0, 0.9, 0xFFFFFF)
    assert fb.get(0, 0) == 0xFFFFFF
    assert fb.get(15, 15) == 0xFFFFFF
    assert fb.get(7, 7) == 0xFFFFFF


def test_draw_line_depth_test():
    fb = Framebuffer(16, 16)
    fb.clear(0)
    draw_line(fb, 0.0, 0.0, 0.5, 15.0, 15.0, 0.5, 0xFFFFFF)
    draw_line(fb, 0.0, 0.0, 0.3, 15.0, 15.0, 0.3, 0xFF0000)
    assert fb.get(7, 7) == 0xFFFFFF
