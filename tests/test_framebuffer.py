"""Regression tests for the framebuffer and reverse-Z depth behavior.

The framebuffer is the render target used by the rasterizer and the Tk viewer,
so these checks validate the exact memory layout and depth semantics that the
rest of the pipeline depends on.
"""

import pytest

from Rasterizer.rendering.framebuffer import Framebuffer


def test_framebuffer_construction():

    for bad in [(0, 10), (10, 0), (-1, 5)]:
        with pytest.raises(ValueError):
            Framebuffer(*bad)
    fb = Framebuffer(4, 3)
    assert fb.size == 12


def test_framebuffer_clear():

    fb = Framebuffer(4, 3)
    fb.clear(0xFF0000)
    assert all(c == 0xFF0000 for c in fb.color)
    assert all(z == 0.0 for z in fb.depth)


def test_framebuffer_put_get():

    fb = Framebuffer(4, 3)
    fb.put(1, 2, 0xAABBCC)
    assert fb.get(1, 2) == 0xAABBCC


def test_framebuffer_out_of_bounds():

    fb = Framebuffer(4, 3)
    fb.put(-1, 0, 0xFFFFFF)
    fb.put(4, 0, 0xFFFFFF)
    assert fb.get(-1, 0) == 0
    assert fb.get(4, 0) == 0


def test_framebuffer_depth_test_reverse_z():

    fb = Framebuffer(4, 4)
    fb.clear(0)
    assert fb.put_depthtested(1, 1, 0x00FF00, 0.5)
    assert not fb.put_depthtested(1, 1, 0x0000FF, 0.25)
    assert fb.put_depthtested(1, 1, 0xFF0000, 0.75)
    assert fb.get(1, 1) == 0xFF0000


def test_framebuffer_depth_strict():

    fb = Framebuffer(4, 4)
    fb.clear(0)
    fb.put_depthtested(1, 1, 0x00FF00, 0.5)
    assert not fb.put_depthtested(1, 1, 0xFFFFFF, 0.5)


def test_framebuffer_ppm_b64():
    import base64
    fb = Framebuffer(2, 1)
    fb.clear(0)
    fb.put(0, 0, 0xFF0000)
    fb.put(1, 0, 0x00FF00)
    raw = base64.b64decode(fb.to_ppm_b64())
    header = b"P6\n2 1\n255\n"
    assert raw.startswith(header)
    body = raw[len(header) :]
    assert body == b"\xff\x00\x00\x00\xff\x00"


def test_framebuffer_png_b64():
    import base64
    import struct
    import zlib
    fb = Framebuffer(2, 1)
    fb.clear(0)
    fb.put(0, 0, 0xFF0000)
    fb.put(1, 0, 0x00FF00)
    png = base64.b64decode(fb.to_png_b64())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    off, chunks = 8, {}
    while off < len(png):
        ln = struct.unpack(">I", png[off : off + 4])[0]
        tag = png[off + 4 : off + 8]
        chunks[tag] = png[off + 8 : off + 8 + ln]
        off += 12 + ln
    assert b"IHDR" in chunks and b"IDAT" in chunks and b"IEND" in chunks
    idat = zlib.decompress(chunks[b"IDAT"])
    assert idat == b"\x00\xff\x00\x00\x00\xff\x00"


def test_upscale_nearest_2x():
    from array import array
    src = Framebuffer(2, 2)
    src.color = array("I", [0x110000, 0x220000, 0x330000, 0x440000])
    dst = Framebuffer(4, 4)
    src.upscale_nearest(dst)

    # Each src pixel fills a 2x2 block
    for x, y in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        assert dst.get(x, y) == 0x110000
    for x, y in [(2, 0), (3, 0), (2, 1), (3, 1)]:
        assert dst.get(x, y) == 0x220000
    for x, y in [(0, 2), (1, 2), (0, 3), (1, 3)]:
        assert dst.get(x, y) == 0x330000
    for x, y in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        assert dst.get(x, y) == 0x440000


def test_upscale_nearest_3x():
    from array import array
    src = Framebuffer(2, 1)
    src.color = array("I", [0xAA0000, 0xBB0000])
    dst = Framebuffer(6, 3)
    src.upscale_nearest(dst)
    for y in range(3):
        for x in range(3):
            assert dst.get(x, y) == 0xAA0000
        for x in range(3, 6):
            assert dst.get(x, y) == 0xBB0000


def test_upscale_nearest_rejects_non_integer():

    src = Framebuffer(3, 2)
    dst = Framebuffer(10, 10)
    with pytest.raises(ValueError):
        src.upscale_nearest(dst)


def test_upscale_nearest_identity():
    from array import array
    src = Framebuffer(4, 3)
    src.color = array("I", range(12))
    dst = Framebuffer(4, 3)
    src.upscale_nearest(dst)
    for i in range(12):
        assert dst.color[i] == src.color[i]
