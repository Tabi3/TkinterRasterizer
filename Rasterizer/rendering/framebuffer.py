"""Framebuffer and depth-buffer implementation.

Purpose:
    Store color and depth data for the rasterizer and provide lightweight
    serialization helpers for Tk, PPM, and PNG output.

Contract:
    - Colors are packed as 24-bit RGB values with no alpha channel.
    - Depth uses reverse-Z semantics, where larger values are closer.
    - Pixel access follows row-major indexing: idx = y * width + x.

Behavior:
    The framebuffer stays intentionally small and allocation-friendly so it can
    be used in the hot path of the renderer without extra Python object churn.

Notes:
    This module intentionally keeps the rendering surface and conversion logic
    separate from the actual rasterization passes in the renderer backend.

Naming convention:
    local_*  short-lived scratch state inside a helper
    pixel_* pixel-oriented values for indexing and conversion
    row_*    row-scoped data produced by buffer iteration helpers

Reference:
    See IMPLEMENTATION_SPEC.md for the architecture, naming rules, and backend
    responsibility split that define this render target.
"""

from __future__ import annotations

import base64
import struct
import sys
import zlib
from array import array


def _as_rgb24(value: int) -> int:
    """Normalize a packed RGB value to the 24-bit framebuffer format."""
    return int(value) & 0xFFFFFF


def _row_slices(data: array, width: int, height: int):
    """Yield each row of a row-major buffer as a contiguous view."""
    for row_index in range(height):
        row_start = row_index * width
        yield data[row_start : row_start + width]


# array('I') is C unsigned int. 4 bytes on every modern ABI (Win/Mac/Linux).
# array('f') is C float. 4 bytes on every modern ABI. Check once at import
# so failures are loud and immediate, not a mystery five frames in.
if array("I").itemsize != 4:
    raise RuntimeError(f"array('I') itemsize is {array('I').itemsize}, expected 4")
if array("f").itemsize != 4:
    raise RuntimeError(f"array('f') itemsize is {array('f').itemsize}, expected 4")


# ---------------------------------------------------------------------------
class Framebuffer:
    """Color + depth render target.

    Depth is float32. Incoming float64 z is rounded to the nearest
    float32 on write; comparisons are still exact because float32 ->
    float64 widening is lossless. Only bit-exact equality against a
    float64 literal (e.g. 0.8) will disagree — compare with tolerance,
    or write depths that are exact in float32 (multiples of powers of 2).
    """

    __slots__ = (
        "_clear_color",
        "_color_template",
        "_depth_template",
        "color",
        "depth",
        "height",
        "size",
        "width",
    )

    def __init__(self, width: int, height: int) -> None:
        if not (isinstance(width, int) and isinstance(height, int)):
            raise TypeError("framebuffer dimensions must be integers")
        if width <= 0 or height <= 0:
            raise ValueError(f"framebuffer size must be positive, got {width}x{height}")
        self.width = int(width)
        self.height = int(height)
        self.size = self.width * self.height

        # Templates for clear(). Pre-built once; reused every frame via
        # C-level slice assignment, which is a memcpy — no per-pixel loop.
        self._clear_color = 0
        self._color_template = array("I", [0]) * self.size
        self._depth_template = array("f", [0.0]) * self.size

        self.color = self._color_template[:]
        self.depth = self._depth_template[:]

    # ----- clearing -------------------------------------------------------
    def clear(self, color: int = 0x000000) -> None:
        """Reset color to `color` and depth to 0.0 (reverse-Z far plane).

        Color change rebuilds the template; same-color clears are pure
        memcpy. Typical renderer calls this once per frame with the
        scene background color.

        Reference:
            IMPLEMENTATION_SPEC.md > Core architecture > 2. Rendering and backend layer
        """
        if not (isinstance(color, int) and not isinstance(color, bool)):
            raise TypeError("color must be an integer pixel value")
        if color != self._clear_color:
            self._clear_color = color
            self._color_template = array("I", [color]) * self.size
        self.color[:] = self._color_template
        self.depth[:] = self._depth_template

    # ----- pixel ops (helper / debug — rasterizer inlines the hot loop) ---
    def put(self, x: int, y: int, rgb: int) -> None:
        """Write a pixel without a depth test. Out-of-bounds is ignored."""
        if not (isinstance(x, int) and isinstance(y, int)):
            raise TypeError("x/y must be integer pixel coordinates")
        if not (isinstance(rgb, int) and not isinstance(rgb, bool)):
            raise TypeError("rgb must be an integer color value")
        if 0 <= x < self.width and 0 <= y < self.height:
            self.color[y * self.width + x] = _as_rgb24(rgb)

    def get(self, x: int, y: int) -> int:
        """Read a pixel's RGB. Out-of-bounds returns 0."""
        if not (isinstance(x, int) and isinstance(y, int)):
            raise TypeError("x/y must be integer pixel coordinates")
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.color[y * self.width + x]
        return 0

    def put_depthtested(self, x: int, y: int, rgb: int, z: float) -> bool:
        """Depth-tested pixel write. Returns True if the write happened.

        The rasterizer's inner loop inlines exactly this logic:
            idx = y*w + x
            if z > fb.depth[idx]:
                fb.depth[idx] = z
                fb.color[idx] = rgb
        This method is for tests and non-hot callers.

        Reference:
            IMPLEMENTATION_SPEC.md > Core architecture > 2. Rendering and backend layer
        """
        if not (isinstance(x, int) and isinstance(y, int)):
            raise TypeError("x/y must be integer pixel coordinates")
        if not (isinstance(rgb, int) and not isinstance(rgb, bool)):
            raise TypeError("rgb must be an integer color value")
        if not (isinstance(z, (int, float)) and not isinstance(z, bool)):
            raise TypeError("z must be a real depth value")
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            if z > self.depth[idx]:
                self.depth[idx] = z
                self.color[idx] = _as_rgb24(rgb)
                return True
        return False

    def get_depth(self, x: int, y: int) -> float:
        """Read the reverse-Z value. 0.0 = far, 1.0 = near."""
        if not (isinstance(x, int) and isinstance(y, int)):
            raise TypeError("x/y must be integer pixel coordinates")
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.depth[y * self.width + x]
        return 0.0

    # ----- Tk blit --------------------------------------------------------
    def to_tk_data(self) -> str:
        """Format the color buffer as PhotoImage.put(data=...) input.

        Format:  "{#rrggbb #rrggbb ...} {#rrggbb ...} ..."
        — one brace group per row, space-separated colors, lowercase hex.

        Perf notes:
            The whole conversion is C-level: array slice (memcpy) ->
            `map(str.__mod__, ...)` -> `str.join`. That's roughly
            ~80 ns/pixel on 3.11+ CPython, so at 640x360 (~230k pixels)
            expect ~18 ms/frame. This *is* the frame budget, which is why
            we target 360p and keep the rasterizer cheap.
        """
        w = self.width
        h = self.height
        buf = self.color
        fmt = "#%06x".__mod__
        rows = ("{" + " ".join(map(fmt, row)) + "}" for row in _row_slices(buf, w, h))
        return " ".join(rows)

    # ----- offline dump (debug / CI) -------------------------------------
    def to_ppm(self, path: str) -> None:
        """Fast P6 PPM write. Byte reorder via extended-slice assignment."""
        w = self.width
        h = self.height
        buf = self.color
        if sys.byteorder != "little":
            buf = array("I", buf)
            buf.byteswap()
        raw = buf.tobytes()  # [B, G, R, 0] * n on LE
        rgb = bytearray(w * h * 3)
        rgb[0::3] = raw[2::4]  # R
        rgb[1::3] = raw[1::4]  # G
        rgb[2::3] = raw[0::4]  # B
        with open(path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(rgb)

    def to_png_b64(self) -> str:
        """Base64-encoded PNG of the color buffer.

        Preferred blit path: Tk's PNG decoder accepts -data and runs in C.
        Filter-byte strategy: 0 (None) per row; zlib level 1 (fast).
        """
        w = self.width
        h = self.height
        buf = self.color
        if sys.byteorder != "little":
            buf = array("I", buf)
            buf.byteswap()
        raw = buf.tobytes()
        rgb = bytearray(w * h * 3)
        rgb[0::3] = raw[2::4]
        rgb[1::3] = raw[1::4]
        rgb[2::3] = raw[0::4]

        stride = w * 3
        scanlines = bytearray((stride + 1) * h)  # +1 for filter byte 0
        for y in range(h):
            s = y * stride
            d = y * (stride + 1) + 1
            scanlines[d : d + stride] = rgb[s : s + stride]

        def chunk(tag, data):
            return (
                struct.pack(">I", len(data))
                + tag
                + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            )

        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(scanlines), 1))
            + chunk(b"IEND", b"")
        )
        return base64.b64encode(png).decode("ascii")

    def to_ppm_depth(self, path):
        """Write the depth buffer as grayscale (near=white, far=black)."""
        w = self.width
        h = self.height
        buf = self.depth
        gray = bytearray(w * h)
        for i, z in enumerate(buf):
            v = 0 if z <= 0.0 else 255 if z >= 1.0 else int(z * 255.0 + 0.5)
            gray[i] = v
        with open(path, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
            f.write(gray)

    def to_ppm_b64(self):
        w, h = self.width, self.height
        n = w * h
        header = f"P6\n{w} {h}\n255\n".encode("ascii")
        src = self.color
        if sys.byteorder != "little":
            src = array("I", src)
            src.byteswap()  # normalize to LE byte order
        raw = src.tobytes()  # [B, G, R, 0] * n on LE
        rgb = bytearray(n * 3)
        rgb[0::3] = raw[2::4]  # R
        rgb[1::3] = raw[1::4]  # G
        rgb[2::3] = raw[0::4]  # B
        return base64.b64encode(header + bytes(rgb)).decode("ascii")

    # ----- misc -----------------------------------------------------------
    def __repr__(self):
        return f"Framebuffer({self.width}x{self.height})"

    def upscale_nearest(self, dst):
        """Nearest-neighbor upscale into `dst`.

        Spec:
            dst dimensions must be integer multiples of self's. Each source
            pixel (sx, sy) maps to the (dw//sw) x (dh//sh) block at
            (sx * dw // sw, sy * dh // sh).

        Behavior:
            Uses array extended-slice assignment for both the horizontal repeat
            and row duplication. The inner loops run in C; only the outer
            row iteration is Python. ~3 ms for 320x180 → 640x360.

        Args:
            dst: Framebuffer at the destination (display) resolution.

        Raises:
            ValueError if dst is not an integer multiple of self in both axes.
        """
        sw, sh = self.width, self.height
        dw, dh = dst.width, dst.height
        if dw % sw or dh % sh:
            raise ValueError(f"dst {dw}x{dh} is not an integer multiple of {sw}x{sh}")
        scale_x = dw // sw
        scale_y = dh // sh

        src = self.color
        dc = dst.color
        up_row = array("I", [0]) * dw
        for sy in range(sh):
            row = src[sy * sw : (sy + 1) * sw]
            # Horizontal repeat: up_row[k::scale_x] = row fills every scale_x-th slot
            for k in range(scale_x):
                up_row[k::scale_x] = row
            # Vertical duplicate: write up_row to scale_y consecutive dst rows
            for k in range(scale_y):
                dy = sy * scale_y + k
                dc[dy * dw : (dy + 1) * dw] = up_row
