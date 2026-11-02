"""Texture storage and sampling helpers.

Spec:
    Represent a pixel buffer with fixed-row-major layout and nearest-neighbor
    sampling for UV-driven shading and procedural texture passes.

Contract:
    - `data` stores packed 32-bit colors in `array('I')` layout.
    - `sample(u, v)` expects normalized coordinates in `[0, 1]` unless wrapping
      is enabled.
    - `wrap` may be `'clamp'` or `'repeat'`.

Behavior:
    The class is used as a small, reusable texture resource by the shader layer.
    The shader callbacks may inline texture sampling for performance, while the
    texture object itself remains convenient for tests and helper use.

Reference:
    See IMPLEMENTATION_SPEC.md for the rendering architecture and the texture
    layer's role in the broader pipeline.
"""

import math
from array import array


class Texture:
    __slots__ = ("data", "height", "width", "wrap")

    def __init__(self, width, height, data=None, fill=0x808080, wrap="clamp"):

        if not (isinstance(width, int) and isinstance(height, int)):
            raise TypeError("texture dimensions must be integers")

        if width <= 0 or height <= 0:
            raise ValueError(f"texture size must be positive, got {width}x{height}")

        if wrap not in {"clamp", "repeat"}:
            raise ValueError(f"unsupported texture wrap: {wrap!r}")

        self.width  = int(width)
        self.height = int(height)
        self.wrap   = wrap

        if data is None:
            data = array("I", [fill]) * (self.width * self.height)

        elif len(data) != self.width * self.height:
            raise ValueError(f"data length {len(data)} != width*height ({self.width * self.height})")

        self.data = data if isinstance(data, array) else array("I", data)

    # ----- sampling ------------------------------------------------------

    def sample(self, u, v):

        if self.wrap == "clamp":
            if      u < 0.0 : u = 0.0
            elif    u > 1.0 : u = 1.0

            if      v < 0.0 : v = 0.0
            elif    v > 1.0 : v = 1.0

        else:  # 'repeat'
            if u < 0.0 or u >= 1.0: u = u - math.floor(u)
            if v < 0.0 or v >= 1.0: v = v - math.floor(v)

        ix = int(u * self.width)

        if ix >= self.width: ix = self.width - 1

        iy = int(v * self.height)

        if iy >= self.height: iy = self.height - 1

        return self.data[iy * self.width + ix]

    # ----- constructors --------------------------------------------------
    @classmethod
    def solid(cls, width, height, color):

        return cls(width, height, None, fill=color)

    @classmethod
    def checkerboard(cls, size=64, cells=8, color_a=0xFFFFFF, color_b=0x404040):

        w = h = size

        cell_size = size / cells

        data = array("I", [0]) * (w * h)

        for y in range(h):
            cy = int(y / cell_size)

            for x in range(w):
                cx = int(x / cell_size)

                data[y * w + x] = color_a if (cx + cy) % 2 == 0 else color_b

        return cls(w, h, data)

    @classmethod
    def from_ppm(cls, path):
        """Load a binary P6 PPM."""

        with open(path, "rb") as f:

            def next_token():

                while True:
                    c = f.read(1)

                    if not c:
                        raise ValueError("unexpected EOF in PPM header")

                    if c == b"#":
                        while True:
                            c = f.read(1)
                            if not c or c == b"\n": break

                    elif not c.isspace(): break

                tok = c

                while True:
                    c = f.read(1)

                    if not c or c.isspace(): return tok

                    tok += c

            magic = next_token()

            if magic != b"P6": raise ValueError(f"expected P6 PPM, got {magic!r}")

            w = int(next_token())
            h = int(next_token())

            maxval = int(next_token())

            if maxval != 255: raise ValueError(f"only maxval=255 supported, got {maxval}")

            rgb = f.read(w * h * 3)

            if len(rgb) != w * h * 3:
                raise ValueError(f"short PPM: {len(rgb)}/{w * h * 3} bytes")

        data = array("I", [0]) * (w * h)

        for i in range(w * h):
            data[i] = (rgb[i * 3] << 16) | (rgb[i * 3 + 1] << 8) | rgb[i * 3 + 2]

        return cls(w, h, data)

    @classmethod
    def from_framebuffer(cls, fb):

        return cls(fb.width, fb.height, array("I", fb.color))

    def __repr__(self):

        return f"Texture({self.width}x{self.height}, {self.wrap})"


# ---------------------------------------------------------------------------
