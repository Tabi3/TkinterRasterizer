"""Clipped opaque UI rectangles and a code-defined 5x7 bitmap font."""

from array import array
from functools import lru_cache

from ..data.font5x7 import ROWS
from .framebuffer import Framebuffer


def _runs(rows: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    """Compile contiguous glyph spans once; runtime painting copies whole runs."""

    result = []

    for y, row in enumerate(rows):
        x = 0

        while x < 5:
            if not row & (16 >> x):
                x += 1
                continue

            start = x

            while x < 5 and row & (16 >> x):
                x += 1

            result.append((start, y, x - start))

    return tuple(result)


# Spec: IMPLEMENTATION_SPEC.md#bulk-pixel-operations.
# Cached arrays are private read-only sources, never exposed as writable targets.
_RUNS = {char: _runs(rows) for char, rows in ROWS.items()}


@lru_cache(maxsize=128)
def _solid_row(width: int, color: int) -> array[int]:
    return array("I", [color]) * width


def fill_rect(fb: Framebuffer, x: int, y: int, width: int, height: int, color: int) -> None:
    """Fill a clipped opaque rectangle, preserving the depth buffer."""

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fb.width, x + width), min(fb.height, y + height)

    if x0 >= x1 or y0 >= y1:
        return

    row = _solid_row(x1 - x0, color & 0xFFFFFF)

    for py in range(y0, y1):
        start = py * fb.width + x0
        fb.color[start : start + len(row)] = row


def draw_text(fb: Framebuffer, text: str, x: int, y: int, color: int, scale: int = 2) -> None:
    """Draw clipped bitmap text using reusable C-backed span copies.

    Spec: IMPLEMENTATION_SPEC.md#bulk-pixel-operations. Newlines advance nine
    source pixels; unsupported characters use the question-mark glyph.
    """

    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError("font scale must be a positive integer")

    origin = x

    for char in text.upper():
        if char == "\n":
            x, y = origin, y + 9 * scale
            continue

        # Spec: IMPLEMENTATION_SPEC.md#bulk-pixel-operations.
        # One packed slice per span replaces per-bit rectangle construction.
        for px, py, length in _RUNS.get(char, _RUNS["?"]):
            fill_rect(fb, x + px * scale, y + py * scale, length * scale, scale, color)

        x += 6 * scale
