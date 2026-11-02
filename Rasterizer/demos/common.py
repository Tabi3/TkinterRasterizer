"""Shared helpers for the interactive demo entry points.

Spec:
    Centralize reusable demo-setup logic so each example script can focus on its
    scene setup and rendering behavior instead of repeating CLI or timing logic.

Contract:
    This module must stay small, dependency-free, and safe to import from any
    example module in the project.

Behavior:
    Animation consumes caller-supplied elapsed seconds, never frame counts.
    The CLI helper accepts a command-line argument list and returns the requested
    window resolution when present, otherwise it falls back to the provided
    default value.

Reference:
    See IMPLEMENTATION_SPEC.md for the repo-level conventions used by the demos
    and the shared helper layer.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from Rasterizer.display.tk_backend import Viewer
    from Rasterizer.rendering.framebuffer import Framebuffer


def _ensure_repo_root() -> None:
    """Ensure the workspace root is importable even when a demo is run directly.

    Direct file execution adds the demo directory to sys.path, not the parent
    project root. The bootstrap keeps package imports available in that mode.
    """

    root = Path(__file__).resolve().parents[2]

    root_str = str(root)

    if root_str not in sys.path:
        sys.path.insert(0, root_str)


_ensure_repo_root()


def parse_resolution(
    argv: list[str] | tuple[str, ...],
    default: tuple[int, int, int] = (640, 360, 1),
) -> tuple[int, int, int]:
    """Parse demo arguments using the shared CLI helper."""

    from Rasterizer.cli import parse_resolution as parse

    return parse(argv, default)


def advance_time(state: dict[str, Any], dt: float) -> float:
    """Advance demo time in seconds, independent of render/presentation count.

    Compensated summation keeps long runs and different frame partitions close.
    Zero delta freezes animation; stalls consume their full elapsed time.
    """

    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise TypeError("dt must be a real number of seconds")

    if not math.isfinite(dt) or dt < 0:
        raise ValueError("dt must be finite and nonnegative")

    elapsed = state.get("elapsed", 0.0)

    if dt == 0:
        state.setdefault("elapsed", elapsed)

        return elapsed

    adjusted = dt - state.get("elapsed_error", 0.0)

    total = elapsed + adjusted

    if not math.isfinite(total):
        raise ValueError("elapsed time must remain finite")

    state["elapsed_error"] = (total - elapsed) - adjusted

    state["elapsed"] = total

    return total


def demo_viewer(draw: Callable[[Framebuffer, float], None], title: str) -> Viewer:
    """Create a CLI-configured viewer; examples only need viewer.run().

    Spec: IMPLEMENTATION_SPEC.md#api-and-organization. Tk is imported only when
    an interactive example launches, keeping headless imports side-effect free.
    """

    from Rasterizer.display.tk_backend import Viewer

    width, height, scale = parse_resolution(sys.argv)

    viewer = Viewer(width, height, title=title, render_scale=scale)

    viewer.set_frame_callback(draw)

    return viewer
