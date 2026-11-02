"""Shared helpers for the interactive demo entry points.

Spec:
    Centralize reusable demo-setup logic so each example script can focus on its
    scene setup and rendering behavior instead of repeating CLI parsing code.

Contract:
    This module must stay small, dependency-free, and safe to import from any
    example module in the project.

Behavior:
    The helper accepts a command-line argument list and returns the requested
    window resolution when present, otherwise it falls back to the provided
    default value.

Reference:
    See IMPLEMENTATION_SPEC.md for the repo-level conventions used by the demos
    and the shared helper layer.
"""

from __future__ import annotations

import sys
from pathlib import Path


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
