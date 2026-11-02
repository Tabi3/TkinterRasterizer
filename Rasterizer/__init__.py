"""Rasterizer package root.

Spec:
    Expose the public rendering API for the project while keeping the runtime
    implementation in purpose-based packages and the examples in the
    `Rasterizer.demos` package.

Contract:
    The package root should remain a thin convenience layer that imports the
    stable, user-facing rendering surface without exposing the internal backend
    details.

Behavior:
    Library users can import the renderer from the package root, while demo
    scripts call the same API through the dedicated example package.

Reference:
    See IMPLEMENTATION_SPEC.md for the repository architecture, project
    conventions, and refactor contract that govern this package layout.
"""

from .cli import parse_resolution
from .rendering.renderer import Camera, Object3D, Renderer, Uniforms

__all__ = [
    "Camera",
    "Object3D",
    "Renderer",
    "Uniforms",
    "parse_resolution",
]

__version__ = "0.1.0"
