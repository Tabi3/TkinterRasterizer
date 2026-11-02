"""Interactive example applications for the Rasterizer library.

Spec:
    Package the demo entry points as lightweight examples that build a scene,
    configure the camera, and invoke the public rendering API.

Contract:
    Demos should remain thin and readable; they are not meant to hold the core
    implementation details of the engine itself.

Behavior:
    A demo script sets up its scene and then hands the actual drawing work to
    the public API from the `Rasterizer` package.

Reference:
    See IMPLEMENTATION_SPEC.md for the demo/package boundary and the project
    conventions that keep examples separate from the engine itself.
"""

from .common import parse_resolution


__all__ = ["parse_resolution"]
