"""Compatibility imports for the UI scene module.

Spec: TextTwister/IMPLEMENTATION_SPEC.md#organization-and-style.
"""

from .ui.scene3d import (
    PROJECTIONS,
    BoxImage,
    Point,
    Scene3D,
    Triangle,
    blit,
    contains,
    labelled_box,
    projection_for_stage,
)


__all__ = [
    "PROJECTIONS",
    "BoxImage",
    "Point",
    "Scene3D",
    "Triangle",
    "blit",
    "contains",
    "labelled_box",
    "projection_for_stage",
]
