"""Mutable per-view widgets. Spec: TextTwister/IMPLEMENTATION_SPEC.md#presentation-and-input."""

from dataclasses import dataclass

from .scene3d import Triangle, contains


@dataclass(slots=True)
class Button:
    label: str

    action: str

    x: int

    y: int

    width: int

    height: int = 36

    enabled: bool = True

    triangles: tuple[Triangle, ...] = ()

    def contains(self, x: int, y: int) -> bool:

        if not (self.x <= x < self.x + self.width and self.y <= y < self.y + self.height):
            return False

        return any(contains(triangle, x - self.x, y - self.y) for triangle in self.triangles)
