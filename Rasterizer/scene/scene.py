"""Scene graph: Node, Light, and spatial queries.

Spec:
    Node forms a tree with local transforms composed top-down during a draw
    traversal. Mesh and Material are optional per node; a node without a mesh
    is a pure transform. Light is a scene-level object, held in a flat list,
    not attached to nodes.

Contract:
    - Node.world_transform is only valid during a traversal (set by parent
      before visitor is invoked).
    - Light.direction must be unit-length; normalized in __init__.
    - Node.local_transform is the source of truth; world_transform is derived.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..algebra.math3d import Mat4, Vec3
from ..geometry.mesh import Mesh
from ..materials.material import Material


class Light:
    """Lighting source for the scene graph.

    A light can be directional or positional. Directional lights are treated as
    infinitely far away and only contribute a direction vector to the shader,
    while point lights also contribute attenuation and a distance-based falloff
    term. The scene keeps these lights in a flat list, which keeps the render API
    simple while leaving the actual shading logic to the per-material shader.
    """

    __slots__ = (
        "color",
        "direction",
        "enabled",
        "intensity",
        "kind",
        "position",
        "radius",
    )

    def __init__(
        self,
        kind: str = "directional",
        position: Vec3 | None = None,
        direction: Vec3 | None = None,
        color: tuple[float, float, float] = (1.0, 0.95, 0.85),
        intensity: float = 1.0,
        radius: float = 10.0,
    ) -> None:
        if kind not in {"directional", "point"}:
            raise ValueError(f"unsupported light kind: {kind!r}")
        if not (isinstance(color, tuple) and len(color) == 3):
            raise TypeError("color must be a 3-tuple of floats")
        if not (
            isinstance(intensity, (int, float)) and not isinstance(intensity, bool)
        ):
            raise TypeError("intensity must be a real numeric value")
        if not (isinstance(radius, (int, float)) and not isinstance(radius, bool)):
            raise TypeError("radius must be a real numeric value")
        self.kind = kind
        self.position = position or Vec3.zero()
        self.direction = (direction or Vec3(0, -1, 0)).normalize()
        self.color = color
        self.intensity = intensity
        self.radius = radius
        self.enabled = True


class Node:
    """Renderable node in a scene graph with parent/child transform composition.

    Each node holds an optional mesh and material, as well as a local transform
    that is composed into a world transform during traversal. This lets the public
    scene API remain simple while the renderer can walk the tree and draw every
    visible object using the same traversal contract.
    """

    __slots__ = (
        "children",
        "local_transform",
        "material",
        "mesh",
        "name",
        "parent",
        "world_transform",
    )

    def __init__(
        self,
        mesh: Mesh | None = None,
        material: Material | None = None,
        local_transform: Mat4 | None = None,
        name: str = "node",
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("node name must be a string")
        self.name = name
        self.mesh = mesh
        self.material = material
        self.local_transform = (
            local_transform if local_transform is not None else Mat4.identity()
        )
        self.world_transform = Mat4.identity()
        self.children: list[Node] = []
        self.parent: Node | None = None

    def add_child(self, child: Node) -> Node:
        child.parent = self
        self.children.append(child)
        return child

    def traverse(self, parent_world: Mat4, visitor: Callable[[Node], None]) -> None:
        """Pre-order traversal. Sets self.world_transform, then calls
        visitor(self), then recurses into children.
        """
        self.world_transform = parent_world @ self.local_transform
        visitor(self)
        for child in self.children:
            child.traverse(self.world_transform, visitor)

    def world_aabb(self):
        """AABB in world space. Requires world_transform to be current
        (only valid inside a traversal visitor, or after a full traversal).
        """
        if self.mesh is None:
            raise ValueError("Mesh is None")
        return self.mesh.aabb.transform(self.world_transform)


def lights_affecting_aabb(world_aabb, lights: Sequence[Light]) -> list[Light]:
    """Return lights that can contribute to geometry inside `world_aabb`.

    Spec:
        Directional lights always contribute. Point lights are rejected if
        their position is farther than `radius` from the AABB.

    Contract:
        world_aabb may be empty; result is a list (possibly the same list
        reference if no light could be rejected).
    """
    out: list[Light] = []
    for light in lights:
        if not light.enabled:
            continue
        if light.kind == "directional":
            out.append(light)
        else:
            if world_aabb.expand(light.radius).contains_point(light.position):
                out.append(light)
    return out
