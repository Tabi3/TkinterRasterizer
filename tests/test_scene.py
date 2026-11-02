"""Scene graph tests covering traversal, transforms, and light visibility.

These tests validate the public scene abstraction used during rendering without
relying on a graphical backend.
"""

from Rasterizer.algebra.math3d import Mat4, Vec3, Vec4
from Rasterizer.geometry.mesh import AABB
from Rasterizer.scene.scene import Light, Node, lights_affecting_aabb


def test_node_traversal_composes_transform():
    root = Node(
        name="root",
        local_transform=Mat4.from_rows(
            (1, 0, 0, 1), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
        ),
    )
    child = Node(
        name="child",
        local_transform=Mat4.from_rows(
            (1, 0, 0, 0), (0, 1, 0, 2), (0, 0, 1, 0), (0, 0, 0, 1)
        ),
    )
    root.add_child(child)

    visited = []
    root.traverse(Mat4.identity(), lambda n: visited.append(n.name))
    assert visited == ["root", "child"]

    p = child.world_transform @ Vec4(0, 0, 0, 1)
    assert abs(p.x - 1.0) < 1e-9
    assert abs(p.y - 2.0) < 1e-9
    assert abs(p.z - 0.0) < 1e-9


def test_light_direction_normalized():
    light = Light(kind="directional", direction=Vec3(3, 0, 0))
    assert abs(light.direction.length() - 1.0) < 1e-9


def test_lights_affecting_aabb_directional():
    box = AABB(0, 0, 0, 1, 1, 1)
    light = Light(kind="directional", direction=Vec3(0, -1, 0))
    result = lights_affecting_aabb(box, [light])
    assert len(result) == 1


def test_lights_affecting_aabb_point_in_range():
    box = AABB(0, 0, 0, 1, 1, 1)
    light = Light(kind="point", position=Vec3(0.5, 0.5, 0.5), radius=2.0)
    result = lights_affecting_aabb(box, [light])
    assert len(result) == 1


def test_lights_affecting_aabb_point_out_of_range():
    box = AABB(0, 0, 0, 1, 1, 1)
    light = Light(kind="point", position=Vec3(100, 100, 100), radius=2.0)
    result = lights_affecting_aabb(box, [light])
    assert len(result) == 0


def test_lights_affecting_aabb_disabled():
    box = AABB(0, 0, 0, 1, 1, 1)
    light = Light(kind="directional")
    light.enabled = False
    result = lights_affecting_aabb(box, [light])
    assert len(result) == 0
