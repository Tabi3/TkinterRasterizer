"""Geometry regression tests for mesh topology, bounds, and procedural primitives.

These checks protect the model-building layer used by the renderer and ensure the
triangle winding and normals remain stable across the generated geometry.
"""

import math

from Rasterizer.algebra.math3d import Mat4, Vec3
from Rasterizer.geometry.mesh import (
    AABB,
    MeshBuilder,
    Vertex,
    make_cone,
    make_cube,
    make_cylinder,
    make_plane,
    make_sphere,
    make_torus,
)


def _triangles_face_outward(mesh, inside_point):
    for t in range(mesh.triangle_count()):
        v0, v1, v2 = mesh.triangle_vertices(t)
        n = (v1.pos - v0.pos).cross(v2.pos - v0.pos)
        centroid = (v0.pos + v1.pos + v2.pos) / 3.0
        if n.dot(centroid - inside_point) <= 0.0:
            return False
    return True


def _triangles_agree_with_vertex_normals(mesh):
    for t in range(mesh.triangle_count()):
        v0, v1, v2 = mesh.triangle_vertices(t)
        if v0.normal is None or v1.normal is None or v2.normal is None:
            continue
        n = (v1.pos - v0.pos).cross(v2.pos - v0.pos)
        avg = (v0.normal + v1.normal + v2.normal) / 3.0
        if n.dot(avg) <= 0.0:
            return False
    return True


def test_aabb_empty():
    box = AABB.empty()
    assert box.is_empty()
    assert not box.intersects(AABB(0, 0, 0, 1, 1, 1))


def test_aabb_from_vertices():
    box = AABB.from_vertices(
        [
            Vertex(Vec3(1, 2, 3)),
            Vertex(Vec3(-1, 0, 1)),
            Vertex(Vec3(0, 4, 2)),
        ]
    )
    assert box.min_x == -1 and box.max_x == 1
    assert box.min_y == 0 and box.max_y == 4
    assert box.min_z == 1 and box.max_z == 3


def test_aabb_union_expand():
    b1 = AABB(0, 0, 0, 1, 1, 1)
    b2 = AABB(2, 2, 2, 3, 3, 3)
    u = b1.union(b2)
    assert u.min_x == 0 and u.max_x == 3
    assert b1.expand(0.5).contains_point(Vec3(-0.4, -0.4, -0.4))


def test_aabb_transform():
    box = AABB(-1, -1, -1, 1, 1, 1)
    translation = Mat4.from_rows((1, 0, 0, 5), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    t = box.transform(translation)
    assert abs(t.min_x - 4.0) < 1e-9
    assert abs(t.max_x - 6.0) < 1e-9


def test_mesh_builder_basic():
    mb = MeshBuilder()
    i0 = mb.vertex(Vec3(0, 0, 0))
    i1 = mb.vertex(Vec3(1, 0, 0))
    i2 = mb.vertex(Vec3(0, 1, 0))
    mb.tri(i0, i1, i2)
    m = mb.build()
    assert m.triangle_count() == 1
    assert m.triangle(0) == (0, 1, 2)
    assert len(m.edges()) == 3


def test_cube_winding():
    cube = make_cube(size=2.0)
    assert cube.triangle_count() == 12
    assert _triangles_face_outward(cube, Vec3.zero())
    assert _triangles_agree_with_vertex_normals(cube)


def test_plane_faces_up():
    plane = make_plane(size=4.0, subdivisions=2)
    for t in range(plane.triangle_count()):
        v0, v1, v2 = plane.triangle_vertices(t)
        n = (v1.pos - v0.pos).cross(v2.pos - v0.pos)
        assert n.y > 0


def test_sphere_normals_radial():
    sphere = make_sphere(radius=1.0, segments=8, rings=6)
    for v in sphere.vertices:
        assert abs(v.pos.length() - 1.0) < 1e-9
        assert abs(v.normal.dot(v.pos.normalize()) - 1.0) < 1e-9


def test_cylinder_winding():
    cyl = make_cylinder(radius=1.0, height=2.0, segments=12)
    assert _triangles_face_outward(cyl, Vec3.zero())


def test_cone_winding():
    cone = make_cone(radius=1.0, height=2.0, segments=12)
    assert _triangles_face_outward(cone, Vec3.zero())


def test_torus_geometry():
    torus = make_torus(
        major_radius=1.0, minor_radius=0.3, major_segments=12, minor_segments=8
    )
    for v in torus.vertices:
        r_xz = math.hypot(v.pos.x, v.pos.z)
        d = math.hypot(r_xz - 1.0, v.pos.y)
        assert abs(d - 0.3) < 1e-9
    assert _triangles_agree_with_vertex_normals(torus)


def test_mesh_transform_scale():
    cube = make_cube(size=2.0)
    scaling = Mat4.from_rows((2, 0, 0, 5), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 1))
    scaled = cube.transform(scaling)
    assert abs(scaled.aabb.min_x - 3.0) < 1e-9
    assert abs(scaled.aabb.max_x - 7.0) < 1e-9
