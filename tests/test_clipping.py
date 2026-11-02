"""Near-plane clipping regression tests."""

from Rasterizer.algebra.math3d import Vec4
from Rasterizer.rendering.backend import clip_near_shaded, clip_near_triangle


def test_clip_near_triangle_keeps_fully_visible_triangle():

    vertices = (
        Vec4(0, 0, 0.1, 1),
        Vec4(1, 0, 0.2, 2),
        Vec4(0, 1, 0.3, 3),
    )
    assert clip_near_triangle(*vertices, near=0.5) == [(vertices[1], vertices[2], vertices[0])]


def test_clip_near_triangle_discards_fully_hidden_triangle():

    vertices = (
        Vec4(0, 0, 0, 0.1),
        Vec4(1, 0, 0, 0.2),
        Vec4(0, 1, 0, 0.3),
    )
    assert clip_near_triangle(*vertices, near=0.5) == []


def test_clip_near_triangle_clips_one_hidden_vertex_into_quad():

    vertices = (
        Vec4(0, 0, 0, 0.25),
        Vec4(2, 0, 0, 1.0),
        Vec4(0, 2, 0, 1.0),
    )
    triangles = clip_near_triangle(*vertices, near=0.5)
    assert len(triangles) == 2
    assert all(vertex.w >= 0.5 for triangle in triangles for vertex in triangle)


def test_clip_near_shaded_interpolates_varyings_on_cut_edges():

    vertices = (
        (Vec4(0, 0, 0, 0.25), (0.0, 2.0)),
        (Vec4(2, 0, 0, 1.0), (1.0, 4.0)),
        (Vec4(0, 2, 0, 1.0), (0.0, 6.0)),
    )
    triangles = clip_near_shaded(vertices, near=0.5)
    clipped_vertices = [vertex for triangle in triangles for vertex in triangle]
    assert len(triangles) == 2
    assert all(clip.w >= 0.5 for clip, _ in clipped_vertices)
    assert (1 / 3, 8 / 3) in [varying for _, varying in clipped_vertices]
