"""Analytical coverage, interpolation, projection, and render-state checks."""

import math
import random

import pytest

from Rasterizer.algebra.math3d import Mat4, Vec3, Vec4
from Rasterizer.geometry.mesh import AABB, Mesh, Vertex
from Rasterizer.materials.material import Material
from Rasterizer.materials.shaders import make_gouraud_shader, make_textured_lit_shader
from Rasterizer.materials.texture import Texture
from Rasterizer.rendering.backend import aabb_visible, clip_near_shaded, fill_gradient
from Rasterizer.rendering.framebuffer import Framebuffer
from Rasterizer.rendering.rasterizer import (
    draw_line,
    rasterize_triangle,
    rasterize_triangle_shaded,
)
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms, _clip_view_segment
from Rasterizer.scene.scene import Light, Node


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_pixel_center_depth_and_perspective_varyings(count, reverse):
    vertices = [(0.0, 0.0, 0.2, 1.0), (15.0, 0.0, 0.8, 0.5), (0.0, 15.0, 0.2, 0.25)]
    varying = [tuple(float(i + k) for k in range(count)) for i in range(3)]
    if reverse:
        vertices[1], vertices[2] = vertices[2], vertices[1]
        varying[1], varying[2] = varying[2], varying[1]
    fb = Framebuffer(16, 16)
    captured = []
    rasterize_triangle_shaded(
        fb, *vertices, *varying, lambda v: captured.append(v) or 0xFFFFFF
    )
    assert fb.get_depth(0, 0) == pytest.approx(0.22)
    # First covered pixel is (0.5, 0.5), with these screen barycentrics.
    weights = (14 / 15, 0.5 / 15, 0.5 / 15)
    denominator = sum(w * vertex[3] for w, vertex in zip(weights, vertices))
    expected = tuple(
        sum(
            w * vertex[3] * values[k]
            for w, vertex, values in zip(weights, vertices, varying)
        )
        / denominator
        for k in range(count)
    )
    assert captured[0] == pytest.approx(expected)


def test_flat_and_shaded_depth_agree_for_random_triangles():
    rng = random.Random(2718)
    for _ in range(30):
        vertices = [
            (rng.uniform(-8, 40), rng.uniform(-8, 40), rng.uniform(0.1, 0.9), 1.0)
            for _ in range(3)
        ]
        flat, shaded = Framebuffer(32, 32), Framebuffer(32, 32)
        rasterize_triangle(flat, *vertices, 0xFFFFFF)
        rasterize_triangle_shaded(shaded, *vertices, (), (), (), lambda v: 0xFFFFFF)
        assert flat.color == shaded.color
        assert list(flat.depth) == pytest.approx(list(shaded.depth), abs=2e-6)


def test_shaded_depth_write_disabled_still_tests_existing_depth():
    fb = Framebuffer(8, 8)
    vertices = ((0, 0, 0.5, 1), (8, 0, 0.5, 1), (0, 8, 0.5, 1))
    fb.put_depthtested(0, 0, 0xFF0000, 0.75)
    before = fb.depth[:]
    rasterize_triangle_shaded(
        fb, *vertices, (), (), (), lambda v: 0xFFFFFF, depth_write=False
    )
    assert fb.depth == before
    assert fb.get(0, 0) == 0xFF0000
    assert fb.get(1, 1) == 0xFFFFFF


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_shaded_rejects_invalid_inverse_w(bad):
    with pytest.raises(ValueError):
        rasterize_triangle_shaded(
            Framebuffer(4, 4),
            (0, 0, 0.5, bad),
            (4, 0, 0.5, 1),
            (0, 4, 0.5, 1),
            (),
            (),
            (),
            lambda v: 1,
        )


def test_mismatched_varyings_rejected():
    with pytest.raises(ValueError, match="varying"):
        rasterize_triangle_shaded(
            Framebuffer(4, 4),
            (0, 0, 0.5, 1),
            (4, 0, 0.5, 1),
            (0, 4, 0.5, 1),
            (0,),
            (),
            (),
            lambda v: 1,
        )


def test_orthographic_has_affine_attributes_and_linear_reverse_depth():
    camera = Camera(
        Vec3.zero(),
        Vec3(0, 0, -1),
        projection="orthographic",
        near=1,
        far=11,
        ortho_scale=1,
    )
    vertices = [
        camera.project(Vec4(-1, 1, -2, 1), 1),
        camera.project(Vec4(1, 1, -8, 1), 1),
        camera.project(Vec4(-1, -1, -2, 1), 1),
    ]
    assert [v.w for v in vertices] == [1, 1, 1]
    assert [v.z for v in vertices] == pytest.approx([0.9, 0.3, 0.9])
    fb, captured = Framebuffer(10, 10), []
    rasterize_triangle_shaded(
        fb,
        *((5 * (v.x + 1), 5 * (1 - v.y), v.z, 1) for v in vertices),
        (0,),
        (1,),
        (0,),
        lambda v: captured.append(v) or 1,
    )
    assert captured[0] == pytest.approx((0.05,))
    assert fb.get_depth(0, 0) == pytest.approx(0.87)


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_scene_honors_cull_depth_write_and_instancing(projection):
    mesh = Mesh(
        [
            Vertex(Vec3(-0.5, -0.5, -2)),
            Vertex(Vec3(0, 0.5, -2)),
            Vertex(Vec3(0.5, -0.5, -2)),
        ]
    )
    material = Material(
        lambda v, u: (u.transform(v.pos), ()),
        lambda v: 0xFFFFFF,
        cull=False,
        depth_write=False,
    )
    renderer, fb = Renderer(32, 32), Framebuffer(32, 32)
    camera = Camera(Vec3.zero(), Vec3(0, 0, -1), projection=projection)
    renderer.draw_scene(fb, Node(mesh, material), camera, Uniforms())
    assert any(fb.color)
    assert not any(fb.depth)
    scene_color = fb.color[:]
    fb.clear(0)
    renderer.draw_instanced(fb, mesh, material, Uniforms(), camera, [Mat4.identity()])
    assert fb.color == scene_color
    assert not any(fb.depth)
    material.cull = True
    fb.clear(0)
    renderer.draw_scene(fb, Node(mesh, material), camera, Uniforms())
    assert not any(fb.color)


def test_invisible_scene_skips_vertex_shader():
    mesh = Mesh([Vertex(Vec3(0, 0, 1))] * 3)

    def unexpected(v, u):
        pytest.fail("hidden geometry reached the vertex shader")

    Renderer(8, 8).draw_scene(
        Framebuffer(8, 8),
        Node(mesh, Material(unexpected, lambda v: 1)),
        Camera(Vec3.zero(), Vec3(0, 0, -1)),
        Uniforms(),
    )


@pytest.mark.parametrize("textured", [False, True])
def test_normals_use_inverse_transpose_and_handle_coincident_light(textured):
    vs, _ = (
        make_textured_lit_shader(Texture.solid(1, 1, 0xFFFFFF))
        if textured
        else make_gouraud_shader(light_color=(1, 1, 1), ambient=0)
    )
    u = Uniforms()
    u.ambient = 0
    u.model = Mat4((2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
    u.lights = [Light(direction=Vec3(-1, 0, 0), color=(1, 1, 1))]
    _, varying = vs(Vertex(Vec3.zero(), normal=Vec3(1, 1, 0)), u)
    assert varying[-3:] == pytest.approx((1 / math.sqrt(5),) * 3)
    assert u.normal_matrix is u.normal_matrix
    u.lights = [Light(kind="point", position=Vec3.zero())]
    _, varying = vs(Vertex(Vec3.zero(), normal=Vec3.unit_y()), u)
    assert varying[-3:] == (0, 0, 0)


def test_lit_texture_reflects_pixel_updates():
    texture = Texture.solid(1, 1, 0xFFFFFF)
    _, fs = make_textured_lit_shader(texture)
    texture.data[0] = 0x123456
    assert fs((0, 0, 1, 1, 1)) == 0x123456


def test_orthographic_clips_near_and_far_and_preserves_varyings():
    vertices = [
        (Vec4(0, 0, 1.5, 1), (1.5,)),
        (Vec4(1, 0, 0.5, 1), (0.5,)),
        (Vec4(0, 1, -0.5, 1), (-0.5,)),
    ]
    clipped = clip_near_shaded(vertices, 0.1, orthographic=True)
    assert clipped
    for triangle in clipped:
        for vertex, varying in triangle:
            assert 0 <= vertex.z <= 1
            assert varying == pytest.approx((vertex.z,))


def test_perspective_preserves_behind_eye_coordinates_for_clipping():
    camera = Camera(Vec3.zero(), Vec3(0, 0, -1))
    behind = camera.project(Vec4(2, 3, 1, 1), 1)
    assert behind.w == -1
    assert behind.x != 0 and behind.y != 0


def test_near_plane_aabb_is_visible():
    assert aabb_visible(AABB(-1, -1, -0.1, 1, 1, -0.1), Mat4.identity(), 0.1)


def test_empty_mesh_and_invalid_indices():
    empty = Mesh([])
    assert empty.aabb.is_empty()
    Renderer(4, 4).draw_mesh_shaded(
        Framebuffer(4, 4),
        empty,
        lambda v, u: None,
        lambda v: 1,
        Uniforms(),
        Camera(Vec3.zero(), Vec3(0, 0, -1)),
        Mat4.identity(),
    )
    with pytest.raises(ValueError, match="index"):
        Mesh([Vertex(Vec3.zero())], [0, 0, 1])


@pytest.mark.parametrize(
    "near,far,fov,scale",
    [
        (0, 10, 60, 1),
        (0.1, 0.1, 60, 1),
        (0.1, 10, 180, 1),
        (0.1, 10, 60, 0),
        (math.nan, 10, 60, 1),
    ],
)
def test_invalid_camera_parameters(near, far, fov, scale):
    with pytest.raises(ValueError):
        Camera(
            Vec3.zero(),
            Vec3(0, 0, -1),
            near=near,
            far=far,
            fov_y_deg=fov,
            ortho_scale=scale,
        )


def test_target_size_mismatch_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        Renderer(4, 4).draw_ndc_triangle(
            Framebuffer(8, 8), (0, 0, 0.5), (1, 0, 0.5), (0, 1, 0.5), 1
        )


def test_line_clips_huge_endpoints_and_interpolates_depth():
    fb = Framebuffer(8, 8)
    draw_line(fb, -1e9, 2, 0.25, 1e9, 2, 0.75, 0xFFFFFF)
    for x in range(8):
        assert fb.get(x, 2) == 0xFFFFFF
        assert fb.get_depth(x, 2) == pytest.approx(0.5)
    assert sum(bool(c) for c in fb.color) == 8


def test_wire_segment_clips_in_view_space():
    camera = Camera(Vec3.zero(), Vec3(0, 0, -1), near=1)
    a, b = _clip_view_segment(Vec4(-1, 0, 1, 1), Vec4(1, 0, -3, 1), camera)
    assert (a.x, a.z, b.x, b.z) == pytest.approx((0, -1, 1, -3))


def test_gradient_top_and_bottom():
    fb = Framebuffer(2, 3)
    fill_gradient(fb, 0xFF0000, 0x0000FF)
    assert fb.get(0, 0) == 0xFF0000
    assert fb.get(0, 2) == 0x0000FF
    assert not any(fb.depth)


def test_scene_reparenting_and_cycles():
    a, b, child = Node(), Node(), Node()
    a.add_child(child)
    b.add_child(child)
    assert child not in a.children
    b.add_child(child)
    assert b.children == [child]
    with pytest.raises(ValueError, match="cycles"):
        child.add_child(b)


@pytest.mark.parametrize("offset", [0.0, 0.25, 0.5])
@pytest.mark.parametrize("reverse", [False, True])
def test_shared_edges_have_exactly_one_owner(offset, reverse):
    corners = [
        (2 + offset, 2 + offset, 0.5, 1),
        (10 + offset, 2 + offset, 0.5, 1),
        (10 + offset, 10 + offset, 0.5, 1),
        (2 + offset, 10 + offset, 0.5, 1),
    ]
    for indices in ((0, 1, 2, 0, 2, 3), (0, 1, 3, 1, 2, 3)):
        ownership = [0] * 256
        for start in (0, 3):
            vertices = [corners[i] for i in indices[start : start + 3]]
            if reverse:
                vertices.reverse()
            fb = Framebuffer(16, 16)
            rasterize_triangle_shaded(fb, *vertices, (), (), (), lambda v: 1)
            for i, color in enumerate(fb.color):
                ownership[i] += bool(color)
        assert max(ownership) == 1
        assert sum(ownership) == 64


def test_singular_normal_matrix_and_cache_invalidation():
    u = Uniforms()
    cached = u.normal_matrix
    u.model = Mat4((0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))
    assert u.normal_matrix is not cached
    assert u.normal_matrix.data == (0.0,) * 9
    u.model = Mat4.identity()
    assert u.normal_matrix.data == cached.data


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan])
def test_invalid_point_light_radius(value):
    with pytest.raises(ValueError, match="radius"):
        Light(kind="point", radius=value)


@pytest.mark.parametrize("projection", ["perspective", "orthographic"])
def test_flat_and_shaded_pipeline_agree_across_depth_clip_planes(projection):
    camera = Camera(
        Vec3.zero(), Vec3(0, 0, -1), projection=projection, near=1, far=5, ortho_scale=2
    )
    mesh = Mesh(
        [Vertex(Vec3(-1, -1, 0.5)), Vertex(Vec3(1, -1, -3)), Vertex(Vec3(0, 1, -7))]
    )
    flat, shaded = Framebuffer(32, 32), Framebuffer(32, 32)
    renderer = Renderer(32, 32)
    renderer.draw_mesh(flat, mesh, Mat4.identity(), 0xFFFFFF, camera)
    renderer.draw_mesh_shaded(
        shaded,
        mesh,
        lambda v, u: (u.transform(v.pos), ()),
        lambda v: 0xFFFFFF,
        Uniforms(),
        camera,
        Mat4.identity(),
    )
    assert any(flat.color)
    assert flat.color == shaded.color
    assert list(flat.depth) == pytest.approx(list(shaded.depth), abs=2e-6)


def test_renderer_preserves_explicit_time_across_repeated_draws():
    mesh = Mesh(
        [
            Vertex(Vec3(-0.5, -0.5, -2)),
            Vertex(Vec3(0.5, -0.5, -2)),
            Vertex(Vec3(0, 0.5, -2)),
        ]
    )
    camera = Camera(Vec3.zero(), Vec3(0, 0, -1))
    renderer, fb, u = Renderer(16, 16), Framebuffer(16, 16), Uniforms()
    u.time, u.delta_time = 2.5, 0.125
    samples = []

    def vertex_shader(vertex, uniforms):
        samples.append((uniforms.time, uniforms.delta_time))
        return uniforms.transform(vertex.pos), ()

    root = Node(mesh, Material(vertex_shader, lambda v: 1))
    outputs = []
    for _ in range(3):
        fb.clear(0)
        renderer.draw_scene(fb, root, camera, u)
        outputs.append(fb.color[:])
    assert any(outputs[0])
    assert outputs[0] == outputs[1] == outputs[2]
    assert samples == [(2.5, 0.125)] * 9
    assert (u.time, u.delta_time) == (2.5, 0.125)
