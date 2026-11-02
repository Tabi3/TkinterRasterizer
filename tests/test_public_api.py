"""Public convenience API, owned mutable scratch, and C-backed pixel copies."""

import math
from array import array

import pytest

from Rasterizer import (
    Camera,
    Framebuffer,
    Mat4,
    Material,
    Node,
    Renderer,
    Texture,
    Uniforms,
    Vec3,
    make_cube,
)
from Rasterizer.algebra.affine import write_trs
from Rasterizer.algebra.transforms import rotate_x, rotate_y, rotate_z, scale, translate
from Rasterizer.data.font5x7 import ROWS
from Rasterizer.rendering.drawing2d import draw_text, fill_rect


def test_public_and_legacy_camera_uniform_imports_are_identical():
    from Rasterizer.rendering.renderer import Camera as LegacyCamera
    from Rasterizer.rendering.renderer import Uniforms as LegacyUniforms
    assert Camera is LegacyCamera
    assert Uniforms is LegacyUniforms


def test_scene_module_compatibility_imports():
    from TextTwister.scene3d import Scene3D as LegacyScene
    from TextTwister.ui.scene3d import Scene3D
    assert LegacyScene is Scene3D


@pytest.mark.parametrize("projection", ["perspective", "orthographic", "panini", "ellipsoid"])
def test_high_level_scene_render_matches_low_level_pipeline(projection):

    renderer = Renderer(64, 64)
    camera = Camera(Vec3(0, 0, 4), Vec3.zero(), projection=projection)
    root = Node(make_cube(), Material.gouraud())
    high, low = Framebuffer(64, 64), Framebuffer(64, 64)
    renderer.render(high, root, camera, background=(0x112233, 0x445566))
    renderer.draw_skybox(low, 0x112233, 0x445566)
    renderer.draw_scene(low, root, camera, renderer.uniforms)
    assert high.color == low.color
    assert high.depth == low.depth


def test_direct_draw_matches_flat_draw_and_reuses_clip_vectors():

    renderer, camera = Renderer(32, 32), Camera(Vec3(0, 0, 4), Vec3.zero())
    mesh = make_cube()
    high, low = Framebuffer(32, 32), Framebuffer(32, 32)
    renderer.draw(high, mesh, camera, color=0x123456)
    renderer.draw_mesh(low, mesh, Mat4.identity(), 0x123456, camera)
    assert high.color == low.color and high.depth == low.depth
    first = tuple(map(id, renderer._clip_buffer))
    renderer.draw(high, mesh, camera, color=0xFFFFFF)
    assert tuple(map(id, renderer._clip_buffer)) == first


def test_material_draw_honors_flags_and_explicit_time():

    renderer, camera = Renderer(32, 32), Camera(Vec3(0, 0, 4), Vec3.zero())
    material = Material.textured(Texture.solid(1, 1, 0xABCDEF), lit=True, cull=False)
    material.depth_write = False
    renderer.uniforms.time = 2.5
    renderer.uniforms.delta_time = 0.125
    fb = Framebuffer(32, 32)
    renderer.draw(fb, make_cube(), camera, material=material)
    assert any(fb.color) and not any(fb.depth)
    assert (renderer.uniforms.time, renderer.uniforms.delta_time) == (2.5, 0.125)


def test_render_can_preserve_existing_background_and_draw_wireframe():

    renderer, camera = Renderer(32, 32), Camera(Vec3(0, 0, 4), Vec3.zero())
    root, fb = Node(make_cube(), Material.gouraud()), Framebuffer(32, 32)
    fb.put(0, 0, 0x123456)
    renderer.render(fb, root, camera, background=None, wireframe=True)
    assert fb.get(0, 0) == 0x123456
    assert any(fb.depth)


@pytest.mark.parametrize(
    "angles,size",
    [
        ((0, 0, 0), (1, 1, 1)),
        ((0.2, -0.3, 0.4), (2, 0.5, 3)),
        ((math.pi, 0.4, -0.1), (-1, 2, 0.7)),
        ((0.2, 0.3, 0.4), (0, 1, 1)),
    ],
)
def test_owned_transform_writer_matches_matrix_composition(angles, size):

    yaw, pitch, roll = angles
    sx, sy, sz = size
    output = Mat4.identity()
    expected = translate(Vec3(1, 2, 3)) @ rotate_y(yaw) @ rotate_x(pitch) @ rotate_z(roll) @ scale(Vec3(*size))
    result = write_trs(output, x=1, y=2, z=3, yaw=yaw, pitch=pitch, roll=roll, sx=sx, sy=sy, sz=sz)
    assert result is output
    assert output.data == pytest.approx(expected.data, abs=1e-12)


def test_mutable_pose_invalidates_normal_cache_without_changing_snapshot():

    output = Mat4.identity()
    uniforms = Uniforms()
    uniforms.model = output
    normal = uniforms.normal_matrix
    old_data = output.data
    write_trs(output, yaw=0.5, sx=2)
    assert uniforms.normal_matrix is not normal
    assert old_data == Mat4.identity().data


def test_blit_overlap_snapshots_source_before_copying_rows():

    fb = Framebuffer(3, 3)
    fb.color[:] = array("I", range(9))
    fb.depth[:] = array("f", (i / 16 for i in range(9)))
    fb.blit(fb, 1, 1)
    assert list(fb.color) == [0, 1, 2, 3, 0, 1, 6, 3, 4]
    assert list(fb.depth) == pytest.approx([0, 1 / 16, 2 / 16, 3 / 16, 0, 1 / 16, 6 / 16, 3 / 16, 4 / 16])


@pytest.mark.parametrize("font_scale", [1, 2, 3, 4])
@pytest.mark.parametrize("position", [(-3, -2), (3, 3), (25, 20)])
def test_span_text_matches_reference_bit_painting(font_scale, position):

    fast, reference = Framebuffer(48, 40), Framebuffer(48, 40)
    text, color = "AB?\n12", 0xFEDCBA
    x, y = position
    draw_text(fast, text, x, y, color, font_scale)
    origin = x
    for char in text:
        if char == "\n":
            x, y = origin, y + 9 * font_scale
            continue
        for row, bits in enumerate(ROWS[char]):
            for column in range(5):
                if bits & (16 >> column):
                    for dy in range(font_scale):
                        for dx in range(font_scale):
                            reference.put(
                                x + column * font_scale + dx,
                                y + row * font_scale + dy,
                                color,
                            )
        x += 6 * font_scale
    assert fast.color == reference.color


def test_cached_fill_sources_never_alias_writable_destination():

    first, second = Framebuffer(4, 4), Framebuffer(4, 4)
    fill_rect(first, 0, 0, 4, 4, 123)
    first.color[0] = 456
    fill_rect(second, 0, 0, 4, 4, 123)
    assert all(color == 123 for color in second.color)
