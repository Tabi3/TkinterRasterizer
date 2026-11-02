"""Geometry, attached labels, projection switching and projected button hits."""

import pytest

from Rasterizer.rendering.framebuffer import Framebuffer
from TextTwister.app import GOLD, Button
from TextTwister.scene3d import (
    PROJECTIONS,
    Scene3D,
    blit,
    contains,
    labelled_box,
    projection_for_stage,
)


@pytest.mark.parametrize(
    "stage,mode",
    [
        (0, "perspective"),
        (1, "perspective"),
        (2, "orthographic"),
        (3, "panini"),
        (4, "ellipsoid"),
        (5, "perspective"),
        (8, "ellipsoid"),
        (9, "perspective"),
    ],
)
def test_projection_cycles_by_stage(stage, mode):

    assert projection_for_stage(stage) == mode


def test_front_mesh_is_subdivided_and_side_uvs_have_no_letter():

    flat, curved = labelled_box(1), labelled_box(3)
    assert flat.triangle_count() == 12
    assert curved.triangle_count() == 28
    for vertex in curved.vertices:
        assert vertex.normal is not None and vertex.uv is not None
        if vertex.normal.z <= 0:
            assert (vertex.uv.x, vertex.uv.y) == (0, 0)
    front = [v for v in curved.vertices if v.normal is not None and v.normal.z > 0]
    assert len(front) == 16
    assert len({v.uv.y for v in front if v.uv is not None}) == 4


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_text_is_rasterized_on_mesh_and_transforms_with_it(projection):

    scene = Scene3D()
    labelled = scene.box("A", 120, 96, GOLD, projection)
    blank = scene.box("", 120, 96, GOLD, projection)
    fb = labelled.framebuffer
    assert any(fb.depth)
    assert fb.depth == blank.framebuffer.depth
    changed = [i for i, (a, b) in enumerate(zip(fb.color, blank.framebuffer.color)) if a != b]
    assert len(changed) > 20
    assert all(fb.depth[i] > 0 for i in changed)
    tilted = scene.box("A", 120, 96, GOLD, projection, yaw=-0.4, pitch=0.3)
    assert tilted.framebuffer.color != fb.color
    assert tilted.framebuffer.depth != fb.depth
    assert tilted.triangles != labelled.triangles


@pytest.mark.parametrize("projection", PROJECTIONS)
def test_button_hit_testing_follows_projected_surface(projection):

    image = Scene3D().box("NEXT", 160, 48, GOLD, projection)
    button = Button("NEXT", "next", 100, 200, 160, 48, triangles=image.triangles)
    assert button.contains(180, 224)
    assert not button.contains(100, 200)
    assert not button.contains(99, 224)
    assert not button.contains(261, 224)


def test_degenerate_triangle_is_not_a_hit():

    assert not contains(((0, 0), (0, 0), (0, 0)), 100, 100)


def test_static_images_and_materials_are_bounded_and_reused():

    scene = Scene3D()
    first = scene.box("NEXT", 80, 36, GOLD, "perspective")
    assert scene.box("NEXT", 80, 36, GOLD, "perspective") is first
    for index in range(70):
        scene.box(str(index), 64, 36, GOLD, "perspective")
    assert len(scene.boxes) == 32
    assert len(scene.materials) == 64


def test_rack_animation_uses_elapsed_seconds_and_reuses_framebuffer():

    scene = Scene3D()
    rack = scene.rack("ABC", 320, 96, "perspective", 0, GOLD)
    initial = rack.color[:]
    assert any(rack.depth)
    assert scene.rack("ABC", 320, 96, "perspective", 0, GOLD) is rack
    assert rack.color == initial
    scene.rack("ABC", 320, 96, "perspective", 2.0, GOLD)
    assert rack.color != initial


def test_blit_clips_both_color_and_mesh_depth():

    source = Framebuffer(4, 4)
    source.clear(123)
    source.depth[5] = 0.5
    target = Framebuffer(4, 4)
    blit(source, target, -1, -1)
    assert target.get(0, 0) == 123
    assert target.get_depth(0, 0) == 0.5
    assert sum(bool(c) for c in target.color) == 9
