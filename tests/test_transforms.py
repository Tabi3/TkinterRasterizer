"""Tests for transforms and projection conventions."""

import math

import pytest

from Rasterizer.algebra.math3d import Mat4, Vec3, Vec4
from Rasterizer.algebra.transforms import (
    Quat,
    compose_trs,
    look_at,
    ndc_ellipsoid,
    ndc_orthographic,
    ndc_panini,
    ndc_perspective,
    perspective_reverse_z,
    viewport,
)


def test_look_at_maps_camera_axes():

    view = look_at(Vec3.zero(), Vec3(0, 0, -1), Vec3.unit_y())
    assert view == Mat4.identity()
    assert view @ Vec4(0, 0, -2, 1) == Vec4(0, 0, -2, 1)


def test_look_at_translates_eye_to_origin():

    eye = Vec3(2, 3, 4)
    view = look_at(eye, Vec3(2, 3, 3), Vec3.unit_y())
    assert view @ eye.to_vec4() == Vec4(0, 0, 0, 1)


def test_reverse_z_projection_maps_near_and_far():

    projection = perspective_reverse_z(math.pi / 2, 1.0, 0.1, 100.0)
    near = projection @ Vec4(0, 0, -0.1, 1)
    far = projection @ Vec4(0, 0, -100.0, 1)
    assert near.z / near.w == pytest.approx(1.0)
    assert far.z / far.w == pytest.approx(0.0, abs=1e-12)


def test_reverse_z_projection_rejects_invalid_planes():

    with pytest.raises(ValueError, match="near"):
        perspective_reverse_z(math.pi / 2, 1.0, 0.0)
    with pytest.raises(ValueError, match="far"):
        perspective_reverse_z(math.pi / 2, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="aspect"):
        perspective_reverse_z(math.pi / 2, 0.0, 0.1)


def test_ndc_projection_modes_agree_at_center():

    assert ndc_perspective(0.0, 0.0, -2.0, 1.0, 1.5) == (0.0, 0.0)
    assert ndc_orthographic(0.0, 0.0, -2.0, 3.0, 1.5) == (0.0, 0.0)
    assert ndc_panini(0.0, 0.0, -2.0, 1.0, 1.5) == (0.0, 0.0)
    assert ndc_ellipsoid(0.0, 0.0, -2.0, 1.0, 1.5) == (0.0, 0.0)


def test_non_linear_projections_handle_points_behind_camera():

    assert ndc_panini(1.0, 1.0, 0.0, 1.0, 1.0) == (0.0, 0.0)
    assert ndc_ellipsoid(1.0, 1.0, 0.0, 1.0, 1.0) == (0.0, 0.0)


def test_viewport_maps_center_and_flips_y():

    screen = viewport(0, 0, 100, 80)
    assert screen @ Vec4(0, 0, 0.75, 1) == Vec4(50, 40, 0.75, 1)
    assert screen @ Vec4(0, 1, 0.75, 1) == Vec4(50, 0, 0.75, 1)


def test_quaternion_rotation_and_trs():

    rotation = Quat.from_axis_angle(Vec3.unit_y(), math.pi / 2)
    transform = compose_trs(Vec3(1, 0, 0), rotation, Vec3(2, 2, 2))
    result = transform @ Vec4(1, 0, 0, 1)
    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(0.0)
    assert result.z == pytest.approx(-2.0)
