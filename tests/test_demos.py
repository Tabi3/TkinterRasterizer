"""Headless smoke tests for runnable demos."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Rasterizer.demos import (
    demo_lit_scene,
    demo_projections,
    demo_textured,
    demo_unshaded,
)
from Rasterizer.demos.common import advance_time
from Rasterizer.rendering.framebuffer import Framebuffer


@pytest.mark.parametrize("name", ["demo_unshaded", "demo_lit_scene", "demo_textured", "demo_projections"])
def test_demo_direct_file_bootstrap_outside_repository(name, tmp_path):

    demo_path = Path(__file__).resolve().parents[1] / "Rasterizer" / "demos" / f"{name}.py"

    # Isolated Python cannot rely on the repository cwd or PYTHONPATH. A custom
    # run name exercises direct-file imports without launching the interactive UI.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            ("import runpy, sys; ns = runpy.run_path(sys.argv[1], run_name='direct_demo_test'); assert callable(ns['draw'])"),
            str(demo_path),
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_unshaded_demo_renders_without_opening_a_window():

    demo_unshaded._state.clear()
    framebuffer = Framebuffer(64, 64)
    demo_unshaded.draw(framebuffer, 0.0)
    assert any(framebuffer.color)


@pytest.mark.parametrize("demo", [demo_lit_scene, demo_textured, demo_projections])
def test_shaded_demos_render_geometry_headlessly(demo):

    demo._state.clear()
    framebuffer = Framebuffer(64, 64)
    try:
        demo.draw(framebuffer, 0.0)
        assert any(framebuffer.depth)
        if demo is demo_projections:
            for projection in demo._proj_names:
                demo._state["camera"].projection = projection
                demo.draw(framebuffer, 0.0)
                assert any(framebuffer.depth), projection
    finally:
        demo._state.clear()


@pytest.mark.parametrize("demo", [demo_unshaded, demo_lit_scene, demo_textured, demo_projections])
@pytest.mark.parametrize("fps", [7, 30, 60, 144, None])
def test_animation_matches_after_same_elapsed_time_at_different_rates(demo, fps):

    def simulate(deltas):
        demo._state.clear()
        fb = Framebuffer(24, 24)
        demo.draw(fb, 0.0)
        for dt in deltas:
            demo.draw(fb, dt)
        assert demo._state["elapsed"] == pytest.approx(1.0)
        if "uniforms" in demo._state:
            assert demo._state["uniforms"].time == pytest.approx(1.0)
            assert demo._state["uniforms"].delta_time == deltas[-1]
        return fb.color[:], list(fb.depth)
    try:
        expected_color, expected_depth = simulate([1.0])

        # An irregular schedule includes a stall; elapsed time is never clamped.
        deltas = [1 / fps] * fps if fps else [0.01, 0.02, 0.67, 0.3]
        color, depth = simulate(deltas)
        assert color == expected_color
        assert depth == pytest.approx(expected_depth, abs=2e-6)
    finally:
        demo._state.clear()


@pytest.mark.parametrize("demo", [demo_unshaded, demo_lit_scene, demo_textured, demo_projections])
def test_zero_delta_freezes_animation(demo):

    demo._state.clear()
    fb = Framebuffer(24, 24)
    try:
        demo.draw(fb, 0.75)
        color, depth = fb.color[:], fb.depth[:]
        for _ in range(4):
            demo.draw(fb, 0)
        assert fb.color == color
        assert fb.depth == depth
        assert demo._state["elapsed"] == 0.75
    finally:
        demo._state.clear()


@pytest.mark.parametrize("dt", [-1, float("nan"), float("inf")])
def test_invalid_delta_rejected_without_advancing_time(dt):

    state = {"elapsed": 1.0}
    with pytest.raises(ValueError):
        advance_time(state, dt)
    assert state == {"elapsed": 1.0}


@pytest.mark.parametrize(
    "batches",
    [
        [("2", "1", "w", "w")],
        [("2",), ("1",), ("w",), ("w",)],
        [("2", "1"), ("w", "w")],
    ],
)
def test_projection_input_order_and_toggles_survive_slow_frames(batches, monkeypatch):

    demo_projections._state.clear()
    fake_viewer = SimpleNamespace(root=Mock(), key_presses=lambda: ())
    monkeypatch.setattr(demo_projections, "_viewer", fake_viewer)
    fb = Framebuffer(24, 24)
    try:
        for events in batches:
            fake_viewer.key_presses = lambda events=events: events
            demo_projections.draw(fb, 0)
        assert demo_projections._state["camera"].projection == "perspective"
        assert not demo_projections._state["wireframe"]
    finally:
        demo_projections._state.clear()
