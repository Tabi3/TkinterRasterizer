"""Headless viewer tests for mailbox ownership, timing and both resource modes."""

import threading
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Rasterizer.display.mailbox import EncodedFrame, FrameMailbox
from Rasterizer.display.tk_backend import Viewer
from Rasterizer.rendering.framebuffer import Framebuffer


@pytest.fixture
def viewer(monkeypatch):

    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.Tk", Mock())
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.Canvas", Mock())
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.PhotoImage", Mock())
    instance = Viewer(4, 4, background_encode=False)
    instance.background_encode = True
    instance._mailbox = FrameMailbox(4, 4)
    yield instance
    instance.close()
    if instance._encode_thread is not None:
        assert not instance._encode_thread.is_alive()


def start_worker(viewer):

    viewer._encode_thread = threading.Thread(target=viewer._encode_worker)
    viewer._encode_thread.start()


def wait_free(mailbox):

    with mailbox.condition:
        assert mailbox.condition.wait_for(lambda: len(mailbox.free) == 3, timeout=2)


def test_mailbox_copies_color_only_and_replaces_pending(viewer):

    mailbox = viewer._mailbox
    assert mailbox is not None
    viewer.fb.clear(1)
    viewer.fb.depth[0] = 0.75
    assert viewer._submit_frame()
    viewer.fb.clear(2)
    assert viewer._submit_frame()
    target = mailbox.take()
    assert target is not None
    assert target.color[0] == 2
    assert not any(target.depth)
    assert len(mailbox.free) == 2
    viewer.fb.color[0] = 3
    assert target.color[0] == 2
    mailbox.release(target)
    assert len(mailbox.free) == 3


def test_slow_worker_retains_buffer_and_encodes_latest_pending(viewer, monkeypatch):

    entered, release, finished = (threading.Event() for _ in range(3))
    encoded = []
    def encode(fb):
        color = fb.color[0]
        if color == 1:
            entered.set()
            assert release.wait(2)
            assert fb.color[0] == 1
        encoded.append(color)
        if color == 7:
            finished.set()
        return str(color)
    monkeypatch.setattr(Framebuffer, "to_png_b64", encode)
    viewer.fb.clear(1)
    viewer._submit_frame()
    start_worker(viewer)
    try:
        assert entered.wait(2)
        for color in range(2, 8):
            viewer.fb.clear(color)
            assert viewer._submit_frame()
        release.set()
        assert finished.wait(2)
        assert encoded == [1, 7]
        mailbox = viewer._mailbox
        assert mailbox is not None
        wait_free(mailbox)
        assert len(mailbox.free) == 3
    finally:
        release.set()


def test_protected_worker_returns_buffer_after_error(viewer, monkeypatch):

    def fail(fb):
        raise ValueError("encoder failure")
    monkeypatch.setattr(Framebuffer, "to_png_b64", fail)
    viewer._submit_frame()
    start_worker(viewer)
    assert viewer._mailbox is not None
    wait_free(viewer._mailbox)
    assert viewer.last_error == "encoder failure"
    assert viewer._encode_thread is not None and viewer._encode_thread.is_alive()


def test_mailbox_keeps_newest_ready_frame(viewer, monkeypatch):

    photo = Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.PhotoImage", photo)
    assert viewer._mailbox is not None
    for value in ("old", "latest"):
        viewer._mailbox.publish(EncodedFrame(value, "png", 0, 0))
    assert viewer._blit()
    photo.assert_called_once_with(data="latest", format="png")
    assert not viewer._blit()


def test_mailbox_stop_wakes_waiting_worker_and_releases_pending(viewer):

    viewer._submit_frame()
    assert viewer._mailbox is not None
    viewer._mailbox.stop()
    assert viewer._mailbox.take() is None
    assert len(viewer._mailbox.free) == 3
    assert not viewer._submit_frame()
    start_worker(viewer)
    viewer.close()
    assert viewer._encode_thread is not None and not viewer._encode_thread.is_alive()


def test_ppm_upload_is_in_memory_and_scaled(viewer, monkeypatch):

    photo = Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.PhotoImage", photo)
    viewer._blit_mode = "PPM"
    viewer.display_fb = Framebuffer(8, 8)
    viewer.fb.clear(0x123456)
    assert viewer._present()
    data = photo.call_args.kwargs["data"]
    assert data == b"P6\n8 8\n255\n" + bytes.fromhex("123456") * 64
    assert photo.call_args.kwargs["format"] == "PPM"


def test_png_failure_falls_back_only_in_protected_mode(viewer, monkeypatch):

    calls = []
    def photo(**kwargs):
        calls.append(kwargs)
        if kwargs["format"] == "png":
            raise tk.TclError("PNG unavailable")
        return Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.PhotoImage", photo)
    assert viewer._present()
    assert [call["format"] for call in calls] == ["png", "PPM"]
    viewer._blit_mode = "png"
    viewer.resource_mode = "unprotected"
    with pytest.raises(tk.TclError):
        viewer._present()


def test_synchronous_constructor_creates_no_worker_or_mailbox(viewer, monkeypatch):

    thread = Mock(side_effect=AssertionError("thread creation"))
    monkeypatch.setattr("Rasterizer.display.tk_backend.threading.Thread", thread)
    root = Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.Tk", lambda: root)
    instance = Viewer(8, 8, background_encode=False)
    try:
        assert instance._mailbox is None and instance._encode_thread is None
        assert instance._present()
        thread.assert_not_called()
        root.protocol.assert_called_with("WM_DELETE_WINDOW", instance.close)
    finally:
        instance.close()


def test_run_resource_modes_cleanup_difference(viewer):

    viewer._run_loop = Mock(side_effect=RuntimeError("run failure"))
    with pytest.raises(RuntimeError):
        viewer.run_protected()
    assert viewer._closed
    viewer._closed = False
    with pytest.raises(RuntimeError):
        viewer.run_unprotected()
    assert not viewer._closed


@pytest.mark.parametrize("mode,cleaned", [("protected", True), ("unprotected", False)])
def test_initialization_resource_modes_cleanup_partial_window(viewer, monkeypatch, mode, cleaned):

    root = Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.Tk", lambda: root)
    monkeypatch.setattr(
        "Rasterizer.display.tk_backend.tk.Canvas",
        Mock(side_effect=tk.TclError("canvas failed")),
    )
    with pytest.raises(tk.TclError):
        Viewer(4, 4, background_encode=False, resource_mode=mode)
    assert root.destroy.called == cleaned


def test_unprotected_encoding_error_propagates(viewer, monkeypatch):

    def fail(fb):
        raise ValueError("encoder failure")
    monkeypatch.setattr(Framebuffer, "to_png_b64", fail)
    viewer.resource_mode = "unprotected"
    with pytest.raises(ValueError):
        viewer._present()


def test_unprotected_viewer_does_not_dispatch_to_protected_functions(viewer, monkeypatch):

    for name in (
        "_initialize_window_protected",
        "_upload_protected",
        "_present_protected",
        "_worker_protected",
        "close_protected",
        "run_protected",
    ):
        monkeypatch.setattr(Viewer, name, Mock(side_effect=AssertionError(name)))
    instance = Viewer(4, 4, background_encode=False, resource_mode="unprotected")
    instance.run()
    assert instance._closed

    # The fixture is protected; its finalizer runs after monkeypatch restoration.
    viewer.resource_mode = "unprotected"


def test_tick_submits_once_and_clears_input(viewer):

    viewer._frame_cb = Mock()
    viewer._keys_this_frame = {"w"}
    viewer._key_press_events = ["w", "w"]
    viewer._submit_frame = Mock()
    viewer._blit = Mock(return_value=False)
    viewer._tick()
    viewer._submit_frame.assert_called_once()
    assert viewer._fps_accum == 0
    assert not viewer._keys_this_frame and not viewer._key_press_events


def test_close_from_callback_is_idempotent(viewer):

    viewer._frame_cb = lambda fb, dt: viewer.close()
    viewer._tick()
    viewer.close()
    viewer.root.destroy.assert_called_once()
    viewer.root.after.assert_not_called()


@pytest.mark.parametrize("presented", [False, True])
def test_elapsed_delta_survives_render_stalls_and_dropped_presentations(viewer, monkeypatch, presented):

    now = [10.0]
    deltas = []
    def render(fb, dt):
        deltas.append(dt)
        now[0] += 0.003  # Simulate render cost without sleeping.
    monkeypatch.setattr("Rasterizer.display.tk_backend.time.perf_counter", lambda: now[0])
    viewer._last_tick = None
    viewer._frame_cb = render
    viewer._keys_this_frame = set()
    viewer._render_time = None
    viewer._fps_time = None
    viewer._fps_accum = 0
    viewer._tick_ms = 0
    viewer._submit_frame = Mock(return_value=False)
    viewer._blit = Mock(return_value=presented)
    for start in (10.0, 10.01, 10.05, 10.3, 11.3):
        now[0] = start
        viewer._tick()
    assert deltas == pytest.approx([0, 0.01, 0.04, 0.25, 1.0])
    assert sum(deltas) == pytest.approx(1.3)
    assert viewer._submit_frame.call_count == 5


@pytest.mark.parametrize("period,cost,delay", [(16, 0.006, 10), (16, 0.02, 0), (0, 0.006, 0)])
def test_scheduler_accounts_for_render_cost_and_has_uncapped_mode(viewer, monkeypatch, period, cost, delay):

    viewer._tick_ms = period
    monkeypatch.setattr("Rasterizer.display.tk_backend.time.perf_counter", lambda: cost)
    viewer._schedule_next_tick(0.0)
    if delay:
        viewer.root.after.assert_called_once_with(delay, viewer._tick)
        viewer.root.after_idle.assert_not_called()
    else:
        viewer.root.after_idle.assert_called_once_with(viewer._queue_tick)
        viewer._queue_tick()
        viewer.root.after.assert_called_once_with(0, viewer._tick)


def test_pending_uncapped_tick_does_not_restart_after_close(viewer):

    viewer._closed = True
    viewer._queue_tick()
    viewer.root.after.assert_not_called()


@pytest.mark.parametrize("fps", [15, 30, 60, 144])
def test_diagnostic_smoothing_uses_seconds_instead_of_frame_count(fps):

    expected = Viewer._ema(0, 1, 1)
    smoothed = 0
    for _ in range(fps):
        smoothed = Viewer._ema(smoothed, 1, 1 / fps)
    assert smoothed == pytest.approx(expected)


def test_key_edges_count_multiple_taps_and_ignore_held_repeats(viewer):

    viewer._keys = set()
    viewer._keys_this_frame = set()
    key = SimpleNamespace(keysym="W")
    viewer._on_key_press(key)
    viewer._on_key_press(key)
    assert viewer.key_presses() == ("w",)
    assert viewer.key_was_pressed("w")
    viewer._on_key_release(key)
    viewer._on_key_press(key)
    assert viewer.key_presses() == ("w", "w")
    assert viewer.is_key_down("w")


@pytest.mark.parametrize("period,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_tick_period_rejected_before_creating_a_window(period, error):

    with pytest.raises(error, match="tick_ms"):
        Viewer(4, 4, tick_ms=period)


def test_terminal_timings_are_periodic_and_can_be_disabled(viewer, monkeypatch, capsys):

    now = [0.0]
    monkeypatch.setattr("Rasterizer.display.tk_backend.time.perf_counter", lambda: now[0])
    viewer._submit_frame = Mock()
    viewer._blit = Mock(return_value=False)
    viewer._tick()
    assert not capsys.readouterr().out
    now[0] = 1.1
    viewer._tick()
    output = capsys.readouterr().out
    for field in ("FPS", "frame", "render", "upscale", "encode", "upload", "4x4 async"):
        assert field in output
    assert "n/a" in output
    now[0] = 1.2
    viewer._tick()
    assert not capsys.readouterr().out
    viewer.log_timings = False
    now[0] = 2.2
    viewer._tick()
    assert not capsys.readouterr().out


def test_timings_show_milliseconds_and_sync_mode(viewer, capsys):

    viewer.background_encode = False
    viewer.fps = 60
    viewer._frame_time = 0.008
    viewer._render_time = 0.003
    viewer._upscale_time = 0
    viewer._encode_time = 0.002
    viewer._upload_time = 0.003
    viewer._report_timings()
    output = capsys.readouterr().out
    assert "FPS 60.0" in output
    assert "frame 8.00ms" in output
    assert "render 3.00ms" in output
    assert "upscale 0.00ms" in output
    assert "encode 2.00ms" in output
    assert "upload 3.00ms" in output
    assert "4x4 sync" in output
