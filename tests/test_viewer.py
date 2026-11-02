"""Exercise frame ownership and backpressure without creating a Tk window."""

import base64
import threading
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Rasterizer.display.tk_backend import Viewer
from Rasterizer.rendering.framebuffer import Framebuffer


@pytest.fixture
def viewer():
    instance = Viewer.__new__(Viewer)
    instance.render_width = instance.display_width = 4
    instance.render_height = instance.display_height = 4
    instance.render_scale = 1
    instance.fb = Framebuffer(4, 4)
    instance.display_fb = None
    instance._snapshot_pool = [Framebuffer(4, 4) for _ in range(3)]
    instance._snapshot_free = list(instance._snapshot_pool)
    instance._snapshot_return = Queue(maxsize=4)
    instance._encode_queue = Queue(maxsize=2)
    instance._encoded_ready = Queue(maxsize=2)
    instance._stop = threading.Event()
    instance._closed = False
    instance._blit_mode = "png"
    instance._temp_path = None
    instance.root = Mock()
    instance.canvas = Mock()
    instance._key_press_events = []
    instance._last_upload_at = None
    instance._image_id = 1
    instance._upscale_time = instance._encode_time = instance._upload_time = None
    instance._encode_thread = threading.Thread(target=instance._encode_worker)
    yield instance
    instance._stop.set()
    if instance._encode_thread.ident is not None:
        instance._encode_thread.join(timeout=2)
        assert not instance._encode_thread.is_alive()


def test_snapshot_submission_copies_color_only_and_reuses_pool(viewer):
    viewer.fb.clear(0x123456)
    viewer.fb.depth[0] = 0.75
    assert viewer._submit_frame()
    snapshot = viewer._encode_queue.get_nowait()
    assert snapshot in viewer._snapshot_pool
    assert snapshot.color == viewer.fb.color
    assert not any(snapshot.depth)
    viewer.fb.color[0] = 0
    assert snapshot.color[0] == 0x123456
    viewer._snapshot_return.put_nowait(snapshot)
    assert viewer._submit_frame()
    assert len(viewer._snapshot_free) == 2


def test_full_queue_replaces_oldest_without_losing_pool_buffer(viewer):
    for color in (1, 2, 3):
        viewer.fb.clear(color)
        assert viewer._submit_frame()
    assert len(viewer._snapshot_free) == 1
    queued = [viewer._encode_queue.get_nowait() for _ in range(2)]
    assert [fb.color[0] for fb in queued] == [2, 3]
    assert {id(fb) for fb in queued + viewer._snapshot_free} == {
        id(fb) for fb in viewer._snapshot_pool
    }
    viewer._snapshot_free.clear()
    assert not viewer._submit_frame()


def test_worker_retains_buffer_until_encoding_finishes(viewer, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = Framebuffer.to_png_b64

    def encode(fb):
        entered.set()
        assert release.wait(timeout=2)
        return original(fb)

    monkeypatch.setattr(Framebuffer, "to_png_b64", encode)
    viewer._submit_frame()
    viewer._encode_thread.start()
    try:
        assert entered.wait(timeout=2)
        assert viewer._snapshot_return.empty()
        release.set()
        data, _, _ = viewer._encoded_ready.get(timeout=2)
        assert base64.b64decode(data).startswith(b"\x89PNG")
        assert viewer._snapshot_return.get(timeout=2) in viewer._snapshot_pool
    finally:
        release.set()


def test_worker_replaces_stale_ready_frame_without_blocking(viewer):
    viewer._encoded_ready.put(("oldest", 0, 0))
    viewer._encoded_ready.put(("older", 0, 0))
    viewer._submit_frame()
    viewer._encode_thread.start()
    viewer._snapshot_return.get(timeout=2)
    assert viewer._encoded_ready.get_nowait()[0] == "older"
    assert base64.b64decode(viewer._encoded_ready.get_nowait()[0]).startswith(
        b"\x89PNG"
    )


def test_worker_returns_buffer_after_encoding_error(viewer, monkeypatch):
    def fail(fb):
        raise ValueError("encoder failure")

    monkeypatch.setattr(Framebuffer, "to_png_b64", fail)
    viewer._submit_frame()
    viewer._encode_thread.start()
    assert viewer._snapshot_return.get(timeout=2) in viewer._snapshot_pool
    assert viewer._encoded_ready.empty()
    assert viewer._encode_thread.is_alive()


def test_ppm_fallback_remains_active_for_later_frames(viewer, monkeypatch, tmp_path):
    path = tmp_path / "frame.ppm"
    viewer._blit_mode = "ppm_file"
    viewer._temp_path = str(path)
    photo = Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.PhotoImage", photo)
    viewer.fb.clear(0x123456)
    viewer._submit_frame()
    viewer._encode_thread.start()
    viewer._snapshot_return.get(timeout=2)
    assert viewer._blit()
    assert path.read_bytes() == b"P6\n4 4\n255\n" + bytes.fromhex("123456") * 16
    photo.assert_called_once_with(file=str(path))


def test_tick_submits_once_and_counts_only_presented_frames(viewer):
    viewer._last_tick = None
    viewer._frame_cb = Mock()
    viewer._keys_this_frame = {"w"}
    viewer._key_press_events = ["w", "w"]
    viewer._render_time = None
    viewer._fps_time = None
    viewer._fps_accum = 0
    viewer._tick_ms = 1
    viewer._submit_frame = Mock()
    viewer._blit = Mock(return_value=False)
    viewer._tick()
    viewer._submit_frame.assert_called_once()
    assert viewer._fps_accum == 0
    viewer._blit.return_value = True
    viewer._tick()
    assert viewer._fps_accum == 1
    assert not viewer._keys_this_frame
    assert not viewer._key_press_events


def test_close_stops_worker_removes_temporary_file_and_is_idempotent(viewer, tmp_path):
    path = tmp_path / "frame.ppm"
    path.write_bytes(b"frame")
    viewer._temp_path = str(path)
    viewer._encode_thread.start()
    viewer.close()
    viewer.close()
    assert not viewer._encode_thread.is_alive()
    assert not path.exists()
    viewer.root.destroy.assert_called_once()


def test_constructor_registers_close_callback_without_calling_it(monkeypatch):
    root = Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.Tk", lambda: root)
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.Canvas", Mock())
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.PhotoImage", Mock())
    instance = Viewer(4, 4)
    try:
        root.protocol.assert_called_once_with("WM_DELETE_WINDOW", instance.close)
        root.destroy.assert_not_called()
        assert not instance._closed
    finally:
        instance.close()


def test_fps_reporting_before_first_upload(viewer, monkeypatch, capsys):
    viewer._last_tick = None
    viewer._frame_cb = None
    viewer._keys_this_frame = set()
    viewer._render_time = None
    viewer._fps_time = 0
    viewer._fps_accum = 0
    viewer._tick_ms = 1
    viewer._submit_frame = Mock()
    viewer._blit = Mock(return_value=False)
    monkeypatch.setattr("Rasterizer.display.tk_backend.time.perf_counter", lambda: 2)
    viewer._tick()
    assert viewer.fps == 0
    assert "n/a" in capsys.readouterr().out


def test_close_from_frame_callback_does_not_schedule_another_tick(viewer):
    viewer._last_tick = None
    viewer._frame_cb = lambda fb, dt: viewer.close()
    viewer._encode_thread.start()
    viewer._tick()
    viewer.root.after.assert_not_called()


def test_encoder_reuses_upscale_target(viewer, monkeypatch):
    viewer.render_scale = 2
    viewer.display_width = viewer.display_height = 8
    allocated = []

    def allocate(width, height):
        target = Framebuffer(width, height)
        allocated.append(target)
        return target

    monkeypatch.setattr("Rasterizer.display.tk_backend.Framebuffer", allocate)
    viewer._submit_frame()
    viewer._submit_frame()
    viewer._encode_thread.start()
    viewer._snapshot_return.get(timeout=2)
    viewer._snapshot_return.get(timeout=2)
    assert len(allocated) == 1
    assert (allocated[0].width, allocated[0].height) == (8, 8)


def test_presentation_uses_newest_ready_frame(viewer, monkeypatch):
    photo = Mock()
    monkeypatch.setattr("Rasterizer.display.tk_backend.tk.PhotoImage", photo)
    viewer._encoded_ready.put(("obsolete", 0, 0))
    viewer._encoded_ready.put(("latest", 0, 0))
    assert viewer._blit()
    photo.assert_called_once_with(data="latest", format="png")
    assert viewer._encoded_ready.empty()


def test_worker_skips_obsolete_pending_frame(viewer, monkeypatch):
    encoded = []

    def encode(fb):
        encoded.append(fb.color[0])
        return "encoded"

    monkeypatch.setattr(Framebuffer, "to_png_b64", encode)
    viewer.fb.clear(1)
    viewer._submit_frame()
    viewer.fb.clear(2)
    viewer._submit_frame()
    viewer._encode_thread.start()
    viewer._snapshot_return.get(timeout=2)
    viewer._snapshot_return.get(timeout=2)
    assert encoded == [2]


def test_slow_encoder_receives_latest_pending_frame_without_buffer_races(
    viewer, monkeypatch
):
    entered, release, latest_encoded = (threading.Event() for _ in range(3))
    encoded = []

    def encode(fb):
        color = fb.color[0]
        if color == 1:
            entered.set()
            assert release.wait(timeout=2)
            # Submission cannot reclaim the buffer the encoder is reading.
            assert fb.color[0] == 1
        encoded.append(color)
        if color == 7:
            latest_encoded.set()
        return str(color)

    monkeypatch.setattr(Framebuffer, "to_png_b64", encode)
    viewer.fb.clear(1)
    viewer._submit_frame()
    viewer._encode_thread.start()
    try:
        assert entered.wait(timeout=2)
        for color in range(2, 8):
            viewer.fb.clear(color)
            assert viewer._submit_frame()
        release.set()
        assert latest_encoded.wait(timeout=2)
        assert encoded == [1, 7]
    finally:
        release.set()


@pytest.mark.parametrize("presented", [False, True])
def test_elapsed_delta_survives_render_stalls_and_dropped_presentations(
    viewer, monkeypatch, presented
):
    now = [10.0]
    deltas = []

    def render(fb, dt):
        deltas.append(dt)
        now[0] += 0.003  # Simulate render cost without sleeping.

    monkeypatch.setattr(
        "Rasterizer.display.tk_backend.time.perf_counter", lambda: now[0]
    )
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


@pytest.mark.parametrize(
    "period,cost,delay", [(16, 0.006, 10), (16, 0.02, 0), (0, 0.006, 0)]
)
def test_scheduler_accounts_for_render_cost_and_has_uncapped_mode(
    viewer, monkeypatch, period, cost, delay
):
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


@pytest.mark.parametrize(
    "period,error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)]
)
def test_invalid_tick_period_rejected_before_creating_a_window(period, error):
    with pytest.raises(error, match="tick_ms"):
        Viewer(4, 4, tick_ms=period)
