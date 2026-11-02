"""Tk presentation with optional background encoding and paired resource modes.

Ordinary input, scheduling and buffer control use explicit state. Only functions
ending in _protected contain exception handling at resource boundaries.
"""

from __future__ import annotations

import base64
import math
import threading
import time
import tkinter as tk
from collections.abc import Callable
from typing import Literal

from ..rendering.framebuffer import Framebuffer
from .mailbox import EncodedFrame, FrameMailbox


ResourceMode = Literal["protected", "unprotected"]


class Viewer:
    def __init__(
        self,
        width: int,
        height: int,
        title: str = "Rasterizer",
        tick_ms: int = 0,
        render_scale: int = 1,
        *,
        background_encode: bool = True,
        resource_mode: ResourceMode = "protected",
        log_timings: bool = True,
    ) -> None:

        if isinstance(width, bool) or isinstance(height, bool) or not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("display dimensions must be integers")

        if width <= 0 or height <= 0:
            raise ValueError("display dimensions must be positive")

        if isinstance(render_scale, bool) or not isinstance(render_scale, int):
            raise TypeError("render_scale must be a positive int")

        if render_scale <= 0 or width % render_scale or height % render_scale:
            raise ValueError("display dimensions must be divisible by positive render_scale")

        if isinstance(tick_ms, bool) or not isinstance(tick_ms, int):
            raise TypeError("tick_ms must be an integer")

        if tick_ms < 0:
            raise ValueError("tick_ms must be nonnegative")

        if not isinstance(background_encode, bool):
            raise TypeError("background_encode must be boolean")

        if resource_mode not in ("protected", "unprotected"):
            raise ValueError("unknown resource mode")

        if not isinstance(log_timings, bool):
            raise TypeError("log_timings must be boolean")

        self.log_timings = log_timings

        self.display_width, self.display_height = width, height

        self.render_scale = render_scale

        self.render_width, self.render_height = (
            width // render_scale,
            height // render_scale,
        )

        self.fb = Framebuffer(self.render_width, self.render_height)

        self.display_fb = Framebuffer(width, height) if render_scale > 1 else None

        self.background_encode = background_encode

        self.resource_mode = resource_mode

        self.title = title

        self.last_error = ""

        self._closed = False

        self._blit_mode = "png" if tk.TkVersion >= 8.6 else "PPM"

        self._frame_cb: Callable[[Framebuffer, float], None] | None = None

        self._tick_ms = tick_ms

        self._last_tick: float | None = None

        self._keys: set[str] = set()

        self._keys_this_frame: set[str] = set()

        self._key_press_events: list[str] = []

        self._fps_accum = 0

        self._fps_time: float | None = None

        self.fps = 0.0

        self._frame_time: float | None = None

        self._render_time: float | None = None

        self._upscale_time: float | None = None

        self._encode_time: float | None = None

        self._upload_time: float | None = None

        self._last_upload_at: float | None = None

        self._mailbox = FrameMailbox(self.render_width, self.render_height) if background_encode else None

        self._encode_thread: threading.Thread | None = None

        if resource_mode == "protected":
            self._initialize_window_protected()

        else:
            self._initialize_window_unprotected()

    def _configure_window(self) -> None:

        self.root.title(self.title)

        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=self.display_width,
            height=self.display_height,
            highlightthickness=0,
            bd=0,
            bg="black",
        )

        self.canvas.pack()

        self.photo = tk.PhotoImage(width=self.display_width, height=self.display_height)

        self._image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        self.root.bind("<Escape>", lambda e: self.close())

        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.root.bind("<KeyPress>", self._on_key_press)

        self.root.bind("<KeyRelease>", self._on_key_release)

        self.root.bind("<FocusOut>", self._on_focus_out)

        self.root.focus_set()

    def _start_worker(self) -> None:

        if self.background_encode:
            self._encode_thread = threading.Thread(target=self._encode_worker, daemon=True)

            self._encode_thread.start()

    def _initialize_window_unprotected(self) -> None:

        self.root = tk.Tk()

        self._configure_window()

        self._start_worker()

        # Protected reference:
        # ready = False
        # try:
        #     self._configure_window()
        #     self._start_worker()
        #     ready = True
        # finally:
        #     if not ready:
        #         self.close_protected()

    def _initialize_window_protected(self) -> None:

        self.root = tk.Tk()

        ready = False

        try:
            self._configure_window()

            self._start_worker()

            ready = True

        finally:
            if not ready:
                self.close_protected()

    def _on_key_press(self, event):

        key = event.keysym.lower()

        if key not in self._keys:
            self._keys_this_frame.add(key)

            self._key_press_events.append(key)

        self._keys.add(key)

    def _on_key_release(self, event):

        self._keys.discard(event.keysym.lower())

    def _on_focus_out(self, _event):

        self._keys.clear()

    def is_key_down(self, keysym: str) -> bool:

        return keysym.lower() in self._keys

    def key_was_pressed(self, keysym: str) -> bool:

        return keysym.lower() in self._keys_this_frame

    def key_presses(self) -> tuple[str, ...]:

        return tuple(self._key_press_events)

    def set_frame_callback(self, cb: Callable[[Framebuffer, float], None]) -> None:
        """Set cb(framebuffer, dt_seconds); the first delta is zero."""

        self._frame_cb = cb

    @staticmethod
    def _ema(
        prev: float | None,
        sample: float,
        elapsed: float,
        smoothing_seconds: float = 0.1,
    ) -> float:

        if prev is None:
            return sample

        return prev + (sample - prev) * -math.expm1(-elapsed / smoothing_seconds)

    def _encode_frame(self, source: Framebuffer, target: Framebuffer | None) -> EncodedFrame:

        started = time.perf_counter()

        if target is not None:
            source.upscale_nearest(target)

            source = target

        upscaled = time.perf_counter()

        mode = self._blit_mode

        data = source.to_png_b64() if mode == "png" else base64.b64decode(source.to_ppm_b64())

        return EncodedFrame(data, mode, upscaled - started, time.perf_counter() - upscaled)

    def _upload_unprotected(self, frame: EncodedFrame) -> bool:

        started = time.perf_counter()

        photo = tk.PhotoImage(data=frame.data, format=frame.format)

        self.canvas.itemconfig(self._image_id, image=photo)

        self.photo = photo

        duration = time.perf_counter() - started

        elapsed = 0 if self._last_upload_at is None else started - self._last_upload_at

        self._last_upload_at = started

        self._upscale_time = self._ema(self._upscale_time, frame.upscale_seconds, elapsed)

        self._encode_time = self._ema(self._encode_time, frame.encode_seconds, elapsed)

        self._upload_time = self._ema(self._upload_time, duration, elapsed)

        return True

        # Protected reference:
        # try:
        #     return self._upload_unprotected(frame)
        # except tk.TclError as error:
        #     self.last_error = str(error)
        #     return False

    def _upload_protected(self, frame: EncodedFrame) -> bool:

        try:
            return self._upload_unprotected(frame)

        except tk.TclError as error:
            self.last_error = str(error)

            if frame.format == "png":
                self._blit_mode = "PPM"

            return False

    def _upload(self, frame: EncodedFrame) -> bool:

        if self.resource_mode == "protected":
            return self._upload_protected(frame)

        return self._upload_unprotected(frame)

    def _present_unprotected(self) -> bool:

        return self._upload_unprotected(self._encode_frame(self.fb, self.display_fb))

    def _present_protected(self) -> bool:

        try:
            frame = self._encode_frame(self.fb, self.display_fb)

            if self._upload_protected(frame):
                return True

            if frame.format == "png":
                return self._upload_protected(self._encode_frame(self.fb, self.display_fb))

            return False

        except (ValueError, RuntimeError, OSError) as error:
            self.last_error = str(error)

            return False

    def _present(self) -> bool:

        if self.resource_mode == "protected":
            return self._present_protected()

        return self._present_unprotected()

    def _init_blit_path(self) -> bool:

        return self._present()

    def _submit_frame(self) -> bool:

        return self._mailbox.submit(self.fb) if self._mailbox is not None else False

    def _blit(self) -> bool:

        if self._mailbox is None:
            return False

        frame = self._mailbox.take_ready()

        return self._upload(frame) if frame is not None else False

    def _encode_worker(self) -> None:

        target = Framebuffer(self.display_width, self.display_height) if self.render_scale > 1 else None

        if self.resource_mode == "protected":
            self._worker_protected(target)

        else:
            self._worker_unprotected(target)

    def _worker_unprotected(self, target: Framebuffer | None) -> None:

        mailbox = self._mailbox

        if mailbox is None:
            return

        source = mailbox.take()

        while source is not None:
            mailbox.publish(self._encode_frame(source, target))

            mailbox.release(source)

            source = mailbox.take()

        # Protected reference: release the source in finally if encoding fails.
        # try:
        #     mailbox.publish(self._encode_frame(source, target))
        # finally:
        #     mailbox.release(source)

    def _worker_protected(self, target: Framebuffer | None) -> None:

        mailbox = self._mailbox

        if mailbox is None:
            return

        source = mailbox.take()

        while source is not None:
            try:
                mailbox.publish(self._encode_frame(source, target))

            except (ValueError, RuntimeError, OSError) as error:
                self.last_error = str(error)

            finally:
                mailbox.release(source)

            source = mailbox.take()

    def _tick(self) -> None:

        if self._closed:
            return

        now = time.perf_counter()

        dt = 0.0 if self._last_tick is None else now - self._last_tick

        self._last_tick = now

        started = time.perf_counter()

        if self._frame_cb is not None:
            self._frame_cb(self.fb, dt)

        if self._closed:
            return

        self._render_time = self._ema(self._render_time, time.perf_counter() - started, dt)

        self._keys_this_frame.clear()

        self._key_press_events.clear()

        if self.background_encode:
            self._submit_frame()

            presented = self._blit()

        else:
            presented = self._present()

        finished = time.perf_counter()

        self._frame_time = self._ema(self._frame_time, finished - now, dt)

        self._fps_accum += int(presented)

        if self._fps_time is None:
            self._fps_time = now

        if finished - self._fps_time >= 1:
            self.fps = self._fps_accum / (finished - self._fps_time)

            self._fps_accum = 0

            self._fps_time = finished

            if self.log_timings:
                self._report_timings()

        self._schedule_next_tick(now)

    def _report_timings(self) -> None:

        def milliseconds(value: float | None) -> str:

            return "n/a" if value is None else f"{value * 1000:.2f}ms"

        mode = "async" if self.background_encode else "sync"

        print(
            f"FPS {self.fps:.1f} | frame {milliseconds(self._frame_time)} | "
            f"render {milliseconds(self._render_time)} | "
            f"upscale {milliseconds(self._upscale_time)} | "
            f"encode {milliseconds(self._encode_time)} | "
            f"upload {milliseconds(self._upload_time)} | "
            f"{self.render_width}x{self.render_height} {mode}"
        )

    def _schedule_next_tick(self, started: float) -> None:

        delay = max(0, math.ceil(self._tick_ms - (time.perf_counter() - started) * 1000))

        if delay:
            self.root.after(delay, self._tick)

        else:
            self.root.after_idle(self._queue_tick)

    def _queue_tick(self) -> None:

        if not self._closed:
            self.root.after(0, self._tick)

    def _stop_worker(self) -> None:

        self._closed = True

        if self._mailbox is not None:
            self._mailbox.stop()

        if self._encode_thread is not None and self._encode_thread.ident is not None:
            self._encode_thread.join(timeout=1)

    def close_unprotected(self) -> None:

        if not self._closed:
            self._stop_worker()

            self.root.destroy()

    def close_protected(self) -> None:

        if not self._closed:
            try:
                self._stop_worker()

            finally:
                try:
                    self.root.destroy()

                except tk.TclError as error:
                    self.last_error = str(error)

    def close(self) -> None:

        if self.resource_mode == "protected":
            self.close_protected()

        else:
            self.close_unprotected()

    def _run_loop(self) -> None:

        if self._init_blit_path():
            self.root.after(0, self._tick)

            self.root.mainloop()

    def run_unprotected(self) -> None:

        self._run_loop()

        self.close_unprotected()

        # Protected reference:
        # try:
        #     self._run_loop()
        # finally:
        #     self.close_protected()

    def run_protected(self) -> None:

        try:
            self._run_loop()

        finally:
            self.close_protected()

    def run(self) -> None:

        if self.resource_mode == "protected":
            self.run_protected()

        else:
            self.run_unprotected()
