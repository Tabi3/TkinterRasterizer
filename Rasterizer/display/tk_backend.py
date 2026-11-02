"""Tk-based viewer backend.

Spec:
    Display the framebuffer inside a lightweight Tk window and upload each frame
    via the fastest available path while preserving stable input handling for the
    demo scripts.

Contract:
    - Prefer in-memory PNG uploads for efficient C-level decoding.
    - Fall back to a temporary PPM file if CGI-style PNG upload is unavailable.
    - Keep the key-handling interface stable for demo-driven input loops.

Behavior:
    The viewer updates each frame by calling the registered callback and then
    replaces the canvas image with the newly rendered framebuffer.

Reference:
    See IMPLEMENTATION_SPEC.md for the repo-level separation between the public
    rendering API and the demo/viewer backend.
"""

from __future__ import annotations

import base64
import math
import os
import tempfile
import threading
import time
import tkinter as tk
from queue import Empty, Full, Queue

from ..rendering.framebuffer import Framebuffer


class Viewer:
    def __init__(self, width, height, title="Rasterizer", tick_ms=0, render_scale=1):
        if not (isinstance(width, int) and isinstance(height, int)):
            raise TypeError("display dimensions must be integers")
        if not (isinstance(render_scale, int) and render_scale > 0):
            raise TypeError("render_scale must be a positive int")
        if width % render_scale or height % render_scale:
            raise ValueError(
                f"display {width}x{height} not divisible by scale {render_scale}"
            )
        if isinstance(tick_ms, bool) or not isinstance(tick_ms, int):
            raise TypeError("tick_ms must be an integer")
        if tick_ms < 0:
            raise ValueError("tick_ms must be nonnegative")
        self.display_width = width
        self.display_height = height
        self.render_scale = render_scale
        self.render_width = width // render_scale
        self.render_height = height // render_scale

        self.fb = Framebuffer(self.render_width, self.render_height)
        if render_scale > 1:
            self.display_fb = Framebuffer(width, height)
        else:
            self.display_fb = None

        self._snapshot_pool = [
            Framebuffer(self.render_width, self.render_height) for _ in range(3)
        ]
        self._snapshot_free = list(self._snapshot_pool)
        self._snapshot_return = Queue(maxsize=4)

        # Canvas and PhotoImage are at display resolution
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            highlightthickness=0,
            bd=0,
            bg="black",
        )
        self.canvas.pack()
        self.photo = tk.PhotoImage(width=width, height=height)
        self._image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self._upscale_time = None
        self._encode_time = None

        self._frame_cb = None
        self._tick_ms = tick_ms
        self._keys = set()
        self._last_tick = None

        self._fps_accum = 0
        self._fps_time = None
        self.fps = 0.0
        self._render_time = None
        self._upload_time = None
        self._last_upload_at = None

        self._blit_mode = None
        self._temp_path = None

        # Double-buffered encode pipeline
        self._encode_queue = Queue(maxsize=2)
        self._encoded_ready = Queue(maxsize=2)
        self._stop = threading.Event()
        self._closed = False
        self._encode_thread = threading.Thread(target=self._encode_worker, daemon=True)
        self._encode_thread.start()

        self._keys_this_frame = set()  # Edge-triggered key taps.
        self._key_press_events = []

        self.root.bind("<Escape>", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _on_key_press(self, event):
        ks = event.keysym.lower()
        if ks not in self._keys:  # only first frame of the press
            self._keys_this_frame.add(ks)
            self._key_press_events.append(ks)
        self._keys.add(ks)

    def _on_key_release(self, event):
        self._keys.discard(event.keysym.lower())

    def is_key_down(self, keysym):
        """True while the key is held."""
        return keysym.lower() in self._keys

    def key_was_pressed(self, keysym):
        """Whether at least one new key-down edge arrived for this callback."""
        return keysym.lower() in self._keys_this_frame

    def key_presses(self):
        """Ordered new key-down edges delivered since the previous callback.

        Preserves multiple taps and their order between slow frames. Repeated
        KeyPress events while a key remains held do not create new edges.
        """
        return tuple(self._key_press_events)

    def _init_blit_path(self):
        try:
            if self.display_fb is not None:
                self.fb.upscale_nearest(self.display_fb)
                data = self.display_fb.to_png_b64()
            else:
                data = self.fb.to_png_b64()
            self.photo = tk.PhotoImage(data=data, format="png")
            self.canvas.itemconfig(self._image_id, image=self.photo)
            self._blit_mode = "png"
            print("blit: PNG via data= (in-memory, C decode)")
            return
        except tk.TclError as e:
            print(f"blit: PNG data= failed ({e}); falling back to PPM file")

        fd, path = tempfile.mkstemp(suffix=".ppm")
        os.close(fd)
        self._temp_path = path
        source = self.display_fb if self.display_fb is not None else self.fb
        source.to_ppm(path)
        self.photo = tk.PhotoImage(file=path)
        self.canvas.itemconfig(self._image_id, image=self.photo)
        self._blit_mode = "ppm_file"
        print(f"blit: PPM file fallback ({path})")

    def set_frame_callback(self, cb):
        """Set cb(framebuffer, dt_seconds); dt includes all intervening work.

        The first callback receives zero. Presentation drops do not skip updates
        or discard elapsed time. Integrate velocities using dt, not frame count.
        """
        self._frame_cb = cb

    @staticmethod
    def _ema(prev, sample, elapsed, smoothing_seconds=0.1):
        """Time-based diagnostic smoothing, independent of sample frequency."""
        if prev is None:
            return sample
        alpha = -math.expm1(-elapsed / smoothing_seconds)
        return prev + (sample - prev) * alpha

    def _blit(self):
        try:
            data, upscale_time, encode_time = self._encoded_ready.get_nowait()
        except Empty:
            return False

        # Present the newest available result instead of one stale frame/tick.
        try:
            data, upscale_time, encode_time = self._encoded_ready.get_nowait()
        except Empty:
            pass
        t0 = time.perf_counter()

        try:
            if self._blit_mode == "ppm_file":
                with open(self._temp_path, "wb") as output:
                    output.write(base64.b64decode(data))
                new_photo = tk.PhotoImage(file=self._temp_path)
            else:
                new_photo = tk.PhotoImage(data=data, format="png")
            self.canvas.itemconfig(self._image_id, image=new_photo)
            self.photo = new_photo
        except (tk.TclError, OSError) as e:
            print(f"blit: upload failed: {e}")
            return False

        blit_time = time.perf_counter() - t0

        elapsed = 0.0 if self._last_upload_at is None else t0 - self._last_upload_at
        self._last_upload_at = t0
        self._upscale_time = self._ema(self._upscale_time, upscale_time, elapsed)
        self._encode_time = self._ema(self._encode_time, encode_time, elapsed)
        self._upload_time = self._ema(self._upload_time, blit_time, elapsed)

        return True

    def _encode_worker(self):
        # Reuse one private upscale target. It never crosses thread boundaries.
        display_snapshot = (
            Framebuffer(self.display_width, self.display_height)
            if self.render_scale > 1
            else None
        )
        while not self._stop.is_set():
            try:
                fb_snapshot = self._encode_queue.get(timeout=0.05)
            except Empty:
                continue
            # Skip an obsolete pending snapshot before doing expensive encoding.
            try:
                newest = self._encode_queue.get_nowait()
            except Empty:
                pass
            else:
                self._snapshot_return.put_nowait(fb_snapshot)
                fb_snapshot = newest
            try:
                upscale_time = 0.0
                source_fb = fb_snapshot
                if display_snapshot is not None:
                    t0 = time.perf_counter()
                    fb_snapshot.upscale_nearest(display_snapshot)
                    upscale_time = time.perf_counter() - t0
                    source_fb = display_snapshot
                t0 = time.perf_counter()
                data = (
                    source_fb.to_ppm_b64()
                    if self._blit_mode == "ppm_file"
                    else source_fb.to_png_b64()
                )
                result = (data, upscale_time, time.perf_counter() - t0)
                # Keep latency bounded: replace the oldest ready frame.
                try:
                    self._encoded_ready.put_nowait(result)
                except Full:
                    try:
                        self._encoded_ready.get_nowait()
                    except Empty:
                        pass
                    self._encoded_ready.put_nowait(result)
            except (ValueError, RuntimeError, OSError) as error:
                print(f"blit: background encoding failed: {error}")
            finally:
                # Ownership returns only after encoding has finished reading.
                self._snapshot_return.put_nowait(fb_snapshot)

    def _submit_frame(self):
        """Submit one color snapshot, replacing obsolete pending work if needed."""
        while True:
            try:
                self._snapshot_free.append(self._snapshot_return.get_nowait())
            except Empty:
                break
        if self._snapshot_free:
            snapshot = self._snapshot_free.pop()
        else:
            # A queued snapshot is safe to reclaim; an encoding snapshot is not.
            try:
                snapshot = self._encode_queue.get_nowait()
            except Empty:
                return False
        snapshot.color[:] = self.fb.color
        try:
            self._encode_queue.put_nowait(snapshot)
        except Full:
            try:
                obsolete = self._encode_queue.get_nowait()
            except Empty:
                pass  # The encoder just freed capacity.
            else:
                self._snapshot_free.append(obsolete)
            # Tk is the sole producer; the worker can only free capacity here.
            self._encode_queue.put_nowait(snapshot)
        return True

    def _tick(self):
        if self._closed:
            return
        now = time.perf_counter()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        # 1. Render into the framebuffer.
        t0 = time.perf_counter()
        if self._frame_cb is not None:
            self._frame_cb(self.fb, dt)
        if self._closed:
            return
        t1 = time.perf_counter()

        self._keys_this_frame.clear()
        self._key_press_events.clear()

        self._submit_frame()
        self._render_time = self._ema(self._render_time, t1 - t0, dt)
        if self._blit():
            self._fps_accum += 1

        if self._fps_time is None:
            self._fps_time = now
        elif now - self._fps_time >= 1.0:
            elapsed = now - self._fps_time
            self.fps = self._fps_accum / elapsed
            self._fps_accum = 0
            self._fps_time = now

            up_scale_str = (
                f"{self._upscale_time * 1000:5.1f}ms"
                if self._upscale_time is not None
                else "  n/a  "
            )
            enc_str = (
                f"{self._encode_time * 1000:5.1f}ms"
                if self._encode_time is not None
                else "  n/a  "
            )

            blit_str = (
                f"{self._upload_time * 1000:5.1f}ms"
                if self._upload_time is not None
                else "  n/a  "
            )
            print(
                f"FPS {self.fps:5.1f} | "
                f"render {self._render_time * 1000:5.1f}ms | "
                f"upscale {up_scale_str} | "
                f"encode {enc_str} | "
                f"blit {blit_str} | "
                f"{self.display_width}x{self.display_height} "
                f"(render {self.render_width}x{self.render_height})"
            )

        self._schedule_next_tick(now)

    def _schedule_next_tick(self, started):
        # tick_ms is an optional start-to-start period, not an added render delay.
        remaining_ms = self._tick_ms - (time.perf_counter() - started) * 1000
        delay = max(0, math.ceil(remaining_ms))
        if delay:
            self.root.after(delay, self._tick)
        else:
            # Give Tk's idle redraws a turn even when rendering without a cap.
            self.root.after_idle(self._queue_tick)

    def _queue_tick(self):
        if not self._closed:
            self.root.after(0, self._tick)

    def close(self):
        """Stop the encoder and release Tk and temporary-file resources."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._encode_thread.join(timeout=1.0)
        if self._temp_path is not None:
            try:
                os.unlink(self._temp_path)
            except FileNotFoundError:
                pass
            self._temp_path = None
        self.root.destroy()

    def run(self):
        try:
            self._init_blit_path()
            self.root.after(0, self._tick)
            self.root.mainloop()
        finally:
            self.close()
