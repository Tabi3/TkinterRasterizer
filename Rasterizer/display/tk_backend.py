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

import os
import tempfile
import threading
import time
import tkinter as tk
from queue import Empty, Full, Queue

from ..rendering.framebuffer import Framebuffer


class Viewer:
    def __init__(self, width, height, title="Rasterizer", tick_ms=1, render_scale=1):
        if not (isinstance(width, int) and isinstance(height, int)):
            raise TypeError("display dimensions must be integers")
        if not (isinstance(render_scale, int) and render_scale > 0):
            raise TypeError("render_scale must be a positive int")
        if width % render_scale or height % render_scale:
            raise ValueError(
                f"display {width}x{height} not divisible by scale {render_scale}"
            )
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

        self._blit_mode = None
        self._temp_path = None

        # Double-buffered encode pipeline
        self._encode_queue = Queue(maxsize=2)
        self._encoded_ready = Queue(maxsize=2)
        self._encode_thread = threading.Thread(target=self._encode_worker, daemon=True)
        self._encode_thread.start()

        self._keys = set()
        self._keys_this_frame = set()  # <-- NEW: edge-triggered taps

        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.focus_set()

    def _on_key_press(self, event):
        ks = event.keysym.lower()
        if ks not in self._keys:  # only first frame of the press
            self._keys_this_frame.add(ks)
        self._keys.add(ks)

    def _on_key_release(self, event):
        self._keys.discard(event.keysym.lower())

    def is_key_down(self, keysym):
        """True while the key is held."""
        return keysym.lower() in self._keys

    def key_was_pressed(self, keysym):
        """True exactly once per physical key tap — call from frame callback."""
        return keysym.lower() in self._keys_this_frame

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
        self.fb.to_ppm(path)
        self.photo = tk.PhotoImage(file=path)
        self.canvas.itemconfig(self._image_id, image=self.photo)
        self._blit_mode = "ppm_file"
        print(f"blit: PPM file fallback ({path})")

    def set_frame_callback(self, cb):
        self._frame_cb = cb

    @staticmethod
    def _ema(prev, sample, alpha=0.2):
        return sample if prev is None else prev * (1 - alpha) + sample * alpha

    def _blit(self):
        try:
            data, upscale_time, encode_time = self._encoded_ready.get_nowait()
        except Empty:
            return False

        t0 = time.perf_counter()

        try:
            new_photo = tk.PhotoImage(data=data, format="png")
            self.canvas.itemconfig(self._image_id, image=new_photo)
            self.photo = new_photo
        except tk.TclError as e:
            print(f"blit: PNG upload failed: {e}")
            return False

        blit_time = time.perf_counter() - t0

        self._upscale_time = self._ema(self._upscale_time, upscale_time)
        self._encode_time = self._ema(self._encode_time, encode_time)
        self._upload_time = self._ema(self._upload_time, blit_time)

        return True

    def _encode_worker(self):
        while True:
            fb_snapshot = self._encode_queue.get()
            if fb_snapshot is None:
                return
            try:
                upscale_time = 0.0
                if self.display_fb is not None:
                    t0 = time.perf_counter()
                    display_snapshot = Framebuffer(
                        self.display_width, self.display_height
                    )
                    fb_snapshot.upscale_nearest(display_snapshot)
                    upscale_time = time.perf_counter() - t0
                    source_fb = display_snapshot
                else:
                    source_fb = fb_snapshot

                # fb_snapshot is no longer needed; return to pool.
                # (Worker holds a reference to self._snapshot_free list — fine.)
                # Actually: better to send it back through a separate queue.
                # Simplest: since we don't share state with the main thread
                # without a lock, use a queue.
                try:
                    self._snapshot_return.put_nowait(fb_snapshot)
                except Full:
                    pass  # Pool will just miss one; next tick allocates.

                t0 = time.perf_counter()
                data = source_fb.to_png_b64()
                encode_time = time.perf_counter() - t0

                self._encoded_ready.put((data, upscale_time, encode_time))
            except Full as e:
                print(f"blit: background processing failed: {e}")

    def _tick(self):
        now = time.perf_counter()
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now

        # 1. Render into the framebuffer.
        t0 = time.perf_counter()
        if self._frame_cb is not None:
            self._frame_cb(self.fb, dt)
        t1 = time.perf_counter()

        self._keys_this_frame.clear()

        # 2. Snapshot the framebuffer and submit it for background encoding.
        snapshot = Framebuffer(self.render_width, self.render_height)
        snapshot.color[:] = self.fb.color
        snapshot.depth[:] = self.fb.depth

        try:
            self._encode_queue.put_nowait(snapshot)
        except Full:
            pass  # Encoding queue is full; drop this frame.

        # 3. Upload an encoded frame through _blit().
        self._blit()
        t2 = time.perf_counter()

        self._render_time = self._ema(self._render_time, t1 - t0)
        self._upload_time = self._ema(self._upload_time, t2 - t1)

        # 4. FPS reporting.
        self._fps_accum += 1

        # 2. Snapshot from the pool; drop the frame if the pool is exhausted.
        if self._snapshot_free:
            snapshot = self._snapshot_free.pop()
            snapshot.color[:] = self.fb.color
            snapshot.depth[:] = self.fb.depth
            try:
                self._encode_queue.put_nowait(snapshot)
            except Full:
                self._snapshot_free.append(snapshot)

        if self._fps_time is None:
            self._fps_time = now
        elif now - self._fps_time >= 1.0:
            elapsed = now - self._fps_time
            self.fps = self._fps_accum / elapsed

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

            print(
                f"FPS {self.fps:5.1f} | "
                f"render {self._render_time * 1000:5.1f}ms | "
                f"upscale {up_scale_str} | "
                f"encode {enc_str} | "
                f"blit {self._upload_time * 1000:5.1f}ms | "
                f"{self.display_width}x{self.display_height} "
                f"(render {self.render_width}x{self.render_height})"
            )

        self.root.after(self._tick_ms, self._tick)
        while True:
            try:
                self._snapshot_free.append(self._snapshot_return.get_nowait())
            except Empty:
                break

    def run(self):
        self._init_blit_path()
        self.root.after(0, self._tick)
        self.root.mainloop()
