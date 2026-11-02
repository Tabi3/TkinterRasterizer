"""Bounded latest-frame handoff. Availability and ownership share one lock."""

from dataclasses import dataclass
from threading import Condition

from ..rendering.framebuffer import Framebuffer


@dataclass(frozen=True)
class EncodedFrame:
    data: str | bytes

    format: str

    upscale_seconds: float

    encode_seconds: float


class FrameMailbox:
    def __init__(self, width: int, height: int) -> None:

        self.condition = Condition()

        self.free = [Framebuffer(width, height) for _ in range(3)]

        self.pending: Framebuffer | None = None

        self.ready: EncodedFrame | None = None

        self.stopped = False

    def submit(self, source: Framebuffer) -> bool:

        with self.condition:
            if self.stopped:
                return False

            if self.pending is not None:
                target = self.pending

            elif self.free:
                target = self.free.pop()

            else:
                return False

            target.color[:] = source.color

            self.pending = target

            self.condition.notify()

            return True

    def take(self) -> Framebuffer | None:

        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.stopped)

            if self.stopped:
                return None

            target = self.pending

            self.pending = None

            return target

    def release(self, target: Framebuffer) -> None:

        with self.condition:
            self.free.append(target)

            self.condition.notify_all()

    def publish(self, result: EncodedFrame) -> None:

        with self.condition:
            if not self.stopped:
                self.ready = result

                self.condition.notify_all()

    def take_ready(self) -> EncodedFrame | None:

        with self.condition:
            result = self.ready

            self.ready = None

            return result

    def stop(self) -> None:

        with self.condition:
            self.stopped = True

            if self.pending is not None:
                self.free.append(self.pending)

                self.pending = None

            self.ready = None

            self.condition.notify_all()
