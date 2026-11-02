"""Input/session controller; framebuffer painting lives in ui/screens.py.

Spec: TextTwister/IMPLEMENTATION_SPEC.md#organization-and-style.
"""

import time
from collections.abc import Callable
from random import Random
from typing import Literal

from Rasterizer.rendering.framebuffer import Framebuffer

from .data.rules import MAX_NAME_LENGTH
from .data.theme import GOLD, PAGE_SIZE
from .game import Game
from .storage import Leaderboard, Record
from .ui.scene3d import Scene3D, projection_for_stage
from .ui.screens import GameView
from .ui.widgets import Button
from .words import WordIndex


__all__ = ["GOLD", "PAGE_SIZE", "Button", "GameApp"]


class GameApp:
    """Own one game session, its persistent scores, and reusable view resources."""

    screen: Literal["start", "playing", "over"]

    buttons: list[Button]

    def __init__(
        self,
        index: WordIndex,
        leaderboard: Leaderboard,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        wall_clock: Callable[[], int] = time.time_ns,
        rng: Random | None = None,
    ) -> None:
        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#organization-and-style.
        # Session services are stable; only the following input/view state mutates.

        self.game = Game(index, clock=clock, rng=rng)

        self.leaderboard = leaderboard

        self.wall_clock = wall_clock

        self.screen = "start"

        self.name = ""

        self.player = "PLAYER"

        self.entry = ""

        self.page = 0

        self.show_answers = False

        self.recorded = False

        self.buttons = []

        self.scene = Scene3D()

        self.view = GameView(self)

        self.visual_origin_ns = clock()

    def start(self) -> None:
        """Begin a fresh run while retaining the selected player and view caches."""

        self.player = self.name.strip() or "PLAYER"

        self.game.start()

        self.entry, self.page = "", 0

        self.recorded = False

        self.screen = "playing"

        self.show_answers = False

    def finish(self) -> None:
        """End and record the run once; repeated close callbacks are harmless."""

        if self.screen != "playing":
            return

        self.game.end()

        self.screen = "over"

        self.show_answers = True

        self.entry, self.page = "", 0

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#leaderboard-format.
        # Mark before IO so duplicate callbacks cannot write duplicate records.
        if not self.recorded:
            self.recorded = True

            game = self.game

            self.leaderboard.add(
                Record(
                    self.player,
                    game.words,
                    game.bonus,
                    game.stage,
                    game.carryover_ns,
                    self.wall_clock(),
                )
            )

    def tick(self) -> bool:
        """Apply deadlines and return whether the visible stage/screen changed."""

        if self.screen != "playing":
            return False

        changed = self.game.update()

        ended = not self.game.active

        if changed or ended:
            self.entry, self.page = "", 0

        if ended:
            self.finish()

        return changed or ended

    def handle_key(self, keysym: str, char: str = "") -> bool:
        """Return True when the application should close. Process in Tk order."""

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#presentation-and-input.
        # Expiration owns the transition; do not reinterpret old input as restart.
        changed = self.tick()

        key = keysym.lower()

        if key == "escape":
            if self.screen == "playing":
                self.finish()

                return False

            return True

        if changed:
            return False

        if self.screen == "start":
            if key == "return":
                self.start()

            elif key == "backspace":
                self.name = self.name[:-1]

            else:
                letter = char or (" " if key == "space" else key if len(key) == 1 else "")

                if (
                    len(letter) == 1
                    and len(self.name) < MAX_NAME_LENGTH
                    and ("a" <= letter <= "z" or "A" <= letter <= "Z" or "0" <= letter <= "9" or letter == " ")
                ):
                    self.name += letter

        elif self.screen == "over":
            if key == "return":
                self.start()

            elif key == "tab":
                self.show_answers = not self.show_answers

                self.page = 0

            elif key in ("prior", "next"):
                self.turn_page(-1 if key == "prior" else 1)

        elif key == "return":
            if self.game.submit(self.entry) == "accepted":
                self.entry = ""

        elif key == "backspace":
            self.entry = self.entry[:-1]

        elif key in ("space", "f3"):
            self.game.shuffle()

        elif key == "f2":
            self.next_stage()

        elif key in ("prior", "next"):
            self.turn_page(-1 if key == "prior" else 1)

        elif len(key) == 1 and "a" <= key <= "z" and len(self.entry) < len(self.game.rack):
            self.entry += key

        return False

    def next_stage(self) -> None:
        """Advance a qualified stage and clear input belonging to its old rack."""

        if self.game.advance():
            self.entry, self.page = "", 0

        else:
            self.game.message = "Find a full-rack word first"

    def displayed_words(self) -> list[str]:
        """Return display order: found words by length, missed words by familiarity."""

        puzzle = self.game.puzzle

        if self.screen == "over" and puzzle is not None:
            return sorted(
                puzzle.answers - self.game.found,
                key=lambda word: (
                    word != puzzle.source_word,
                    word not in self.game.index.familiar_words,
                    len(word),
                    word,
                ),
            )

        return sorted(self.game.found, key=lambda word: (len(word), word))

    def turn_page(self, delta: int) -> None:
        """Clamp the requested page to the current answer list."""

        last = max(0, (len(self.displayed_words()) - 1) // PAGE_SIZE)

        self.page = min(last, max(0, self.page + delta))

    def click(self, x: int, y: int) -> bool:
        """Dispatch a click only after deadlines and projected surface hit tests."""

        if self.tick():
            return False

        for button in self.buttons:
            if button.enabled and button.contains(x, y):
                if button.action == "start":
                    self.start()

                elif button.action == "submit":
                    self.handle_key("return")

                elif button.action == "next":
                    self.next_stage()

                elif button.action == "shuffle":
                    self.game.shuffle()

                elif button.action == "previous":
                    self.turn_page(-1)

                elif button.action == "following":
                    self.turn_page(1)

                elif button.action == "answers":
                    self.show_answers = not self.show_answers

                    self.page = 0

                elif button.action == "end":
                    self.finish()

                return True

        return False

    @property
    def projection(self) -> str:

        return projection_for_stage(self.game.stage)

    def render(self, fb: Framebuffer, _dt: float = 0.0) -> None:
        """Tick absolute deadlines and paint through the reusable game view.

        Spec: TextTwister/IMPLEMENTATION_SPEC.md#presentation-and-input.
        Rendering does not advance the game clock by frame count.
        """

        self.tick()

        self.buttons.clear()

        self.view.render(fb)
