"""Framebuffer screen painting, separate from rules and controller input.

Spec: TextTwister/IMPLEMENTATION_SPEC.md#presentation-and-input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from Rasterizer.rendering.drawing2d import draw_text, fill_rect
from Rasterizer.rendering.framebuffer import Framebuffer

from ..data.rules import SECOND
from ..data.theme import (
    BACKGROUND,
    DISABLED,
    GOLD,
    GREEN,
    MUTED,
    PAGE_SIZE,
    PANEL,
    RED,
    START_LINES,
    TEXT,
)
from ..vocabulary import difficulty_label
from .scene3d import blit
from .widgets import Button


if TYPE_CHECKING:
    from ..app import GameApp


class GameView:
    """Paint controller state with persistent widgets and reusable 3D resources."""

    def __init__(self, app: GameApp) -> None:

        self.app = app

        self._widgets: dict[tuple[str, str, int, int, int, int], Button] = {}

    def _button(
        self,
        fb: Framebuffer,
        label: str,
        action: str,
        x: int,
        y: int,
        width: int,
        height: int = 36,
        enabled: bool = True,
    ) -> None:
        """Reuse button records and their cached projected hit geometry."""

        key = (label, action, x, y, width, height)

        if key not in self._widgets:
            self._widgets[key] = Button(label, action, x, y, width, height)

        button = self._widgets[key]

        button.enabled = enabled

        image = self.app.scene.box(label, width, height, GOLD if enabled else DISABLED, self.app.projection)

        button.triangles = image.triangles

        self.app.buttons.append(button)

        blit(image.framebuffer, fb, x, y)

    def _word_panel(self, fb: Framebuffer, x: int, y: int, column_width: int) -> None:

        words = self.app.displayed_words()

        last = max(0, (len(words) - 1) // PAGE_SIZE)

        self.app.page = min(self.app.page, last)

        for index, word in enumerate(words[self.app.page * PAGE_SIZE : (self.app.page + 1) * PAGE_SIZE]):
            draw_text(
                fb,
                word,
                x + (index // 8) * column_width,
                y + (index % 8) * 22,
                GOLD if len(word) == len(self.app.game.rack) else TEXT,
                2,
            )

        draw_text(fb, f"PAGE {self.app.page + 1}/{last + 1}", x + 185, 560, MUTED, 2)

        self._button(fb, "<", "previous", x, 548, 64, enabled=self.app.page > 0)

        self._button(fb, ">", "following", x + 80, 548, 64, enabled=self.app.page < last)

    def render(self, fb: Framebuffer) -> None:
        """Paint one screen; the controller has already applied its deadlines."""

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#presentation-and-input.
        # Shared header precedes screen-specific panels and persistent IO status.
        fb.clear(BACKGROUND)

        fill_rect(fb, 0, 0, 960, 64, PANEL)

        title = self.app.scene.box("TEXTTWISTER", 430, 56, GOLD, self.app.projection, yaw=-0.06, pitch=0.16)

        blit(title.framebuffer, fb, 16, 4)

        draw_text(fb, "PROJECTION: " + self.app.projection.upper(), 500, 28, MUTED, 2)

        if self.app.screen == "start":
            self._render_start(fb)

        elif self.app.screen == "playing":
            self._render_game(fb)

        else:
            self._render_over(fb)

        if self.app.leaderboard.error:
            draw_text(
                fb,
                "SCORES NOT SAVED: " + self.app.leaderboard.error[:115],
                24,
                620,
                RED,
                1,
            )

    def _render_start(self, fb: Framebuffer) -> None:

        for row, line in enumerate(START_LINES):
            draw_text(fb, line, 100, 130 + row * 32, TEXT, 2)

        draw_text(fb, "PLAYER NAME (16 LETTERS, DIGITS OR SPACES)", 100, 290, MUTED, 2)

        fill_rect(fb, 100, 320, 760, 56, PANEL)

        draw_text(fb, (self.app.name or "PLAYER") + "_", 120, 337, GOLD, 3)

        self._button(fb, "START / ENTER", "start", 340, 420, 280, 48)

        draw_text(fb, "3+ LETTER WORDS / 2 MINUTES + CARRYOVER", 100, 520, MUTED, 2)

        draw_text(fb, "RACKS GROW: 6 LETTERS, THEN 7, THEN 8", 100, 550, MUTED, 2)

    def _render_game(self, fb: Framebuffer) -> None:
        """Draw HUD, animated rack, input actions, then the paginated word panel."""

        game, puzzle = self.app.game, self.app.game.puzzle

        if puzzle is None:
            return

        draw_text(
            fb,
            f"RACK DIFFICULTY: {difficulty_label(puzzle.difficulty)} {puzzle.difficulty}/100",
            500,
            50,
            MUTED,
            1,
        )

        seconds = (game.remaining_ns() + SECOND - 1) // SECOND

        draw_text(fb, f"STAGE {game.stage}", 32, 88, TEXT, 3)

        draw_text(
            fb,
            f"TIME {seconds // 60}:{seconds % 60:02d}",
            330,
            88,
            RED if seconds < 20 else GOLD,
            3,
        )

        draw_text(fb, f"SCORE {game.score_text}", 650, 88, TEXT, 2)

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#allocation-and-rendering.
        # Reuse the 3D rack viewport; animation samples clock time, never dt steps.
        elapsed = max(0, self.app.game.clock() - self.app.visual_origin_ns) / SECOND

        rack = self.app.scene.rack(game.rack, 960, 120, self.app.projection, elapsed, GOLD)

        blit(rack, fb, 0, 110)

        # Input and action widgets stay legible beside the animated letters.
        fill_rect(fb, 32, 230, 464, 48, PANEL)

        draw_text(fb, self.app.entry + "_", 48, 244, TEXT, 3)

        self._button(fb, "SUBMIT", "submit", 512, 230, 160, 48)

        self._button(fb, "NEXT / F2", "next", 704, 230, 224, 48, game.qualified)

        draw_text(fb, game.message, 32, 296, GREEN if game.qualified else MUTED, 2)

        if game.last_stage is not None:
            previous = game.last_stage

            draw_text(
                fb,
                f"PREVIOUS STAGE {previous.stage} - RACK {previous.rack}: {previous.source_word}",
                32,
                316,
                GOLD,
                1,
            )

        # Counts and paginated answers are independent of projection distortion.
        fill_rect(fb, 32, 328, 196, 210, PANEL)

        draw_text(fb, "LENGTH FOUND/TOTAL", 44, 344, MUTED, 1)

        for row, length in enumerate(range(3, len(game.rack) + 1)):
            found = sum(len(word) == length for word in game.found)

            draw_text(
                fb,
                f"{length}: {found}/{puzzle.counts.get(length, 0)}",
                48,
                368 + row * 24,
                GOLD if length == len(game.rack) else TEXT,
                2,
            )

        draw_text(fb, f"FOUND WORDS: {len(game.found)}", 248, 332, GREEN, 2)

        self._word_panel(fb, 248, 364, 220)

        self._button(fb, "SHUFFLE", "shuffle", 32, 548, 196)

        self._button(fb, "END", "end", 816, 548, 112)

        draw_text(
            fb,
            "ENTER SUBMIT  SPACE SHUFFLE  F2 NEXT  PGUP/PGDN WORDS  ESC END",
            32,
            596,
            MUTED,
            2,
        )

    def _render_over(self, fb: Framebuffer) -> None:
        """Show the archived intended answer before the scores/answers toggle."""

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#presentation-and-input.
        game = self.app.game

        draw_text(fb, f"RUN COMPLETE - {self.app.player}", 32, 90, GOLD, 3)

        draw_text(
            fb,
            f"STAGE {game.stage}  WORDS {game.words}  BONUS {game.bonus}",
            32,
            132,
            TEXT,
            2,
        )

        draw_text(
            fb,
            f"SCORE {game.score_text}  CARRYOVER {game.carryover_ns // SECOND} SECONDS",
            32,
            166,
            GREEN,
            2,
        )

        if game.last_stage is not None:
            result = game.last_stage

            draw_text(
                fb,
                f"ANSWER: {result.source_word}  (RACK: {result.rack})",
                32,
                192,
                GOLD,
                2,
            )

            alternatives = "FULL-RACK WORDS: " + ", ".join(result.full_words)

            draw_text(fb, alternatives[:148], 32, 264, MUTED, 1)

        self._button(fb, "PLAY AGAIN / ENTER", "start", 32, 216, 300)

        self._button(fb, "SCORES / ANSWERS", "answers", 360, 216, 300)

        if self.app.show_answers:
            draw_text(fb, "MISSED WORDS - FAMILIAR FIRST", 32, 280, GOLD, 2)

            self._word_panel(fb, 32, 320, 290)

        else:
            draw_text(fb, "TOP SCORES", 32, 280, GOLD, 2)

            for row, record in enumerate(self.app.leaderboard.records[:10]):
                draw_text(
                    fb,
                    f"{row + 1:2d}. {record.player:<16} {record.score_text}  STAGE {record.stage}",
                    32,
                    316 + row * 24,
                    TEXT,
                    2,
                )

            if not self.app.leaderboard.records:
                draw_text(fb, "NO SAVED SCORES YET", 32, 328, MUTED, 2)

        draw_text(
            fb,
            "TAB TOGGLE SCORES/ANSWERS  PGUP/PGDN WORDS  ESC EXIT",
            32,
            596,
            MUTED,
            2,
        )
