"""Pure deadline-driven game rules. No UI, file IO, or exception control flow."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from random import Random

from .data.rules import RACK_LENGTHS, SECOND, STAGE_TIME, STAGES_PER_TIER
from .words import Puzzle, PuzzlePicker, WordIndex


__all__ = ["SECOND", "STAGE_TIME", "Game", "StageResult"]


@dataclass(frozen=True)
class StageResult:
    """Immutable completed-stage snapshot; later shuffles cannot alter reveals."""

    stage: int

    rack: str

    source_word: str

    full_words: tuple[str, ...]


class Game:
    """Own mutable session state driven solely by an injected absolute clock.

    Spec: TextTwister/IMPLEMENTATION_SPEC.md#game-rules.
    """

    puzzle: Puzzle | None

    found: set[str]

    last_stage: StageResult | None

    def __init__(
        self,
        index: WordIndex,
        *,
        rng: Random | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:

        if any(not index.candidates[length] for length in RACK_LENGTHS):
            raise ValueError("word list must contain familiar six-, seven-, and eight-letter rack seeds")

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#game-rules.
        # Stable services and mutable per-run state have separate lifetimes.
        self.index = index

        self.rng = rng if rng is not None else Random()

        self.clock = clock

        self.picker = PuzzlePicker(index, self.rng)

        self.active = False

        self.stage = 0

        self.words = 0

        self.bonus = 0

        self.carryover_ns = 0

        self.deadline_ns = 0

        self.puzzle = None

        self.rack = ""

        self.found = set()

        self.full_found = 0

        self.message = ""

        self.last_stage = None

    @property
    def qualified(self) -> bool:

        return self.full_found > 0

    @property
    def score_numerator(self) -> int:

        return (self.words + self.bonus) * self.stage * (STAGE_TIME + self.carryover_ns)

    @property
    def score_text(self) -> str:

        cents = (self.score_numerator * 100 + STAGE_TIME // 2) // STAGE_TIME

        return f"{cents // 100}.{cents % 100:02d}"

    def start(self) -> None:
        """Reset the session and seed its first deadline from the absolute clock."""

        self.stage = self.words = self.bonus = self.carryover_ns = 0

        self.picker = PuzzlePicker(self.index, self.rng)

        self.active = True

        self.puzzle = None

        self.last_stage = None

        self._new_stage(self.clock(), 0)

    def _new_stage(self, started_ns: int, carry_ns: int) -> None:

        self._archive_stage()

        self.stage += 1

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#game-rules.
        # Progress through frozen rack tiers, then remain at the largest rack.
        tier = min((self.stage - 1) // STAGES_PER_TIER, len(RACK_LENGTHS) - 1)

        length = RACK_LENGTHS[tier]

        self.puzzle = self.picker.pick(length, stage=self.stage)

        self.rack = self.puzzle.rack

        self.found = set()

        self.full_found = 0

        self.deadline_ns = started_ns + STAGE_TIME + carry_ns

        self.message = "Find a full-rack word to unlock Next"

    def update(self, now_ns: int | None = None) -> bool:
        """Return whether expiration changed stages or ended the run."""

        now = self.clock() if now_ns is None else now_ns

        changed = False

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#game-rules.
        # Auto-advance uses the old deadline, so a stalled frame grants no time.
        while self.active and now >= self.deadline_ns:
            changed = True

            if self.qualified:
                self._new_stage(self.deadline_ns, 0)

            else:
                self.end()

                self.message = "Time is up"

        return changed

    def remaining_ns(self, now_ns: int | None = None) -> int:
        """Return nonnegative remaining time without advancing session state."""

        now = self.clock() if now_ns is None else now_ns

        return max(0, self.deadline_ns - now) if self.active else 0

    def submit(self, raw: str) -> str:
        """Validate a normalized guess after applying every expired deadline."""

        now = self.clock()

        if self.update(now) or not self.active:
            return "expired"

        # Dictionary membership and per-stage duplicate checks precede scoring.
        word = raw.strip().lower()

        if self.puzzle is None or word not in self.puzzle.answers:
            self.message = "Not a valid rack word"

            return "invalid"

        if word in self.found:
            self.message = "Already found"

            return "duplicate"

        # First full-rack answer qualifies; subsequent anagrams add a bonus.
        self.found.add(word)

        self.words += 1

        if len(word) == len(self.rack):
            if self.full_found:
                self.bonus += 1

            self.full_found += 1

        self.message = "Anagram bonus +1" if len(word) == len(self.rack) and self.full_found > 1 else "Word accepted"

        return "accepted"

    def advance(self) -> bool:
        """Bank remaining time and advance only a qualified live stage."""

        now = self.clock()

        if self.update(now) or not self.active or not self.qualified:
            return False

        remaining = self.remaining_ns(now)

        self.carryover_ns += remaining

        self._new_stage(now, remaining)

        return True

    def shuffle(self) -> None:
        """Shuffle visible letters while preserving answers and the intended word."""

        if self.update() or not self.active:
            return

        letters = list(self.rack)

        self.rng.shuffle(letters)

        shuffled = "".join(letters)

        self.rack = shuffled if shuffled != self.rack else self.rack[1:] + self.rack[:1]

    def _archive_stage(self) -> None:

        if self.puzzle is not None:
            self.last_stage = StageResult(self.stage, self.rack, self.puzzle.source_word, self.puzzle.full_words)

    def end(self) -> None:
        """Archive the current answer once and mark the session inactive."""

        if self.active:
            self._archive_stage()

        self.active = False
