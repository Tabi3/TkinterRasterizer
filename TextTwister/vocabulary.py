"""Difficulty algorithms. Spec: TextTwister/IMPLEMENTATION_SPEC.md#difficulty-and-selection."""

from .data.rules import (
    DIFFICULTY_BASE,
    DIFFICULTY_MAX,
    DIFFICULTY_STEP,
    RARE_LETTER_WEIGHTS,
    SHORT_ANSWER_TARGET,
)
from .data.vocabulary import COMMON_WORDS


__all__ = ["COMMON_WORDS", "complexity", "difficulty_label", "difficulty_limit"]


def complexity(word: str, familiar_short_answers: int) -> int:
    """0-100 scrambling estimate; familiarity is enforced separately by the pool."""

    # Spec: TextTwister/IMPLEMENTATION_SPEC.md#difficulty-and-selection.
    # Inspect spelling separately from the familiar-answer availability score.
    vowels = sum(c in "aeiou" for c in word)

    longest = run = 0

    for letter in word:
        run = 0 if letter in "aeiouy" else run + 1

        longest = max(longest, run)

    length_cost = 6 * max(0, len(word) - 6)

    rare_cost = 2 * sum(RARE_LETTER_WEIGHTS.get(c, 0) for c in word)

    repeated_cost = 2 * (len(word) - len(set(word)))

    vowel_cost = 5 * max(0, 2 - vowels) + 3 * max(0, vowels - 3)

    consonant_cost = 4 * max(0, longest - 2)

    answer_cost = 2 * max(0, SHORT_ANSWER_TARGET - familiar_short_answers)

    # Keep this an estimate: familiarity is enforced by seed eligibility itself.
    return min(
        100,
        sum(
            (
                length_cost,
                rare_cost,
                repeated_cost,
                vowel_cost,
                consonant_cost,
                answer_cost,
            )
        ),
    )


def difficulty_limit(stage: int) -> int:
    """Return the progressively widening cap; never admit non-familiar seeds."""

    return min(DIFFICULTY_MAX, DIFFICULTY_BASE + DIFFICULTY_STEP * max(0, stage - 1))


def difficulty_label(score: int) -> str:
    """Map the numeric scrambling estimate to its UI label."""

    return "EASY" if score <= 24 else "MEDIUM" if score <= 40 else "HARD"
