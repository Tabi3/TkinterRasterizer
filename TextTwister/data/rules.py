"""Frozen game rules. Spec: TextTwister/IMPLEMENTATION_SPEC.md#game-rules."""

SECOND = 1_000_000_000

STAGE_SECONDS = 120

STAGE_TIME = STAGE_SECONDS * SECOND

RACK_LENGTHS = (6, 7, 8)

STAGES_PER_TIER = 3

MIN_WORD_LENGTH = 3

MAX_SHORT_LENGTH = 5

MAX_NAME_LENGTH = 16

DIFFICULTY_BASE = 8

DIFFICULTY_STEP = 2

DIFFICULTY_MAX = 60

SHORT_ANSWER_TARGET = 8


PRIMES = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
)

RARE_LETTER_WEIGHTS = {"j": 6, "q": 8, "x": 6, "z": 8, "k": 2, "v": 2, "w": 1, "y": 1}
