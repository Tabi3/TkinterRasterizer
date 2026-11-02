"""Dictionary indexing and subset lookup without parser libraries."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from random import Random

from .data.rules import PRIMES, RACK_LENGTHS
from .vocabulary import COMMON_WORDS, complexity, difficulty_limit


def signature(word: str) -> int:

    product = 1

    for letter in word:
        if not "a" <= letter <= "z":
            raise ValueError("words must contain lowercase ASCII letters")

        product *= PRIMES[ord(letter) - 97]

    return product


@dataclass(frozen=True)
class Puzzle:
    """Per-stage answer snapshot; word collections are immutable and reusable."""

    signature: int

    rack: str

    groups: dict[int, tuple[str, ...]]

    answers: frozenset[str]

    counts: dict[int, int]

    source_word: str = ""

    difficulty: int = 0

    familiar_short_answers: int = 0

    @property
    def full_words(self) -> tuple[str, ...]:

        return self.groups.get(self.signature, ())


@dataclass(frozen=True)
class RackCandidate:
    """Precomputed familiar source and its deterministic difficulty metadata."""

    signature: int

    source_word: str

    difficulty: int

    familiar_short_answers: int


class WordIndex:
    """Index the dictionary once; guesses and stages perform bounded lookups.

    Spec: TextTwister/IMPLEMENTATION_SPEC.md#dictionary-index.
    """

    def __init__(self, words: Iterable[str], *, common_words: Iterable[str] = COMMON_WORDS) -> None:
        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#dictionary-index.
        # Validate and group dictionary words before evaluating familiarity.

        buckets: dict[int, set[str]] = defaultdict(set)

        for line, raw in enumerate(words, 1):
            word = raw.strip()

            if not word:
                continue

            if not all("a" <= c <= "z" for c in word):
                raise ValueError(f"word list line {line}: expected lowercase ASCII")

            if 3 <= len(word) <= 8:
                buckets[signature(word)].add(word)

        self.buckets = {key: tuple(sorted(values)) for key, values in buckets.items()}

        self.racks = {length: tuple(key for key, values in self.buckets.items() if len(values[0]) == length) for length in RACK_LENGTHS}

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#difficulty-and-selection.
        # Intersect curated data with actual accepted words, not just signatures.
        familiar: dict[int, set[str]] = defaultdict(set)

        for word in common_words:
            key = signature(word)

            if word in self.buckets.get(key, ()):
                familiar[key].add(word)

        self.familiar_words = frozenset(word for values in familiar.values() for word in values)

        # Profile work belongs to startup, never to the frame or guess loops.
        short_products = [key for key, values in familiar.items() for word in values if 3 <= len(word) <= 5]

        self.rack_profiles: dict[int, RackCandidate] = {}

        for key, values in familiar.items():
            if not 6 <= len(next(iter(values))) <= 8:
                continue

            short_count = sum(key % short == 0 for short in short_products)

            source = min(values, key=lambda word: (complexity(word, short_count), word))

            self.rack_profiles[key] = RackCandidate(key, source, complexity(source, short_count), short_count)

        self.candidates = {
            length: tuple(
                sorted(
                    (profile for profile in self.rack_profiles.values() if len(profile.source_word) == length),
                    key=lambda profile: (profile.difficulty, profile.source_word),
                )
            )
            for length in RACK_LENGTHS
        }

    def puzzle(self, rack: str, *, source_word: str | None = None) -> Puzzle:
        """Resolve all multiset subsets and preserve the intended full-rack word."""

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#dictionary-index.
        # Enumerate counts per letter; repeated letters do not duplicate subsets.
        rack_signature = signature(rack)

        subsets = [(1, 0)]

        for letter, count in Counter(rack).items():
            prime = PRIMES[ord(letter) - 97]

            subsets = [(product * prime**copies, length + copies) for product, length in subsets for copies in range(count + 1)]

        groups = {key: self.buckets[key] for key, length in subsets if length >= 3 and key in self.buckets}

        answers = frozenset(word for values in groups.values() for word in values)

        # Reveal metadata must match the exact rack, including letter counts.
        profile = self.rack_profiles.get(rack_signature)

        full = self.buckets.get(rack_signature, ())

        source = source_word if source_word is not None else (profile.source_word if profile is not None else full[0] if full else "")

        if source and source not in full:
            raise ValueError("source word must be a dictionary anagram of the rack")

        short_count = profile.familiar_short_answers if profile is not None else 0

        return Puzzle(
            rack_signature,
            rack,
            groups,
            answers,
            dict(Counter(map(len, answers))),
            source,
            complexity(source, short_count) if source else 0,
            short_count,
        )


class PuzzlePicker:
    """Choose familiar signature buckets without replacement within the stage cap."""

    def __init__(self, index: WordIndex, rng: Random) -> None:

        self.index = index

        self.rng = rng

        self.used: dict[int, set[int]] = {length: set() for length in RACK_LENGTHS}

    def pick(self, length: int, *, stage: int = 1) -> Puzzle:
        """Choose an eligible unused signature, then jumble its familiar source."""

        # Spec: TextTwister/IMPLEMENTATION_SPEC.md#difficulty-and-selection.
        profiles = self.index.candidates.get(length, ())

        if not profiles:
            raise ValueError(f"word list has no familiar {length}-letter racks")

        limit = difficulty_limit(stage)

        pool = [profile for profile in profiles if profile.difficulty <= limit]

        if not pool:
            easiest = profiles[0].difficulty

            pool = [profile for profile in profiles if profile.difficulty == easiest]

        # Exhaustion resets only this eligible pool; harder seeds stay gated.
        available = [profile for profile in pool if profile.signature not in self.used[length]]

        if not available:
            self.used[length].difference_update(profile.signature for profile in pool)

            available = pool

        selected = self.rng.choice(available)

        self.used[length].add(selected.signature)

        word = selected.source_word

        letters = list(word)

        self.rng.shuffle(letters)

        rack = "".join(letters)

        if rack == word and len(set(word)) > 1:
            rack = word[1:] + word[:1]

        return self.index.puzzle(rack, source_word=word)
