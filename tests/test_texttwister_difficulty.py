"""Familiar seed eligibility, complexity progression and visible answer reveals."""

import random
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest

from Rasterizer.rendering.framebuffer import Framebuffer
from TextTwister.app import GameApp
from TextTwister.game import SECOND, STAGE_TIME, Game
from TextTwister.storage import Leaderboard
from TextTwister.vocabulary import COMMON_WORDS, complexity, difficulty_limit
from TextTwister.words import PuzzlePicker, WordIndex


@pytest.fixture(scope="module")
def index():

    path = Path(__file__).resolve().parents[1] / "TextTwister" / "word_list.txt"
    return WordIndex(path.read_text(encoding="ascii").splitlines())


def test_complexity_is_explicit_and_rewards_more_familiar_answers():

    assert complexity("listen", 12) < complexity("quartz", 12)
    assert complexity("listen", 12) < complexity("listen", 0)
    assert complexity("triangle", 12) > complexity("listen", 12)
    assert 0 <= complexity("quizzing", 0) <= 100
    assert difficulty_limit(1) == 8
    assert difficulty_limit(4) == 14
    assert difficulty_limit(7) == 20
    assert difficulty_limit(1000) == 60


def test_opening_pool_has_variety_and_excludes_obscure_dictionary_seeds(index):

    profiles = index.candidates[6]
    eligible = [p for p in profiles if p.difficulty <= difficulty_limit(1)]
    assert len(eligible) >= 75
    assert all(p.source_word in COMMON_WORDS for p in profiles)
    assert len(profiles) < len(index.racks[6])
    assert all(p.source_word != "quartz" for p in eligible)
    for seed in range(30):
        puzzle = PuzzlePicker(index, random.Random(seed)).pick(6)
        assert puzzle.source_word in COMMON_WORDS
        assert puzzle.difficulty <= difficulty_limit(1)
        assert puzzle.source_word in puzzle.full_words
        assert Counter(puzzle.rack) == Counter(puzzle.source_word)


def test_no_repeat_until_eligible_opening_pool_exhausted(index):

    picker = PuzzlePicker(index, random.Random(3))
    count = sum(p.difficulty <= difficulty_limit(1) for p in index.candidates[6])
    signatures = [picker.pick(6).signature for _ in range(count)]
    assert len(set(signatures)) == count
    assert picker.pick(6).signature in signatures


def test_growing_cap_unlocks_harder_familiar_racks(index):

    early = {p.signature for p in index.candidates[8] if p.difficulty <= difficulty_limit(7)}
    later = {p.signature for p in index.candidates[8] if p.difficulty <= difficulty_limit(30)}
    assert early < later
    picker = PuzzlePicker(index, random.Random(3))
    for _ in range(20):
        puzzle = picker.pick(8, stage=7)
        assert puzzle.difficulty <= difficulty_limit(7)
        assert puzzle.source_word in COMMON_WORDS


def test_custom_dictionary_can_supply_its_own_familiar_vocabulary():

    index = WordIndex(["aarrgh", "aahed", "aahs"], common_words=["aarrgh", "aahed"])
    assert PuzzlePicker(index, random.Random(1)).pick(6).source_word == "aarrgh"
    unknown = WordIndex(["aarrgh"])
    with pytest.raises(ValueError, match="familiar"):
        PuzzlePicker(unknown, random.Random(1)).pick(6)


def test_unknown_bonus_anagrams_remain_accepted(index):

    puzzle = index.puzzle("teacher")
    assert "hectare" in puzzle.answers
    assert puzzle.source_word in COMMON_WORDS
    assert puzzle.source_word != "hectare"


def test_source_word_must_match_dictionary_and_letter_counts(index):

    with pytest.raises(ValueError, match="anagram"):
        index.puzzle("listen", source_word="letter")
    puzzle = index.puzzle("listen", source_word="silent")
    assert puzzle.source_word == "silent"


def test_answer_record_survives_shuffle_advancement_timeout_and_restart(index):

    now = [0]
    game = Game(index, clock=lambda: now[0], rng=random.Random(6))
    game.start()
    assert game.puzzle is not None
    source = game.puzzle.source_word
    game.shuffle()
    shuffled = game.rack
    game.submit(source)
    assert game.advance()
    assert game.last_stage is not None
    assert game.last_stage.source_word == source
    assert game.last_stage.rack == shuffled
    assert game.last_stage.stage == 1
    assert game.puzzle is not None
    next_source = game.puzzle.source_word
    now[0] = game.deadline_ns
    game.update()
    assert not game.active
    assert game.last_stage.source_word == next_source
    game.end()
    assert game.last_stage.stage == 2
    game.start()
    assert game.last_stage is None


def test_qualified_auto_advance_reveals_previous_answer(index):

    now = [0]
    game = Game(index, clock=lambda: now[0], rng=random.Random(4))
    game.start()
    assert game.puzzle is not None
    source = game.puzzle.source_word
    game.submit(source)
    now[0] = STAGE_TIME
    game.update()
    assert game.active and game.stage == 2
    assert game.last_stage is not None and game.last_stage.source_word == source
    assert game.remaining_ns() == 120 * SECOND


def test_answer_is_drawn_prominently_without_tab_on_game_over(index, tmp_path, monkeypatch):

    now = [0]
    app = GameApp(
        index,
        Leaderboard(tmp_path / "scores.txt"),
        clock=lambda: now[0],
        rng=random.Random(2),
    )
    app.start()
    assert app.game.puzzle is not None
    source = app.game.puzzle.source_word
    now[0] = STAGE_TIME
    app.tick()
    assert app.screen == "over" and app.show_answers
    drawn = Mock()
    monkeypatch.setattr("TextTwister.ui.screens.draw_text", drawn)
    app.render(Framebuffer(960, 640))
    texts = [call.args[1] for call in drawn.call_args_list]
    assert any(text.startswith("ANSWER: " + source) for text in texts)
    assert any(text.startswith("FULL-RACK WORDS: ") for text in texts)


def test_previous_answer_is_visible_during_next_stage(index, tmp_path, monkeypatch):

    app = GameApp(
        index,
        Leaderboard(tmp_path / "scores.txt"),
        clock=lambda: 0,
        rng=random.Random(2),
    )
    app.start()
    assert app.game.puzzle is not None
    source = app.game.puzzle.source_word
    app.game.submit(source)
    app.next_stage()
    drawn = Mock()
    monkeypatch.setattr("TextTwister.ui.screens.draw_text", drawn)
    app.render(Framebuffer(960, 640))
    texts = [call.args[1] for call in drawn.call_args_list]
    assert any(text.startswith("PREVIOUS STAGE 1") and source in text for text in texts)
    assert any(text.startswith("RACK DIFFICULTY: ") for text in texts)


def test_missed_words_show_intended_and_familiar_answers_before_rare_ones(index, tmp_path):

    app = GameApp(index, Leaderboard(tmp_path / "scores.txt"), clock=lambda: 0)
    app.start()
    app.game.puzzle = index.puzzle("theory")
    app.game.rack = "theory"
    app.finish()
    words = app.displayed_words()
    assert words[0] == "theory"
    assert "eth" in words  # Full dictionary answers are retained, not discarded.
    familiar = [i for i, word in enumerate(words) if word in index.familiar_words]
    rare = [i for i, word in enumerate(words) if word not in index.familiar_words]
    assert max(familiar) < min(rare)
