"""Deterministic rules, handwritten formats, resources and headless game UI."""

import random
from collections import Counter
from pathlib import Path

import pytest

from Rasterizer.rendering.drawing2d import draw_text, fill_rect
from Rasterizer.rendering.framebuffer import Framebuffer
from TextTwister.__main__ import resource_mode
from TextTwister.app import PAGE_SIZE, GameApp
from TextTwister.game import SECOND, STAGE_TIME, Game
from TextTwister.storage import (
    HEADER,
    Leaderboard,
    Record,
    parse_records,
    ranked,
    read_protected,
    read_unprotected,
    serialize_records,
    write_protected,
    write_unprotected,
)
from TextTwister.words import PuzzlePicker, WordIndex, signature


@pytest.fixture
def index():

    return WordIndex(
        [
            "listen",
            "silent",
            "enlist",
            "inlets",
            "tinsel",
            "tin",
            "sin",
            "sit",
            "lie",
            "let",
            "net",
            "ten",
            "tie",
            "lit",
            "set",
            "line",
            "nest",
            "tile",
            "lent",
            "lens",
            "teacher",
            "cheater",
            "hectare",
            "triangle",
            "integral",
            "altering",
        ]
    )


@pytest.fixture
def clock():

    return [0]


@pytest.fixture
def game(index, clock):

    return Game(index, clock=lambda: clock[0], rng=random.Random(7))


def test_prime_signature_and_subsets(index):

    assert signature("listen") == signature("silent")
    assert signature("letter") != signature("leter")
    puzzle = index.puzzle("listen")
    expected = {w for bucket in index.buckets.values() for w in bucket if not Counter(w) - Counter("listen")}
    assert puzzle.answers == expected
    assert puzzle.counts[6] == 5
    assert sum(puzzle.counts.values()) == len(expected)
    assert len(puzzle.groups[signature("listen")]) == 5


def test_repeated_letters_and_duplicate_lines():

    index = WordIndex(["letter", "letter", "tee", "eel", "tree", "teller", "eee"])
    assert index.buckets[signature("letter")] == ("letter",)
    assert index.puzzle("letter").answers == {"letter", "tee", "eel", "tree"}


@pytest.mark.parametrize("word", ["UPPER", "can't", "naïve", "abc1"])
def test_invalid_dictionary_text_is_explicit(word):

    with pytest.raises(ValueError, match="line 1"):
        WordIndex([word])


def test_picker_uses_each_bucket_before_refill():

    index = WordIndex(["listen", "silent", "letter", "planet"])
    picker = PuzzlePicker(index, random.Random(2))
    first = [picker.pick(6, stage=20).signature for _ in range(3)]
    assert len(set(first)) == 3
    assert picker.pick(6, stage=20).signature in first


def test_guess_validation_qualification_and_bonus(game):

    game.start()
    assert not game.advance()
    assert game.submit("tin") == "accepted"
    assert game.submit(" TIN ") == "duplicate"
    assert game.submit("tint") == "invalid"
    assert game.submit("listen") == "accepted"
    assert game.qualified and game.bonus == 0
    assert game.submit("silent") == "accepted"
    assert game.words == 3 and game.bonus == 1
    assert game.score_text == "4.00"


def test_manual_time_carryover_is_cumulative(game, clock):

    game.start()
    game.submit("listen")
    clock[0] = 10 * SECOND
    assert game.advance()
    assert game.stage == 2
    assert game.carryover_ns == 110 * SECOND
    assert game.remaining_ns() == 230 * SECOND
    game.submit("listen")  # A word can score again in a different stage.
    clock[0] = 50 * SECOND
    assert game.advance()
    assert game.carryover_ns == 300 * SECOND
    assert game.remaining_ns() == 310 * SECOND
    assert game.score_text == "21.00"


def test_exact_deadline_expires_before_guess(game, clock):

    game.start()
    clock[0] = STAGE_TIME
    assert game.submit("listen") == "expired"
    assert not game.active and game.words == 0


def test_qualified_timeout_advances_without_adding_time_or_carry(game, clock):

    game.start()
    game.submit("listen")
    clock[0] = STAGE_TIME + 5 * SECOND
    assert game.update()
    assert game.stage == 2 and game.active
    assert game.remaining_ns() == 115 * SECOND
    assert game.carryover_ns == 0
    assert not game.qualified


def test_stall_can_expire_qualified_and_following_unqualified_stage(game, clock):

    game.start()
    game.submit("listen")
    clock[0] = 500 * SECOND
    game.update()
    assert game.stage == 2 and not game.active


def test_rack_growth_and_reset(game):

    game.start()
    for stage in range(1, 10):
        expected = 6 if stage <= 3 else 7 if stage <= 6 else 8
        assert len(game.rack) == expected
        assert game.puzzle is not None
        full_word = game.puzzle.groups[game.puzzle.signature][0]
        game.submit(full_word)
        assert game.advance()
    game.start()
    assert game.stage == 1 and len(game.rack) == 6
    assert game.words == game.bonus == game.carryover_ns == 0


@pytest.mark.parametrize("fps", [7, 30, 60, 144])
def test_clock_outcome_is_independent_of_frame_schedule(index, fps):

    clock = [0]
    game = Game(index, clock=lambda: clock[0], rng=random.Random(7))
    game.start()
    game.submit("listen")
    for frame in range(fps):
        clock[0] = (frame + 1) * 125 * SECOND // fps
        game.update()
    assert game.stage == 2 and game.remaining_ns() == 115 * SECOND


def test_format_roundtrip_and_exact_ranking():

    records = [
        Record("PLAYER 1", 10, 2, 3, 30 * SECOND, 100),
        Record("EARLIER", 10, 2, 3, 30 * SECOND, 99),
        Record("ZERO", 0, 0, 1, 0, 90),
    ]
    assert records[0].score_text == "45.00"
    parsed, error = parse_records(serialize_records(records))
    assert not error and parsed == ranked(records)
    assert parsed[0].player == "EARLIER"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"TEXTTWISTER|2\n",
        b"\xff",
        b"{}",
        (HEADER + "\nA|1|0|1|0\n").encode(),
        (HEADER + "\nA|1|0|0|0|1\n").encode(),
        (HEADER + "\nA|1|2|1|0|1\n").encode(),
        (HEADER + "\nA|-1|0|1|0|1\n").encode(),
        (HEADER + "\nA|nan|0|1|0|1\n").encode(),
        (HEADER + "\nA|" + "9" * 33 + "|0|1|0|1\n").encode(),
        (HEADER + "\n A|1|0|1|0|1\n").encode(),
        b"A" * 65537,
    ],
)
def test_malformed_save_returns_error(data):

    records, error = parse_records(data)
    assert not records and error


def test_bad_record_reports_line():

    _, error = parse_records((HEADER + "\nA|1|0|1|0|1\nbroken\n").encode())
    assert "line 3" in error


@pytest.mark.parametrize("mode", ["protected", "unprotected"])
def test_persistence_and_top_hundred(tmp_path, mode):

    path = tmp_path / "leaderboard.txt"
    board = Leaderboard(path, mode)
    for i in range(103):
        assert board.add(Record("PLAYER", i, 0, 1, 0, i))
    loaded = Leaderboard(path, mode)
    assert not loaded.error and len(loaded.records) == 100
    assert loaded.records[0].words == 102
    assert not list(tmp_path.glob(".leaderboard-*.tmp"))


def test_corrupt_file_is_preserved(tmp_path):

    path = tmp_path / "leaderboard.txt"
    path.write_bytes(b"broken")
    board = Leaderboard(path)
    assert board.error
    assert not board.add(Record("PLAYER", 1, 0, 1, 0, 1))
    assert path.read_bytes() == b"broken"


def test_bounded_file_reads_and_oversized_leaderboard(tmp_path):

    path = tmp_path / "large.txt"
    path.write_bytes(b"a" * 1_000_000)
    assert read_unprotected(path, limit=4) == (b"aaaa", "")
    board = Leaderboard(path)
    assert "64 KiB" in board.error
    assert not board.records


def test_file_resource_modes(tmp_path, monkeypatch):

    path = tmp_path / "missing"
    data, error = read_protected(path)
    assert data is None and error
    with pytest.raises(FileNotFoundError):
        read_unprotected(path)
    def fail_replace(self, target):
        raise OSError("replace failed")
    monkeypatch.setattr(Path, "replace", fail_replace)
    assert "replace failed" in write_protected(path, b"data")
    assert not list(tmp_path.glob(".leaderboard-*.tmp"))
    with pytest.raises(OSError, match="replace failed"):
        write_unprotected(path, b"data")
    assert list(tmp_path.glob(".leaderboard-*.tmp"))


def test_existence_check_failure_respects_resource_mode(tmp_path, monkeypatch):

    def fail_exists(self):
        raise PermissionError("directory inaccessible")
    monkeypatch.setattr(Path, "exists", fail_exists)
    assert "directory inaccessible" in Leaderboard(tmp_path / "scores.txt").error
    with pytest.raises(PermissionError):
        Leaderboard(tmp_path / "scores.txt", "unprotected")


def test_drawing_clips_and_preserves_depth():

    fb = Framebuffer(20, 20)
    fb.depth[0] = 0.5
    fill_rect(fb, -5, -5, 7, 7, 0xFFFFFF)
    assert sum(bool(c) for c in fb.color) == 4
    draw_text(fb, "ABC?\n12", -2, 12, 0xFF0000, 1)
    assert fb.depth[0] == 0.5
    with pytest.raises(ValueError):
        draw_text(fb, "A", 0, 0, 1, 0)


@pytest.fixture
def app(index, clock, tmp_path):

    return GameApp(
        index,
        Leaderboard(tmp_path / "leaderboard.txt"),
        clock=lambda: clock[0],
        wall_clock=lambda: 123,
        rng=random.Random(7),
    )


def test_ui_typing_guess_next_and_record_once(app):

    for char in "ALICE":
        app.handle_key(char.lower(), char)
    app.handle_key("return")
    assert app.player == "ALICE"
    for char in "listen":
        app.handle_key(char)
    app.handle_key("return")
    assert app.game.qualified and not app.entry
    app.handle_key("f2")
    assert app.game.stage == 2
    app.handle_key("escape")
    app.finish()
    assert app.screen == "over" and len(app.leaderboard.records) == 1
    assert app.handle_key("escape")
    app.handle_key("return")
    assert app.screen == "playing" and app.game.stage == 1


def test_ui_timeout_clears_stale_entry_without_restarting(app, clock):

    app.start()
    app.entry = "listen"
    clock[0] = STAGE_TIME
    app.handle_key("return")
    assert app.screen == "over" and not app.entry
    assert len(app.leaderboard.records) == 1


def test_expiration_during_submission_does_not_turn_next_enter_into_restart(app):

    app.start()
    app.entry = "listen"
    samples = iter((STAGE_TIME - 1, STAGE_TIME, STAGE_TIME))
    app.game.clock = lambda: next(samples)
    app.handle_key("return")
    assert not app.game.active
    app.handle_key("return")
    assert app.screen == "over"
    assert len(app.leaderboard.records) == 1


def test_headless_screens_clicks_and_no_thread_creation(app, monkeypatch):

    def forbidden(*args, **kwargs):
        pytest.fail("game created a thread")
    monkeypatch.setattr("threading.Thread", forbidden)
    fb = Framebuffer(960, 640)
    app.render(fb)
    assert any(fb.color)
    start = app.buttons[0]
    assert app.click(start.x + start.width // 2, start.y + start.height // 2)
    app.render(fb)
    assert app.screen == "playing"
    assert any(b.action == "next" and not b.enabled for b in app.buttons)
    app.finish()
    app.render(fb)
    assert app.show_answers
    app.handle_key("tab")
    app.render(fb)
    assert not app.show_answers
    app.handle_key("tab")
    assert app.show_answers


def test_3d_rendering_keeps_explicit_game_time_and_projection_rules(app):

    app.start()
    fb = Framebuffer(960, 640)
    app.render(fb, 0)
    initial = fb.color[:]
    assert app.projection == "perspective" and any(fb.depth)
    app.render(fb, 1.0 / 144)
    assert fb.color == initial
    assert app.game.stage == 1 and app.game.words == 0
    app.game.stage = 2
    app.render(fb, 1.0 / 7)
    assert app.projection == "orthographic"
    assert fb.color != initial


def test_real_wordlist_and_large_solution_pagination(tmp_path):

    path = Path(__file__).resolve().parents[1] / "TextTwister" / "word_list.txt"
    index = WordIndex(path.read_text(encoding="ascii").splitlines())
    puzzle = index.puzzle("triangle")
    assert {"triangle", "integral", "altering"} <= puzzle.answers
    assert len(puzzle.answers) > PAGE_SIZE
    app = GameApp(index, Leaderboard(tmp_path / "scores.txt"))
    app.start()
    app.game.puzzle, app.game.rack = puzzle, puzzle.rack
    app.finish()
    app.show_answers = True
    app.turn_page(1)
    assert app.page == 1
    fb = Framebuffer(960, 640)
    app.render(fb)
    app.turn_page(10000)
    assert app.page == (len(puzzle.answers) - 1) // PAGE_SIZE


def test_manual_cli():

    assert resource_mode([]) == "protected"
    assert resource_mode(["--resource-mode=unprotected"]) == "unprotected"
    with pytest.raises(ValueError):
        resource_mode(["--resource-mode", "bad"])


def test_unprotected_storage_does_not_dispatch_to_protected_functions(tmp_path, monkeypatch):

    def forbidden(*args, **kwargs):
        pytest.fail("unprotected mode called a protected resource function")
    for name in ("read_protected", "remove_protected", "write_protected"):
        monkeypatch.setattr("TextTwister.storage." + name, forbidden)
    monkeypatch.setattr(Leaderboard, "_load_protected", forbidden)
    path = tmp_path / "scores.txt"
    board = Leaderboard(path, "unprotected")
    assert board.add(Record("PLAYER", 1, 0, 1, 0, 1))
    assert Leaderboard(path, "unprotected").records == board.records
