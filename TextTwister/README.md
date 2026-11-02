# TextTwister

A single-threaded word game drawn entirely into the rasterizer framebuffer.
Tk presents pixels and delivers input; there are no native text or button widgets.
Letters, title, buttons, and the supporting rack tray are rendered as actual 3D
boxes. Bitmap text is baked into face textures and projected with its mesh.
Python 3.12+ with Tk is required. No additional runtime packages are needed.

```bash
python -m TextTwister
python -m TextTwister --resource-mode unprotected
```

Enter a name, or leave it blank for `PLAYER`, then press Enter. Names allow up to
16 ASCII letters, digits, and spaces. Clickable buttons mirror the keyboard controls.

| Control | Action |
| --- | --- |
| Letters / Backspace | Edit the guess or player name |
| Enter | Start, submit a guess, or play again |
| Space / F3 | Shuffle the rack during play |
| F2 | Advance after finding a full-rack word |
| Page Up / Page Down | Browse word pages |
| Tab | Switch scores and missed answers after the run |
| Escape | End the current run; exit from the start/game-over screen |

## Rules

Stages 1–3 use six letters, stages 4–6 use seven, and later stages use eight.
Find words of at least three letters using the available letter counts. The
supplied `word_list.txt` is authoritative, including its less common words.
Each accepted word scores once per stage; it may score again in a later stage.

One full-rack word unlocks Next. Each additional full-rack anagram earns one bonus
point in addition to its normal word credit. Continue finding shorter words or
advance early to bank time. Counts by length, full-rack totals, and found words
are visible. After advancing, the previous rack's intended word is shown beneath
the status line. Game over prominently reveals the final intended word and all
full-rack alternatives, and opens the missed-answers view automatically. The
missed list puts the intended word first, then familiar answers, then the rest
of the dictionary answers.

A stage begins with 120 seconds plus the previous stage's remaining time. Manual
advancement adds that remainder to cumulative carryover. At timeout a qualified
stage advances automatically with zero carryover; an unqualified stage ends the
run. Timers continue while unfocused. Stalls never grant extra time, and input
at the deadline is processed after expiration. Automatic advancement can therefore
be followed by game over if a long stall also exhausted the following stage.

The score is:

```text
(accepted_words + anagram_bonus_points) * stage_reached
    * (1 + cumulative_carryover_seconds / 120)
```

`stage_reached` is the highest stage entered. Carryover is the sum banked at
manual transitions, including inherited time still remaining then. Timing is
stored in nanoseconds; ranking uses an exact integer numerator. Displayed scores
round to two decimal places. Zero carryover is valid and never divides by zero.

## 3D stages and frame timings

Stages cycle through these projections, independently of the rack-length tiers:

| Stage in each four-stage cycle | Projection |
| --- | --- |
| 1 | Perspective |
| 2 | Orthographic |
| 3 | Panini |
| 4 | Ellipsoid |

The current projection is shown in the header. Letters tilt and bob using elapsed
seconds, not frame counts. Labels are UV-mapped onto the box fronts, so they tilt,
foreshorten, and warp with the geometry. Nonlinear projections use subdivided
front faces; they remain tessellated approximations of curved projection.
Sides sample the solid texture rim. Buttons use their projected triangles for
mouse hits, including curved/tessellated outlines. The timer, guess field, word
lists, and score stay readable as 2D overlays.

Static textured button/title images use a 32-entry cache; textures/materials use
a separate 64-entry cache. Rack buffers, meshes, and viewport renderers are reused.
The game continues to run on one thread. Rendering costs do not change deadlines.

The terminal reports approximately once per second:

```text
FPS 14.2 | frame 70.40ms | render 61.80ms | upscale 0.00ms | encode 4.10ms | upload 4.50ms | 960x640 sync
```

Numbers are illustrative. FPS counts successful uploads; frame time covers the
current tick's work before scheduling. Individual stage timings use time-based
smoothing. Asynchronous viewers may report stages from different completed
frames, so their stage averages are not necessarily additive. Library callers
can disable reporting with `Viewer(..., log_timings=False)`.

## Index and persistence

Letters map to the first 26 primes (`a=2`, `z=101`). Anagram words share an integer
product. Valid guesses still come from the full dictionary, but rack seeds come
from the hand-curated everyday vocabulary in `data/vocabulary.py`. Every seed is
checked against the supplied dictionary. Obscure anagrams remain optional bonus
answers; the intended word is always a familiar seed.

Each familiar signature gets a deterministic 0-100 scrambling estimate. The
score includes length, uncommon letters, repeated letters, vowel imbalance,
consonant runs, and the number of familiar three-to-five-letter answers. The
opening score cap is 8, increasing by two per stage up to 60. Stage 4 has a cap
of 14 and stage 7 a cap of 20. Selection is uniform among eligible signature
buckets, avoiding repeats until the eligible pool is exhausted. Growing caps
unlock additional familiar racks; they never unlock arbitrary obscure seeds.

The current score and easy/medium/hard label appear below the projection label.
This is a transparent gameplay heuristic and manually curated familiarity list,
not a measured corpus-frequency or linguistic difficulty ranking. The supplied
list currently offers 164 eligible six-letter opening racks. Custom dictionaries
may pass their own `common_words` iterable to `WordIndex`; a rack length with no
familiar seeds is rejected rather than silently selecting obscure words. If a
small custom list has no candidate under the cap, its easiest familiar candidates
are used.

The chosen familiar source word is retained when its letters are shuffled, so
reveals identify the actual intended word. Every full-rack dictionary anagram
still qualifies. Subsets are generated by letter multiplicities and looked up
directly, without scanning the word list per guess.

The dictionary loads before the timer starts, relative to the package directory.
Blank lines are ignored, invalid non-lowercase-ASCII entries report their line,
duplicates are collapsed, and lengths outside 3–8 are excluded from the index.

Scores save beside this file in `leaderboard.txt`, which Git ignores. The custom
ASCII format has one header and six pipe-separated fields per record:

```text
TEXTTWISTER|1
PLAYER|10|2|3|30000000000|1793620800000000000
```

Fields are player name, word count, bonus count, stage, cumulative carryover in
nanoseconds, and UTC Unix completion time in nanoseconds. Scores are derived,
not parsed from floating-point text. The best 100 runs are retained; ten are
shown. Ties favor earlier completion. Each run is recorded once on timeout or
manual exit. Missed answers refer to the final rack.

The handwritten reader validates ASCII, the version, six fields, allowed names,
decimal digits, and counter ranges. It rejects files larger than 64 KiB or more
than 100 records. Malformed files are preserved and persistence is disabled for
that session. Protected IO failures show a status message while the game remains
usable. Writes use a temporary file and atomic replacement.

## Resource modes and source policy

`protected` is the default. Narrow resource handlers cover window initialization,
presentation, cleanup, and file access. Rendering, game rules, math, parsing,
input handling, and mailbox availability do not use exception-driven control flow.

`unprotected` selects callable alternatives without authored `try` blocks. Their
IO/Tk errors propagate; failure can leave a temporary file or partially initialized
window. Context managers still close file handles. Commented protected reference
patterns accompany the unprotected implementations. This mode is not a change
to Python's managed-memory model.

Authored production code does not use JSON, CSV, TOML/YAML, regex parsers,
`argparse`, pickle, AST parsing, or eval-based formats. String helpers and numeric
conversion after ASCII validation are allowed. Image decoders and the internal
parsers of development tools remain permitted.

## Verification

From the repository root:

```bash
python -m pytest -q
python -m ruff check .
python -m basedpyright --warnings
```

Tests cover prime products, repeated letters, progression, anagram bonuses,
deadlines and stalls, familiar seed selection and complexity caps, source-word
reveals, handwritten formats, both resource modes, buffer ownership,
single-threaded presentation, input, pagination, and headless UI rendering. A
lightweight handwritten source audit guards parser imports and confines active
`try` blocks to audited protected resource functions.

## Code organization and extension

This README covers playing and running the game. Developer contracts live in
[IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md), separate from the rasterizer's
specification. Docstrings/annotations describe interfaces; block comments point
to the relevant specification before dense computations.

- `data/` holds the static vocabulary, prime/rule tables, theme, instructions,
  and save-format constants. Edit these declarations instead of algorithm bodies.
- `game.py`, `words.py`, and `vocabulary.py` implement rules, lookup and difficulty.
- `app.py` handles input/session coordination; `ui/` paints screens, caches 3D
  viewports and owns reusable mutable widgets.
- `storage.py` handles the handwritten save format and resource modes.

The original word_list.txt and saved leaderboard location are unchanged. The old
scene3d import path remains a compatibility facade. Animated transforms, cameras,
widgets and viewport buffers are reused; answer sets and completed-stage metadata
remain snapshots. No parser library or additional runtime dependency was added.

Source follows the Framebuffer layout: statements in a block stay adjacent, one
blank line separates computation phases, and two separate methods or sections.
Assignments and declaration rows are grouped and aligned where useful. The
preferred width is 150 characters. See the root
[code style specification](../IMPLEMENTATION_SPEC.md#documentation-and-code-style).
