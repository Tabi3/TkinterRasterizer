# TextTwister implementation specification

The game is a single-threaded client of the rasterizer. This specification owns
game behavior; the root implementation spec owns rendering math, public renderer
interfaces, and buffer operations. User instructions and controls are in README.md.

## Organization and style

- `data/`: static everyday vocabulary, prime/rule tables, theme, instructions,
  save-format limits. Data modules contain declarations, not game algorithms.
- `words.py`: dictionary indexing, familiar rack profiles, and stage selection.
  `vocabulary.py` evaluates complexity using the data module's word set/weights.
- `game.py`: mutable run state and immutable completed-stage snapshots.
- `app.py`: session/input controller and leaderboard coordination.
- `ui/`: screen painting, mutable widget records, and cached 3D scene resources.
  The former `scene3d.py` import path remains a compatibility facade.
- `storage.py`: handwritten records and paired protected/unprotected IO paths.
- `__main__.py`: package-relative loading and synchronous Tk composition.

Keep algorithms out of data modules and screen drawing out of the controller.
The original user-supplied `word_list.txt` stays at its existing package-root path.
Generated scores remain separate and Git-ignored; no save migration is required.

Documentation has two layers. Docstrings and annotations describe inputs,
outputs, ownership, and failure behavior. Comments divide dense calculations into
logical blocks, explain the next block's intent, and reference its specification:

```python
# Spec: TextTwister/IMPLEMENTATION_SPEC.md#difficulty-and-selection.
# Profile costs belong to startup; frame callbacks only read the result.
```

Use the manual block spacing and alignment below throughout the game.
Runtime lists, widgets, matrices, and caches are mutable and owned by a session
or view. Answer sets, candidate metadata, and stage archives remain snapshots.

Source layout follows the root [code style contract](../IMPLEMENTATION_SPEC.md#documentation-and-code-style):
statements stay adjacent within a block, one blank line separates computation phases,
and two separate methods or sections. Operators and declaration rows are aligned where
useful, compact conditions remain readable, and the preferred width is 150.

## Game rules

`data/rules.py` defines the 120-second base stage, nanosecond units, rack lengths,
three-stage length tiers, difficulty progression, and input limits. Stages 1-3
use six letters, 4-6 use seven, and later stages use eight.

Accept any dictionary word of at least three letters using the available letter
multiplicities. Count it once per stage; it may score again in a later stage.
The first full-rack word qualifies Next. Each further full-rack anagram adds one
bonus point on top of its normal word credit.

Manual advancement banks the remaining nanoseconds and sets the new deadline
to `now + 120 seconds + remainder`. Qualified expiration automatically advances
with zero newly banked time; unqualified expiration ends the run. Absolute
`monotonic_ns()` deadlines expire before input is processed. Automatic advancement
uses the previous deadline, so stalls cannot grant time. Timers do not pause
on focus loss. A long stall can expire the following unqualified stage as well.

Score = `(words + bonus) * stage * (1 + carryover_ns / 120000000000)`.
Ranking compares the integer numerator; only display rounds to two decimals.
`stage` is the highest stage entered. Carryover sums every manually banked
remainder, including any inherited time that remained at the next transition.

`Game.last_stage` records the completed rack, intended source, and full-rack
alternatives before advancement or ending. Restart clears the archive. The
controller records the run once even if several close/timeout callbacks arrive.

## Dictionary index

Load the authoritative lowercase ASCII dictionary once, before starting time.
Ignore blank lines, reject invalid text with a line number, collapse duplicates,
and retain lengths 3-8. Letters map to primes from a=2 through z=101; signatures
are exact Python integers.

For a rack, enumerate multiplicities for each distinct letter and look up each
subset signature. Do not scan dictionary words or enumerate permutations during
a guess. `Puzzle` retains read-only word tuples, answer membership, counts by
length, `source_word`, `difficulty`, and `familiar_short_answers`. `full_words`
exposes the full-rack alternatives. Shuffling changes only visible letter order.

The intended source must be a dictionary member of the exact full-rack signature.
All dictionary anagrams still qualify, even when they are not familiar seeds.

## Difficulty and selection

`data/vocabulary.py` contains a hand-curated everyday word set, not measured
corpus-frequency ranks. Intersect it with actual dictionary members. An obscure
word sharing a familiar signature remains an optional answer, never the required
intended source. Custom indices may supply `common_words` explicitly.

The 0-100 complexity estimate sums and caps these components:

- 6 per letter beyond six.
- Twice the rare-letter weight: j=6, q=8, x=6, z=8, k/v=2, w/y=1.
- 2 per repeated letter.
- 5 per missing vowel below two; 3 per vowel above three.
- 4 per consonant in the longest run beyond two.
- 2 per missing familiar short answer below eight.

Vowels are a/e/i/o/u, with y also breaking consonant runs. Familiar short answers
are validated three-to-five-letter words respecting the rack's multiplicities.
Compute profiles once at index construction. For a signature with multiple
familiar sources, use minimum complexity then alphabetical tie-breaking.

The cap is `min(60, 8 + 2 * (stage - 1))`. Choose uniformly among eligible unused
signatures. Exhaustion resets only that eligible pool; growing caps unlock new
familiar candidates. If a small custom vocabulary has none below the cap, use
its minimum-complexity familiar candidates. Reject lengths with no familiar seeds.

## Presentation and input

`GameApp` owns the session, input, and reusable view. `GameView` paints screens
without changing deadlines. The controller ticks before painting and processes
Tk keyboard/mouse events in arrival order. A screen transition consumes stale
input rather than treating it as an immediate restart command.

The 960x640 framebuffer contains actual lit 3D letter boxes, a supporting tray,
title, and buttons. Stage projections cycle perspective, orthographic, Panini,
ellipsoid every four stages. Bitmap labels are front-face textures, so transforms
and perspective-correct interpolation move the text with its surface. Nonlinear
fronts use three subdivisions; other faces remain quads. Curved projection is
still a tessellated approximation. Side UVs sample the solid texture rim.

The timer, score, input field and word lists remain legible 2D overlays. Button
hits use the same projected, culled triangles as their visible surfaces. Lists
paginate 24 words. The next stage reveals the previous intended word; game over
shows the final source prominently and opens missed answers. Missed words sort
by intended source, familiarity, length, then alphabet. Found words sort by length
then alphabet. Reveals introduce no extra pause or time.

## Allocation and rendering

One view owns reusable meshes, stationary cameras, rack framebuffers, pose matrices,
and mutable widgets. Widget records and the visible button list are updated in
place. They are not shared between sessions. Completed-stage archives do not
refer to these mutable structures.

`write_trs()` overwrites an owned matrix with direct affine coefficients instead
of constructing and multiplying temporary translation/rotation/scale matrices.
Tuple snapshots of matrix data preserve normal-cache invalidation. Cameras are
cached per projection/aspect, and the tray transform changes only with rack size.

Static 3D images/hit triangles have a 32-entry cache; materials/textures have a
64-entry cache. The pixel compositor uses `Framebuffer.blit()` typed memoryviews
for C-backed color/depth copies. Glyph spans and solid fill rows are reused by
the renderer's bitmap painter. Rendering time never advances game time.

`Viewer(background_encode=False)` creates no worker or mailbox. Main-thread work
includes rendering, optional scaling, encoding, Tk upload and file IO. Terminal
reports show actual upload FPS and smoothed per-stage timings once per second.

## Leaderboard format

`data/save_format.py` defines the header and limits. The ASCII format is:

```text
TEXTTWISTER|1
PLAYER|words|bonus|stage|carryover_ns|finished_unix_ns
```

There are exactly six fields. Names are trimmed, up to 16 ASCII letters, digits
or spaces. Numeric fields are nonnegative ASCII decimals of at most 32 digits;
stage must be positive and bonus cannot exceed words. Validate before conversion.
Read at most 64 KiB plus one detection byte and retain at most 100 records.

Derive scores rather than parsing floating-point text. Rank descending by exact
score, then earlier completion. Display ten scores. Missing files start empty;
malformed or unsupported saves are preserved and disable persistence for that
session. Writes use a same-directory temporary file and atomic replacement.

## Resource policy

Authored production code uses no JSON/CSV/TOML/YAML parsers, argparse, regex
parsers, pickle, AST parsing or eval-based formats. String helpers and validated
numeric conversion are allowed. Data modules use Python declarations; no new
format parser is required. Image decoders and development-tool internals remain
outside this restriction.

Ordinary game, rendering and input logic has no try blocks. Protected mode is
the default. Only audited `_protected` resource functions in storage and the Tk
viewer may catch narrow IO/image failures or use finally for cleanup. Protected
score errors report status while the game remains usable.

Callable unprotected alternatives contain no authored try blocks. Select them
with `--resource-mode unprotected`. Unexpected errors propagate and may leave
temporary files, windows, or buffer state unfinished. Context managers still
close files. Commented protected reference patterns accompany the alternatives.
Neither mode introduces raw-memory access or a new memory-management model.

## Validation

Run pytest, Ruff lint checks, and basedpyright from the repository root.
The existing CI checks both packages and tests on Windows/Linux and Python
3.12/3.14. The handwritten source audit guards parser imports and allowed resource
handlers. Tests cover dictionary signatures, repeated letters, seed difficulty,
timing boundaries/stalls, carryover, score arithmetic, reveals, manual records,
resource modes, projected hit shapes, cache bounds, 3D labels and headless screens.

Performance checks are observational. Preserve output and lifetime contracts;
do not use wall-clock thresholds as correctness gates. Root PERFORMANCE.md records
the reproducible benchmark command and observed changes.
