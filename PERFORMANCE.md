# Performance measurements

Run from the repository root:

```bash
python -m benchmarks.benchmark_rendering --number 10 --repeat 5
```

This standard-library harness reports the best elapsed time per call across
repeated batches. Rendering cases include framebuffer clear. The lit sphere uses
the scene pipeline; the triangle cases isolate shaded fill and interpolation.
PNG encoding excludes Tk decoding/upload. Index cases materialize the same list
of triangle index tuples.

## Observed changes

Measured locally on CPython 3.14.7, Windows, at 320x180. For the kernel comparison,
the previous implementation's half-pixel bug was corrected before timing, so both
versions sample the same positions. Each result is the best of four batches of
five draws, using a constant fragment callback and unequal inverse-w values.

| Varying components | Previous kernel | Incremental kernel | Time reduction |
| --- | ---: | ---: | ---: |
| 2 | 20.986 ms | 14.774 ms | 30% |
| 3 | 23.043 ms | 18.515 ms | 20% |
| 5 | 28.805 ms | 21.770 ms | 24% |
| 8 | 71.983 ms | 68.365 ms | 5% |

An independent index microbenchmark measured 0.084 seconds for repeated triangle
method calls versus 0.012 seconds for grouped iterator reads over 1,000 sphere
index-list constructions (best of five). That is about 7x faster for this
operation; it does not imply a 7x improvement in complete scene rendering.

One subsequent harness run (`--number 5 --repeat 3`) measured a lit sphere at
13.533 ms and PNG encoding at 0.637 ms. These are current measurements, without
an end-to-end baseline. CPU, interpreter, geometry, shader cost, occlusion, and
display backend all affect results; these numbers are observations, not CI gates.

## Implementation choices

- Increment attribute numerators and reciprocal-w denominator across spans.
  Keep scalar specializations for common varying sizes and private mutable
  lists for other sizes. Recompute starts each row to bound accumulated error.
- Group index reads with a shared iterator and `zip`, and collect vertex shader
  results with a list comprehension. These built-ins avoid Python dispatch.
- Cache the inverse-transpose normal matrix by model data. Retain the existing
  baked perspective MVP and mutable viewport scratch.
- Reuse viewer snapshots and the worker's upscale framebuffer. Copy only color,
  submit once per tick, and replace obsolete pending pixels under a condition
  lock. A single ready slot always presents the newest encoded result.
- The viewer defaults to uncapped scheduling (`tick_ms=0`). An optional period
  includes render cost. This removes deliberate pacing delay; it does not remove
  CPU, encoding, Tk, or OS scheduling costs. Animation uses elapsed seconds, so
  benchmark FPS never determines its speed.
- Clip offscreen lines before stepping so runtime follows visible span length.

Keep correctness tests independent of wall-clock speed. Re-run the harness on
representative workloads before adding specialization, caching mutable resources,
or replacing loops with comprehensions or itertools. Public math objects retain
their existing semantics; scratch storage must stay private to a draw or worker.

## TextTwister workload

TextTwister indexes the supplied dictionary once before gameplay, retaining
signature buckets for lengths 3-8. Earlier local measurement found 88,296 relevant
signatures with roughly 0.33 seconds of indexing. An eight-distinct-letter rack
has only 256 subset signatures to inspect; the supplied TRIANGLE rack has 415
answers, so the UI paginates rather than drawing an unbounded word list.

The game uses synchronous 960x640 framebuffer rendering, image encoding, and Tk
upload on one thread. Bitmap glyph coordinates are precomputed; clipped UI
rectangles use packed array slices. This trades asynchronous throughput for the
required single-threaded execution model. Nanosecond deadlines keep game time
independent of frame rate and encoding cost. No timing threshold is a CI gate.

A local hidden-window smoke test successfully uploaded both PNG and in-memory PPM
frames at 960x640 with zero additional threads. Headless tests check the same
ownership/timing paths without requiring a platform display.

## 3D TextTwister presentation

The letter rack now uses animated, depth-tested boxes and a supporting tray.
Text labels are face textures, with front-only tessellation in nonlinear stages.
Buttons and the title use cached 3D viewport images, limiting repeat work to the
animated rack and readable overlays. Material and image caches are bounded, and
rack framebuffers/renderers are reused. Mouse hit geometry is cached with images.

An initial local 960x640 six-letter measurement observed warm game-render times
around 57-70 ms (cold projection/image-cache fills around 126-138 ms). These are
workload observations, not FPS promises or regression gates. Actual rates depend
on projection, rack length, CPU, and upload cost. The terminal now prints measured
FPS and per-stage timings once per second by default. `log_timings=False` disables
that output for embedding/measurement. Gameplay deadlines remain independent of
those rendering costs.

## Familiar rack selection

Seed candidates now come from a hand-curated everyday vocabulary intersected with
the dictionary. Complexity metadata and familiar short-answer counts are built
once with the index, not in the render or guess loops. The current dictionary has
318 familiar six-letter signature candidates, 164 of which meet the opening
complexity cap of 8; seven- and eight-letter familiar pools contain 336 and 262
signatures. Selection scans these small candidate pools, while accepted guesses
continue to use the complete dictionary answer sets. No frequency service,
external parser, extra worker, or runtime dependency is introduced.

## Organization and allocation refactor

Measured locally on the same CPython 3.14 process configuration at 960x640, using
three batches of three warm renders per projection and fixed clock/racks. Before
and after framebuffers were byte-compared: all four projection frames were
pixel-identical. Timing remains observational rather than a CI gate.

| Projection | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Perspective | 74.85 ms | 69.86 ms | 7% |
| Orthographic | 72.97 ms | 66.66 ms | 9% |
| Panini | 66.05 ms | 62.03 ms | 6% |
| Ellipsoid | 79.29 ms | 75.03 ms | 5% |

A 25-character bitmap text sample at scale 3 changed from 0.428 ms to 0.260 ms
(best of three batches of 200 calls), about 39% less time. Glyph spans and cached
solid rows replace per-bit rectangle allocation. Typed memoryview framebuffer
copies avoid per-row temporary pixel arrays. These are C-backed operations, not
a claim that the Python shader loop runs natively.

Owned transform matrices and stationary cameras are reused, eliminating chains
of temporary matrices/vectors from rack poses. Matrix data remains a tuple
snapshot so existing normal-cache invalidation is preserved. Flat draws reuse
private clip-vector capacity. Static box geometry, glyphs and vocabulary now live
in dedicated data modules; this is an organization change, not itself a speedup.

The benchmark harness also reports composed versus owned TRS, bitmap text, and
framebuffer blits. Re-run on the target machine; ratios can vary with load and
Python version. Public convenience APIs retain the existing low-level rendering
paths, and no runtime package or new parser was introduced.
