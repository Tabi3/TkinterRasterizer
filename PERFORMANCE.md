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
  submit once per tick, and replace obsolete queued frames under backpressure.
  The encoder skips stale pending work and presentation selects the newest result.
- The viewer defaults to uncapped scheduling (`tick_ms=0`). An optional period
  includes render cost. This removes deliberate pacing delay; it does not remove
  CPU, encoding, Tk, or OS scheduling costs. Animation uses elapsed seconds, so
  benchmark FPS never determines its speed.
- Clip offscreen lines before stepping so runtime follows visible span length.

Keep correctness tests independent of wall-clock speed. Re-run the harness on
representative workloads before adding specialization, caching mutable resources,
or replacing loops with comprehensions or itertools. Public math objects retain
their existing semantics; scratch storage must stay private to a draw or worker.
