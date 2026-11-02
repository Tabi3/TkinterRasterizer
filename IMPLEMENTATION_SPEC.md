# Implementation specification

## Project goals

The rasterizer is a screen-space triangle-fill pipeline. Given a triangle
with clip-space vertices c_i = (x_i, y_i, z_i, w_i), it computes the set of
integer pixels P = {(px, py)} whose centers (px+0.5, py+0.5) lie inside the
triangle in screen space, subject to the reverse-Z depth test:

    z_ndc(px, py) > depth[py, px]   =>  write color, optionally update depth

where z_ndc is affine in screen-space barycentric weights (derived below).

Layers, ordered by data flow:

- `Rasterizer/algebra/`: `Vec2/3/4`, `Mat3/4`, quaternions, and coordinate
  transforms. Math operations return value objects; vector components can be
  mutated by callers.
- `Rasterizer/geometry/`: vertices, meshes, bounds, and procedural builders.
- `Rasterizer/materials/` and `Rasterizer/scene/`: shader/material resources,
  lights, and transform hierarchy.
- `Rasterizer/rendering/`: `Renderer` orchestrates vertex → clip → viewport →
  cull → rasterize. The backend clips/culls; the rasterizer handles spans and
  interpolation; `Framebuffer` owns packed color (`'I'`) and depth (`'f'`)
  arrays.
- `Rasterizer/display/` and `Rasterizer/demos/`: window integration and runnable
  examples. Keep display code out of the rendering core.

The rasterizer keeps mutable scratch values local to a draw call. Per-pixel
work uses scalar state (edge functions, depth gradients, and varyings) rather
than allocating math objects in the inner loop.

The stage contracts are:

- Vertex shader: vs(vertex, u) → (c ∈ ℝ⁴ clip, v ∈ ℝⁿ varyings)
- Clipping: perspective cuts at `w_clip = near`; orthographic cuts at
  `z_clip = w_clip` (near) and `z_clip = 0` (far). One shared polygon
  clipper interpolates positions and varyings at the same intersections.
- Rasterizer: fills pixels satisfying edge tests, interpolates v by
  perspective correction, calls fs(v) → 0xRRGGBB.

## API and organization

The package root exports the ordinary application types: Camera, Framebuffer,
Renderer, Material, Mesh, Node, Light, Texture, vectors and Mat4. Viewer stays in
`Rasterizer.display` so headless imports do not pull in windowing code.

The convenience API has two operations:

- `Renderer.draw(fb, mesh, camera, *, material=None, transform=None, color=...,
uniforms=None)` draws one mesh without clearing or advancing time. A missing
  material selects the flat-color path. A material supplies shaders and flags.
- `Renderer.render(fb, root, camera, *, background=..., uniforms=None,
wireframe=False, wireframe_color=...)` prepares a background and draws a scene.
  An integer clears to a solid color, a pair draws a gradient, and None preserves
  existing pixels/depth. Wireframe uses the established projection-aware edge path.

`renderer.uniforms` is persistent default state, initialized with one directional
light. Callers own its lighting and timing. Explicit uniforms override it. The
low-level draw_mesh/draw_mesh_shaded/draw_scene APIs remain available unchanged.
`Material.gouraud()` and `Material.textured(texture, lit=...)` wrap the built-in
shader factories, which remain supported for custom/material-building code.

Camera and projection policy live in `rendering/camera.py`; reusable uniforms
live in `rendering/uniforms.py`. The former imports from rendering.renderer remain
compatible. The dispatch module owns orchestration, not camera/state definitions.
Static fonts, box-face bases, projections and defaults live in `Rasterizer/data/`.
Those scalar tables never become shared mutable mesh/vector instances.

Demos use these public entry points, initialize resources once, reuse poses, and
share `demo_viewer()` for CLI/window setup. They still support direct-file and
module execution. The game keeps its own user README and implementation spec.

## Allocation and mutable scratch

Reused scratch is explicitly owned and must not leak across calls or sessions.
Renderer keeps flat-path clip-vector capacity; vectors are overwritten for active
vertices, and stale capacity is never traversed by mesh indices. The shaded
vertex-shader callback contract still returns independent results; do not alias
shader output to a single scratch vector.

`write_trs(output, ...)` computes T * Ry * Rx * Rz * S directly into an owned Mat4.
It reuses the matrix object but replaces its data with a tuple snapshot. Ordinary
matrix operators retain their existing value behavior, and Uniforms normal-cache
comparison still detects pose changes. One private matrix per animated object
avoids constructing chains of temporary matrices and vectors each frame.

Game cameras, viewports, meshes and static transforms are cached by the owning
view. Mutable widget records and lists are reused; answer sets and archived run
metadata remain snapshots. Prefer measured changes to blanket mutability or
replacing every loop with a comprehension.

## Bulk pixel operations

Framebuffer colors and depths remain packed arrays. `Framebuffer.blit(source,x,y)`
uses typed memoryview slice assignment for clipped C-backed color/depth copies.
Ordinary distinct-buffer copies allocate views, not per-row pixel arrays. Alias
copies snapshot source arrays first so overlapping moves do not corrupt later rows.

Font glyph rows live in data/font5x7.py. The drawing module compiles contiguous
spans once and copies cached solid array rows rather than painting individual
bits with new rectangles. Cached rows are private read-only sources; writable
framebuffers never alias them. The row cache is bounded at 128 entries.

The unit-box face table is scalar data. Mesh builders create their own normal
vectors, avoiding shared mutable constants. Unit-box orientations/UV winding and
existing output are preserved when this table is used by engine and game meshes.

## Documentation and code style

Docstrings and annotations define purpose, inputs/outputs, ownership, and failure
behavior. Comments divide dense computations into phases, introduce the next
block, and cite the relevant heading in this specification. Example:

```python
# Spec: IMPLEMENTATION_SPEC.md#allocation-and-mutable-scratch.
# Only private flat-path vertices are overwritten; shader outputs stay independent.
```

Source layout uses the `Framebuffer` module as its reference, with a preferred
line length of 150. A blank line separates related computation phases; two
blank lines separate methods, section dividers, and top-level declarations:

```python
class Framebuffer:
    __slots__ = (
        "_clear_color",
        "_color_template",
        "_depth_template",
        "color",
        "depth",
        "height",
        "size",
        "width",
    )

    def __init__(self, width: int, height: int) -> None:
        if not isinstance(width, int):
            raise TypeError("width must be int")
        if width <= 0:
            raise ValueError("width must be positive")

        self.width = int(width)
        self.height = int(height)
        self.size = self.width * self.height

        # Spec: IMPLEMENTATION_SPEC.md#documentation-and-code-style.
        # Build reusable storage once; frame operations use C-backed slices.
        self.color = array("I", [0]) * self.size
        self.depth = array("f", [0.0]) * self.size
```

- Keep statements in one computation block adjacent. Separate related phases
  with one empty line; separate methods, sections, and top-level blocks with
  two empty lines. Within a related assignment group, align operators.
  A declaration block such as `__slots__` or a tuple/list may use grouped rows;
  align its rows to the data shape rather than forcing one item per line.
- Put one empty line before `elif` and `else` branches when they contain a
  separate computation block; keep decorators attached to declarations.
- Put short conditional bodies on one line when the result stays readable.
  Align the conditions and colons of compact `if`/`elif` chains. Long conditions
  or bodies remain multiline; do not introduce semicolon-separated statements.
- Lay out long tuples, lists, slots and function parameters in grouped rows,
  like NumPy array declarations. Matrix rows should follow their dimensions
  where they fit the preferred width. This is a layout convention, not a NumPy dependency.
- Use section dividers (`# ----- name -----`) between major method groups. Keep
  related helpers together so a section reads as one computation block.
- A short guard may be written as `if   (condition): statement`; align the
  condition and colon across an `if`/`elif` chain. Use ordinary multiline blocks
  when the condition or statement is long, nested, or has more than one action.
- Keep docstrings attached to their declarations. Computation comments provide
  a preamble and a specification reference, rather than repeating assignments.

Ruff lint remains enabled. Ruff formatting is the baseline for spacing, while
the configured line length and small alignment exceptions preserve readability.
CI runs lint, types and tests after formatting changes are reviewed.
Static data belongs in dedicated data modules, not giant algorithm-file literals.

## Numerical conventions

- Column-vector math: (M · v)_i = Σ_j M[i, j] · v_j. Mat4 storage is
  row-major flat, M[i, j] = data[i·4 + j].
- Perspective clip space: `w_clip = -Z_view`, `z_clip = near`. Preserve x/y
  even for vertices behind the eye so homogeneous clipping finds correct edges.
- Orthographic clip space: `w_clip = 1`, `z_clip = (far + Z_view)/(far - near)`.
  Attributes and depth interpolate affinely in screen space. `Camera.far`
  defaults to 1000 world units and affects orthographic rendering only.
- Panini/ellipsoid: use `w_clip = -Z_view`, `z_clip = near`. Filled triangles
  approximate curved projected surfaces with straight chords; their clipping
  and attribute interpolation are approximations, especially near the eye.
- NDC: (x_ndc, y_ndc, z_ndc) = (x_clip/w_clip, y_clip/w_clip, z_clip/w_clip),
  with z_ndc ∈ [0, 1] reverse-Z: near → 1, far → 0.
- Depth test: z_ndc_new > depth[px, py]. Reverse-Z means "greater wins".
- Viewport: (px, py) = (W/2 + x_ndc·W/2, H/2 - y_ndc·H/2). Y is flipped so
  screen Y grows downward.
- Winding: CCW viewed from outside the surface. The viewport Y-flip makes
  front faces have negative signed area in screen space; cull iff area ≥ 0.

## Analytical foundations of the renderer

### 1. Camera and projection math

The renderer follows the standard right-handed camera convention used by many
real-time pipelines: the camera looks along the negative Z axis in view space,
so a visible point satisfies $z_{view} < 0$. For perspective the denominator is
$w_{clip} = -z_{view} > 0$.

For a view-space point $(x, y, z)$, perspective projection is:

$n_x = \frac{x}{-z} \cdot \frac{f}{aspect}, \quad
n_y = \frac{y}{-z} \cdot f,
\quad
w_{clip} = -z,
\quad
z_{clip} = near,$

with $f = 1 / \tan(\theta_{fov}/2)$ for perspective projection. The result is
then converted to NDC with:

$x_{ndc} = \frac{x_{clip}}{w_{clip}}, \quad
y_{ndc} = \frac{y_{clip}}{w_{clip}}, \quad
z_{ndc} = \frac{z_{clip}}{w_{clip}}.$

Because reverse-Z stores the near plane as 1 and the far plane as 0, the depth
comparison is simply:

$z_{new} > z_{depth}[px, py].$

This is the exact mirror of the standard depth test in forward-Z pipelines and
makes the clear value equal to 0.0 while preserving the rule that closer objects
win.

### 2. Edge-function rasterization

For a triangle with screen-space vertices $p_0, p_1, p_2$, the edge function for
edge $e_i$ is defined as:

$E_i(x, y) = a_i x + b_i y + c_i,$

where the constants are derived from the two vertices that define the edge. A
point lies inside the triangle if and only if all three edge tests evaluate as
positive under the chosen winding convention. The renderer uses the standard
"top-left" rule to consume boundary pixels consistently when $E_i = 0$.

This yields a scanline loop of the form:

$E_i(p + \Delta x) = E_i(p) + a_i,
\quad
E_i(p + \Delta y) = E_i(p) + b_i,$

which means the inner loop is dominated by additions rather than expensive
recalculation of the full edge equation at every pixel. In practice this is the
primary reason the rasterizer remains fast in Python: the hot loop avoids a full
multiply-heavy per-pixel setup.

### 3. Perspective-correct interpolation

Let $iw_i = 1 / w_{clip,i}$ and let $E_i(p)$ be the triangle edge function for the
vertex opposite $i$. The barycentric weight for vertex $i$ is:

$\lambda_i(p) = \frac{E_i(p)}{A}, \quad A = \sum_i E_i(p).$

The correct perspective-correct interpolant for varying data is:

$v_k(p) = \frac{\sum_i E_i(p)\, v_i[k] \, iw_i}{\sum_i E_i(p) \, iw_i}.$

This is why the rasterizer stores a per-vertex inverse-w value and computes the
weighted varying numerator and denominator in the same screen-space span loop. The
inverse-area factor cancels, so the fragment shader sees the correct value without
a costly division per varying channel.

### 4. Depth interpolation and reverse-Z

The depth buffer stores the NDC depth value; for a triangle, the screen-space
linear depth is:

$z_{ndc}(p) = \sum_i \lambda_i(p) z_i.$

Because the depth is affine in barycentric coordinates, the rasterizer can update
it incrementally across a scanline using the same edge-function evaluation scheme.
The reverse-Z convention preserves the common rule:

$\text{write if } z_{new} > z_{stored},$

so the final visible surface is the one with the greatest depth in the reverse-Z
space, which corresponds to the nearest object in view space.

### 5. Lighting and shading model

The shader pipeline takes a world-space normal $\hat{n}$ and a light direction
$\hat{l}$ and evaluates the diffuse term:

$d = \max(0, \hat{n} \cdot \hat{l}).$

The per-light contribution is then:

$\text{shade}_k = ambient_k + d \cdot light_k,$

with the color channel being multiplied by the sampled texture value when a
textured shader is active. This makes the lighting model a low-cost linear
function evaluated at vertices; resulting colors are interpolated for Gouraud
shading. Normals use the cached inverse transpose of the model linear transform,
so nonuniform scale is supported. Singular transforms produce zero normals
(ambient only). A coincident point light contributes zero diffuse illumination.

## Selected references and literature

- Foley, van Dam, Feiner, Hughes, and Phillips. Computer Graphics: Principles
  and Practice.
- Marschner and Shirley. Fundamentals of Computer Graphics.
- Haines and Akenine-Möller. Real-Time Rendering.
- Akenine-Möller, Haines, and Hoffman. Real-Time Rendering, 3rd edition.
- Angel and Shreiner. Interactive Computer Graphics.
- The reverse-Z convention used here is a standard real-time rendering pattern
  described in modern graphics texts and widely used in deferred and forward
  pipelines when depth writes are optimized for the near plane.

## Repository layout

Source files are grouped by their job, with static tables separated from algorithms:

- `Rasterizer/algebra/` — vectors, matrices, quaternions, and coordinate transforms
- `Rasterizer/geometry/` — vertices, meshes, bounds, and procedural primitives
- `Rasterizer/rendering/` — renderer, clipping/culling backend, rasterizer, framebuffer
- `Rasterizer/materials/` — material, texture, and shader definitions
- `Rasterizer/scene/` — scene graph and lights
- `Rasterizer/display/` — Tk viewer and input handling
- `Rasterizer/demos/` — runnable examples, including direct unshaded drawing
- `tests/` — isolated behavior and regression tests

`Rasterizer` re-exports the supported convenience API: `Camera`, `Object3D`,
`Renderer`, `Uniforms`, and `parse_resolution`. Implementation modules use
lowercase, purpose-based import paths. Internal paths may change as the
architecture evolves.

## Layer contracts

1. **Algebra and geometry:** value-oriented math types and mesh data. These
   modules do not depend on windowing or frame state.
2. **Materials and scene:** material callbacks, texture resources, lights, and
   transform hierarchy.
3. **Rendering:** transform vertices, clip triangles at the near plane, cull,
   rasterize, and write reverse-Z depth to a framebuffer.
4. **Display and demos:** adapt completed frames to Tk and show supported
   library behavior. Demos are entry points, not alternate implementations of
   engine logic.

The public renderer delegates clipping and raster work to private backend
modules. Keep the rendering order and numerical conventions above stable. Add
tests for each behavior change, and preserve the package-root API unless an
intentional breaking change is documented.

## Correctness and code quality

- Production code uses explicit exceptions for invalid input. Python
  `assert` statements belong in tests; optimized Python can remove runtime
  assertions.
- Test boundary conditions as well as normal output: clipping, winding,
  shared triangle edges, depth ordering, projection limits, and framebuffer
  dimensions.
- Remove unused implementation helpers. Keep public features only when they
  have a clear caller and a runnable demo when they are user-facing.
- Keep overloads and type annotations aligned with the actual supported input
  and output shapes.
- Ruff is configured in `pyproject.toml`. Run `python -m ruff check .` before
  changing behavior or imports. Preserve the manual block layout described above.

## Render state and input contracts

- `Material.cull` and `Material.depth_write` are mutable flags honored by scene
  and instanced draws. `draw_mesh_shaded(..., *, depth_write=True)` also exposes
  the latter directly. Disabling depth writes still performs the depth test.
- `cast_shadow` is reserved metadata; there is no shadow pass or alpha blending.
- Filled scene traversal rejects bounds entirely before the near plane, with
  boundary equality visible. Bounds must describe shader-deformed positions if
  a custom shader moves geometry. Refresh mesh bounds after vertex edits.
  No side-frustum or occlusion culling is currently implemented.
- `Uniforms.normal_matrix` lazily caches a `Mat3` inverse transpose keyed by
  model data. Shader projection callbacks accept homogeneous `Vec4` positions.
- Both textured shaders observe in-place edits to pixels. Rebuild shaders after
  changing texture dimensions, wrap mode, or the data object.
- Camera parameters must be finite: `0 < near < far`, positive ortho scale,
  `0 < fov_y_deg < 180`, ellipsoid strength in `[0, 1]`. Perspective retains its
  infinite far plane; finite `far` is used only by orthographic projection.
- Targets must match renderer dimensions. Empty meshes are supported;
  out-of-range indices and inconsistent varying lengths are rejected.
- Shaded screen vertices must be finite with positive inverse w. Coverage and
  interpolation both sample exactly `(x + 0.5, y + 0.5)`.
- Nodes reject cycles, detach from the old parent when reparented, and ignore
  duplicate attachment to the same parent.
- Wire segments clip in view space before projection. DDA lines clip to the
  pixel-index rectangle before stepping, bounding offscreen work.
  `curve_segments` must be a positive integer.
- Sky gradients put `color_top` in the first row and `color_bottom` in the last;
  a single-row framebuffer uses the top color. Depth resets to zero.

## Performance and allocation

Common 2-, 3-, and 5-component varyings use scalar numerator and denominator
increments across each span. Other sizes reuse a private mutable numerator list.
Each scanline recomputes its start to limit roundoff. Depth is screen-linear;
only attributes divide by interpolated inverse w.

Draws group the index iterator with `zip(indices, indices, indices)` instead of
repeated triangle method calls. Vertex results use a list comprehension. The
perspective path retains its baked MVP and private mutable viewport scratch.
Public vector/matrix semantics are unchanged. Framebuffer bulk work uses arrays
and slices rather than per-pixel Python copies.

Use built-ins, comprehensions, or itertools only where they reduce measured work.
Avoid temporary math objects in pixel loops. See `PERFORMANCE.md` for timings,
measurement limits, and the repeatable headless benchmark command.

## Viewer ownership and lifecycle

`Viewer(..., background_encode=True, resource_mode="protected")` preserves the
asynchronous demo path. A `FrameMailbox` owns three reusable snapshots, one pending
snapshot, and one ready result. A condition lock protects buffer availability and
ownership together. Submission replaces pending pixels under that lock; encoding
owns its snapshot exclusively until release. Publishing replaces a stale ready
result. No Queue.Empty/Full exceptions or racy empty-then-get checks are used.

The worker reuses one private upscale target and performs no Tk calls. Protected
encoding returns buffers in finally even when it fails. The unprotected worker
propagates failures and does not promise to return an in-flight buffer after an
error. Closing sets the stop predicate and wakes waiters; join waits up to one
second. Both close paths are idempotent on normal execution.

`background_encode=False` allocates no mailbox and creates no encoder thread.
Encoding, optional upscaling, and upload run on the Tk thread. TextTwister uses
this path. PNG is preferred on Tk 8.6+; PPM uploads use in-memory image bytes with
no temporary display files. Protected PNG failures select PPM; unprotected uploads
use the selected format and propagate failures. Upload errors do not increment
presented FPS. Timing telemetry is exposed through Viewer fields. With `log_timings=True`
(the default), a once-per-second console report includes successful-upload FPS,
whole-tick work time, render, upscale, encoding, upload, resolution, and mode.
Frame/FPS windows end after upload work, before scheduling. Values use time-based
smoothing; asynchronous stage averages need not correspond to the same frame.

Protected window initialization and run-loop wrappers clean up partial windows
or failed runs. Unprotected counterparts perform cleanup only on the success
path. See the resource policy below for the deliberately different guarantees.

## Parser and resource policy

Authored runtime and benchmark code uses no general-purpose format parser:
JSON, CSV, TOML/YAML, argparse, regex parsing, pickle, AST parsing, and eval-based
formats are excluded. String helpers are allowed. Validate ASCII decimal text
before numeric conversion. Image decoders and development-tool internals are
outside this restriction. The benchmark retains --number/--repeat through the
shared handwritten option reader; malformed/missing/duplicate options are rejected.
The existing resolution parser retains its API and validates numeric syntax;
viewer construction checks render-scale divisibility.

Ordinary rendering, game, input, math and mailbox operations contain no try blocks.
The normal-matrix cache checks the determinant before inverse computation.
Active try blocks are confined to audited functions ending in `_protected`:

- Viewer window initialization, encoding resource ownership, image presentation,
  run lifecycle, and close operations in `display/tk_backend.py`.
- Score/dictionary file reads, atomic writes, and temporary-file cleanup in
  `TextTwister/storage.py`.

Protected mode is the default and may use narrow exception handlers and finally
for resource cleanup. Corresponding `_unprotected` functions share validation
and normal algorithms but contain no authored try blocks. Their failures propagate
and can leave windows, temporary files, or buffer state unfinished. Context
managers still close files. Commented protected reference patterns accompany
unprotected functions. No raw-memory access is introduced by either mode.

## TextTwister integration

The game's independent behavior, data layout, scoring, dictionary selection,
resource policy and UI contracts live in
[TextTwister/IMPLEMENTATION_SPEC.md](TextTwister/IMPLEMENTATION_SPEC.md).
The game uses the renderer's public draw API and a synchronous Viewer. Keep
renderer math here and game rules there rather than duplicating specifications.

## Time, scheduling, and input

- The renderer, geometry, materials, and rasterizer have no frame-rate clock or
  built-in frame limiter. Draw calls consume the supplied scene state without
  advancing it. `Uniforms.time` (elapsed seconds) and `Uniforms.delta_time`
  (seconds since the last update) belong to the caller and default to zero.
- `Viewer.set_frame_callback(cb)` calls `cb(framebuffer, dt_seconds)` once per
  render tick. The first delta is zero; later deltas use monotonic start-to-start
  time and include rendering, encoding contention, event-loop waits, and stalls.
  Failed or skipped presentation never suppresses callbacks or clamps delta.
- `Viewer(..., tick_ms=0)` is the default uncapped mode. Tk idle redraws get a
  turn before queuing the next immediate timer. An explicit positive `tick_ms`
  requests a minimum start-to-start period; work already spent on the current
  frame is subtracted before scheduling. It is not an extra post-render sleep.
  Timer precision and actual rendering/upload costs still determine throughput.
- Demos advance a shared compensated elapsed-time accumulator using the supplied
  delta, with rates in radians per second. Zero delta freezes animation; long
  deltas consume their full duration. Negative/nonfinite deltas are rejected.
  Animation starts at time zero rather than the process-wide clock epoch.
- Built-in demo shaders receive both elapsed time and delta in uniforms. Custom
  applications should multiply continuous velocities by delta, or evaluate
  animation from explicit elapsed time. There is no physics integration loop.
- `Viewer.key_presses()` returns ordered new key-down edges, including multiple
  taps between callbacks. The projection demo processes them in arrival order,
  so multiple toggles and projection changes retain their meaning at low FPS.
  `key_was_pressed()` remains the existing boolean convenience query, and
  `is_key_down()` reports held state. Input history clears after each callback.
- Diagnostic smoothing uses elapsed seconds rather than a fixed per-frame alpha.
  Presented FPS remains a throughput metric and never controls simulation time.

## Validation

`.github/workflows/quality.yml` enforces lint and basedpyright on
pushes and pull requests. Its test matrix covers Windows/Linux and Python
3.12/3.14. Install the
pinned tools from `requirements-dev.txt` locally so editor and CI diagnostics
agree. Update tool pins deliberately and run all checks when upgrading them.
CI preserves manual formatting; local automatic lint fixes require diff review.
The compact-statement exceptions are intentional; other lint rules remain enabled.

Basedpyright checks `Rasterizer/`, `TextTwister/`, `tests/`, and `benchmarks/` in standard mode
targeting Python 3.12. Missing generic arguments, implicit relative imports, and
unnecessary type-ignore comments are errors. The CI command uses `--warnings`
so new warnings also fail. Standard mode is the typing baseline for this partly
annotated codebase; it does not require complete annotation of every helper.
Do not skip unannotated functions or exclude tests to pass this check. Intentional
invalid-input tests may use a documented, diagnostic-specific ignore on the call
that violates a static contract; production paths should be fixed directly.

Screen vertices are four-float tuples and varyings are float tuples. Specialized
raster kernels initialize scalar slots before branching; only active components
are read, and their gradients are computed once per triangle. Optional results
and file paths must be checked before use. Mesh index arrays must use unsigned
`I` storage; other array formats are rejected. Demos bootstrap the package root
for direct file execution and then use explicit package imports.

Run the full regression and style checks from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m basedpyright --warnings
```

Regression tests include analytical pixel-center values, deterministic random
flat/shaded comparisons, both windings, varying sizes, orthographic interpolation
and clipping, material state, normals, empty geometry, bounds, scene cycles,
and headless viewer ownership/lifecycle checks with actual encoder threads.
Timing regressions compare identical animation at 7/30/60/144 FPS and irregular
schedules, including stalls and zero delta. They also check scheduler cost
compensation, uncapped mode, ordered input, explicit shader time, and latest-frame
selection under a blocked encoder, without wall-clock speed assertions.

Interactive demos need Tk support. Headless tests mock Tk and do not establish
visual quality or validate the platform decoder.

The handwritten source-policy test audits authored parser imports and protected
try-block placement. Game tests cover repeated letters, signature bags, stage
progression, exact deadline boundaries and long stalls, score arithmetic,
persistence corruption/IO failure, both resource modes, and framebuffer screens.
