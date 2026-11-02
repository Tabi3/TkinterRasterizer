# Implementation specification

## Project goals

The rasterizer is a screen-space triangle-fill pipeline. Given a triangle
with clip-space vertices c_i = (x_i, y_i, z_i, w_i), it computes the set of
integer pixels P = {(px, py)} whose centers (px+0.5, py+0.5) lie inside the
triangle in screen space, subject to the reverse-Z depth test:

    z_ndc(px, py) > depth[py, px]   =>  write color, update depth

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
- Clipping: cuts triangle edges at w_clip = near, preserving v by linear
  clip-space lerp (correct because clip space is a linear embedding of ℝ³).
- Rasterizer: fills pixels satisfying edge tests, interpolates v by
  perspective correction, calls fs(v) → 0xRRGGBB.

## Numerical conventions

- Column-vector math: (M · v)_i = Σ_j M[i, j] · v_j. Mat4 storage is
  row-major flat, M[i, j] = data[i·4 + j].
- Clip space: w_clip > 0 for all visible geometry. w_clip = -Z_view for all
  projections. Therefore the near-plane test is w_clip ≥ near.
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
so a visible point satisfies $z_{view} < 0$ and the clip-space denominator is
$w_{clip} = -z_{view} > 0$.

For a view-space point $(x, y, z)$, the projection used by the renderer is:

$$

n_x = \frac{x}{-z} \cdot f, \quad
n_y = \frac{y}{-z} \cdot f,
\quad
w_{clip} = -z,
\quad
z_{clip} = near,

$$

with $f = 1 / \tan(\theta_{fov}/2)$ for perspective projection. The result is
then converted to NDC with:

$$

x_{ndc} = \frac{x_{clip}}{w_{clip}}, \quad
y_{ndc} = \frac{y_{clip}}{w_{clip}}, \quad
z_{ndc} = \frac{z_{clip}}{w_{clip}}.

$$

Because reverse-Z stores the near plane as 1 and the far plane as 0, the depth
comparison is simply:

$$

z_{new} > z_{depth}[px, py].

$$

This is the exact mirror of the standard depth test in forward-Z pipelines and
makes the clear value equal to 0.0 while preserving the rule that closer objects
win.

### 2. Edge-function rasterization

For a triangle with screen-space vertices $p_0, p_1, p_2$, the edge function for
edge $e_i$ is defined as:

$$

E_i(x, y) = a_i x + b_i y + c_i,

$$

where the constants are derived from the two vertices that define the edge. A
point lies inside the triangle if and only if all three edge tests evaluate as
positive under the chosen winding convention. The renderer uses the standard
"top-left" rule to consume boundary pixels consistently when $E_i = 0$.

This yields a scanline loop of the form:

$$

E_i(p + \Delta x) = E_i(p) + a_i,
\quad
E_i(p + \Delta y) = E_i(p) + b_i,

$$

which means the inner loop is dominated by additions rather than expensive
recalculation of the full edge equation at every pixel. In practice this is the
primary reason the rasterizer remains fast in Python: the hot loop avoids a full
multiply-heavy per-pixel setup.

### 3. Perspective-correct interpolation

Let $iw_i = 1 / w_{clip,i}$ and let $E_i(p)$ be the triangle edge function for the
vertex opposite $i$. The barycentric weight for vertex $i$ is:

$$

\lambda_i(p) = \frac{E_i(p)}{A}, \quad A = \sum_i E_i(p).

$$

The correct perspective-correct interpolant for varying data is:

$$

v_k(p) = \frac{\sum_i E_i(p)\, v_i[k] \, iw_i}{\sum_i E_i(p) \, iw_i}.

$$

This is why the rasterizer stores a per-vertex inverse-w value and computes the
weighted varying numerator and denominator in the same screen-space span loop. The
inverse-area factor cancels, so the fragment shader sees the correct value without
a costly division per varying channel.

### 4. Depth interpolation and reverse-Z

The depth buffer stores the NDC depth value; for a triangle, the screen-space
linear depth is:

$$

z_{ndc}(p) = \sum_i \lambda_i(p) z_i.

$$

Because the depth is affine in barycentric coordinates, the rasterizer can update
it incrementally across a scanline using the same edge-function evaluation scheme.
The reverse-Z convention preserves the common rule:

$$

\text{write if } z_{new} > z_{stored},

$$

so the final visible surface is the one with the greatest depth in the reverse-Z
space, which corresponds to the nearest object in view space.

### 5. Lighting and shading model

The shader pipeline takes a view-space normal $\hat{n}$ and a light direction
$\hat{l}$ and evaluates the diffuse term:

$$

d = \max(0, \hat{n} \cdot \hat{l}).

$$

The per-light contribution is then:

$$

\text{shade}_k = ambient_k + d \cdot light_k,

$$

with the color channel being multiplied by the sampled texture value when a
textured shader is active. This makes the lighting model a low-cost linear
function of interpolated surface data and keeps the fragment kernel small enough
for Python-level control flow.

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

Source files are grouped by their job, not under a catch-all math directory:

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
- Ruff is configured in `pyproject.toml`. Run `python -m ruff check .` and
  `python -m ruff format --check .` before changing behavior or imports.

## Performance and allocation

The rasterizer uses incrementally stepped edge and depth values, and specializes
common varying sizes to keep per-pixel work small. The framebuffer uses packed
arrays and slice operations for bulk clears and nearest-neighbor scaling, so
those copies run in C rather than through Python pixel loops.

Prefer scalar locals and built-in operations in measured hot loops. Avoid
creating temporary vectors, matrices, or closures inside per-pixel work. Reuse
private mutable scratch storage only when it cannot leak across calls or change
the public value semantics. Keep Python loops where callbacks or per-pixel
conditions require them. Measure representative scenes before and after each
optimization; preserve image output and test coverage.

## Validation

Run the full regression and style checks from the repository root:

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The interactive demos additionally need a Python build with Tk support.
