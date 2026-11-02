# Rasterizer

A Python software rasterizer with reverse-Z depth, programmable materials, four
camera projections, and a Tk viewer. The accompanying game has its own
[TextTwister README](TextTwister/README.md) and
[game implementation spec](TextTwister/IMPLEMENTATION_SPEC.md).

## Small rendering API

```python
from Rasterizer import Camera, Framebuffer, Material, Node, Renderer, Vec3, make_cube

fb = Framebuffer(640, 360)
renderer = Renderer(fb.width, fb.height)
camera = Camera(Vec3(0, 1.5, 5), Vec3.zero())
scene = Node(make_cube(), Material.gouraud())

renderer.render(fb, scene, camera)
fb.to_ppm("cube.ppm")
```

Use `renderer.draw(fb, mesh, camera, color=...)` for a flat mesh, or supply
`material=Material.textured(texture, lit=True)` and an optional transform.
`render()` draws a scene and prepares its background: an integer is a solid
color, a color pair is a gradient, and None preserves existing pixels/depth.
Set `wireframe=True` to draw scene edges.

Lighting and explicit time are in reusable `renderer.uniforms`. Existing
`draw_mesh`, `draw_mesh_shaded`, `draw_scene`, shader factories and former
camera/uniform import paths remain supported. Import Viewer from
`Rasterizer.display.tk_backend`; core imports do not initialize Tk.

## Demos and game

```bash
python -m Rasterizer.demos.demo_unshaded
python -m Rasterizer.demos.demo_textured
python -m Rasterizer.demos.demo_lit_scene
python -m Rasterizer.demos.demo_projections
python -m TextTwister
```

Demos accept `--res 640x360 --scale 2`. The projection example uses 1-4 to select
perspective/orthographic/Panini/ellipsoid and W for wireframe. See the
[demo guide](Rasterizer/demos/README.md) and the separate
[game controls and rules](TextTwister/README.md). Interactive programs require Tk;
tests and image exports can run headlessly.

## Layout and contracts

- `Rasterizer/data/`: font glyphs, unit-box face bases and shared constants.
- `Rasterizer/algebra/`: math values and an explicit owned-output transform writer.
- `Rasterizer/geometry/`, `materials/`, `scene/`: resources and transform hierarchy.
- `Rasterizer/rendering/`: camera, uniforms, dispatch, fill kernels and framebuffer.
- `Rasterizer/display/`, `demos/`: window integration and concise runnable examples.
- `TextTwister/`: independent game rules/controller, static data, UI and persistence.

[IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md) defines the numerical and API
contracts, scratch ownership, bulk pixel operations, and documentation style.
Ordinary matrix operations retain value behavior. `write_trs()` explicitly reuses
an owned matrix, and `Framebuffer.blit()` copies color/depth with typed views.
Bitmap text uses precompiled spans and private cached fill rows.

Perspective uses infinite reverse-Z. Orthographic uses linear reverse depth over
`Camera.near` to `Camera.far` (default 1000). Nonlinear projections approximate
curved surfaces with tessellated triangles. Rendering targets must match renderer
dimensions. Materials honor culling and depth-write flags; shadows are reserved.

Viewer defaults to uncapped scheduling and prints FPS plus frame/render/upscale/
encode/upload timings approximately once per second. Pass `log_timings=False`
to disable printing. `background_encode=False` creates a synchronous viewer with
no worker or mailbox; TextTwister uses this mode. Timing is in elapsed seconds,
and presentation drops never advance or discard simulation time.

## Validation and performance

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m basedpyright --warnings
python -m benchmarks.benchmark_rendering --number 10 --repeat 5
```

CI runs pinned Ruff and basedpyright checks, plus Windows/Linux tests on Python
3.12/3.14. See [PERFORMANCE.md](PERFORMANCE.md) for observed timings and limits.

Application/benchmark parsing is handwritten; general-purpose parser libraries
are excluded, with image decoding permitted. Ordinary control flow avoids try
blocks. Narrow resource handlers are confined to audited `_protected` functions,
with explicit unprotected alternatives. The game README documents their behavior.

Source follows the Framebuffer layout: statements in a block stay adjacent, one
blank line separates computation phases, and two separate methods or sections.
Assignments and declaration rows are grouped and aligned where useful. The
preferred width is 150 characters. See the root
[code style specification](IMPLEMENTATION_SPEC.md#documentation-and-code-style).
