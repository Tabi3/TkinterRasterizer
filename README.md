# Rasterizer

A small Python 3D rasterizer focused on reverse-Z depth, edge-function triangle
fills, and lightweight rendering demos.

## Project layout

- `Rasterizer/algebra/` — vectors, matrices, quaternions, and transforms
- `Rasterizer/geometry/` — mesh data, bounds, and procedural primitives
- `Rasterizer/rendering/` — renderer, clipping, rasterizer, and framebuffer
- `Rasterizer/materials/` — materials, textures, and shader factories
- `Rasterizer/scene/` — scene graph and lights
- `Rasterizer/display/` — Tk interactive viewer
- `Rasterizer/demos/` — runnable rendering examples
- `tests/` — behavior and regression tests
- `IMPLEMENTATION_SPEC.md` — rendering conventions and contributor contract

The package root keeps the small public API:

```python
from Rasterizer import Camera, Object3D, Renderer, Uniforms
```

## Run tests and lint

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The code-quality workflow runs Ruff lint and formatting checks on pushes and
pull requests, plus tests on Windows and Linux with Python 3.12 and 3.14.
Development tool versions are pinned in `requirements-dev.txt`; editor Ruff
integrations should use the same version and the repository's `pyproject.toml`.
CI checks files without rewriting them. To apply formatting locally, run
`python -m ruff format .`; review fixes from `python -m ruff check --fix .`
before committing. Do not disable rules just to pass a check.

## Run demos

```bash
python -m Rasterizer.demos.demo_lit_scene
python -m Rasterizer.demos.demo_textured
python -m Rasterizer.demos.demo_projections
python -m Rasterizer.demos.demo_unshaded
```

The interactive demos need Tk support in the Python installation. The renderer
and test suite do not need a display.

## Rendering behavior

Perspective uses infinite reverse-Z depth. Orthographic uses affine attribute
interpolation and linear reverse-Z depth between `Camera.near` and `Camera.far`
(default 1000). This corrects older orthographic images and changes their depth
values. Panini and ellipsoid filled triangles remain approximations; subdivide
meshes for better curved-projection results.

Scene and instanced draws honor `Material.cull` and `Material.depth_write`.
Direct shaded draws accept the keyword `depth_write=False`. Lighting supports
nonuniform model scale and coincident point lights. Textured shaders observe
in-place pixel edits. `cast_shadow` remains reserved metadata.

Renderer and framebuffer dimensions must match. Empty meshes are valid; invalid
indices and camera ranges raise explicit errors. Nodes support reparenting and
reject cycles. See [IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md) for contracts.

The Tk viewer reuses bounded buffers, drops frames under backpressure, and
reports presented FPS. PNG and PPM fallback uploads are supported. Call
`Viewer.close()` to stop a viewer programmatically.

## Animation and scheduling

The viewer renders without a configured FPS cap by default (`tick_ms=0`). Pass
`tick_ms=16`, for example, to request an approximately 16 ms start-to-start period;
render time is included in that period. Rendering and Tk upload costs still
limit actual throughput.

Frame callbacks receive `(framebuffer, dt)` with delta in seconds, including
stalls; the first delta is zero. All demos animate from accumulated delta rather
than frame counts or a process-wide clock. For custom animation, use elapsed
seconds or `velocity * dt`. Set `Uniforms.time` and `Uniforms.delta_time` explicitly
for time-aware shaders; rendering never advances either field.

The viewer replaces stale pending frames and presents the newest available
result. Dropped display frames do not discard animation time. Use
`Viewer.key_presses()` for ordered discrete input, including repeated taps between
slow frames, and `is_key_down()` with delta-scaled rates for continuous movement.

## Performance checks

```bash
python -m benchmarks.benchmark_rendering --number 10 --repeat 5
```

The benchmark needs no window or third-party packages. It measures triangle
kernels, a lit sphere, index iteration, and PNG encoding separately. See
[PERFORMANCE.md](PERFORMANCE.md) for observed improvements and measurement limits.
