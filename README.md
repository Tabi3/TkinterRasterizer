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
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

## Run demos

```bash
python -m Rasterizer.demos.demo_lit_scene
python -m Rasterizer.demos.demo_textured
python -m Rasterizer.demos.demo_projections
python -m Rasterizer.demos.demo_unshaded
```

The interactive demos need Tk support in the Python installation. The renderer
and test suite do not need a display.
