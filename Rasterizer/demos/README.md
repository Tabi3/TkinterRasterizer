# Rasterizer demos

Run modules from the repository root, or launch individual files directly:

```bash
python -m Rasterizer.demos.demo_unshaded --res 640x360
python -m Rasterizer.demos.demo_textured --res 640x360 --scale 2
python -m Rasterizer.demos.demo_lit_scene
python -m Rasterizer.demos.demo_projections
```

- Unshaded: `Renderer.draw()` with flat color and a reusable pose.
- Textured: `Material.textured(..., lit=True)` and one scene node.
- Lit scene: `Material.gouraud()` with directional and point lights.
- Projections: keys 1-4 change projection; W toggles wireframe.

Each example initializes resources once and updates its owned transform from
elapsed seconds. `demo_viewer()` shares CLI/window setup; `Renderer.render()`
shares background and scene dispatch. Frame callbacks never advance time by
frame count. Escape closes the viewer. Terminal timing reports are enabled.

The numerical/API contracts are in
[the root implementation spec](../../IMPLEMENTATION_SPEC.md#api-and-organization).
The game has separate instructions in [TextTwister](../../TextTwister/README.md).
