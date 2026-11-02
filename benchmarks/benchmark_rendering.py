"""Repeatable headless timings; run from the repository root as a module."""

import argparse
import platform
import timeit

from Rasterizer.algebra.math3d import Mat4, Vec3
from Rasterizer.geometry.mesh import make_sphere
from Rasterizer.materials.material import Material
from Rasterizer.materials.shaders import make_gouraud_shader
from Rasterizer.rendering.framebuffer import Framebuffer
from Rasterizer.rendering.rasterizer import rasterize_triangle_shaded
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms
from Rasterizer.scene.scene import Light, Node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--number", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    if args.number <= 0 or args.repeat <= 0:
        parser.error("--number and --repeat must be positive")
    fb = Framebuffer(320, 180)
    mesh = make_sphere()
    renderer = Renderer(320, 180)
    camera = Camera(Vec3(0, 0, 4), Vec3.zero())
    uniforms = Uniforms()
    uniforms.lights = [Light()]
    root = Node(mesh, Material(*make_gouraud_shader()), Mat4.identity())

    def scene():
        fb.clear(0)
        renderer.draw_scene(fb, root, camera, uniforms)

    def indexed():
        return [mesh.triangle(i) for i in range(mesh.triangle_count())]

    def grouped():
        indices = iter(mesh.indices)
        return list(zip(indices, indices, indices))

    cases = [
        ("index methods", indexed),
        ("grouped indices", grouped),
        ("lit sphere", scene),
        ("PNG encoding", fb.to_png_b64),
    ]
    for count in (2, 3, 5, 8):

        def triangle(count=count):
            fb.clear(0)
            rasterize_triangle_shaded(
                fb,
                (0.0, 0.0, 0.5, 1.0),
                (320.0, 0.0, 0.5, 0.5),
                (0.0, 180.0, 0.5, 0.25),
                (0.0,) * count,
                (1.0,) * count,
                (0.5,) * count,
                lambda v: 0xFFFFFF,
            )

        cases.append((f"triangle {count} varyings", triangle))

    print(f"{platform.python_implementation()} {platform.python_version()}, 320x180")
    print("Best milliseconds per call (includes clear for rendering):")
    for name, callback in cases:
        callback()  # Warm caches and produce a representative frame for encoding.
        samples = timeit.repeat(callback, number=args.number, repeat=args.repeat)
        print(f"{name:24s} {min(samples) * 1000 / args.number:9.3f}")


if __name__ == "__main__":
    main()
