"""Repeatable headless timings; run from the repository root as a module."""

import platform
import sys
import timeit

from Rasterizer.algebra.affine import write_trs
from Rasterizer.algebra.math3d import Mat4, Vec3
from Rasterizer.algebra.transforms import rotate_x, rotate_y, rotate_z, scale, translate
from Rasterizer.cli import option_values, positive_int
from Rasterizer.geometry.mesh import make_sphere
from Rasterizer.materials.material import Material
from Rasterizer.materials.shaders import make_gouraud_shader
from Rasterizer.rendering.drawing2d import draw_text
from Rasterizer.rendering.framebuffer import Framebuffer
from Rasterizer.rendering.rasterizer import rasterize_triangle_shaded
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms
from Rasterizer.scene.scene import Light, Node


def main():

    if sys.argv[1:] == ["--help"]:
        print("Usage: python -m benchmarks.benchmark_rendering [--number N] [--repeat N]")

        return

    options = option_values(sys.argv[1:], {"--number", "--repeat"})

    number = positive_int(options.get("--number", "10"))

    repeat = positive_int(options.get("--repeat", "5"))

    fb = Framebuffer(320, 180)

    mesh = make_sphere()

    renderer = Renderer(320, 180)

    camera = Camera(Vec3(0, 0, 4), Vec3.zero())

    uniforms = Uniforms()

    uniforms.lights = [Light()]

    root = Node(mesh, Material(*make_gouraud_shader()), Mat4.identity())

    def scene():

        renderer.render(fb, root, camera, background=0, uniforms=uniforms)

    def indexed():

        return [mesh.triangle(i) for i in range(mesh.triangle_count())]

    def grouped():

        indices = iter(mesh.indices)

        return list(zip(indices, indices, indices))

    owned_pose = Mat4.identity()

    source_fb = Framebuffer(160, 90)

    def composed_pose():

        return translate(Vec3(1, 2, 3)) @ rotate_y(0.2) @ rotate_x(0.3) @ rotate_z(0.4) @ scale(Vec3(2, 1, 0.5))

    def mutable_pose():

        return write_trs(owned_pose, x=1, y=2, z=3, yaw=0.2, pitch=0.3, roll=0.4, sx=2, sy=1, sz=0.5)

    def text():

        draw_text(fb, "THE QUICK BROWN FOX 12345", 20, 20, 0xFFFFFF, 3)

    def blit():

        fb.blit(source_fb)

    cases = [
        ("index methods", indexed),
        ("grouped indices", grouped),
        ("lit sphere", scene),
        ("PNG encoding", fb.to_png_b64),
        ("composed TRS", composed_pose),
        ("owned TRS", mutable_pose),
        ("bitmap text", text),
        ("framebuffer blit", blit),
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

        samples = timeit.repeat(callback, number=number, repeat=repeat)

        print(f"{name:24s} {min(samples) * 1000 / number:9.3f}")


if __name__ == "__main__":
    main()
