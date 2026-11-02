"""Renderer API regression tests for camera projection, NDC draws, and scene output.

The tests cover the public renderer surface and ensure the backend pipeline still
matches the camera and depth conventions described by the implementation spec.
"""

from Rasterizer.algebra.math3d import Mat4, Vec3, Vec4
from Rasterizer.geometry.mesh import Vertex, make_cube
from Rasterizer.materials.shaders import make_gouraud_shader
from Rasterizer.rendering.framebuffer import Framebuffer
from Rasterizer.rendering.renderer import Camera, Renderer, Uniforms
from Rasterizer.scene.scene import Light


def test_camera_project_center():
    cam = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov_y_deg=60, near=0.1)
    view = cam.view_matrix()
    p = cam.project(view @ Vec4(0, 0, 0, 1), 16 / 9)
    assert p.w > 0
    assert abs(p.x / p.w) < 1e-9
    assert abs(p.y / p.w) < 1e-9


def test_camera_project_near_plane():
    cam = Camera(Vec3(0, 0, 5), Vec3(0, 0, 0), fov_y_deg=60, near=0.1)
    view = cam.view_matrix()
    p = cam.project(view @ Vec4(0, 0, 4.9, 1), 16 / 9)
    assert abs(p.z / p.w - 1.0) < 1e-6


def test_camera_view_matrix_reuses_cache_until_pose_changes():
    camera = Camera(Vec3(0, 0, 5), Vec3.zero())
    view = camera.view_matrix()
    key = camera._view_key

    assert camera.view_matrix() is view
    assert camera._view_key is key

    camera.position.x += 1
    assert camera.view_matrix() is not view


def test_camera_all_projections():
    for proj in ("perspective", "orthographic", "panini", "ellipsoid"):
        cam = Camera(
            Vec3(0, 0, 5), Vec3(0, 0, 0), fov_y_deg=90, near=0.1, projection=proj
        )
        view = cam.view_matrix()
        p = cam.project(view @ Vec4(0, 0, 0, 1), 16 / 9)
        assert p.w > 0
        assert abs(p.x / p.w) < 1e-9
        assert abs(p.y / p.w) < 1e-9


def test_uniforms_has_required_slots():
    u = Uniforms()
    for name in ("model", "view", "project", "aspect", "camera_pos", "time"):
        assert hasattr(u, name), f"Uniforms missing {name}"
    # Lighting moved from a single light_dir to a lights list. Accept either
    # representation, but require one of them to exist.
    assert hasattr(u, "lights") or hasattr(u, "light_dir"), (
        "Uniforms has no lighting field"
    )


def test_draw_ndc_triangle():
    fb = Framebuffer(160, 90)
    r = Renderer(160, 90)
    r.clear(fb, 0)
    r.draw_ndc_triangle(
        fb, (0.0, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), 0xFF0000
    )
    count = sum(1 for c in fb.color if c != 0)
    assert count > 100


def test_backface_cull():
    fb = Framebuffer(160, 90)
    r = Renderer(160, 90)
    r.clear(fb, 0)
    r.draw_ndc_triangle(
        fb, (-0.5, -0.5, 0.5), (0.0, 0.5, 0.5), (0.5, -0.5, 0.5), 0xFF0000
    )
    assert sum(1 for c in fb.color if c != 0) == 0


def test_unshaded_draw_mesh_renders_cube():
    fb = Framebuffer(64, 64)
    renderer = Renderer(64, 64)
    camera = Camera(Vec3(0, 0, 4), Vec3.zero())

    renderer.draw_mesh(fb, make_cube(), Mat4.identity(), 0xFFFFFF, camera)

    assert sum(1 for color in fb.color if color) > 100


def test_gouraud_shader_uses_factory_tint_and_ambient():
    vertex_shader, _ = make_gouraud_shader(light_color=(0.25, 0.5, 0.75), ambient=0.0)
    uniforms = Uniforms()
    uniforms.ambient = 0.0
    uniforms.lights = [
        Light(kind="directional", direction=Vec3(0, -1, 0), color=(1, 1, 1))
    ]
    vertex = Vertex(Vec3.zero(), normal=Vec3.unit_y(), color=0xFFFFFF)

    _, varyings = vertex_shader(vertex, uniforms)

    assert varyings == (0.25, 0.5, 0.75)


def test_gouraud_shader_uses_factory_ambient_floor():
    vertex_shader, _ = make_gouraud_shader(ambient=0.4)
    uniforms = Uniforms()
    uniforms.lights = []
    vertex = Vertex(Vec3.zero(), normal=Vec3.unit_y(), color=0xFFFFFF)

    _, varyings = vertex_shader(vertex, uniforms)

    assert varyings == (0.4, 0.4, 0.4)


def test_half_res_renderer_pipeline():
    from Rasterizer.algebra.math3d import Vec3
    from Rasterizer.geometry.mesh import make_cube
    from Rasterizer.materials.material import Material
    from Rasterizer.materials.shaders import make_gouraud_shader
    from Rasterizer.rendering.framebuffer import Framebuffer
    from Rasterizer.rendering.renderer import Camera, Renderer
    from Rasterizer.scene.scene import Node

    render_fb = Framebuffer(160, 90)  # half of 320x180
    display_fb = Framebuffer(320, 180)

    r = Renderer(render_fb.width, render_fb.height)
    cube = make_cube(size=2.0)
    mat = Material(*make_gouraud_shader(), name="t")
    root = Node(mesh=cube, material=mat, name="cube")
    cam = Camera(Vec3(0, 1.5, 5), Vec3(0, 0, 0), fov_y_deg=55, near=0.1)
    u = Uniforms()
    u.lights = []

    r.draw_skybox(render_fb, 0x000000, 0x000000)
    r.draw_scene(render_fb, root, cam, u)

    drawn = sum(1 for c in render_fb.color if c != 0)
    assert drawn > 100, f"half-res cube drew {drawn} px"

    render_fb.upscale_nearest(display_fb)
    upscaled = sum(1 for c in display_fb.color if c != 0)
    # 2x2 blocks: 4x the pixel count, +/- edge pixels
    assert upscaled >= 4 * drawn - 100, f"upscale loss: {upscaled} vs 4x{drawn}"
