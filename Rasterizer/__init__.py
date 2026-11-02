"""Small public API. Spec: IMPLEMENTATION_SPEC.md#api-and-organization.

Import Camera, Framebuffer, Material, Mesh, Node and Renderer here. Algebra and
resources are also re-exported for concise examples. Import Viewer explicitly
from Rasterizer.display; headless imports do not initialize Tk.
"""

from .algebra.math3d import Mat4, Vec2, Vec3, Vec4
from .cli import parse_resolution
from .geometry.mesh import Mesh, make_cube
from .materials.material import Material
from .materials.texture import Texture
from .rendering.camera import Camera
from .rendering.framebuffer import Framebuffer
from .rendering.renderer import Object3D, Renderer
from .rendering.uniforms import Uniforms
from .scene.scene import Light, Node


__all__ = [
    "Camera",
    "Framebuffer",
    "Light",
    "Mat4",
    "Material",
    "Mesh",
    "Node",
    "Object3D",
    "Renderer",
    "Texture",
    "Uniforms",
    "Vec2",
    "Vec3",
    "Vec4",
    "make_cube",
    "parse_resolution",
]

__version__ = "0.1.0"
