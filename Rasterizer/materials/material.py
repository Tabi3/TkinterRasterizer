"""Material: shader pair + per-material render state.

Spec:
    A Material bundles a (vs, fs) shader pair with the render state that is
    constant for a draw (cull mode, depth-write). Per-frame uniforms
    (model, view, lights, camera_pos) are NOT here — they live in Uniforms.

Contract:
    Render flags are mutable and read on each draw. Textures are captured at factory
    time; in-place pixel edits are visible. Rebuild shader callbacks when texture
    dimensions, wrapping, or the data object change. cast_shadow is reserved;
    no shadow pass is implemented.

    Shader pair contract (same as before):
        vs(vertex, u) -> (clip: Vec4, varyings: tuple[float])
        fs(varyings) -> 0xRRGGBB
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .texture import Texture


class Material:
    """Shader pair and render-state bundle for a mesh.

    Materials keep the per-draw call contract surface small: a vertex shader,
    a fragment shader, and a small set of render flags such as culling and depth
    writes. The actual frame data such as model/view matrices and scene lights
    are passed through Uniforms, which keeps the material reusable across many
    objects and draw calls.
    """

    __slots__ = ("cast_shadow", "cull", "depth_write", "fs", "name", "vs")

    @classmethod
    def gouraud(cls, *, ambient: float = 0.15, cull: bool = True) -> Material:
        """Create the built-in vertex-lit material; lights come from Uniforms.

        Spec: IMPLEMENTATION_SPEC.md#api-and-organization.
        """

        from .shaders import make_gouraud_shader

        return cls(*make_gouraud_shader(ambient=ambient), cull=cull, name="gouraud")

    @classmethod
    def textured(cls, texture: Texture, *, lit: bool = False, cull: bool = True) -> Material:
        """Create an unlit or vertex-lit textured material without shader wiring."""

        from .shaders import make_textured_lit_shader, make_textured_shader

        factory = make_textured_lit_shader if lit else make_textured_shader

        return cls(*factory(texture), cull=cull, name="textured_lit" if lit else "textured")

    def __init__(
        self,
        vs: Callable[..., tuple[Any, tuple[float, ...]]],
        fs: Callable[[tuple[float, ...]], int],
        cull: bool = True,
        depth_write: bool = True,
        cast_shadow: bool = True,
        name: str = "material",
    ) -> None:

        if not callable(vs): raise TypeError("vertex shader must be callable")

        if not callable(fs): raise TypeError("fragment shader must be callable")

        if not isinstance(cull, bool): raise TypeError("cull must be boolean")

        if not isinstance(depth_write, bool): raise TypeError("depth_write must be boolean")

        if not isinstance(cast_shadow, bool): raise TypeError("cast_shadow must be boolean")

        self.vs     = vs
        self.fs     = fs
        self.name   = name
        self.cull   = cull
        self.depth_write = depth_write
        self.cast_shadow = cast_shadow
