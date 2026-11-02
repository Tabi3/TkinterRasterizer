"""Material: shader pair + per-material render state.

Spec:
    A Material bundles a (vs, fs) shader pair with the render state that is
    constant for a draw (cull mode, depth-write). Per-frame uniforms
    (model, view, lights, camera_pos) are NOT here — they live in Uniforms.

Contract:
    Immutable after construction. Textures are captured in fs at factory
    time; rebuild the Material when textures change, not when lights move.

    Shader pair contract (same as before):
        vs(vertex, u) -> (clip: Vec4, varyings: tuple[float])
        fs(varyings) -> 0xRRGGBB
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Material:
    """Shader pair and render-state bundle for a mesh.

    Materials keep the per-draw call contract surface small: a vertex shader,
    a fragment shader, and a small set of render flags such as culling and depth
    writes. The actual frame data such as model/view matrices and scene lights
    are passed through Uniforms, which keeps the material reusable across many
    objects and draw calls.
    """

    __slots__ = ("cast_shadow", "cull", "depth_write", "fs", "name", "vs")

    def __init__(
        self,
        vs: Callable[..., tuple[Any, tuple[float, ...]]],
        fs: Callable[[tuple[float, ...]], int],
        cull: bool = True,
        depth_write: bool = True,
        cast_shadow: bool = True,
        name: str = "material",
    ) -> None:
        if not callable(vs):
            raise TypeError("vertex shader must be callable")
        if not callable(fs):
            raise TypeError("fragment shader must be callable")
        if not isinstance(cull, bool):
            raise TypeError("cull must be boolean")
        if not isinstance(depth_write, bool):
            raise TypeError("depth_write must be boolean")
        if not isinstance(cast_shadow, bool):
            raise TypeError("cast_shadow must be boolean")
        self.vs = vs
        self.fs = fs
        self.cull = cull
        self.depth_write = depth_write
        self.cast_shadow = cast_shadow
        self.name = name
