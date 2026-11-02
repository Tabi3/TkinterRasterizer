"""Mesh, vertex, and procedural geometry utilities.

Spec:
    This module defines the mesh data model used by the renderer: vertices,
    triangle indices, bounding boxes, and primitive constructors. The geometry
    layer is intentionally data-oriented and free of frame-state or rendering
    logic so that scene construction remains separate from rasterization.

Contract:
    - Use right-handed Y-up space consistently for all generated primitives.
    - Store triangle indices as flat triplets in a single `array('I')` buffer.
    - Treat normals and UVs as optional per-vertex metadata.
    - Keep mesh construction deterministic so tests can verify winding and face
      orientation without any display pipeline involvement.

Behavior:
    The geometry API produces topological primitives, not screen output. The
    renderer consumes these meshes and performs the actual projection and fill
    work in the raster backend.

Reference:
    See the implementation spec in the repository root for the broader system
    architecture and naming conventions.
"""

from __future__ import annotations

import itertools
import math
from array import array
from collections.abc import Sequence

from ..algebra.math3d import Mat3, Vec2, Vec3, Vec4
from ..data.box_faces import BOX_FACES


# ===========================================================================
# AABB
# ===========================================================================


class AABB:
    """Axis-aligned bounding box in mesh-local space.

    Spec:
        Represents the minimum and maximum extents of a mesh or transformed set of
        points. An empty box uses ``(min=+inf, max=-inf)`` per axis, which makes
        ``union`` and ``intersects`` behave predictably for empty inputs.

    Args:
        min_x, min_y, min_z: lower bounds on each axis.
        max_x, max_y, max_z: upper bounds on each axis.

    Returns:
        A new bounding box with the provided extents.
    """

    __slots__ = ("max_x", "max_y", "max_z", "min_x", "min_y", "min_z")

    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z):

        for name, value in (
            ("min_x", min_x),
            ("min_y", min_y),
            ("min_z", min_z),
            ("max_x", max_x),
            ("max_y", max_y),
            ("max_z", max_z),
        ):
            if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
                raise TypeError(f"{name} must be a real number, got {type(value).__name__}")

        self.min_x = float(min_x)

        self.min_y = float(min_y)

        self.min_z = float(min_z)

        self.max_x = float(max_x)

        self.max_y = float(max_y)

        self.max_z = float(max_z)

    @classmethod
    def empty(cls):
        """Return an empty axis-aligned bounding box.

        Returns:
            An AABB whose bounds are invalid on the lower/upper checks used by
            ``is_empty`` and ``union``.
        """

        inf = math.inf

        return cls(inf, inf, inf, -inf, -inf, -inf)

    @classmethod
    def from_vertices(cls, verts):
        """Compute a bounding box from a sequence of vertices.

        Args:
            verts: Iterable of vertex objects exposing a ``pos`` attribute.

        Returns:
            A bounding box spanning the minimum and maximum positions.
        """

        vertex_list = tuple(verts)

        if not vertex_list:
            return cls.empty()

        xs = (vertex.pos.x for vertex in vertex_list)

        ys = (vertex.pos.y for vertex in vertex_list)

        zs = (vertex.pos.z for vertex in vertex_list)

        return cls(
            min(xs),
            min(ys),
            min(zs),
            max(vertex.pos.x for vertex in vertex_list),
            max(vertex.pos.y for vertex in vertex_list),
            max(vertex.pos.z for vertex in vertex_list),
        )

    def is_empty(self):
        """Return True when the box has no valid extents."""

        return self.min_x > self.max_x or self.min_y > self.max_y or self.min_z > self.max_z

    def center(self):
        """Return the center of the box as a Vec3."""

        return Vec3(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )

    def extents(self):
        """Return the full size of the box along each axis.

        Returns:
            A Vec3 containing (width, height, depth).
        """

        return Vec3(self.max_x - self.min_x, self.max_y - self.min_y, self.max_z - self.min_z)

    def half_extents(self):
        """Return half the box extents."""

        return self.extents() * 0.5

    def union(self, other):
        """Return the smallest AABB containing both boxes."""

        return AABB(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            min(self.min_z, other.min_z),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
            max(self.max_z, other.max_z),
        )

    def expand(self, margin):
        """Expand the box by the provided margin on every side."""

        if not isinstance(margin, (int, float)):
            raise TypeError("margin must be a real number")

        return AABB(
            self.min_x - margin,
            self.min_y - margin,
            self.min_z - margin,
            self.max_x + margin,
            self.max_y + margin,
            self.max_z + margin,
        )

    def intersects(self, other):
        """Return True when two boxes overlap in space."""

        return not (
            self.min_x > other.max_x
            or self.max_x < other.min_x
            or self.min_y > other.max_y
            or self.max_y < other.min_y
            or self.min_z > other.max_z
            or self.max_z < other.min_z
        )

    def contains_point(self, p):
        """Return True when the point lies inside the box volume."""

        if not (hasattr(p, "x") and hasattr(p, "y") and hasattr(p, "z")):
            raise TypeError("point must provide x/y/z components")

        return self.min_x <= p.x <= self.max_x and self.min_y <= p.y <= self.max_y and self.min_z <= p.z <= self.max_z

    def transform(self, mat):
        """Transform the box by a Mat4 and recompute its bounds.

        Args:
            mat: A 4x4 transform matrix, applied to each of the eight corners.

        Returns:
            A new AABB that encloses the transformed cube.

        Reference:
            IMPLEMENTATION_SPEC.md > Core architecture > 1. Math and geometry layer
        """

        corners = [
            Vec4(self.min_x, self.min_y, self.min_z, 1.0),
            Vec4(self.max_x, self.min_y, self.min_z, 1.0),
            Vec4(self.min_x, self.max_y, self.min_z, 1.0),
            Vec4(self.max_x, self.max_y, self.min_z, 1.0),
            Vec4(self.min_x, self.min_y, self.max_z, 1.0),
            Vec4(self.max_x, self.min_y, self.max_z, 1.0),
            Vec4(self.min_x, self.max_y, self.max_z, 1.0),
            Vec4(self.max_x, self.max_y, self.max_z, 1.0),
        ]

        out = AABB.empty()

        for corner in corners:
            transformed = mat @ corner

            out.min_x = min(out.min_x, transformed.x)

            out.min_y = min(out.min_y, transformed.y)

            out.min_z = min(out.min_z, transformed.z)

            out.max_x = max(out.max_x, transformed.x)

            out.max_y = max(out.max_y, transformed.y)

            out.max_z = max(out.max_z, transformed.z)

        return out

    def __repr__(self):

        if self.is_empty():
            return "AABB(empty)"

        return f"AABB(({self.min_x:.3g},{self.min_y:.3g},{self.min_z:.3g}) .. ({self.max_x:.3g},{self.max_y:.3g},{self.max_z:.3g}))"


# ===========================================================================
# Vertex / Mesh
# ===========================================================================


class Vertex:
    """Single mesh vertex.

    Args:
        pos: Position in model space as a Vec3.
        normal: Optional shading normal as Vec3.
        uv: Optional texture coordinate as Vec2.
        color: Optional packed RGB color as an integer or None.

    Returns:
        A vertex object storing the given attributes.
    """

    __slots__ = ("color", "normal", "pos", "uv")

    def __init__(self, pos, normal=None, uv=None, color=None):

        if not (hasattr(pos, "x") and hasattr(pos, "y") and hasattr(pos, "z")):
            raise TypeError("pos must provide x/y/z components")

        if normal is not None and not (hasattr(normal, "x") and hasattr(normal, "y") and hasattr(normal, "z")):
            raise TypeError("normal must provide x/y/z components")

        if uv is not None and not (hasattr(uv, "x") and hasattr(uv, "y")):
            raise TypeError("uv must provide x/y components")

        self.pos = pos

        self.normal = normal

        self.uv = uv

        self.color = color

    def __repr__(self):

        return f"Vertex({self.pos!r})"


class Mesh:
    """Triangle mesh container with index-driven topology.

    Spec:
        The mesh stores vertices and flat triangle indices; each triangle is an
        ordered triplet of vertex indices. The renderer and geometry helpers treat
        the index buffer as the canonical topology.

    Args:
        vertices: Iterable of `Vertex` objects.
        indices: Optional flat triangle index list or array('I').

    Returns:
        A mesh object with an index buffer and computed bounds.
    """

    __slots__ = ("aabb", "indices", "vertices")

    def __init__(
        self,
        vertices: list[Vertex] | tuple[Vertex, ...],
        indices: Sequence[int] | array[int] | None = None,
    ) -> None:

        if not isinstance(vertices, (list, tuple)):
            raise TypeError("vertices must be a sequence of Vertex objects")

        self.vertices: list[Vertex] = list(vertices)

        if indices is None:
            self.indices: array[int] = array("I", range(len(self.vertices)))

        elif isinstance(indices, array):
            if indices.typecode != "I":
                raise ValueError("mesh indices must use array typecode I")

            self.indices = indices

        else:
            self.indices = array("I", indices)

        if len(self.indices) % 3 != 0:
            raise ValueError(f"indices length {len(self.indices)} not divisible by 3")

        if any(i >= len(self.vertices) for i in self.indices):
            raise ValueError("mesh index is outside the vertex buffer")

        self.aabb = AABB.from_vertices(self.vertices)

    def triangle_count(self):
        """Return the number of triangles in the mesh."""

        return len(self.indices) // 3

    def triangle(self, t):
        """Return the integer indices for triangle `t`.

        Args:
            t: Triangle index in the flat index buffer.

        Returns:
            A tuple ``(i0, i1, i2)`` representing a single triangle.
        """

        index = 3 * t

        return self.indices[index], self.indices[index + 1], self.indices[index + 2]

    triangle_indices = triangle

    def triangle_vertices(self, t):
        """Return the three Vertex objects for triangle `t`."""

        i0, i1, i2 = self.triangle(t)

        vertex_list = self.vertices

        return vertex_list[i0], vertex_list[i1], vertex_list[i2]

    def recompute_bounds(self):
        """Refresh the cached bounding box from the current vertices."""

        self.aabb = AABB.from_vertices(self.vertices)

    def edges(self):
        """Return the unique mesh edges as ``(i, j)`` pairs with ``i < j``.

        Returns:
            A list of unique undirected edges suitable for wireframe overlays.
        """

        seen = set()

        out = []

        index_buffer = self.indices

        for triangle_base in range(0, len(index_buffer), 3):
            i0, i1, i2 = (
                index_buffer[triangle_base],
                index_buffer[triangle_base + 1],
                index_buffer[triangle_base + 2],
            )

            for a, b in ((i0, i1), (i1, i2), (i2, i0)):
                if a > b:
                    a, b = b, a

                key = a * 0x100000000 + b

                if key not in seen:
                    seen.add(key)

                    out.append((a, b))

        return out

    def validate_indices(self):
        """Ensure all indices point to a valid vertex in the mesh."""

        vertex_count = len(self.vertices)

        for index in self.indices:
            if index >= vertex_count:
                raise ValueError(f"index {index} out of range for {vertex_count} vertices")

    def transform(self, mat):
        """Return a transformed copy of the mesh.

        Spec:
            Apply the 4x4 transform to all positions and transform normals by the
            inverse-transpose of the upper-left 3x3 block for non-uniform scale.

        Args:
            mat: The transform matrix used to move vertices into the new space.

        Returns:
            A new Mesh with transformed positions and transformed normals.

        Reference:
            IMPLEMENTATION_SPEC.md > Core architecture > 1. Math and geometry layer
        """

        matrix_data = mat.data

        matrix_3x3 = Mat3(
            (
                matrix_data[0],
                matrix_data[1],
                matrix_data[2],
                matrix_data[4],
                matrix_data[5],
                matrix_data[6],
                matrix_data[8],
                matrix_data[9],
                matrix_data[10],
            )
        )

        if matrix_3x3.determinant() == 0.0:
            normal_matrix = matrix_3x3

        else:
            normal_matrix = matrix_3x3.inverse().transpose()

        transformed = []

        for vertex in self.vertices:
            transformed_position = mat @ vertex.pos.to_vec4(1.0)

            new_position = Vec3(transformed_position.x, transformed_position.y, transformed_position.z)

            new_normal = normal_matrix @ vertex.normal if vertex.normal is not None else None

            if new_normal is not None:
                new_normal = new_normal.normalize()

            transformed.append(Vertex(new_position, new_normal, vertex.uv, vertex.color))

        return Mesh(transformed, self.indices)

    def __repr__(self):

        return f"Mesh({len(self.vertices)}v, {self.triangle_count()}t)"


# ===========================================================================
# MeshBuilder — the ergonomic construction API
# ===========================================================================


class MeshBuilder:
    """Ergonomic mesh constructor.

    Spec:
        Accumulate vertices and triangle indices in a simple builder pattern,
        then materialize the final mesh in one pass. This keeps primitive
        constructors readable without exposing low-level `array('I')` details to
        the caller.
    """

    __slots__ = ("indices", "verts")

    def __init__(self):

        self.verts = []

        self.indices = array("I")

    def vertex(self, pos, normal=None, uv=None, color=None):
        """Append a vertex and return its index in the builder buffer."""

        if not (hasattr(pos, "x") and hasattr(pos, "y") and hasattr(pos, "z")):
            raise TypeError("pos must provide x/y/z components")

        if not isinstance(color, (int, type(None))):
            raise TypeError("color must be an integer or None")

        vertex_index = len(self.verts)

        self.verts.append(Vertex(pos, normal, uv, color))

        return vertex_index

    def tri(self, i0, i1, i2):
        """Append one triangle to the builder's index stream."""

        if not (isinstance(i0, int) and isinstance(i1, int) and isinstance(i2, int)):
            raise TypeError("triangle indices must be integers")

        self.indices.extend((i0, i1, i2))

    def quad(self, i0, i1, i2, i3):
        """Split a quad, assumed to be CCW, into two triangles."""

        # Spec: a quad is always two triangles sharing a diagonal. The chosen
        # diagonal matches the winding conventions used by the primitive helpers:
        # the builder adds the first triangle as (i0, i1, i2) and the second as
        # (i0, i2, i3), preserving a consistent outward-facing normal.
        self.tri(i0, i1, i2)

        self.tri(i0, i2, i3)

    def build(self):
        """Return a finalized Mesh from the stored vertices and indices."""

        return Mesh(self.verts, self.indices)


# ===========================================================================
# Primitives
# ===========================================================================


def make_cube(size: float = 2.0, center: Vec3 | None = None, subdivisions: int = 1) -> Mesh:
    """Cube with edge length `size`, centered at `center`.

    subdivisions >= 1 controls tessellation of each face:
        verts = 6 * (subdivisions+1)²
        tris  = 12 * subdivisions²
    Each face still has uniform normal (flat shading).

    Needed for nonlinear projections (panini, ellipsoid) where straight
    3D edges become curves — subdividing reduces the chord-approximation
    error. Perspective and orthographic don't need it (lines project to
    lines exactly).
    """

    if center is None:
        center = Vec3.zero()

    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")

    cx, cy, cz = center.x, center.y, center.z

    n = subdivisions

    # Spec: IMPLEMENTATION_SPEC.md#bulk-pixel-operations.
    # Shared scalar bases avoid constructing temporary direction/corner vectors.
    mb = MeshBuilder()

    for normal_data, right, up, corner in BOX_FACES:
        normal = Vec3(*normal_data)

        ox, oy, oz = (value * size for value in corner)

        grid = [
            [
                mb.vertex(
                    Vec3(
                        cx + ox + size * (u * right[0] + v * up[0]),
                        cy + oy + size * (u * right[1] + v * up[1]),
                        cz + oz + size * (u * right[2] + v * up[2]),
                    ),
                    normal,
                    Vec2(u, v),
                )
                for u in (iu / n for iu in range(n + 1))
            ]
            for v in (iv / n for iv in range(n + 1))
        ]

        for iv, iu in itertools.product(range(n), range(n)):
            a = grid[iv][iu]  # BL

            b = grid[iv][iu + 1]  # BR

            c = grid[iv + 1][iu + 1]  # TR

            d = grid[iv + 1][iu]  # TL

            mb.quad(a, b, c, d)

    return mb.build()


def make_plane(size=10.0, subdivisions=1, center=None):
    """Axis-aligned XZ plane with +Y normal. Centered at `center`.

    `subdivisions` divides the plane into a grid of that many quads per axis,
    giving UV coordinates from (0,0) to (1,1) across the full extent.
    """

    if center is None:
        center = Vec3.zero()

    if subdivisions < 1:
        raise ValueError("subdivisions must be >= 1")

    h = size * 0.5

    cx, cy, cz = center.x, center.y, center.z

    n = subdivisions

    mb = MeshBuilder()

    norm = Vec3.unit_y()

    grid = [
        [
            mb.vertex(
                Vec3(cx - h + size * ix / n, cy, cz - h + size * iz / n),
                norm,
                Vec2(ix / n, iz / n),
            )
            for ix in range(n + 1)
        ]
        for iz in range(n + 1)
    ]

    # CCW from above (+Y): (ix, iz) -> (ix+1, iz) -> (ix+1, iz+1) -> (ix, iz+1)
    # CCW from above (+Y): traverse z-forward last so cross product points +Y
    for iz, ix in itertools.product(range(n), range(n)):
        a = grid[iz][ix]

        b = grid[iz][ix + 1]

        c = grid[iz + 1][ix + 1]

        d = grid[iz + 1][ix]

        mb.quad(a, d, c, b)  # was (a, b, c, d)

    return mb.build()


def make_sphere(radius=1.0, segments=16, rings=12, center=None):
    """UV sphere: segments around Y, rings from north pole to south."""

    if center is None:
        center = Vec3.zero()

    if segments < 3 or rings < 2:
        raise ValueError("sphere needs segments >= 3 and rings >= 2")

    cx, cy, cz = center.x, center.y, center.z

    mb = MeshBuilder()

    # Vertices laid out as (rings+1) rows of (segments+1) columns
    # (the last column duplicates the first for seam-free UVs).
    grid = []

    for r in range(rings + 1):
        theta = math.pi * r / rings  # 0 = north pole

        sin_t = math.sin(theta)

        cos_t = math.cos(theta)

        row = []

        for s in range(segments + 1):
            phi = 2.0 * math.pi * s / segments

            nx = sin_t * math.cos(phi)

            ny = cos_t

            nz = sin_t * math.sin(phi)

            p = Vec3(cx + radius * nx, cy + radius * ny, cz + radius * nz)

            n = Vec3(nx, ny, nz)

            u = s / segments

            v = 1.0 - r / rings

            row.append(mb.vertex(p, n, Vec2(u, v)))

        grid.append(row)

    for r in range(rings):
        for s in range(segments):
            a = grid[r][s]

            b = grid[r][s + 1]

            c = grid[r + 1][s + 1]

            d = grid[r + 1][s]

            # Skip the degenerate triangle at the poles.
            if r != 0:
                mb.tri(a, b, c)

            if r != rings - 1:
                mb.tri(a, c, d)

    return mb.build()


def make_cylinder(radius=1.0, height=2.0, segments=16, center=None):
    """Y-axis cylinder with flat top and bottom caps.

    Winding: CCW viewed from outside (matches the cube / plane tests).
    Cap tri order (asymmetric between top and bottom!) is what makes the
    winding correct; see inline notes.
    """

    if center is None:
        center = Vec3.zero()

    if segments < 3:
        raise ValueError("cylinder needs segments >= 3")

    cx, cy, cz = center.x, center.y, center.z

    hh = height * 0.5

    mb = MeshBuilder()

    # ---- side rings (segments + 1 verts so the UV seam closes) -----------
    bottom_side = []

    top_side = []

    for s in range(segments + 1):
        phi = 2.0 * math.pi * s / segments

        nx, nz = math.cos(phi), math.sin(phi)

        n = Vec3(nx, 0.0, nz)

        u = s / segments

        bottom_side.append(mb.vertex(Vec3(cx + radius * nx, cy - hh, cz + radius * nz), n, Vec2(u, 0.0)))

        top_side.append(mb.vertex(Vec3(cx + radius * nx, cy + hh, cz + radius * nz), n, Vec2(u, 1.0)))

    # ---- side quads ------------------------------------------------------
    # Traversal (a, d, c, b) — not the "natural" (a, b, c, d) — so both
    # sub-triangles come out CCW viewed from outside the cylinder.
    for s in range(segments):
        a = bottom_side[s]

        b = bottom_side[s + 1]

        c = top_side[s + 1]

        d = top_side[s]

        mb.quad(a, d, c, b)

    # ---- cap rings (fresh vertices — cap normals differ from side) ------
    top_cap_center = mb.vertex(Vec3(cx, cy + hh, cz), Vec3.unit_y(), Vec2(0.5, 0.5))

    bot_cap_center = mb.vertex(Vec3(cx, cy - hh, cz), Vec3(0, -1, 0), Vec2(0.5, 0.5))

    top_cap_ring = []

    bot_cap_ring = []

    for s in range(segments + 1):
        phi = 2.0 * math.pi * s / segments

        nx, nz = math.cos(phi), math.sin(phi)

        u = 0.5 + 0.5 * nx

        v = 0.5 + 0.5 * nz

        top_cap_ring.append(
            mb.vertex(
                Vec3(cx + radius * nx, cy + hh, cz + radius * nz),
                Vec3.unit_y(),
                Vec2(u, v),
            )
        )

        bot_cap_ring.append(
            mb.vertex(
                Vec3(cx + radius * nx, cy - hh, cz + radius * nz),
                Vec3(0, -1, 0),
                Vec2(u, v),
            )
        )

    # ---- cap fans --------------------------------------------------------
    # IMPORTANT: the two lines have *opposite* ring ordering.  That's not
    # a typo — a top cap and a bottom cap face opposite directions, so
    # the winding that yields "+Y outward" flips to "-Y outward".
    for s in range(segments):
        # Top cap: CCW viewed from +Y → ring order (s+1) then s.

        mb.tri(top_cap_center, top_cap_ring[s + 1], top_cap_ring[s])

        # Bottom cap: CCW viewed from -Y → ring order s then (s+1).
        mb.tri(bot_cap_center, bot_cap_ring[s], bot_cap_ring[s + 1])

    return mb.build()


def make_cone(radius=1.0, height=2.0, segments=16, center=None):
    """Y-axis cone, apex at +Y, base cap at -Y."""

    if center is None:
        center = Vec3.zero()

    if segments < 3:
        raise ValueError("cone needs segments >= 3")

    cx, cy, cz = center.x, center.y, center.z

    hh = height * 0.5

    mb = MeshBuilder()

    apex_pos = Vec3(cx, cy + hh, cz)

    slope = radius / height

    # ---- side triangles (apex + base ring) -------------------------------
    for s in range(segments):
        phi0 = 2.0 * math.pi * s / segments

        phi1 = 2.0 * math.pi * (s + 1) / segments

        n0 = Vec3(math.cos(phi0), slope, math.sin(phi0)).normalize()

        n1 = Vec3(math.cos(phi1), slope, math.sin(phi1)).normalize()

        n_apex = (n0 + n1).normalize()

        apex_i = mb.vertex(apex_pos, n_apex, Vec2((s + 0.5) / segments, 1.0))

        b0 = mb.vertex(
            Vec3(cx + radius * math.cos(phi0), cy - hh, cz + radius * math.sin(phi0)),
            n0,
            Vec2(s / segments, 0.0),
        )

        b1 = mb.vertex(
            Vec3(cx + radius * math.cos(phi1), cy - hh, cz + radius * math.sin(phi1)),
            n1,
            Vec2((s + 1) / segments, 0.0),
        )

        # (b1, b0, apex) — not (b0, b1, apex) — so the outward normal
        # points away from the axis (verified by the face-outward test).
        mb.tri(b1, b0, apex_i)

    # ---- base cap (fan, facing -Y) ---------------------------------------
    cap_center = mb.vertex(Vec3(cx, cy - hh, cz), Vec3(0, -1, 0), Vec2(0.5, 0.5))

    cap_ring = []

    for s in range(segments + 1):
        phi = 2.0 * math.pi * s / segments

        nx, nz = math.cos(phi), math.sin(phi)

        cap_ring.append(
            mb.vertex(
                Vec3(cx + radius * nx, cy - hh, cz + radius * nz),
                Vec3(0, -1, 0),
                Vec2(0.5 + 0.5 * nx, 0.5 + 0.5 * nz),
            )
        )

    for s in range(segments):
        # Base faces -Y → ring order (s, s+1), not (s+1, s).

        mb.tri(cap_center, cap_ring[s], cap_ring[s + 1])

    return mb.build()


def make_torus(
    major_radius=1.0,
    minor_radius=0.3,
    major_segments=24,
    minor_segments=12,
    center=None,
):
    """Y-axis torus. Ring plane is XZ."""

    if center is None:
        center = Vec3.zero()

    if major_segments < 3 or minor_segments < 3:
        raise ValueError("torus needs major_segments >= 3 and minor_segments >= 3")

    cx, cy, cz = center.x, center.y, center.z

    mb = MeshBuilder()

    grid = []

    for i in range(major_segments + 1):
        phi = 2.0 * math.pi * i / major_segments

        cos_phi, sin_phi = math.cos(phi), math.sin(phi)

        row = []

        for j in range(minor_segments + 1):
            theta = 2.0 * math.pi * j / minor_segments

            cos_t, sin_t = math.cos(theta), math.sin(theta)

            # Position
            r = major_radius + minor_radius * cos_t

            px = r * cos_phi

            py = minor_radius * sin_t

            pz = r * sin_phi

            # Normal
            nx = cos_t * cos_phi

            ny = sin_t

            nz = cos_t * sin_phi

            u = i / major_segments

            v = j / minor_segments

            row.append(mb.vertex(Vec3(cx + px, cy + py, cz + pz), Vec3(nx, ny, nz), Vec2(u, v)))

        grid.append(row)

    for i in range(major_segments):
        for j in range(minor_segments):
            a = grid[i][j]

            b = grid[i + 1][j]

            c = grid[i + 1][j + 1]

            d = grid[i][j + 1]

            mb.quad(a, d, c, b)

    return mb.build()
