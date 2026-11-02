"""Math library regression tests for vectors, matrices, and free functions.

This file verifies the algebraic conventions used by the renderer: row-major
matrices, right-handed coordinates, and the exact transform behavior expected by
camera and geometry code.
"""

import pytest

from Rasterizer.algebra.math3d import (
    Mat3,
    Mat4,
    Vec3,
    Vec4,
    clamp,
    cross,
    dot,
    length,
    length_sq,
    lerp,
    normalize,
)


def test_vec3_arithmetic():

    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)
    assert a + b == Vec3(5, 7, 9)
    assert b - a == Vec3(3, 3, 3)
    assert a * 2 == Vec3(2, 4, 6)
    assert 2 * a == Vec3(2, 4, 6)
    assert -a == Vec3(-1, -2, -3)


def test_vec3_dot_cross():

    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)
    assert a.dot(b) == 32
    assert a.cross(b) == Vec3(-3, 6, -3)
    assert b.cross(a) == Vec3(3, -6, 3)


def test_vec3_length_normalize():

    assert Vec3(3, 0, 4).length() == 5.0
    assert Vec3(3, 0, 4).length_sq() == 25.0
    n = Vec3(3, 0, 4).normalize()
    assert abs(n.length() - 1.0) < 1e-12
    assert Vec3.zero().normalize() == Vec3.zero()


def test_vec3_mul_raises_on_vec3():

    with pytest.raises(TypeError):
        _ = Vec3(1, 2, 3) * Vec3(4, 5, 6)


def test_vec4_perspective_divide():

    assert Vec4(2, 4, 6, 2).to_vec3() == Vec3(1, 2, 3)
    assert Vec4(2, 4, 6, 2).to_xyz() == Vec3(2, 4, 6)


def test_mat4_identity():

    identity = Mat4.identity()
    v = Vec4(1, 2, 3, 1)
    assert identity @ v == v
    assert identity @ identity == identity


def test_mat4_mul_vec4_into_updates_existing_output():

    output = Vec4(10, 10, 10, 10)
    Mat4.identity().mul_vec4_into(1, 2, 3, 1, output)
    assert output == Vec4(1, 2, 3, 1)


def test_mat4_inverse_identity():
    # Well-conditioned: scale (2,3,4) + translate (3,4,5). det = 24.

    matrix = Mat4.from_rows((2, 0, 0, 3), (0, 3, 0, 4), (0, 0, 4, 5), (0, 0, 0, 1))
    inverse = matrix.inverse()
    product = matrix @ inverse
    identity = Mat4.identity()
    for a, b in zip(product.data, identity.data):
        assert abs(a - b) < 1e-12


def test_mat4_translate_inverse():

    translation = Mat4.from_rows((1, 0, 0, 5), (0, 1, 0, 6), (0, 0, 1, 7), (0, 0, 0, 1))
    p = Vec4(1, 1, 1, 1)
    assert translation.inverse() @ (translation @ p) == p


def test_mat4_transpose_involution():

    matrix = Mat4.from_rows((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
    assert matrix.transpose().transpose() == matrix


def test_mat4_transpose_product():

    matrix_a = Mat4.identity()
    matrix_b = Mat4.from_rows((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 1))
    assert (matrix_a @ matrix_b).transpose() == (matrix_b.transpose() @ matrix_a.transpose())


def test_mat4_associativity():

    matrix_a = Mat4.from_rows((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16))
    matrix_b = Mat4.from_rows((2, 0, 0, 0), (0, 2, 0, 0), (0, 0, 2, 0), (0, 0, 0, 1))
    v = Vec4(1, 2, 3, 1)
    assert ((matrix_a @ matrix_b) @ v) == (matrix_a @ (matrix_b @ v))


def test_mat4_indexing():

    identity = Mat4.identity()
    assert identity[0, 0] == 1.0
    assert identity.row(0) == (1.0, 0.0, 0.0, 0.0)
    assert identity.col(3) == (0.0, 0.0, 0.0, 1.0)


def test_free_functions():

    a = Vec3(1, 0, 0)
    b = Vec3(0, 1, 0)
    assert dot(a, b) == 0
    assert cross(a, b) == Vec3(0, 0, 1)
    assert length(a) == 1.0
    assert length_sq(Vec3(3, 4, 0)) == 25
    assert normalize(a) == a
    assert lerp(Vec3(0, 0, 0), Vec3(2, 4, 6), 0.5) == Vec3(1, 2, 3)
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
    assert clamp(1.5, 0, 3) == 1.5


def test_mat3_identity_inverse():

    identity = Mat3.identity()
    m = Mat3((2, 0, 0, 0, 3, 0, 0, 0, 4))
    minv = m.inverse()
    prod = m @ minv
    for a, b in zip(prod.data, identity.data):
        assert abs(a - b) < 1e-12


def test_mat3_determinant():

    m = Mat3((2, 0, 0, 0, 3, 0, 0, 0, 4))
    assert abs(m.determinant() - 24.0) < 1e-12


def test_runtime_asserts_reject_bad_numeric_inputs():

    with pytest.raises(TypeError):
        # Deliberately violate the static contract to exercise runtime validation.

        Vec3("oops", 0.0, 0.0)  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError):
        Vec3(1.0, True, 0.0)
    with pytest.raises(ValueError):
        Mat4((1, 2, 3))
    with pytest.raises(ValueError):
        Mat3((1, 2, 3))
