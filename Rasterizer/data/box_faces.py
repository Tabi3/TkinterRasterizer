"""Unit-box face bases; normal, right, up, lower-left corner.

Spec: IMPLEMENTATION_SPEC.md#bulk-pixel-operations. Consumers create their own
mutable normal vectors; these scalar tuples never become shared mesh objects.
"""

BOX_FACES = (
    ((1, 0, 0), (0, 0, -1), (0, 1, 0), (0.5, -0.5, 0.5)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0), (-0.5, -0.5, -0.5)),
    ((0, 1, 0), (1, 0, 0), (0, 0, -1), (-0.5, 0.5, 0.5)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1), (-0.5, -0.5, -0.5)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0), (-0.5, -0.5, 0.5)),
    ((0, 0, -1), (-1, 0, 0), (0, 1, 0), (0.5, -0.5, -0.5)),
)
