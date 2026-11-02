"""Tests for demo command-line options."""

import pytest

from Rasterizer.cli import parse_resolution


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ([], (640, 360, 1)),
        (["--res", "800x600"], (800, 600, 1)),
        (["--res=800x600", "--scale=2"], (800, 600, 2)),
        (["--scale", "3"], (640, 360, 3)),
    ],
)
def test_parse_resolution(arguments, expected):
    assert parse_resolution(arguments) == expected


def test_parse_resolution_uses_given_default():
    assert parse_resolution([], (320, 200, 2)) == (320, 200, 2)


def test_parse_resolution_rejects_malformed_dimensions():
    with pytest.raises(ValueError):
        parse_resolution(["--res", "800by600"])
