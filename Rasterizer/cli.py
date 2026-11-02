"""Command-line parsing shared by the library and demo entry points."""

from collections.abc import Sequence


def parse_resolution(
    argv: Sequence[str],
    default: tuple[int, int, int] = (640, 360, 1),
) -> tuple[int, int, int]:
    """Parse ``--res WxH`` and ``--scale N`` into width, height, and scale."""
    width, height, scale = default
    for index, argument in enumerate(argv):
        if argument == "--res" and index + 1 < len(argv):
            width, height = map(int, argv[index + 1].split("x"))
        elif argument.startswith("--res="):
            width, height = map(int, argument[6:].split("x"))
        elif argument == "--scale" and index + 1 < len(argv):
            scale = int(argv[index + 1])
        elif argument.startswith("--scale="):
            scale = int(argument[8:])
    return width, height, scale
