"""Handwritten option and validated decimal parsing; no parser libraries."""

from collections.abc import Sequence


def option_values(argv: Sequence[str], allowed: set[str]) -> dict[str, str]:

    values: dict[str, str] = {}

    index = 0

    while index < len(argv):
        argument = argv[index]

        fields = argument.split("=", 1)

        name = fields[0]

        if name not in allowed:
            raise ValueError(f"unknown option: {name}")

        if name in values:
            raise ValueError(f"duplicate option: {name}")

        if len(fields) == 2:
            value = fields[1]

        else:
            index += 1

            if index >= len(argv) or argv[index].startswith("--"):
                raise ValueError(f"missing value for {name}")

            value = argv[index]

        if not value:
            raise ValueError(f"missing value for {name}")

        values[name] = value

        index += 1

    return values


def positive_int(text: str) -> int:

    if not text or len(text) > 9 or not all("0" <= c <= "9" for c in text):
        raise ValueError("expected a positive ASCII decimal integer")

    value = int(text)

    if value <= 0:
        raise ValueError("expected a positive integer")

    return value


def parse_resolution(argv: Sequence[str], default: tuple[int, int, int] = (640, 360, 1)) -> tuple[int, int, int]:

    arguments = argv[1:] if argv and not argv[0].startswith("--") else argv

    values = option_values(arguments, {"--res", "--scale"})

    width, height, scale = default

    if "--res" in values:
        fields = values["--res"].split("x")

        if len(fields) != 2:
            raise ValueError("resolution must be WIDTHxHEIGHT")

        width, height = positive_int(fields[0]), positive_int(fields[1])

    if "--scale" in values:
        scale = positive_int(values["--scale"])

    if width <= 0 or height <= 0 or scale <= 0:
        raise ValueError("dimensions and scale must be positive")

    return width, height, scale
