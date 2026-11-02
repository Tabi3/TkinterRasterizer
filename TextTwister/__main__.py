"""Launch with python -m TextTwister [--resource-mode protected|unprotected]."""

import sys
from pathlib import Path

from Rasterizer.cli import option_values
from Rasterizer.display.tk_backend import Viewer

from .app import GameApp
from .data.theme import SCREEN_HEIGHT, SCREEN_WIDTH
from .storage import Leaderboard, ResourceMode, read_protected, read_unprotected
from .words import WordIndex


def resource_mode(arguments: list[str]) -> ResourceMode:

    options = option_values(arguments, {"--resource-mode"})

    mode = options.get("--resource-mode", "protected")

    if mode not in ("protected", "unprotected"):
        raise ValueError("resource mode must be protected or unprotected")

    return mode


def main() -> None:

    if sys.argv[1:] == ["--help"]:
        print(__doc__)

        return

    mode = resource_mode(sys.argv[1:])

    directory = Path(__file__).resolve().parent

    reader = read_protected if mode == "protected" else read_unprotected

    data, error = reader(directory / "word_list.txt")

    if data is None:
        print("Cannot load word list: " + error, file=sys.stderr)

        raise SystemExit(1)

    if any(byte > 127 for byte in data):
        print("Word list must contain lowercase ASCII words", file=sys.stderr)

        raise SystemExit(1)

    index = WordIndex(data.decode("ascii").splitlines())

    app = GameApp(index, Leaderboard(directory / "leaderboard.txt", mode))

    viewer = Viewer(
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        title="TextTwister",
        background_encode=False,
        resource_mode=mode,
    )

    def close():

        app.tick()

        app.finish()

        viewer.close()

    def key(event):

        if app.handle_key(event.keysym, event.char):
            close()

    viewer.root.bind("<Escape>", key)

    viewer.root.bind("<KeyPress>", key, add="+")

    viewer.canvas.bind("<Button-1>", lambda event: app.click(event.x, event.y))

    viewer.root.protocol("WM_DELETE_WINDOW", close)

    viewer.set_frame_callback(app.render)

    viewer.run()


if __name__ == "__main__":
    main()
