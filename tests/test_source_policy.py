"""Small handwritten source-policy checks; no AST/token/parser dependency."""

from pathlib import Path


def test_specification_comment_links_resolve():
    """Keep computation preambles connected to existing specification sections."""
    root = Path(__file__).resolve().parents[1]
    references = 0
    for package in ("Rasterizer", "TextTwister"):
        for source in (root / package).rglob("*.py"):
            for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                if "# Spec: " not in line:
                    continue
                target = line.split("# Spec: ", 1)[1].split()[0].rstrip(".")
                if ".md#" not in target:
                    continue
                filename, anchor = target.split("#", 1)
                document = root / filename
                location = f"{source.relative_to(root)}:{number}"
                assert document.is_file(), location
                headings = {
                    heading.lstrip("# ").strip().lower().replace(" ", "-")
                    for heading in document.read_text(encoding="utf-8").splitlines()
                    if heading.startswith("#")
                }
                assert anchor in headings, location
                references += 1
    assert references > 0


def test_authored_production_sources_follow_parser_and_exception_policy():

    root = Path(__file__).resolve().parents[1]
    forbidden = {
        "json",
        "csv",
        "tomllib",
        "tomli",
        "yaml",
        "argparse",
        "re",
        "pickle",
        "ast",
        "configparser",
        "shlex",
    }
    protected_files = {
        root / "Rasterizer" / "display" / "tk_backend.py": {
            "_initialize_window_protected",
            "_upload_protected",
            "_present_protected",
            "_worker_protected",
            "close_protected",
            "run_protected",
        },
        root / "TextTwister" / "storage.py": {
            "read_protected",
            "remove_protected",
            "write_protected",
            "_load_protected",
        },
    }
    for package in ("Rasterizer", "TextTwister", "benchmarks"):
        for path in (root / package).rglob("*.py"):
            functions: list[tuple[int, str]] = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                text = line.lstrip()
                if not text or text.startswith("#"):
                    continue
                location = f"{path.relative_to(root)}:{number}"
                indent = len(line) - len(text)
                while functions and indent <= functions[-1][0]:
                    functions.pop()
                if text.startswith("def "):
                    functions.append((indent, text[4:].split("(", 1)[0]))
                if text == "try:":
                    assert path in protected_files, location
                    assert functions and functions[-1][1] in protected_files[path], location
                if text.startswith("import "):
                    modules = [part.strip().split()[0].split(".")[0] for part in text[7:].split(",")]
                    assert not forbidden.intersection(modules), location
                elif text.startswith("from "):
                    module = text.split()[1].split(".")[0]
                    assert module not in forbidden, location
