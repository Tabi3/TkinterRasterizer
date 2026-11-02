"""Pytest configuration for the Rasterizer project.

Spec:
    Ensure the repository root and the implementation modules are importable when
    tests are executed from a fresh checkout or from the VS Code terminal.

Contract:
    Keep the import path setup minimal and deterministic so the test suite checks
    the project code itself rather than any ambient system installation.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
