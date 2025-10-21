# ruff: noqa: INP001,S101
"""Smoke tests to ensure repository structure and Python files parse."""

from __future__ import annotations

import compileall
import pathlib


def test_smoke_repo_has_core_files() -> None:
    """Verify key root files and CI directories exist."""
    root = pathlib.Path()
    assert (root / "README.md").exists()
    assert (root / "requirements.txt").exists()
    # Ensure CI-critical folders exist
    assert (root / ".github" / "workflows").exists()


def test_smoke_python_files_parse() -> None:
    """Compile all .py files to catch syntax errors without importing packages."""
    ok = compileall.compile_dir(".", force=False, quiet=1)
    assert ok
