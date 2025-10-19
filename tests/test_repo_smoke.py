from __future__ import annotations

import pathlib


def test_smoke_repo_has_core_files():
    root = pathlib.Path(".")
    assert (root / "README.md").exists()
    assert (root / "requirements.txt").exists()
    # Ensure CI-critical folders exist
    assert (root / ".github" / "workflows").exists()


def test_smoke_python_files_parse():
    # Compile all .py files just to catch syntax errors without importing heavy libs
    import compileall

    ok = compileall.compile_dir(".", force=False, quiet=1)
    assert ok
