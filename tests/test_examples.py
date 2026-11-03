"""Run every script in examples/, so the examples can't silently break when the API changes.

    python -m pytest tests
    python tests/test_examples.py

03_polymod.py is skipped when data/processed/ has not been built (see README).
"""
import contextlib
import io
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("[0-9]*.py"))
POLYMOD_CSV = ROOT / "data" / "processed" / "polymod_contact_matrix.csv"


def run_example(path):
    """Run one example from the project root, hiding its printed output."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        os.chdir(cwd)


def test_examples():
    assert EXAMPLES, "no examples found"
    for path in EXAMPLES:
        if "polymod" in path.name and not POLYMOD_CSV.exists():
            print(f"skipped {path.name}: run scripts/make_dataset.py first")
            continue
        run_example(path)


if __name__ == "__main__":
    for path in EXAMPLES:
        if "polymod" in path.name and not POLYMOD_CSV.exists():
            print(f"skip {path.name}")
            continue
        run_example(path)
        print(f"ok  {path.name}")
