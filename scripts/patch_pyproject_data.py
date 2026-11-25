#!/usr/bin/env python3
"""Add or remove the data = "dist-data" directive in pyproject.toml.

maturin fails if the data directive is present but dist-data/ doesn't exist,
so we can't commit it permanently. This script patches it in before building
a bundled wheel, and restores the original afterward.

Usage:
    python scripts/patch_pyproject_data.py patch    # add directive, save backup
    python scripts/patch_pyproject_data.py restore  # restore from backup
"""

import shutil
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    # tomllib is new in 3.11; tomli is the backport it was adopted from, so the
    # `loads` call below is identical. Keeps this script runnable on the 3.10
    # floor in requires-python, not just the 3.12 in .python-version.
    import tomli as tomllib

PYPROJECT = Path("pyproject.toml")
BACKUP = Path("pyproject.toml.bak")

DIRECTIVE = 'data = "dist-data"'


def patch():
    content = PYPROJECT.read_text()

    if DIRECTIVE in content:
        print("Already patched, skipping.")
        return

    # Validate structure before patching
    config = tomllib.loads(content)
    maturin = config.get("tool", {}).get("maturin", {})
    if "python-packages" not in maturin:
        raise ValueError("[tool.maturin] python-packages not found in pyproject.toml")

    # Insert data directive after the python-packages line
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("python-packages"):
            lines.insert(i + 1, f"{DIRECTIVE}\n")
            break
    else:
        raise RuntimeError(
            "Could not locate python-packages line to insert data directive"
        )

    shutil.copy2(PYPROJECT, BACKUP)
    PYPROJECT.write_text("".join(lines))
    print(f"Patched {PYPROJECT} (backup: {BACKUP})")


def restore():
    if not BACKUP.exists():
        raise FileNotFoundError(f"No backup found at {BACKUP}")

    BACKUP.replace(PYPROJECT)
    print(f"Restored {PYPROJECT} from backup")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("patch", "restore"):
        raise SystemExit(f"Usage: {sys.argv[0]} {{patch|restore}}")

    {"patch": patch, "restore": restore}[sys.argv[1]]()


if __name__ == "__main__":
    main()
