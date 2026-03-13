#!/usr/bin/env bash
# Run a maturin command with the data = "dist-data" directive temporarily
# patched into pyproject.toml.
#
# Usage:
#   scripts/maturin_with_data.sh "build --release"
#   scripts/maturin_with_data.sh "develop --release"
set -euo pipefail

uv run python scripts/patch_pyproject_data.py patch
maturin $@ || { uv run python scripts/patch_pyproject_data.py restore; exit 1; }
uv run python scripts/patch_pyproject_data.py restore
