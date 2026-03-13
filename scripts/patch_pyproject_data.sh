#!/usr/bin/env bash
# Thin wrapper around patch_pyproject_data.py.
# Keeps the same CLI interface for justfile/CI compatibility.
set -euo pipefail
exec python3 "$(dirname "$0")/patch_pyproject_data.py" "$@"
