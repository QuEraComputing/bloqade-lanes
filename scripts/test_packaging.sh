#!/usr/bin/env bash
# Thin wrapper around test_packaging.py.
set -euo pipefail
exec python3 "$(dirname "$0")/test_packaging.py" "$@"
