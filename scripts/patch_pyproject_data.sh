#!/usr/bin/env bash
# Add or remove the data = "dist-data" directive in pyproject.toml.
#
# maturin fails if the data directive is present but dist-data/ doesn't exist,
# so we can't commit it permanently. This script patches it in before building
# a bundled wheel, and restores the original afterward.
#
# Usage:
#   scripts/patch_pyproject_data.sh patch    # add directive, save backup
#   scripts/patch_pyproject_data.sh restore  # restore from backup
set -euo pipefail

PYPROJECT="pyproject.toml"
BACKUP="pyproject.toml.bak"

case "${1:-}" in
    patch)
        if grep -q '^data = "dist-data"' "$PYPROJECT"; then
            echo "Already patched, skipping."
            exit 0
        fi
        cp "$PYPROJECT" "$BACKUP"
        python3 -c "
import sys, tomllib

with open('$PYPROJECT', 'rb') as f:
    config = tomllib.load(f)

maturin = config.get('tool', {}).get('maturin', {})
if 'python-packages' not in maturin:
    print('ERROR: [tool.maturin] python-packages not found in $PYPROJECT', file=sys.stderr)
    sys.exit(1)

with open('$PYPROJECT') as f:
    content = f.read()

# Insert data directive after the python-packages line in [tool.maturin]
lines = content.splitlines(True)
patched = False
for i, line in enumerate(lines):
    if line.startswith('python-packages'):
        lines.insert(i + 1, 'data = \"dist-data\"\n')
        patched = True
        break

if not patched:
    print('ERROR: Could not locate python-packages line to insert data directive', file=sys.stderr)
    sys.exit(1)

with open('$PYPROJECT', 'w') as f:
    f.writelines(lines)
"
        echo "Patched $PYPROJECT (backup: $BACKUP)"
        ;;
    restore)
        if [ -f "$BACKUP" ]; then
            mv "$BACKUP" "$PYPROJECT"
            echo "Restored $PYPROJECT from backup"
        else
            echo "No backup found at $BACKUP" >&2
            exit 1
        fi
        ;;
    *)
        echo "Usage: $0 {patch|restore}" >&2
        exit 1
        ;;
esac
