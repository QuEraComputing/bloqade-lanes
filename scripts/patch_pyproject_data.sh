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
        cp "$PYPROJECT" "$BACKUP"
        # Insert data directive after python-packages line
        python3 -c "
p = open('$PYPROJECT').read()
p = p.replace(
    'python-packages = [\"bloqade\"]',
    'python-packages = [\"bloqade\"]\ndata = \"dist-data\"',
)
open('$PYPROJECT', 'w').write(p)
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
