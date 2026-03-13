#!/usr/bin/env bash
# Verify the committed C header matches what cbindgen generates.
#
# Usage: scripts/check_header.sh
set -euo pipefail

HEADER="crates/bloqade-lanes-bytecode-cli/bloqade_lanes_bytecode.h"

# Rebuild (cbindgen runs in build.rs)
cargo build -p bloqade-lanes-bytecode-cli

# Freshness: committed header must match regenerated output
if ! git diff --exit-code "$HEADER" >/dev/null 2>&1; then
    echo "ERROR: $HEADER is out of date. Run 'cargo build -p bloqade-lanes-bytecode-cli' and commit the updated header."
    git diff "$HEADER"
    exit 1
fi

# Syntax: header must be valid C
cc -fsyntax-only -x c "$HEADER"
echo "Header OK: up-to-date and valid C syntax."
