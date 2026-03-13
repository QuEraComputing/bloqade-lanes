#!/usr/bin/env bash
# Stage CLI binary, C library, and headers for Python wheel packaging.
# All artifacts go into dist-data/ which maturin includes in the wheel:
#   dist-data/scripts/          -> installed to bin/ (on PATH)
#   dist-data/data/lib/         -> installed to sys.prefix/lib/
#   dist-data/data/include/     -> installed to sys.prefix/include/
#
# Usage:
#   scripts/stage_clib.sh                          # uses target/release
#   scripts/stage_clib.sh target/aarch64-.../release  # cross-compilation
set -euo pipefail

RELEASE_DIR="${1:-target/release}"

# Detect platform-specific binary and library names
case "$(uname -s)" in
    Darwin)
        CLI_BIN="bloqade-bytecode"
        SHARED_LIB="libbloqade_lanes_bytecode.dylib"
        STATIC_LIB="libbloqade_lanes_bytecode.a"
        ;;
    Linux)
        CLI_BIN="bloqade-bytecode"
        SHARED_LIB="libbloqade_lanes_bytecode.so"
        STATIC_LIB="libbloqade_lanes_bytecode.a"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        CLI_BIN="bloqade-bytecode.exe"
        SHARED_LIB="bloqade_lanes_bytecode.dll"
        STATIC_LIB="bloqade_lanes_bytecode.lib"
        ;;
    *)
        echo "Unsupported platform: $(uname -s)"
        exit 1
        ;;
esac

SCRIPTS_DIR="dist-data/scripts"
LIB_DIR="dist-data/data/lib"
INCLUDE_DIR="dist-data/data/include"

# Clean previous staging
rm -rf dist-data
mkdir -p "$SCRIPTS_DIR" "$LIB_DIR" "$INCLUDE_DIR"

# Stage CLI binary
cp "$RELEASE_DIR/$CLI_BIN" "$SCRIPTS_DIR/$CLI_BIN"
chmod +x "$SCRIPTS_DIR/$CLI_BIN"

# Stage C shared library (required)
if [ ! -f "$RELEASE_DIR/$SHARED_LIB" ]; then
    echo "Error: Shared library '$SHARED_LIB' not found in '$RELEASE_DIR'."
    echo "       Please ensure the C library is built before running 'stage-clib'."
    exit 1
fi
cp "$RELEASE_DIR/$SHARED_LIB" "$LIB_DIR/$SHARED_LIB"

# Stage C static library (required)
if [ ! -f "$RELEASE_DIR/$STATIC_LIB" ]; then
    echo "Error: Static library '$STATIC_LIB' not found in '$RELEASE_DIR'."
    echo "       Please ensure the C library is built before running 'stage-clib'."
    exit 1
fi
cp "$RELEASE_DIR/$STATIC_LIB" "$LIB_DIR/$STATIC_LIB"

# Stage header file
cp crates/bloqade-lanes-bytecode-cli/bloqade_lanes_bytecode.h "$INCLUDE_DIR/"

echo "Staged artifacts:"
echo "  Scripts:  $SCRIPTS_DIR/"
ls -la "$SCRIPTS_DIR/"
echo "  C lib:    $LIB_DIR/"
ls -la "$LIB_DIR/"
echo "  Headers:  $INCLUDE_DIR/"
ls -la "$INCLUDE_DIR/"
