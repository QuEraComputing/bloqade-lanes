#!/usr/bin/env bash
# Build and run the C-FFI smoke test via CMake.
#
# Usage: scripts/test_c_ffi.sh
set -euo pipefail

ROOT_DIR="$(pwd)"
BUILD_DIR="target/c_smoke_build"

# Build the Rust library (debug mode for testing)
cargo build -p bloqade-lanes-bytecode-cli

LIB_DIR="$ROOT_DIR/target/debug"
INCLUDE_DIR="$ROOT_DIR/crates/bloqade-lanes-bytecode-cli"

# Configure, build, and test
cmake -S tests/c_smoke -B "$BUILD_DIR" \
    -DBLOQADE_LIB_DIR="$LIB_DIR" \
    -DBLOQADE_INCLUDE_DIR="$INCLUDE_DIR"
cmake --build "$BUILD_DIR"
ctest --test-dir "$BUILD_DIR" --output-on-failure
