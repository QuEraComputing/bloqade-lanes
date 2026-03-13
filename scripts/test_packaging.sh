#!/usr/bin/env bash
# Integration tests for Python wheel packaging (CLI + C FFI artifacts).
# Usage: ./scripts/test_packaging.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

PASSED=0
FAILED=0

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1"; FAILED=$((FAILED + 1)); }

echo "=== Step 1: Stage CLI + C library artifacts ==="
just stage-clib

[ -x dist-data/scripts/bloqade-bytecode ] && pass "CLI binary staged" || fail "CLI binary missing"
[ -f dist-data/data/lib/libbloqade_lanes_bytecode.a ] && pass "static lib staged" || fail "static lib missing"
[ -f dist-data/data/include/bloqade_lanes_bytecode.h ] && pass "header staged" || fail "header missing"

# Shared lib name is platform-dependent
case "$(uname -s)" in
    Darwin) SHARED_LIB="libbloqade_lanes_bytecode.dylib" ;;
    Linux)  SHARED_LIB="libbloqade_lanes_bytecode.so" ;;
    *)      SHARED_LIB="bloqade_lanes_bytecode.dll" ;;
esac
[ -f "dist-data/data/lib/$SHARED_LIB" ] && pass "shared lib staged ($SHARED_LIB)" || fail "shared lib missing ($SHARED_LIB)"

echo ""
echo "=== Step 2: Build wheel and inspect contents ==="
just build-wheel

# Pick the most recently modified wheel
WHEEL=$(ls -t target/wheels/bloqade_lanes_bytecode-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] && pass "wheel built: $(basename "$WHEEL")" || { fail "no wheel found"; exit 1; }

WHEEL_LISTING=$(unzip -l "$WHEEL")

check_wheel() {
    local pattern="$1"
    local desc="$2"
    if echo "$WHEEL_LISTING" | grep -q "$pattern"; then
        pass "wheel contains $desc"
    else
        fail "wheel missing $desc"
    fi
}

check_wheel "\.data/scripts/bloqade-bytecode" "CLI binary"
check_wheel "\.data/data/lib/.*bloqade_lanes_bytecode" "C library"
check_wheel "\.data/data/include/bloqade_lanes_bytecode\.h" "C header"
check_wheel "bytecode/_clib_path\.py" "_clib_path.py helper"

echo ""
echo "=== Step 3: Install wheel into temp venv and verify Python helpers ==="
# Extract the Python version from the wheel filename (e.g. cp313 -> 3.13)
WHEEL_CPVER=$(basename "$WHEEL" | sed -n 's/.*-cp\([0-9]*\)-.*/\1/p')
WHEEL_PYVER="${WHEEL_CPVER:0:1}.${WHEEL_CPVER:1}"
TEST_VENV=$(mktemp -d)
uv venv "$TEST_VENV/venv" --python "$WHEEL_PYVER" --quiet
uv pip install "$WHEEL" --python "$TEST_VENV/venv/bin/python" --quiet
PYTHON="$TEST_VENV/venv/bin/python"

HAS_CLIB=$("$PYTHON" -c "from bloqade.lanes.bytecode import has_clib; print(has_clib())")
[ "$HAS_CLIB" = "True" ] && pass "has_clib() returns True" || fail "has_clib() returned $HAS_CLIB"

LIB_EXISTS=$("$PYTHON" -c "from bloqade.lanes.bytecode import lib_path; print(lib_path().exists())")
[ "$LIB_EXISTS" = "True" ] && pass "lib_path() file exists" || fail "lib_path() file does not exist"

HEADER_EXISTS=$("$PYTHON" -c "from bloqade.lanes.bytecode import include_dir; print((include_dir() / 'bloqade_lanes_bytecode.h').exists())")
[ "$HEADER_EXISTS" = "True" ] && pass "header file exists at include_dir()" || fail "header file not found at include_dir()"

echo ""
echo "=== Step 4: Verify CLI binary on PATH ==="
if "$TEST_VENV/venv/bin/bloqade-bytecode" --help | grep -q "Lane-move bytecode"; then
    pass "bloqade-bytecode --help works"
else
    fail "bloqade-bytecode --help failed"
fi

rm -rf "$TEST_VENV"

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
