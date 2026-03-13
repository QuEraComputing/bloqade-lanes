#!/usr/bin/env bash
# Smoke tests for bytecode validation CLI.
# Runs .sst test files through the CLI and checks exit codes / error output.
# Usage: ./scripts/test_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ARCH="examples/arch/full.json"
PASSED=0
FAILED=0

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1"; FAILED=$((FAILED + 1)); }

# Build CLI
echo "=== Building CLI ==="
cargo build -p bloqade-lanes-bytecode-cli 2>&1 | tail -1
CLI="target/debug/bloqade-bytecode"

# --- Helper: expect success ---
expect_pass() {
    local desc="$1"
    shift
    if "$CLI" "$@" >/dev/null 2>&1; then
        pass "$desc"
    else
        fail "$desc (expected pass, got exit code $?)"
    fi
}

# --- Helper: expect failure with optional error pattern ---
expect_fail() {
    local desc="$1"
    local pattern="$2"
    shift 2
    local stderr_out
    if stderr_out=$("$CLI" "$@" 2>&1); then
        fail "$desc (expected failure, got exit 0)"
    elif [ -n "$pattern" ] && ! echo "$stderr_out" | grep -qi "$pattern"; then
        fail "$desc (missing pattern '$pattern' in output)"
        echo "    output: $stderr_out"
    else
        pass "$desc"
    fi
}

echo ""
echo "=== Category A: Address validation (--arch) ==="

expect_pass "addr_locations" \
    validate examples/programs/valid/addr_locations.sst --arch "$ARCH"

expect_pass "addr_lanes_site_bus" \
    validate examples/programs/valid/addr_lanes_site_bus.sst --arch "$ARCH"

expect_pass "addr_lanes_word_bus" \
    validate examples/programs/valid/addr_lanes_word_bus.sst --arch "$ARCH"

expect_pass "addr_zone" \
    validate examples/programs/valid/addr_zone.sst --arch "$ARCH"

expect_fail "addr_invalid_word" "invalid location" \
    validate examples/programs/invalid/addr_invalid_word.sst --arch "$ARCH"

expect_fail "addr_invalid_site" "invalid location" \
    validate examples/programs/invalid/addr_invalid_site.sst --arch "$ARCH"

expect_fail "addr_invalid_bus" "invalid lane" \
    validate examples/programs/invalid/addr_invalid_bus.sst --arch "$ARCH"

expect_fail "addr_invalid_zone" "invalid zone" \
    validate examples/programs/invalid/addr_invalid_zone.sst --arch "$ARCH"

echo ""
echo "=== Category B: Stack simulation (--simulate-stack) ==="

expect_pass "stack_basic" \
    validate examples/programs/valid/stack_basic.sst --simulate-stack

expect_pass "stack_full_pipeline" \
    validate examples/programs/valid/stack_full_pipeline.sst --simulate-stack --arch "$ARCH"

expect_fail "stack_underflow" "stack underflow" \
    validate examples/programs/invalid/stack_underflow.sst --simulate-stack

expect_fail "stack_type_mismatch_fill" "type mismatch" \
    validate examples/programs/invalid/stack_type_mismatch_fill.sst --simulate-stack

expect_fail "stack_type_mismatch_cz" "type mismatch" \
    validate examples/programs/invalid/stack_type_mismatch_cz.sst --simulate-stack

expect_fail "stack_initial_fill_not_first" "initial_fill" \
    validate examples/programs/invalid/stack_initial_fill_not_first.sst

expect_fail "stack_move_wrong_type" "type mismatch" \
    validate examples/programs/invalid/stack_move_wrong_type.sst --simulate-stack

echo ""
echo "=== Category C: Lane group validation (--simulate-stack --arch) ==="

expect_pass "group_consistent_site_bus" \
    validate examples/programs/valid/group_consistent_site_bus.sst --simulate-stack --arch "$ARCH"

expect_pass "group_consistent_word_bus" \
    validate examples/programs/valid/group_consistent_word_bus.sst --simulate-stack --arch "$ARCH"

expect_pass "group_rectangle" \
    validate examples/programs/valid/group_rectangle.sst --simulate-stack --arch "$ARCH"

expect_fail "group_inconsistent_bus" "inconsistent" \
    validate examples/programs/invalid/group_inconsistent_bus.sst --simulate-stack --arch "$ARCH"

expect_fail "group_inconsistent_dir" "inconsistent" \
    validate examples/programs/invalid/group_inconsistent_dir.sst --simulate-stack --arch "$ARCH"

expect_fail "group_not_rectangle" "AOD constraint" \
    validate examples/programs/invalid/group_not_rectangle.sst --simulate-stack --arch "$ARCH"

expect_fail "group_word_not_in_bus_list" "words_with_site_buses" \
    validate examples/programs/invalid/group_word_not_in_bus_list.sst --simulate-stack --arch "$ARCH"

expect_fail "group_site_not_in_bus_list" "sites_with_word_buses" \
    validate examples/programs/invalid/group_site_not_in_bus_list.sst --simulate-stack --arch "$ARCH"

echo ""
echo "=== Category D: Multi-error collection ==="

expect_fail "kitchen_sink (multiple errors)" "validation error" \
    validate examples/programs/invalid/kitchen_sink.sst --simulate-stack --arch "$ARCH"

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
