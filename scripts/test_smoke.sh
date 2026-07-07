#!/usr/bin/env bash
# Smoke tests for the bytecode CLI (vihaco-backed).
#
# Self-contained: programs are generated inline in the *native* text format
# (const.i64/const.f64 etc.), so this does not depend on the legacy example
# .sst files. Covers assemble/disassemble round-trip plus validation:
# structural checks, arch-dependent capability + address checks, and
# stack-type simulation (--simulate-stack).
#
# Usage: ./scripts/test_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

ARCH="examples/arch/simple.json"   # zone 0, word 0, 5 sites; feed_forward=false, atom_reloading=false
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PASSED=0
FAILED=0

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1"; FAILED=$((FAILED + 1)); }

echo "=== Building CLI ==="
cargo build -p bloqade-lanes-bytecode-cli 2>&1 | tail -1
CLI="target/debug/bloqade-bytecode"

# Write a program to $WORK/$1.sst from stdin.
prog() { cat > "$WORK/$1.sst"; }

expect_pass() {
    local desc="$1"; shift
    if "$CLI" "$@" >/dev/null 2>&1; then pass "$desc"; else fail "$desc (expected pass, exit $?)"; fi
}

expect_fail() {
    local desc="$1" pattern="$2"; shift 2
    local out
    if out=$("$CLI" "$@" 2>&1); then
        fail "$desc (expected failure, got exit 0)"
    elif [ -n "$pattern" ] && ! echo "$out" | grep -qi "$pattern"; then
        fail "$desc (missing '$pattern' in output)"; echo "    output: $out"
    else
        pass "$desc"
    fi
}

# A minimal valid program reused across cases.
prog valid <<'EOF'
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  const_loc 0x0000000001000000
  initial_fill 2
  const.f64 1.5708
  global_rz
  return
}
EOF

echo ""
echo "=== Category A: assemble / disassemble round-trip ==="
"$CLI" assemble "$WORK/valid.sst" -o "$WORK/valid.bin" >/dev/null 2>&1 \
    && "$CLI" disassemble "$WORK/valid.bin" > "$WORK/valid.out.sst" 2>/dev/null \
    && "$CLI" assemble "$WORK/valid.out.sst" -o "$WORK/valid2.bin" >/dev/null 2>&1 \
    && cmp -s "$WORK/valid.bin" "$WORK/valid2.bin" \
    && pass "round-trip assemble->disassemble->assemble is stable" \
    || fail "round-trip"

echo ""
echo "=== Category B: structural validation ==="
expect_pass "valid program" validate "$WORK/valid.sst"

prog no_terminator <<'EOF'
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  initial_fill 1
}
EOF
expect_fail "missing terminator" "return or halt" validate "$WORK/no_terminator.sst"

prog fill_not_first <<'EOF'
version 1.0;
fn @main() {
  global_r
  initial_fill 1
  return
}
EOF
expect_fail "initial_fill not first" "initial_fill" validate "$WORK/fill_not_first.sst"

echo ""
echo "=== Category C: capability validation (--arch) ==="
prog multi_measure <<'EOF'
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  initial_fill 1
  const_zone 0x00000000
  measure 1
  const_zone 0x00000000
  measure 1
  return
}
EOF
expect_fail "multiple measure (feed_forward=false)" "feed_forward" \
    validate "$WORK/multi_measure.sst" --arch "$ARCH"

prog fill_reload <<'EOF'
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  initial_fill 1
  const_loc 0x0000000000000000
  fill 1
  return
}
EOF
expect_fail "fill without atom_reloading" "atom_reloading" \
    validate "$WORK/fill_reload.sst" --arch "$ARCH"

# Note: the feed_forward control-flow rule (rejecting br/cond_br/call) is not
# exercised here because vihaco-cpu's branch mnemonics are parser-deferred and
# cannot be expressed in text; that rule is covered by unit tests in
# isa::validate. See bloqade-lanes#769.

echo ""
echo "=== Category D: address validation (--arch) ==="
expect_pass "valid addresses" validate "$WORK/valid.sst" --arch "$ARCH"

prog bad_zone <<'EOF'
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  initial_fill 1
  const_zone 0x00000005
  measure 1
  return
}
EOF
expect_fail "invalid zone" "invalid zone" validate "$WORK/bad_zone.sst" --arch "$ARCH"

prog bad_site <<'EOF'
version 1.0;
fn @main() {
  const_loc 0x0000000063000000
  initial_fill 1
  return
}
EOF
expect_fail "invalid site" "invalid location" validate "$WORK/bad_site.sst" --arch "$ARCH"

echo ""
echo "=== Category E: stack-type simulation (--simulate-stack) ==="
# Stack-balanced: the two locations are consumed by initial_fill, leaving the
# stack empty at the `halt` terminator (halt pops nothing).
prog typed_ok <<'EOF'
version 1.0;
fn @main() {
  const_loc 0x0000000000000000
  const_loc 0x0000000001000000
  initial_fill 2
  halt
}
EOF
expect_pass "well-typed program" validate "$WORK/typed_ok.sst" --simulate-stack

prog underflow <<'EOF'
version 1.0;
fn @main() {
  pop
  return
}
EOF
expect_fail "stack underflow" "underflow" validate "$WORK/underflow.sst" --simulate-stack

prog mismatch <<'EOF'
version 1.0;
fn @main() {
  const.f64 1.0
  initial_fill 1
  return
}
EOF
expect_fail "type mismatch" "type mismatch" validate "$WORK/mismatch.sst" --simulate-stack

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
