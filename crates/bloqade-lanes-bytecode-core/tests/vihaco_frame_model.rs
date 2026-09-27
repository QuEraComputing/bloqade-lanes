//! Programs that need vihaco#110's frame model, and fail on vihaco 0.4.1 alone.
//!
//! vihaco#110 ("Update CPU frame model", merged 2026-09-22, after the v0.4.1
//! tag) gives each function a locals region below its operands:
//! `local_count` slots, parameters first, zero-filled at entry. `store` no
//! longer grows the stack, and operand ops cannot consume locals.
//!
//! `LanesMachine` runs that model on 0.4.1 already (#1038): it reserves the
//! slots, floors every operand pop at them, and reads an unwritten one as
//! zero. `simulate_stack` checks against the same frame. So each program here
//! must validate clean and run to `halt` now — and go on doing so when the
//! bump swaps the machine's emulation for vihaco's own frames. That makes
//! these the acceptance test for the port.
//!
//! What the bump still has to do:
//!
//! - Pass each function's `local_count` to the CPU in its `FunctionInfo`
//!   message, on every call and at entry, and delete the emulation in
//!   `isa/machine.rs` (its `TODO(vihaco#110)`s).
//! - Rewrite these programs for vihaco#85's untyped CPU, where
//!   `cpu::cpu.load u64, 0` is spelled `cpu::cpu.load_u64 0`. Until then they
//!   fail to parse — loudly, and not as a frame-model failure.

use bloqade_lanes_bytecode_core::isa::Program;
use bloqade_lanes_bytecode_core::isa::machine::{LanesMachine, Stopped};
use bloqade_lanes_bytecode_core::isa::text::parse_text;
use bloqade_lanes_bytecode_core::isa::validate::{simulate_stack, validate_structure};

/// Wrap a module body in the `sst v1` container.
fn module(body: &str) -> Program {
    parse_text(&format!(
        "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n\
         .text(root):\n{body}.text(root).\n.section(root).\n"
    ))
    .expect("the module should parse")
}

/// Assert a clean validation and a run to `halt`.
fn check(body: &str) {
    let program = module(body);
    let mut errors = validate_structure(&program);
    errors.extend(simulate_stack(&program, None));
    assert_eq!(errors, vec![], "should validate clean on the frame model");
    let run = LanesMachine::new()
        .run(&program, 1_000)
        .unwrap_or_else(|e| panic!("should run on the frame model: {e:#}"));
    assert_eq!(run.stopped, Stopped::Halted);
}

/// A scratch local is reserved and zeroed at entry, so reading one before any
/// `store` is fine — and reads zero: the other arm returns instead of
/// halting. On 0.4.1 the slot does not exist until something writes it, so
/// the `load` fails.
#[test]
fn a_scratch_local_starts_zeroed() {
    check(
        "fn @main() {\n  cpu::cpu.load i64, 0\n  cpu::cpu.const i64, 0\n  \
         cpu::cpu.eq i64\n  cpu::cpu.cond_br @zero, @other\n\
         cpu::cpu.label @zero\n  cpu::cpu.halt\n\
         cpu::cpu.label @other\n  cpu::cpu.ret 0\n}\n",
    );
}

/// A local has its own slot below the operands, so storing to one cannot
/// clobber a value already pushed. On 0.4.1 local 0 *is* the bottom of the
/// frame — the zone — so `cz` gets the `i64` that overwrote it.
#[test]
fn a_scratch_local_does_not_alias_the_operands() {
    check(
        "fn @main() {\n  lanes::lanes.const_zone 0x00000000\n  cpu::cpu.const i64, 5\n  \
         cpu::cpu.store i64, 0\n  lanes::lanes.cz\n  cpu::cpu.halt\n}\n",
    );
}

/// The same past a callee's parameters: local 1 comes after `z`, not on top of
/// the copy of `z` the body pushed.
#[test]
fn a_scratch_local_after_the_parameters_does_not_alias_the_operands() {
    check(
        "fn @main() {\n  lanes::lanes.const_zone 0x00000000\n  cpu::cpu.call 1, helper\n  \
         cpu::cpu.halt\n}\n\n\
         fn @helper(z: u32) {\n  cpu::cpu.load u32, 0\n  cpu::cpu.const i64, 5\n  \
         cpu::cpu.store i64, 1\n  lanes::lanes.cz\n  cpu::cpu.ret 0\n}\n",
    );
}

/// The canonical scratch variable: a counter in `@main`, which has no
/// parameters to hold one. On 0.4.1 the first `store` grows the stack, so the
/// loop looks unbalanced to the validator, and the first `load` fails at run
/// time.
#[test]
fn a_counter_in_a_scratch_local_keeps_the_loop_balanced() {
    check(
        "fn @main() {\n\
           cpu::cpu.label @loop\n\
           cpu::cpu.load i64, 0\n  cpu::cpu.const i64, 3\n  cpu::cpu.lt i64\n  \
           cpu::cpu.cond_br @body, @done\n\
           cpu::cpu.label @body\n\
           cpu::cpu.load i64, 0\n  cpu::cpu.const i64, 1\n  cpu::cpu.add i64\n  \
           cpu::cpu.store i64, 0\n  cpu::cpu.br @loop\n\
           cpu::cpu.label @done\n\
           cpu::cpu.halt\n}\n",
    );
}
