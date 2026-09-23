//! Programs that fail on vihaco 0.4.1 and must pass on any later release.
//!
//! vihaco#110 ("Update CPU frame model", merged 2026-09-22, after the v0.4.1
//! tag) gives each function a locals region below its operands:
//! `local_count` slots, parameters first, zero-filled at entry. `store` no
//! longer grows the stack, and operand ops cannot consume locals.
//!
//! Each test states what that model makes true, and a probe of the linked
//! vihaco decides what is asserted — see [`HAS_FRAME_MODEL`]:
//!
//! - **Without the frame model** (vihaco 0.4.1): today's failure, exactly —
//!   the validation errors and the run-time error. Strict, so a change in
//!   behaviour on the pinned version is noticed rather than absorbed.
//! - **With it**: the program validates clean and runs to `halt`.
//!
//! Nothing needs flipping at the bump: the expectation follows the vihaco
//! that is linked.
//! What the bump does need before these pass:
//!
//! - `resolve` computing `local_count` as `max(arity, every load/store index
//!   + 1)`. It is 0 today, and #110's `call` rejects a count below the arity.
//! - The machine passing that count on every call and at entry.
//! - `simulate_stack` seeding each frame with that many locals, and flooring
//!   pops at `base + local_count` (#1038).
//! - These programs rewritten for vihaco#85's untyped CPU, where
//!   `cpu::cpu.load u64, 0` is spelled `cpu::cpu.load_u64 0`. Until then they
//!   fail to parse — loudly, and not as the failure pinned below.

use std::sync::LazyLock;

use bloqade_lanes_bytecode_core::isa::Program;
use bloqade_lanes_bytecode_core::isa::machine::{LanesMachine, Stopped};
use bloqade_lanes_bytecode_core::isa::text::parse_text;
use bloqade_lanes_bytecode_core::isa::validate::{
    ValidationError, simulate_stack, tag, validate_structure,
};

/// Whether the linked vihaco has vihaco#110's frame model.
///
/// Probed, not read off a version number: a version would assume every
/// release after 0.4.1 contains #110, and a patch cut from a maintenance
/// branch need not. The probe `store`s past the top of the frame and then
/// `pop`s. On 0.4.1 the `store` grows the stack to reach its slot, so the
/// `pop` finds the value; under the frame model it writes a reserved local,
/// and the operands are empty. Only the old model runs it to `halt`.
static HAS_FRAME_MODEL: LazyLock<bool> = LazyLock::new(|| {
    let probe = module(
        "fn @main() {\n  cpu::cpu.const u64, 7\n  cpu::cpu.store u64, 1\n  \
         lanes::lanes.pop\n  cpu::cpu.halt\n}\n",
    );
    !matches!(
        LanesMachine::new().run(&probe, 100),
        Ok(run) if run.stopped == Stopped::Halted
    )
});

/// What a program does on vihaco 0.4.1.
struct Today {
    /// Every error the structural and stack passes report, in order.
    validation: Vec<ValidationError>,
    /// A fragment of the error the run stops with.
    run_error: &'static str,
}

/// Wrap a module body in the `sst v1` container.
fn module(body: &str) -> Program {
    parse_text(&format!(
        "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n\
         .text(root):\n{body}.text(root).\n.section(root).\n"
    ))
    .expect("the module should parse")
}

/// Assert `today` on vihaco 0.4.1, and a clean validation and run after it.
fn check(body: &str, today: Today) {
    let program = module(body);
    let mut errors = validate_structure(&program);
    errors.extend(simulate_stack(&program, None));
    let run = LanesMachine::new().run(&program, 1_000);

    if *HAS_FRAME_MODEL {
        assert_eq!(errors, vec![], "should validate clean on the frame model");
        let run = run.unwrap_or_else(|e| panic!("should run on the frame model: {e:#}"));
        assert_eq!(run.stopped, Stopped::Halted);
    } else {
        assert_eq!(
            errors, today.validation,
            "validation on vihaco 0.4.1 changed"
        );
        match run {
            Ok(run) => panic!(
                "expected the vihaco 0.4.1 run to fail with {:?}, but it stopped as {:?}",
                today.run_error, run.stopped
            ),
            Err(e) => assert!(
                format!("{e:#}").contains(today.run_error),
                "the vihaco 0.4.1 run failed differently: {e:#}"
            ),
        }
    }
}

/// A scratch local is reserved and zeroed at entry, so reading one before any
/// `store` is fine. On 0.4.1 the slot does not exist until something writes
/// it — which the validator cannot see.
#[test]
fn a_scratch_local_starts_zeroed() {
    check(
        "fn @main() {\n  cpu::cpu.load u64, 0\n  lanes::lanes.pop\n  cpu::cpu.halt\n}\n",
        Today {
            validation: vec![],
            run_error: "local index out of bounds",
        },
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
        Today {
            validation: vec![ValidationError::TypeMismatch {
                pc: 4,
                expected: tag::ZONE,
                got: tag::INT,
            }],
            run_error: "expected a zone address (u32), got I64(5)",
        },
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
        Today {
            validation: vec![ValidationError::TypeMismatch {
                pc: 9,
                expected: tag::ZONE,
                got: tag::INT,
            }],
            run_error: "expected a zone address (u32), got I64(5)",
        },
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
        Today {
            validation: vec![ValidationError::StackDepthMismatch {
                pc: 1,
                expected: 0,
                got: 1,
            }],
            run_error: "local index out of bounds",
        },
    );
}
