//! Architecture-dependent validation for the vihaco-backed ISA.
//!
//! These checks gate a [`Program`] against an [`ArchSpec`]'s device
//! capabilities. They are **arch-dependent by design**: when no arch spec is
//! supplied ([`validate`] called with `None`), every check here is skipped and
//! an empty error list is returned — an arch-free program is considered
//! structurally fine at this layer.
//!
//! Capability rules (all active only when an arch spec is present):
//!
//! - **`feed_forward` → CPU control flow.** Without mid-circuit classical
//!   feedback the hardware can only run straight-line code, so any nested
//!   [`vihaco_cpu`] branch/call (`br`, `cond_br`, `call`, `call_indirect`) is
//!   rejected. (This rule exists because the [`Cpu`](super::Instruction::Cpu)
//!   variant makes those opcodes representable in a lanes program.)
//! - **`feed_forward` → multiple measurements.** Without feed-forward at most
//!   one `measure` may appear.
//! - **`atom_reloading` → `fill`.** Without atom reloading, refilling atoms
//!   after the initial fill is unsupported.

use std::fmt;

use vihaco_cpu::Instruction as Cpu;

use super::{Instruction, Program};
use crate::arch::types::ArchSpec;

/// An arch-dependent validation failure, tagged with the offending
/// instruction's program counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A CPU control-flow instruction appears but `feed_forward` is disabled.
    ControlFlowRequiresFeedForward {
        pc: usize,
        /// The offending mnemonic (e.g. `"cond_br"`).
        mnemonic: &'static str,
    },
    /// More than one `measure` appears but `feed_forward` is disabled.
    MultipleMeasuresRequireFeedForward { pc: usize },
    /// A `fill` appears but `atom_reloading` is disabled.
    FillRequiresAtomReloading { pc: usize },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::ControlFlowRequiresFeedForward { pc, mnemonic } => write!(
                f,
                "pc {pc}: control-flow instruction '{mnemonic}' requires feed_forward capability"
            ),
            ValidationError::MultipleMeasuresRequireFeedForward { pc } => write!(
                f,
                "pc {pc}: multiple measure instructions require feed_forward capability"
            ),
            ValidationError::FillRequiresAtomReloading { pc } => {
                write!(
                    f,
                    "pc {pc}: fill instruction requires atom_reloading capability"
                )
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// If `cpu` is a control-flow instruction, return its canonical mnemonic.
///
/// `Return` is deliberately excluded: it is a terminator, not a feed-forward
/// branch (and lanes uses its own [`Return`](super::Instruction::Return)
/// rather than the nested CPU one).
fn control_flow_mnemonic(cpu: &Cpu) -> Option<&'static str> {
    match cpu {
        Cpu::Branch(_) => Some("br"),
        Cpu::ConditionalBranch(_, _) => Some("cond_br"),
        Cpu::Call(_, _) => Some("call"),
        Cpu::IndirectCall => Some("call_indirect"),
        _ => None,
    }
}

/// Validate a program's arch-dependent capability constraints.
///
/// When `arch` is `None`, all checks are skipped and an empty list is returned.
/// Otherwise every violation is collected (the validator does not stop at the
/// first error), so callers can report them all at once.
pub fn validate(program: &Program, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
    let Some(arch) = arch else {
        return Vec::new();
    };

    let mut errors = Vec::new();
    let mut measure_count = 0u32;

    for (pc, inst) in program.instructions.iter().enumerate() {
        match inst {
            Instruction::Cpu(cpu) if !arch.feed_forward => {
                if let Some(mnemonic) = control_flow_mnemonic(cpu) {
                    errors.push(ValidationError::ControlFlowRequiresFeedForward { pc, mnemonic });
                }
            }
            Instruction::Measure(_) => {
                measure_count += 1;
                if !arch.feed_forward && measure_count > 1 {
                    errors.push(ValidationError::MultipleMeasuresRequireFeedForward { pc });
                }
            }
            Instruction::Fill(_) if !arch.atom_reloading => {
                errors.push(ValidationError::FillRequiresAtomReloading { pc });
            }
            _ => {}
        }
    }

    errors
}

impl Program {
    /// Validate this program's arch-dependent capability constraints; see
    /// [`validate`]. Passing `None` skips all arch-dependent checks.
    pub fn validate_capabilities(&self, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
        validate(self, arch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::Version;

    /// Minimal arch spec carrying only the capability flags the checks read;
    /// topology is empty (these checks never touch it).
    fn arch(feed_forward: bool, atom_reloading: bool) -> ArchSpec {
        ArchSpec {
            version: Version::new(2, 0),
            words: vec![],
            zones: vec![],
            zone_buses: vec![],
            modes: vec![],
            paths: None,
            feed_forward,
            atom_reloading,
            blockade_radius: None,
        }
    }

    fn program(instructions: Vec<Instruction>) -> Program {
        Program {
            version: Version::new(1, 0),
            instructions,
        }
    }

    #[test]
    fn no_arch_skips_all_checks() {
        // A program full of things that would fail under any real arch.
        let p = program(vec![
            Instruction::Cpu(Cpu::ConditionalBranch(0, 1)),
            Instruction::Fill(1),
            Instruction::Measure(1),
            Instruction::Measure(1),
        ]);
        assert!(p.validate_capabilities(None).is_empty());
    }

    #[test]
    fn branching_rejected_without_feed_forward() {
        let p = program(vec![
            Instruction::Cpu(Cpu::Branch(0)),
            Instruction::Cpu(Cpu::ConditionalBranch(0, 1)),
            Instruction::Cpu(Cpu::Call(1, 0)),
            Instruction::Cpu(Cpu::IndirectCall),
        ]);
        let errors = p.validate_capabilities(Some(&arch(false, false)));
        assert_eq!(
            errors,
            vec![
                ValidationError::ControlFlowRequiresFeedForward {
                    pc: 0,
                    mnemonic: "br"
                },
                ValidationError::ControlFlowRequiresFeedForward {
                    pc: 1,
                    mnemonic: "cond_br"
                },
                ValidationError::ControlFlowRequiresFeedForward {
                    pc: 2,
                    mnemonic: "call"
                },
                ValidationError::ControlFlowRequiresFeedForward {
                    pc: 3,
                    mnemonic: "call_indirect"
                },
            ]
        );
    }

    #[test]
    fn branching_allowed_with_feed_forward() {
        let p = program(vec![
            Instruction::Cpu(Cpu::ConditionalBranch(0, 1)),
            Instruction::Measure(1),
            Instruction::Measure(1),
        ]);
        assert!(p.validate_capabilities(Some(&arch(true, false))).is_empty());
    }

    #[test]
    fn non_control_flow_cpu_ops_are_fine() {
        use vihaco::value::Value;
        let p = program(vec![
            Instruction::Cpu(Cpu::Const(Value::I64(1))),
            Instruction::Cpu(Cpu::Dup),
            Instruction::Cpu(Cpu::Halt),
        ]);
        assert!(
            p.validate_capabilities(Some(&arch(false, false)))
                .is_empty()
        );
    }

    #[test]
    fn single_measure_ok_but_second_rejected_without_feed_forward() {
        let p = program(vec![Instruction::Measure(1), Instruction::Measure(1)]);
        let errors = p.validate_capabilities(Some(&arch(false, false)));
        assert_eq!(
            errors,
            vec![ValidationError::MultipleMeasuresRequireFeedForward { pc: 1 }]
        );
    }

    #[test]
    fn fill_requires_atom_reloading() {
        let p = program(vec![Instruction::Fill(1)]);
        assert_eq!(
            p.validate_capabilities(Some(&arch(false, false))),
            vec![ValidationError::FillRequiresAtomReloading { pc: 0 }]
        );
        // Allowed once the capability is present.
        assert!(p.validate_capabilities(Some(&arch(false, true))).is_empty());
    }

    #[test]
    fn all_violations_collected_together() {
        let p = program(vec![
            Instruction::Cpu(Cpu::Branch(0)),
            Instruction::Fill(1),
            Instruction::Measure(1),
            Instruction::Measure(1),
        ]);
        let errors = p.validate_capabilities(Some(&arch(false, false)));
        assert_eq!(
            errors.len(),
            3,
            "expected one error per violation: {errors:?}"
        );
    }
}
