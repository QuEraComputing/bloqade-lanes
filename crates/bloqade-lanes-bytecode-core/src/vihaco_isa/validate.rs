//! Architecture-dependent validation for the vihaco-backed ISA.
//!
//! These checks gate a [`Program`] against an [`ArchSpec`]. They are
//! **arch-dependent by design**: when no arch spec is supplied ([`validate`]
//! called with `None`), every check here is skipped and an empty error list is
//! returned — an arch-free program is considered fine at this layer.
//!
//! Two families of checks run (both only when an arch spec is present):
//!
//! ## Capability checks
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
//!
//! ## Address checks
//!
//! Every `const_loc` / `const_lane` / `const_zone` operand is decoded and
//! checked against the architecture's topology via
//! [`ArchSpec::check_location`] / [`check_lane`](ArchSpec::check_lane) /
//! [`check_zone`](ArchSpec::check_zone) — invalid zones, words, sites, lanes,
//! and AOD constraints are reported with the arch layer's own message.

use std::fmt;

use vihaco_cpu::Instruction as Cpu;

use super::{Instruction, Program};
use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
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
    /// A `const_loc` operand does not name a valid location in the arch spec.
    InvalidLocation { pc: usize, message: String },
    /// A `const_lane` operand does not name a valid lane in the arch spec.
    InvalidLane { pc: usize, message: String },
    /// A `const_zone` operand does not name a valid zone in the arch spec.
    InvalidZone { pc: usize, message: String },
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
            ValidationError::InvalidLocation { pc, message } => {
                write!(f, "pc {pc}: invalid location: {message}")
            }
            ValidationError::InvalidLane { pc, message } => {
                write!(f, "pc {pc}: invalid lane: {message}")
            }
            ValidationError::InvalidZone { pc, message } => {
                write!(f, "pc {pc}: invalid zone: {message}")
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

/// Validate a program's arch-dependent constraints (capabilities + addresses).
///
/// When `arch` is `None`, all checks are skipped and an empty list is returned.
/// Otherwise every violation is collected in program order (the validator does
/// not stop at the first error), so callers can report them all at once.
pub fn validate(program: &Program, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
    let Some(arch) = arch else {
        return Vec::new();
    };

    let mut errors = Vec::new();
    let mut measure_count = 0u32;

    for (pc, inst) in program.instructions.iter().enumerate() {
        match inst {
            // ---- capability checks ----
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

            // ---- address checks ----
            Instruction::ConstLoc(bits) => {
                if let Some(message) = arch.check_location(&LocationAddr::decode(*bits)) {
                    errors.push(ValidationError::InvalidLocation { pc, message });
                }
            }
            Instruction::ConstLane(bits) => {
                for message in arch.check_lane(&LaneAddr::decode_u64(*bits)) {
                    errors.push(ValidationError::InvalidLane { pc, message });
                }
            }
            Instruction::ConstZone(bits) => {
                if let Some(message) = arch.check_zone(&ZoneAddr::decode(*bits)) {
                    errors.push(ValidationError::InvalidZone { pc, message });
                }
            }

            _ => {}
        }
    }

    errors
}

impl Program {
    /// Validate this program's arch-dependent constraints; see [`validate`].
    /// Passing `None` skips all arch-dependent checks.
    pub fn validate(&self, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
        validate(self, arch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::Version;

    /// A small, valid arch: one word of 5 sites, one zone.
    const SIMPLE_ARCH_JSON: &str = include_str!("../../../../examples/arch/simple.json");

    fn simple_arch() -> ArchSpec {
        ArchSpec::from_json(SIMPLE_ARCH_JSON).expect("examples/arch/simple.json should parse")
    }

    /// Minimal arch carrying only capability flags; topology is empty (the
    /// capability checks never touch it).
    fn caps_arch(feed_forward: bool, atom_reloading: bool) -> ArchSpec {
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

    fn loc(zone_id: u32, word_id: u32, site_id: u32) -> u64 {
        LocationAddr {
            zone_id,
            word_id,
            site_id,
        }
        .encode()
    }

    // ---- capability checks ----

    #[test]
    fn no_arch_skips_all_checks() {
        let p = program(vec![
            Instruction::Cpu(Cpu::ConditionalBranch(0, 1)),
            Instruction::Fill(1),
            Instruction::Measure(1),
            Instruction::Measure(1),
            Instruction::ConstZone(99), // also an invalid address
        ]);
        assert!(p.validate(None).is_empty());
    }

    #[test]
    fn branching_rejected_without_feed_forward() {
        let p = program(vec![
            Instruction::Cpu(Cpu::Branch(0)),
            Instruction::Cpu(Cpu::ConditionalBranch(0, 1)),
            Instruction::Cpu(Cpu::Call(1, 0)),
            Instruction::Cpu(Cpu::IndirectCall),
        ]);
        let errors = p.validate(Some(&caps_arch(false, false)));
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
        assert!(p.validate(Some(&caps_arch(true, false))).is_empty());
    }

    #[test]
    fn non_control_flow_cpu_ops_are_fine() {
        use vihaco::value::Value;
        let p = program(vec![
            Instruction::Cpu(Cpu::Const(Value::I64(1))),
            Instruction::Cpu(Cpu::Dup),
            Instruction::Cpu(Cpu::Halt),
        ]);
        assert!(p.validate(Some(&caps_arch(false, false))).is_empty());
    }

    #[test]
    fn single_measure_ok_but_second_rejected_without_feed_forward() {
        let p = program(vec![Instruction::Measure(1), Instruction::Measure(1)]);
        assert_eq!(
            p.validate(Some(&caps_arch(false, false))),
            vec![ValidationError::MultipleMeasuresRequireFeedForward { pc: 1 }]
        );
    }

    #[test]
    fn fill_requires_atom_reloading() {
        let p = program(vec![Instruction::Fill(1)]);
        assert_eq!(
            p.validate(Some(&caps_arch(false, false))),
            vec![ValidationError::FillRequiresAtomReloading { pc: 0 }]
        );
        assert!(p.validate(Some(&caps_arch(false, true))).is_empty());
    }

    // ---- address checks ----

    #[test]
    fn valid_addresses_pass() {
        let arch = simple_arch();
        // zone 0, word 0, sites 0 and 4 are in range; zone 0 exists.
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::ConstLoc(loc(0, 0, 4)),
            Instruction::ConstZone(ZoneAddr { zone_id: 0 }.encode()),
        ]);
        assert!(p.validate(Some(&arch)).is_empty(), "expected no errors");
    }

    #[test]
    fn invalid_location_rejected() {
        let arch = simple_arch();
        // site 99 is out of range for a 5-site word.
        let p = program(vec![Instruction::ConstLoc(loc(0, 0, 99))]);
        let errors = p.validate(Some(&arch));
        assert!(
            matches!(
                errors.as_slice(),
                [ValidationError::InvalidLocation { pc: 0, .. }]
            ),
            "got {errors:?}"
        );
    }

    #[test]
    fn invalid_zone_rejected() {
        let arch = simple_arch();
        // zone 5 does not exist (only zone 0).
        let p = program(vec![Instruction::ConstZone(
            ZoneAddr { zone_id: 5 }.encode(),
        )]);
        let errors = p.validate(Some(&arch));
        assert!(
            matches!(
                errors.as_slice(),
                [ValidationError::InvalidZone { pc: 0, .. }]
            ),
            "got {errors:?}"
        );
    }

    #[test]
    fn invalid_lane_rejected() {
        let arch = simple_arch();
        // A lane in a nonexistent zone is invalid.
        let bad = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 9,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let p = program(vec![Instruction::ConstLane(bad.encode_u64())]);
        let errors = p.validate(Some(&arch));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidLane { pc: 0, .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn capability_and_address_errors_collected_together() {
        let arch = simple_arch(); // feed_forward = false, atom_reloading = false
        let p = program(vec![
            Instruction::Cpu(Cpu::Branch(0)), // pc 0: control flow
            Instruction::Fill(1),             // pc 1: reloading
            Instruction::ConstZone(ZoneAddr { zone_id: 5 }.encode()), // pc 2: bad zone
        ]);
        let errors = p.validate(Some(&arch));
        assert_eq!(
            errors.len(),
            3,
            "one error per violation, in pc order: {errors:?}"
        );
        assert!(matches!(
            errors[0],
            ValidationError::ControlFlowRequiresFeedForward { pc: 0, .. }
        ));
        assert!(matches!(
            errors[1],
            ValidationError::FillRequiresAtomReloading { pc: 1 }
        ));
        assert!(matches!(
            errors[2],
            ValidationError::InvalidZone { pc: 2, .. }
        ));
    }
}
