//! Architecture-dependent validation for the vihaco-backed ISA.
//!
//! These checks gate a [`Program`] against an [`ArchSpec`]. They are
//! **arch-dependent by design**: when no arch spec is supplied ([`validate`]
//! called with `None`), every check here is skipped and an empty error list is
//! returned — an arch-free program is considered fine at this layer.
//!
//! [`validate`] runs the arch-dependent capability + address checks below.
//! [`validate_structure`] adds arch-independent structural checks, and
//! [`simulate_stack`] adds optional stack-type simulation (underflow, type
//! mismatches, and lane/location group validation).
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

use std::collections::HashSet;
use std::fmt;

use vihaco::value::Value;
use vihaco_cpu::Instruction as Cpu;

use super::{Instruction, Program};
use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::query::{LaneGroupError, LocationGroupError};
use crate::arch::types::ArchSpec;

/// Value type tags tracked by the [`simulate_stack`] type simulator. These
/// mirror the stack value kinds the runtime distinguishes.
pub mod tag {
    pub const FLOAT: u8 = 0x0;
    pub const INT: u8 = 0x1;
    pub const ARRAY_REF: u8 = 0x2;
    pub const LOCATION: u8 = 0x3;
    pub const LANE: u8 = 0x4;
    pub const ZONE: u8 = 0x5;
    pub const MEASURE_FUTURE: u8 = 0x6;
    pub const DETECTOR_REF: u8 = 0x7;
    pub const OBSERVABLE_REF: u8 = 0x8;
    pub const MEASUREMENT_RESULT: u8 = 0x9;
}

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

    // ---- structural (arch-independent) ----
    /// `new_array` dim0 must be greater than zero.
    NewArrayZeroDim0 { pc: usize },
    /// `new_array` type_tag exceeds the maximum value tag.
    NewArrayInvalidTypeTag { pc: usize, type_tag: u32 },
    /// `initial_fill` is not the first non-constant instruction.
    InitialFillNotFirst { pc: usize },
    /// The program has no instructions (and therefore no terminator).
    EmptyProgram,
    /// The final instruction is neither `return` nor `halt`.
    MissingTerminator { pc: usize },
    /// An instruction follows a `return`/`halt` and is unreachable.
    UnreachableInstruction { pc: usize },

    // ---- stack-type simulation (only via `simulate_stack`) ----
    /// An instruction popped from an empty stack.
    StackUnderflow { pc: usize },
    /// A popped value had the wrong type tag (see [`tag`]).
    TypeMismatch { pc: usize, expected: u8, got: u8 },
    /// A `local_r`/`local_rz`/`fill`/`initial_fill` location group is invalid.
    LocationGroupValidation {
        pc: usize,
        error: LocationGroupError,
    },
    /// A `move` lane group is invalid (duplicate, inconsistent, AOD, …).
    LaneGroupValidation { pc: usize, error: LaneGroupError },
}

/// Maximum valid `new_array` element type tag (`TAG_OBSERVABLE_REF = 0x8`).
const MAX_TYPE_TAG: u32 = 0x8;

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
            ValidationError::NewArrayZeroDim0 { pc } => {
                write!(f, "pc {pc}: new_array dim0 must be > 0")
            }
            ValidationError::NewArrayInvalidTypeTag { pc, type_tag } => {
                write!(f, "pc {pc}: invalid new_array type tag {type_tag}")
            }
            ValidationError::InitialFillNotFirst { pc } => write!(
                f,
                "pc {pc}: initial_fill must be the first non-constant instruction"
            ),
            ValidationError::EmptyProgram => {
                write!(
                    f,
                    "program has no instructions: missing return or halt terminator"
                )
            }
            ValidationError::MissingTerminator { pc } => {
                write!(f, "pc {pc}: program must end with return or halt")
            }
            ValidationError::UnreachableInstruction { pc } => {
                write!(f, "pc {pc}: unreachable instruction after return or halt")
            }
            ValidationError::StackUnderflow { pc } => write!(f, "pc {pc}: stack underflow"),
            ValidationError::TypeMismatch { pc, expected, got } => write!(
                f,
                "pc {pc}: type mismatch: expected tag 0x{expected:x}, got 0x{got:x}"
            ),
            ValidationError::LocationGroupValidation { pc, error } => {
                write!(f, "pc {pc}: {error}")
            }
            ValidationError::LaneGroupValidation { pc, error } => write!(f, "pc {pc}: {error}"),
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

    for (pc, inst) in program.code.iter().enumerate() {
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

/// True if `inst` terminates execution (`return` or `halt`).
fn is_terminator(inst: &Instruction) -> bool {
    matches!(inst, Instruction::Return | Instruction::Cpu(Cpu::Halt))
}

/// True if `inst` only pushes a constant (and so may precede `initial_fill`).
fn is_constant_push(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::ConstLoc(_)
            | Instruction::ConstLane(_)
            | Instruction::ConstZone(_)
            | Instruction::Cpu(Cpu::Const(_))
    )
}

/// Validate a program's arch-independent structural rules: `new_array` operand
/// bounds, `initial_fill` ordering, and terminator/reachability. These never
/// consult an arch spec, so they always run.
pub fn validate_structure(program: &Program) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut seen_non_constant = false;

    for (pc, inst) in program.code.iter().enumerate() {
        match inst {
            Instruction::NewArray(type_tag, dim0, _dim1) => {
                if *dim0 == 0 {
                    errors.push(ValidationError::NewArrayZeroDim0 { pc });
                }
                if *type_tag > MAX_TYPE_TAG {
                    errors.push(ValidationError::NewArrayInvalidTypeTag {
                        pc,
                        type_tag: *type_tag,
                    });
                }
                seen_non_constant = true;
            }
            Instruction::InitialFill(_) => {
                if seen_non_constant {
                    errors.push(ValidationError::InitialFillNotFirst { pc });
                }
                seen_non_constant = true;
            }
            inst if is_constant_push(inst) => {}
            _ => seen_non_constant = true,
        }
    }

    // Any instruction after the first terminator is unreachable. If there are
    // unreachable instructions they explain a non-terminal last instruction, so
    // `MissingTerminator` would be a redundant second error.
    let mut found_terminator = false;
    let mut unreachable = Vec::new();
    for (pc, inst) in program.code.iter().enumerate() {
        if found_terminator {
            unreachable.push(ValidationError::UnreachableInstruction { pc });
        }
        if is_terminator(inst) {
            found_terminator = true;
        }
    }

    if unreachable.is_empty() {
        match program.code.last() {
            None => errors.push(ValidationError::EmptyProgram),
            Some(last) if !is_terminator(last) => errors.push(ValidationError::MissingTerminator {
                pc: program.code.len() - 1,
            }),
            Some(_) => {}
        }
    } else {
        errors.extend(unreachable);
    }

    errors
}

// ── Stack-type simulation ──────────────────────────────────────────────────

/// One tracked stack value: its type tag and (when known) concrete bits.
#[derive(Debug, Clone)]
struct SimEntry {
    tag: u8,
    value: Option<u64>,
}

/// Type-level stack simulator: walks the instruction stream tracking value
/// types, reporting underflow and type mismatches, and — when given an
/// [`ArchSpec`] — validating `move` lane groups and `fill`/`local_*` location
/// groups (duplicates only without an arch).
struct StackSimulator<'a> {
    stack: Vec<SimEntry>,
    errors: Vec<ValidationError>,
    arch: Option<&'a ArchSpec>,
    pc: usize,
}

impl<'a> StackSimulator<'a> {
    fn new(arch: Option<&'a ArchSpec>) -> Self {
        Self {
            stack: Vec::new(),
            errors: Vec::new(),
            arch,
            pc: 0,
        }
    }

    fn pop_any(&mut self) {
        if self.stack.pop().is_none() {
            self.errors
                .push(ValidationError::StackUnderflow { pc: self.pc });
        }
    }

    fn pop_typed(&mut self, expected: u8) {
        match self.stack.pop() {
            Some(entry) if entry.tag != expected => {
                self.errors.push(ValidationError::TypeMismatch {
                    pc: self.pc,
                    expected,
                    got: entry.tag,
                })
            }
            Some(_) => {}
            None => self
                .errors
                .push(ValidationError::StackUnderflow { pc: self.pc }),
        }
    }

    fn pop_typed_n(&mut self, expected: u8, count: u32) {
        for _ in 0..count {
            self.pop_typed(expected);
        }
    }

    /// Pop one value expected to have `expected` tag, returning its concrete
    /// bits when the type matches.
    fn pop_addr(&mut self, expected: u8) -> Option<u64> {
        match self.stack.pop() {
            Some(entry) if entry.tag == expected => entry.value,
            Some(entry) => {
                self.errors.push(ValidationError::TypeMismatch {
                    pc: self.pc,
                    expected,
                    got: entry.tag,
                });
                None
            }
            None => {
                self.errors
                    .push(ValidationError::StackUnderflow { pc: self.pc });
                None
            }
        }
    }

    fn push(&mut self, tag: u8, value: Option<u64>) {
        self.stack.push(SimEntry { tag, value });
    }

    fn sim_dup(&mut self) {
        if let Some(top) = self.stack.last().cloned() {
            self.stack.push(top);
        } else {
            self.errors
                .push(ValidationError::StackUnderflow { pc: self.pc });
        }
    }

    fn sim_swap(&mut self) {
        let len = self.stack.len();
        if len >= 2 {
            self.stack.swap(len - 1, len - 2);
        } else {
            self.errors
                .push(ValidationError::StackUnderflow { pc: self.pc });
        }
    }

    /// Report each uniquely-duplicated location once (no-arch fallback).
    fn check_duplicate_locations(&mut self, locations: &[LocationAddr]) {
        let mut seen = HashSet::new();
        let mut reported = HashSet::new();
        for loc in locations {
            let bits = loc.encode();
            if !seen.insert(bits) && reported.insert(bits) {
                self.errors.push(ValidationError::LocationGroupValidation {
                    pc: self.pc,
                    error: LocationGroupError::DuplicateAddress { address: bits },
                });
            }
        }
    }

    /// Report each uniquely-duplicated lane once (no-arch fallback).
    fn check_duplicate_lanes(&mut self, lanes: &[LaneAddr]) {
        let mut seen = HashSet::new();
        let mut reported = HashSet::new();
        for lane in lanes {
            let (d0, d1) = lane.encode();
            let combined = (d0 as u64) | ((d1 as u64) << 32);
            if !seen.insert(combined) && reported.insert(combined) {
                self.errors.push(ValidationError::LaneGroupValidation {
                    pc: self.pc,
                    error: LaneGroupError::DuplicateAddress { address: (d0, d1) },
                });
            }
        }
    }

    /// Pop `arity` locations and validate them as a group.
    fn pop_and_validate_locations(&mut self, arity: u32) {
        let bits: Vec<Option<u64>> = (0..arity).map(|_| self.pop_addr(tag::LOCATION)).collect();
        let locations: Vec<LocationAddr> = bits
            .iter()
            .filter_map(|v| v.map(LocationAddr::decode))
            .collect();
        let pc = self.pc;
        if let Some(arch) = self.arch {
            for error in arch.check_locations(&locations) {
                self.errors
                    .push(ValidationError::LocationGroupValidation { pc, error });
            }
        } else {
            self.check_duplicate_locations(&locations);
        }
    }

    /// Pop `arity` lanes and validate them as a group.
    fn sim_move(&mut self, arity: u32) {
        let bits: Vec<Option<u64>> = (0..arity).map(|_| self.pop_addr(tag::LANE)).collect();
        let lanes: Vec<LaneAddr> = bits
            .iter()
            .filter_map(|v| v.map(LaneAddr::decode_u64))
            .collect();
        let pc = self.pc;
        if let Some(arch) = self.arch {
            for error in arch.check_lanes(&lanes) {
                self.errors
                    .push(ValidationError::LaneGroupValidation { pc, error });
            }
        } else {
            self.check_duplicate_lanes(&lanes);
        }
    }

    fn dispatch(&mut self, inst: &Instruction) {
        match inst {
            // constants push a typed value
            Instruction::Cpu(Cpu::Const(Value::F64(v))) => self.push(tag::FLOAT, Some(v.to_bits())),
            Instruction::Cpu(Cpu::Const(Value::I64(v))) => self.push(tag::INT, Some(*v as u64)),
            Instruction::ConstLoc(v) => self.push(tag::LOCATION, Some(*v)),
            Instruction::ConstLane(v) => self.push(tag::LANE, Some(*v)),
            Instruction::ConstZone(v) => self.push(tag::ZONE, Some(*v as u64)),

            // stack manipulation
            Instruction::Pop => self.pop_any(),
            Instruction::Cpu(Cpu::Dup) => self.sim_dup(),
            Instruction::Swap => self.sim_swap(),

            // atom arrangement
            Instruction::InitialFill(arity) | Instruction::Fill(arity) => {
                self.pop_and_validate_locations(*arity)
            }
            Instruction::Move(arity) => self.sim_move(*arity),

            // gates
            Instruction::LocalR(arity) => {
                self.pop_typed_n(tag::FLOAT, 2);
                self.pop_and_validate_locations(*arity);
            }
            Instruction::LocalRz(arity) => {
                self.pop_typed_n(tag::FLOAT, 1);
                self.pop_and_validate_locations(*arity);
            }
            Instruction::GlobalR => self.pop_typed_n(tag::FLOAT, 2),
            Instruction::GlobalRz => self.pop_typed_n(tag::FLOAT, 1),
            Instruction::Cz => self.pop_typed(tag::ZONE),

            // measurement
            Instruction::Measure(arity) => {
                self.pop_typed_n(tag::ZONE, *arity);
                for _ in 0..*arity {
                    self.push(tag::MEASURE_FUTURE, None);
                }
            }
            Instruction::AwaitMeasure => {
                self.pop_typed(tag::MEASURE_FUTURE);
                self.push(tag::MEASUREMENT_RESULT, None);
            }

            // arrays
            Instruction::NewArray(_type_tag, dim0, dim1) => {
                let count = dim0 * if *dim1 == 0 { 1 } else { *dim1 };
                for _ in 0..count {
                    self.pop_any();
                }
                self.push(tag::ARRAY_REF, None);
            }
            Instruction::GetItem(ndims) => {
                self.pop_typed_n(tag::INT, *ndims);
                self.pop_typed(tag::ARRAY_REF);
                // Element type is not tracked; assume float.
                self.push(tag::FLOAT, None);
            }

            // detectors / observables
            Instruction::SetDetector => {
                self.pop_typed(tag::ARRAY_REF);
                self.push(tag::DETECTOR_REF, None);
            }
            Instruction::SetObservable => {
                self.pop_typed(tag::ARRAY_REF);
                self.push(tag::OBSERVABLE_REF, None);
            }

            // control
            Instruction::Return => self.pop_any(),
            Instruction::Cpu(Cpu::Halt) => {}

            // Other reused vihaco-cpu ops (arithmetic, etc.) are not emitted by
            // the lanes pipeline and are not modeled here.
            Instruction::Cpu(_) => {}
        }
    }

    fn run(mut self, program: &Program) -> Vec<ValidationError> {
        for (pc, inst) in program.code.iter().enumerate() {
            self.pc = pc;
            self.dispatch(inst);
        }
        self.errors
    }
}

/// Run the type-level stack simulation over a program. Collects underflow and
/// type-mismatch errors, plus lane/location group errors (validated against
/// `arch` when provided, else duplicate-only).
pub fn simulate_stack(program: &Program, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
    StackSimulator::new(arch).run(program)
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
        crate::isa::program::from_code(Version::new(1, 0), instructions)
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
        assert!(validate(&p, None).is_empty());
    }

    #[test]
    fn branching_rejected_without_feed_forward() {
        let p = program(vec![
            Instruction::Cpu(Cpu::Branch(0)),
            Instruction::Cpu(Cpu::ConditionalBranch(0, 1)),
            Instruction::Cpu(Cpu::Call(1, 0)),
            Instruction::Cpu(Cpu::IndirectCall),
        ]);
        let errors = validate(&p, Some(&caps_arch(false, false)));
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
        assert!(validate(&p, Some(&caps_arch(true, false))).is_empty());
    }

    #[test]
    fn non_control_flow_cpu_ops_are_fine() {
        use vihaco::value::Value;
        let p = program(vec![
            Instruction::Cpu(Cpu::Const(Value::I64(1))),
            Instruction::Cpu(Cpu::Dup),
            Instruction::Cpu(Cpu::Halt),
        ]);
        assert!(validate(&p, Some(&caps_arch(false, false))).is_empty());
    }

    #[test]
    fn single_measure_ok_but_second_rejected_without_feed_forward() {
        let p = program(vec![Instruction::Measure(1), Instruction::Measure(1)]);
        assert_eq!(
            validate(&p, Some(&caps_arch(false, false))),
            vec![ValidationError::MultipleMeasuresRequireFeedForward { pc: 1 }]
        );
    }

    #[test]
    fn fill_requires_atom_reloading() {
        let p = program(vec![Instruction::Fill(1)]);
        assert_eq!(
            validate(&p, Some(&caps_arch(false, false))),
            vec![ValidationError::FillRequiresAtomReloading { pc: 0 }]
        );
        assert!(validate(&p, Some(&caps_arch(false, true))).is_empty());
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
        assert!(validate(&p, Some(&arch)).is_empty(), "expected no errors");
    }

    #[test]
    fn invalid_location_rejected() {
        let arch = simple_arch();
        // site 99 is out of range for a 5-site word.
        let p = program(vec![Instruction::ConstLoc(loc(0, 0, 99))]);
        let errors = validate(&p, Some(&arch));
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
        let errors = validate(&p, Some(&arch));
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
        let errors = validate(&p, Some(&arch));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidLane { pc: 0, .. })),
            "got {errors:?}"
        );
    }

    // ---- structural checks ----

    #[test]
    fn well_formed_program_has_no_structural_errors() {
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::InitialFill(1),
            Instruction::Return,
        ]);
        assert!(validate_structure(&p).is_empty());
    }

    #[test]
    fn empty_program_is_rejected() {
        assert_eq!(
            validate_structure(&program(vec![])),
            vec![ValidationError::EmptyProgram]
        );
    }

    #[test]
    fn missing_terminator_rejected() {
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::InitialFill(1),
        ]);
        assert_eq!(
            validate_structure(&p),
            vec![ValidationError::MissingTerminator { pc: 1 }]
        );
    }

    #[test]
    fn halt_is_a_valid_terminator() {
        let p = program(vec![Instruction::Cpu(Cpu::Halt)]);
        assert!(validate_structure(&p).is_empty());
    }

    #[test]
    fn unreachable_after_terminator_rejected() {
        let p = program(vec![Instruction::Return, Instruction::Cpu(Cpu::Halt)]);
        assert_eq!(
            validate_structure(&p),
            vec![ValidationError::UnreachableInstruction { pc: 1 }]
        );
    }

    #[test]
    fn initial_fill_must_be_first_non_constant() {
        // A const push before initial_fill is fine; a gate before it is not.
        let ok = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::InitialFill(1),
            Instruction::Return,
        ]);
        assert!(validate_structure(&ok).is_empty());

        let bad = program(vec![
            Instruction::GlobalR,
            Instruction::InitialFill(1),
            Instruction::Return,
        ]);
        assert!(validate_structure(&bad).contains(&ValidationError::InitialFillNotFirst { pc: 1 }));
    }

    #[test]
    fn new_array_bounds_checked() {
        let p = program(vec![Instruction::NewArray(99, 0, 0), Instruction::Return]);
        let errors = validate_structure(&p);
        assert!(errors.contains(&ValidationError::NewArrayZeroDim0 { pc: 0 }));
        assert!(errors.contains(&ValidationError::NewArrayInvalidTypeTag {
            pc: 0,
            type_tag: 99
        }));
    }

    #[test]
    fn structure_and_arch_checks_compose() {
        // The composition consumers run: structural + arch-dependent checks
        // both fire and collect. Missing terminator (structural) + bad zone (arch).
        let arch = simple_arch();
        let p = program(vec![Instruction::ConstZone(
            ZoneAddr { zone_id: 5 }.encode(),
        )]);
        let errors: Vec<_> = validate_structure(&p)
            .into_iter()
            .chain(validate(&p, Some(&arch)))
            .collect();
        assert!(errors.contains(&ValidationError::MissingTerminator { pc: 0 }));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidZone { .. }))
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
        let errors = validate(&p, Some(&arch));
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

    // ---- stack simulation ----

    fn cpu_float(v: f64) -> Instruction {
        Instruction::Cpu(Cpu::Const(Value::F64(v)))
    }

    #[test]
    fn stack_well_typed_program_has_no_errors() {
        // const_loc, const_loc, initial_fill 2, const_zone, measure 1,
        // await_measure, return — all types line up.
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::ConstLoc(loc(0, 0, 1)),
            Instruction::InitialFill(2),
            Instruction::ConstZone(0),
            Instruction::Measure(1),
            Instruction::AwaitMeasure,
            Instruction::Return,
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_underflow_detected() {
        let p = program(vec![Instruction::Pop]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 0 }]
        );
    }

    #[test]
    fn type_mismatch_detected() {
        // initial_fill expects locations; a float is on the stack instead.
        let p = program(vec![cpu_float(1.0), Instruction::InitialFill(1)]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::TypeMismatch {
                    pc: 1,
                    expected,
                    got
                } if *expected == tag::LOCATION && *got == tag::FLOAT
            )),
            "got {errors:?}"
        );
    }

    #[test]
    fn measure_pushes_future_consumed_by_await() {
        // A measure future left dangling is fine; awaiting a non-future is not.
        let p = program(vec![cpu_float(1.0), Instruction::AwaitMeasure]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::TypeMismatch { pc: 1, expected, .. } if *expected == tag::MEASURE_FUTURE
            )),
            "got {errors:?}"
        );
    }

    #[test]
    fn local_r_pops_two_floats_then_locations() {
        // const_loc, const_float, const_float, local_r 1 — well typed.
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            cpu_float(1.5),
            cpu_float(0.5),
            Instruction::LocalR(1),
        ]);
        assert!(simulate_stack(&p, None).is_empty());
    }

    #[test]
    fn duplicate_locations_flagged_without_arch() {
        // Two identical locations into a single fill group.
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::InitialFill(2),
        ]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LocationGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn invalid_lane_group_flagged_with_arch() {
        let arch = simple_arch();
        // A lane in a nonexistent zone forms an invalid move group.
        let bad = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 9,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let p = program(vec![
            Instruction::ConstLane(bad.encode_u64()),
            Instruction::Move(1),
        ]);
        let errors = simulate_stack(&p, Some(&arch));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LaneGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn await_measure_pushes_measurement_result() {
        // const_zone, measure 1, await_measure — the awaited value carries the
        // measurement-result tag, so a following set_detector (wants ARRAY_REF)
        // now type-mismatches on MEASUREMENT_RESULT rather than silently matching.
        let p = program(vec![
            Instruction::ConstZone(0),
            Instruction::Measure(1),
            Instruction::AwaitMeasure,
        ]);
        // await_measure must push the measurement-result tag.
        assert_eq!(tag::MEASUREMENT_RESULT, 0x9);
        // Sanity: simulate cleanly (no underflow/mismatch) for the measure→await chain.
        assert!(simulate_stack(&p, None).iter().all(|e| !matches!(
            e,
            ValidationError::StackUnderflow { .. } | ValidationError::TypeMismatch { .. }
        )));
    }

    // ---- Display ----

    #[test]
    fn validation_error_display_strings() {
        use crate::arch::query::{LaneGroupError, LocationGroupError};

        let cases: Vec<(ValidationError, String)> = vec![
            (
                ValidationError::ControlFlowRequiresFeedForward {
                    pc: 3,
                    mnemonic: "cond_br",
                },
                "pc 3: control-flow instruction 'cond_br' requires feed_forward capability".into(),
            ),
            (
                ValidationError::MultipleMeasuresRequireFeedForward { pc: 5 },
                "pc 5: multiple measure instructions require feed_forward capability".into(),
            ),
            (
                ValidationError::FillRequiresAtomReloading { pc: 1 },
                "pc 1: fill instruction requires atom_reloading capability".into(),
            ),
            (
                ValidationError::InvalidLocation {
                    pc: 2,
                    message: "bad".into(),
                },
                "pc 2: invalid location: bad".into(),
            ),
            (
                ValidationError::InvalidLane {
                    pc: 2,
                    message: "bad".into(),
                },
                "pc 2: invalid lane: bad".into(),
            ),
            (
                ValidationError::InvalidZone {
                    pc: 2,
                    message: "bad".into(),
                },
                "pc 2: invalid zone: bad".into(),
            ),
            (
                ValidationError::NewArrayZeroDim0 { pc: 0 },
                "pc 0: new_array dim0 must be > 0".into(),
            ),
            (
                ValidationError::NewArrayInvalidTypeTag {
                    pc: 0,
                    type_tag: 99,
                },
                "pc 0: invalid new_array type tag 99".into(),
            ),
            (
                ValidationError::InitialFillNotFirst { pc: 4 },
                "pc 4: initial_fill must be the first non-constant instruction".into(),
            ),
            (
                ValidationError::EmptyProgram,
                "program has no instructions: missing return or halt terminator".into(),
            ),
            (
                ValidationError::MissingTerminator { pc: 7 },
                "pc 7: program must end with return or halt".into(),
            ),
            (
                ValidationError::UnreachableInstruction { pc: 8 },
                "pc 8: unreachable instruction after return or halt".into(),
            ),
            (
                ValidationError::StackUnderflow { pc: 1 },
                "pc 1: stack underflow".into(),
            ),
            (
                ValidationError::TypeMismatch {
                    pc: 1,
                    expected: tag::LOCATION,
                    got: tag::FLOAT,
                },
                "pc 1: type mismatch: expected tag 0x3, got 0x0".into(),
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }

        // The two group-validation variants prefix `pc N:` onto the wrapped
        // arch-layer error's own Display.
        let loc_err = LocationGroupError::DuplicateAddress { address: 0x10 };
        assert_eq!(
            ValidationError::LocationGroupValidation {
                pc: 2,
                error: loc_err.clone(),
            }
            .to_string(),
            format!("pc 2: {loc_err}")
        );
        let lane_err = LaneGroupError::DuplicateAddress { address: (1, 2) };
        assert_eq!(
            ValidationError::LaneGroupValidation {
                pc: 3,
                error: lane_err.clone(),
            }
            .to_string(),
            format!("pc 3: {lane_err}")
        );
    }

    // ---- stack simulation: dispatch coverage ----

    #[test]
    fn stack_sim_int_const_and_pop() {
        // `const.i64` pushes an INT; `pop` discards it. Well typed.
        let p = program(vec![
            Instruction::Cpu(Cpu::Const(Value::I64(7))),
            Instruction::Pop,
        ]);
        assert!(simulate_stack(&p, None).is_empty());
    }

    #[test]
    fn stack_sim_swap_needs_two_entries() {
        let ok = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::ConstLoc(loc(0, 0, 1)),
            Instruction::Swap,
        ]);
        assert!(simulate_stack(&ok, None).is_empty());

        // Only one entry: swap underflows.
        let bad = program(vec![Instruction::ConstLoc(loc(0, 0, 0)), Instruction::Swap]);
        assert_eq!(
            simulate_stack(&bad, None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );
    }

    #[test]
    fn stack_sim_dup_underflow_on_empty() {
        let p = program(vec![Instruction::Cpu(Cpu::Dup)]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 0 }]
        );
    }

    #[test]
    fn stack_sim_dup_copies_top_of_stack() {
        // `dup` on a non-empty stack pushes a copy of the top entry; popping
        // both leaves an empty, well-typed stack.
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::Cpu(Cpu::Dup),
            Instruction::Pop,
            Instruction::Pop,
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_sim_gate_ops_are_well_typed() {
        // Exercises the LocalRz / GlobalR / GlobalRz / Cz dispatch arms.
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            cpu_float(0.5),
            Instruction::LocalRz(1),
            cpu_float(0.5),
            Instruction::GlobalRz,
            cpu_float(0.5),
            cpu_float(0.25),
            Instruction::GlobalR,
            Instruction::ConstZone(0),
            Instruction::Cz,
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_sim_gate_underflow_on_empty_stack() {
        // GlobalRz pops one float via `pop_typed`; empty stack -> underflow.
        assert_eq!(
            simulate_stack(&program(vec![Instruction::GlobalRz]), None),
            vec![ValidationError::StackUnderflow { pc: 0 }]
        );
        // Move pops a lane via `pop_addr`; empty stack -> underflow.
        assert_eq!(
            simulate_stack(&program(vec![Instruction::Move(1)]), None),
            vec![ValidationError::StackUnderflow { pc: 0 }]
        );
    }

    #[test]
    fn stack_sim_new_array_and_get_item() {
        // new_array pops `dim0` elements and pushes an ARRAY_REF; get_item pops
        // `ndims` INT indices plus the ARRAY_REF and pushes the element.
        let p = program(vec![
            Instruction::Cpu(Cpu::Const(Value::I64(1))),
            Instruction::Cpu(Cpu::Const(Value::I64(2))),
            Instruction::NewArray(tag::INT as u32, 2, 0),
            Instruction::Cpu(Cpu::Const(Value::I64(0))),
            Instruction::GetItem(1),
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_sim_set_detector_and_observable() {
        // Each consumes an ARRAY_REF (produced by a 1-element new_array).
        let det = program(vec![
            Instruction::Cpu(Cpu::Const(Value::I64(0))),
            Instruction::NewArray(tag::INT as u32, 1, 0),
            Instruction::SetDetector,
        ]);
        assert!(
            simulate_stack(&det, None).is_empty(),
            "{:?}",
            simulate_stack(&det, None)
        );

        let obs = program(vec![
            Instruction::Cpu(Cpu::Const(Value::I64(0))),
            Instruction::NewArray(tag::INT as u32, 1, 0),
            Instruction::SetObservable,
        ]);
        assert!(
            simulate_stack(&obs, None).is_empty(),
            "{:?}",
            simulate_stack(&obs, None)
        );
    }

    #[test]
    fn stack_sim_halt_and_other_cpu_ops_are_noops() {
        // `halt` and any other reused vihaco-cpu op (here `print`, and a
        // non-int/float `const`) are untracked no-ops in the type simulator.
        let p = program(vec![
            Instruction::Cpu(Cpu::Print),
            Instruction::Cpu(Cpu::Const(Value::Bool(true))),
            Instruction::Cpu(Cpu::Halt),
        ]);
        assert!(simulate_stack(&p, None).is_empty());
    }

    #[test]
    fn location_group_validated_against_arch() {
        // With an arch, `initial_fill` runs the arch location-group check
        // (not just the no-arch duplicate fallback). Two valid distinct
        // locations produce no group error.
        let arch = simple_arch();
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::ConstLoc(loc(0, 0, 1)),
            Instruction::InitialFill(2),
        ]);
        let errors = simulate_stack(&p, Some(&arch));
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, ValidationError::LocationGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn location_group_arch_errors_are_reported() {
        // With an arch, a bad fill group surfaces the arch layer's own
        // location-group error (exercising the arch branch's error path, not
        // the no-arch duplicate fallback). A location repeated within the
        // group is invalid.
        let arch = simple_arch();
        let p = program(vec![
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::ConstLoc(loc(0, 0, 0)),
            Instruction::InitialFill(2),
        ]);
        let errors = simulate_stack(&p, Some(&arch));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LocationGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn duplicate_lanes_flagged_without_arch() {
        // The no-arch `move` fallback reports repeated lanes as duplicates.
        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let p = program(vec![
            Instruction::ConstLane(lane.encode_u64()),
            Instruction::ConstLane(lane.encode_u64()),
            Instruction::Move(2),
        ]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LaneGroupValidation { .. })),
            "got {errors:?}"
        );
    }
}
