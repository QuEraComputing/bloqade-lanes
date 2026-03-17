use std::collections::HashSet;
use std::fmt;

use super::instruction::{
    ArrayInstruction, AtomArrangementInstruction, CpuInstruction, DetectorObservableInstruction,
    Instruction, LaneConstInstruction, MeasurementInstruction, QuantumGateInstruction,
};
use super::program::Program;
use super::value::{
    TAG_ARRAY_REF, TAG_DETECTOR_REF, TAG_FLOAT, TAG_INT, TAG_LANE, TAG_LOCATION,
    TAG_MEASURE_FUTURE, TAG_OBSERVABLE_REF, TAG_ZONE,
};
use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::query::{LaneGroupError, LocationGroupError};
use crate::arch::types::ArchSpec;

/// Maximum valid type tag (TAG_OBSERVABLE_REF = 0x8).
const MAX_TYPE_TAG: u8 = 0x8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// NewArray dim0 must be > 0.
    NewArrayZeroDim0 { pc: usize },
    /// NewArray type_tag is invalid.
    NewArrayInvalidTypeTag { pc: usize, type_tag: u8 },
    /// InitialFill is not the first non-constant instruction.
    InitialFillNotFirst { pc: usize },
    /// Stack underflow at the given program counter.
    StackUnderflow { pc: usize },
    /// Type mismatch on stack.
    TypeMismatch { pc: usize, expected: u8, got: u8 },
    /// Invalid zone address per arch spec.
    InvalidZone { pc: usize, zone_id: u32 },
    /// Location group validation error (invalid address, duplicate, etc.).
    LocationGroupValidation {
        pc: usize,
        error: LocationGroupError,
    },
    /// Lane group validation error (invalid address, duplicate, inconsistent, AOD constraint, etc.).
    LaneGroupValidation { pc: usize, error: LaneGroupError },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::NewArrayZeroDim0 { pc } => {
                write!(f, "pc {}: new_array dim0 must be > 0", pc)
            }
            ValidationError::NewArrayInvalidTypeTag { pc, type_tag } => {
                write!(f, "pc {}: invalid type tag 0x{:x}", pc, type_tag)
            }
            ValidationError::InitialFillNotFirst { pc } => {
                write!(
                    f,
                    "pc {}: initial_fill must be the first non-constant instruction",
                    pc
                )
            }
            ValidationError::StackUnderflow { pc } => {
                write!(f, "pc {}: stack underflow", pc)
            }
            ValidationError::TypeMismatch { pc, expected, got } => {
                write!(
                    f,
                    "pc {}: type mismatch: expected tag 0x{:x}, got 0x{:x}",
                    pc, expected, got
                )
            }
            ValidationError::InvalidZone { pc, zone_id } => {
                write!(f, "pc {}: invalid zone_id={}", pc, zone_id)
            }
            ValidationError::LocationGroupValidation { pc, error } => {
                write!(f, "pc {}: {}", pc, error)
            }
            ValidationError::LaneGroupValidation { pc, error } => {
                write!(f, "pc {}: {}", pc, error)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Run structural validation on a program. Returns collected errors.
pub fn validate_structure(program: &Program) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut seen_non_constant = false;

    for (pc, instr) in program.instructions.iter().enumerate() {
        match instr {
            Instruction::Array(ArrayInstruction::NewArray {
                type_tag,
                dim0,
                dim1: _,
            }) => {
                if *dim0 == 0 {
                    errors.push(ValidationError::NewArrayZeroDim0 { pc });
                }
                if *type_tag > MAX_TYPE_TAG {
                    errors.push(ValidationError::NewArrayInvalidTypeTag {
                        pc,
                        type_tag: *type_tag,
                    });
                }
            }
            Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { .. }) => {
                if seen_non_constant {
                    errors.push(ValidationError::InitialFillNotFirst { pc });
                }
                seen_non_constant = true;
            }
            Instruction::AtomArrangement(AtomArrangementInstruction::Fill { .. })
            | Instruction::AtomArrangement(AtomArrangementInstruction::Move { .. })
            | Instruction::QuantumGate(QuantumGateInstruction::LocalR { .. })
            | Instruction::QuantumGate(QuantumGateInstruction::LocalRz { .. })
            | Instruction::Measurement(MeasurementInstruction::Measure { .. }) => {
                seen_non_constant = true;
            }
            // Constants don't set seen_non_constant
            Instruction::Cpu(CpuInstruction::ConstFloat(_))
            | Instruction::Cpu(CpuInstruction::ConstInt(_))
            | Instruction::LaneConst(LaneConstInstruction::ConstLoc(_))
            | Instruction::LaneConst(LaneConstInstruction::ConstLane(_, _))
            | Instruction::LaneConst(LaneConstInstruction::ConstZone(_)) => {}
            // All other instructions are non-constant
            _ => {
                seen_non_constant = true;
            }
        }
    }

    errors
}

/// Validate addresses in the program against an architecture specification.
/// Checks each constant location, lane, and zone instruction to ensure the
/// encoded address refers to valid entries in the arch spec.
pub fn validate_addresses(program: &Program, arch: &ArchSpec) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    for (pc, instr) in program.instructions.iter().enumerate() {
        match instr {
            Instruction::LaneConst(LaneConstInstruction::ConstLoc(bits)) => {
                let addr = LocationAddr::decode(*bits);
                if arch.check_location(&addr).is_some() {
                    errors.push(ValidationError::LocationGroupValidation {
                        pc,
                        error: LocationGroupError::InvalidAddress {
                            word_id: addr.word_id,
                            site_id: addr.site_id,
                        },
                    });
                }
            }
            Instruction::LaneConst(LaneConstInstruction::ConstLane(d0, d1)) => {
                let addr = LaneAddr::decode(*d0, *d1);
                for msg in arch.check_lane(&addr) {
                    errors.push(ValidationError::LaneGroupValidation {
                        pc,
                        error: LaneGroupError::InvalidLane { message: msg },
                    });
                }
            }
            Instruction::LaneConst(LaneConstInstruction::ConstZone(bits)) => {
                let addr = ZoneAddr::decode(*bits);
                if arch.check_zone(&addr).is_some() {
                    errors.push(ValidationError::InvalidZone {
                        pc,
                        zone_id: addr.zone_id,
                    });
                }
            }
            _ => {}
        }
    }

    errors
}

/// Stack simulation entry — tracks type tag and optionally the concrete value.
#[derive(Debug, Clone)]
struct SimEntry {
    tag: u8,
    #[allow(dead_code)]
    value: Option<u64>,
}

/// Type-level stack simulator that walks an instruction stream, tracking types
/// and concrete values. Validates stack discipline (underflow, type mismatches)
/// and, when given an `ArchSpec`, validates lane groups at `Move` instructions.
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

    // --- Primitives ---

    /// Pop one value of any type. Records `StackUnderflow` if the stack is empty.
    fn pop_any(&mut self) {
        if self.stack.pop().is_none() {
            self.errors
                .push(ValidationError::StackUnderflow { pc: self.pc });
        }
    }

    /// Pop one value and check its type tag. Records `StackUnderflow` or `TypeMismatch`.
    fn pop_typed(&mut self, expected_tag: u8) {
        match self.stack.pop() {
            Some(entry) if entry.tag != expected_tag => {
                self.errors.push(ValidationError::TypeMismatch {
                    pc: self.pc,
                    expected: expected_tag,
                    got: entry.tag,
                });
            }
            Some(_) => {}
            None => {
                self.errors
                    .push(ValidationError::StackUnderflow { pc: self.pc });
            }
        }
    }

    /// Pop `count` values, each checked against `expected_tag`.
    fn pop_typed_n(&mut self, expected_tag: u8, count: u32) {
        for _ in 0..count {
            self.pop_typed(expected_tag);
        }
    }

    /// Pop one lane-typed value, returning the concrete bits if valid.
    fn pop_lane(&mut self) -> Option<u64> {
        match self.stack.pop() {
            Some(entry) => {
                if entry.tag != TAG_LANE {
                    self.errors.push(ValidationError::TypeMismatch {
                        pc: self.pc,
                        expected: TAG_LANE,
                        got: entry.tag,
                    });
                    None
                } else {
                    entry.value
                }
            }
            None => {
                self.errors
                    .push(ValidationError::StackUnderflow { pc: self.pc });
                None
            }
        }
    }

    /// Pop one location-typed value, returning the concrete bits if valid.
    fn pop_location(&mut self) -> Option<u64> {
        match self.stack.pop() {
            Some(entry) => {
                if entry.tag != TAG_LOCATION {
                    self.errors.push(ValidationError::TypeMismatch {
                        pc: self.pc,
                        expected: TAG_LOCATION,
                        got: entry.tag,
                    });
                    None
                } else {
                    entry.value
                }
            }
            None => {
                self.errors
                    .push(ValidationError::StackUnderflow { pc: self.pc });
                None
            }
        }
    }

    /// Wrap `LocationGroupError`s as `ValidationError`s with the current `pc`.
    fn push_location_group_errors(&mut self, errors: Vec<LocationGroupError>) {
        let pc = self.pc;
        for error in errors {
            self.errors
                .push(ValidationError::LocationGroupValidation { pc, error });
        }
    }

    /// Wrap `LaneGroupError`s as `ValidationError`s with the current `pc`.
    fn push_lane_group_errors(&mut self, errors: Vec<LaneGroupError>) {
        let pc = self.pc;
        for error in errors {
            self.errors
                .push(ValidationError::LaneGroupValidation { pc, error });
        }
    }

    /// Check for duplicate locations without an arch spec (standalone duplicate check).
    /// Reports each unique duplicated address once.
    fn check_duplicate_locations_standalone(&mut self, locations: &[LocationAddr]) {
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

    /// Check for duplicate lanes without an arch spec (standalone duplicate check).
    /// Reports each unique duplicated address once.
    fn check_duplicate_lanes_standalone(&mut self, lanes: &[LaneAddr]) {
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

    // --- Instruction handlers ---

    /// Push a typed constant value onto the simulation stack.
    fn push_const(&mut self, tag: u8, bits: u64) {
        self.stack.push(SimEntry {
            tag,
            value: Some(bits),
        });
    }

    /// Duplicate the top stack entry.
    fn sim_dup(&mut self) {
        if let Some(top) = self.stack.last().cloned() {
            self.stack.push(top);
        } else {
            self.errors
                .push(ValidationError::StackUnderflow { pc: self.pc });
        }
    }

    /// Swap the top two stack entries.
    fn sim_swap(&mut self) {
        let len = self.stack.len();
        if len >= 2 {
            self.stack.swap(len - 1, len - 2);
        } else {
            self.errors
                .push(ValidationError::StackUnderflow { pc: self.pc });
        }
    }

    /// Pop `arity` location-typed values and validate the group
    /// (duplicates + arch spec checks if available).
    fn pop_and_validate_locations(&mut self, arity: u32) {
        let loc_values: Vec<Option<u64>> = (0..arity).map(|_| self.pop_location()).collect();
        let locations: Vec<LocationAddr> = loc_values
            .iter()
            .filter_map(|v| v.map(|bits| LocationAddr::decode(bits as u32)))
            .collect();
        if let Some(arch) = self.arch {
            self.push_location_group_errors(arch.check_locations(&locations));
        } else {
            self.check_duplicate_locations_standalone(&locations);
        }
    }

    /// Simulate `initial_fill` or `fill`: pop `arity` location-typed values and validate.
    fn sim_fill(&mut self, arity: u32) {
        self.pop_and_validate_locations(arity);
    }

    /// Simulate `move`: pop `arity` lane-typed values. When an `ArchSpec` is
    /// available, validates duplicates, consistency, membership, and AOD constraints
    /// via the unified `check_lanes`. Without arch, only checks duplicates.
    fn sim_move(&mut self, arity: u32) {
        let lane_values: Vec<Option<u64>> = (0..arity).map(|_| self.pop_lane()).collect();
        let lanes: Vec<LaneAddr> = lane_values
            .iter()
            .filter_map(|v| {
                v.map(|bits| {
                    let d0 = bits as u32;
                    let d1 = (bits >> 32) as u32;
                    LaneAddr::decode(d0, d1)
                })
            })
            .collect();
        if let Some(arch) = self.arch {
            self.push_lane_group_errors(arch.check_lanes(&lanes));
        } else {
            self.check_duplicate_lanes_standalone(&lanes);
        }
    }

    /// Simulate `local_r`: pop 2 float parameters (rotation_angle, axis_angle) then validate locations.
    fn sim_local_r(&mut self, arity: u32) {
        self.pop_typed_n(TAG_FLOAT, 2);
        self.pop_and_validate_locations(arity);
    }

    /// Simulate `local_rz`: pop 1 float parameter (rotation_angle) then validate locations.
    fn sim_local_rz(&mut self, arity: u32) {
        self.pop_typed_n(TAG_FLOAT, 1);
        self.pop_and_validate_locations(arity);
    }

    /// Simulate `global_r`: pop 2 float parameters (rotation_angle, axis_angle) for a global rotation.
    fn sim_global_r(&mut self) {
        self.pop_typed_n(TAG_FLOAT, 2);
    }

    /// Simulate `global_rz`: pop 1 float parameter (rotation_angle) for a global Z-rotation.
    fn sim_global_rz(&mut self) {
        self.pop_typed_n(TAG_FLOAT, 1);
    }

    /// Simulate `cz`: pop 1 zone value for a controlled-Z gate.
    fn sim_cz(&mut self) {
        self.pop_typed(TAG_ZONE);
    }

    /// Simulate `measure`: pop `arity` zone values and push `arity` measure futures.
    fn sim_measure(&mut self, arity: u32) {
        self.pop_typed_n(TAG_ZONE, arity);
        for _ in 0..arity {
            self.stack.push(SimEntry {
                tag: TAG_MEASURE_FUTURE,
                value: None,
            });
        }
    }

    /// Simulate `await_measure`: pop 1 measure future and push an array reference.
    fn sim_await_measure(&mut self) {
        self.pop_typed(TAG_MEASURE_FUTURE);
        self.stack.push(SimEntry {
            tag: TAG_ARRAY_REF,
            value: None,
        });
    }

    /// Simulate `new_array`: pop `dim0 * max(dim1, 1)` values of any type and push an array reference.
    fn sim_new_array(&mut self, dim0: u16, dim1: u16) {
        let count = dim0 * if dim1 == 0 { 1 } else { dim1 };
        for _ in 0..count {
            self.pop_any();
        }
        self.stack.push(SimEntry {
            tag: TAG_ARRAY_REF,
            value: None,
        });
    }

    /// Simulate `get_item`: pop `ndims` int indices and 1 array reference, then push the
    /// element value (assumed float since element type is not tracked).
    fn sim_get_item(&mut self, ndims: u16) {
        self.pop_typed_n(TAG_INT, ndims.into());
        self.pop_typed(TAG_ARRAY_REF);
        self.stack.push(SimEntry {
            tag: TAG_FLOAT,
            value: None,
        });
    }

    /// Simulate `set_detector` or `set_observable`: pop 1 array reference and push a
    /// typed reference (detector or observable) identified by `out_tag`.
    fn sim_set_ref(&mut self, out_tag: u8) {
        self.pop_typed(TAG_ARRAY_REF);
        self.stack.push(SimEntry {
            tag: out_tag,
            value: None,
        });
    }

    /// Simulate `return`: pop 1 value of any type as the return value.
    fn sim_return(&mut self) {
        self.pop_any();
    }

    // --- Main simulation loop ---

    /// Dispatch a single instruction to the appropriate handler.
    fn dispatch(&mut self, inst: &Instruction) {
        match inst {
            Instruction::Cpu(CpuInstruction::ConstFloat(v)) => {
                self.push_const(TAG_FLOAT, v.to_bits());
            }
            Instruction::Cpu(CpuInstruction::ConstInt(v)) => {
                self.push_const(TAG_INT, *v as u64);
            }
            Instruction::LaneConst(LaneConstInstruction::ConstLoc(v)) => {
                self.push_const(TAG_LOCATION, *v as u64);
            }
            Instruction::LaneConst(LaneConstInstruction::ConstLane(d0, d1)) => {
                let combined = (*d0 as u64) | ((*d1 as u64) << 32);
                self.push_const(TAG_LANE, combined);
            }
            Instruction::LaneConst(LaneConstInstruction::ConstZone(v)) => {
                self.push_const(TAG_ZONE, *v as u64);
            }

            Instruction::Cpu(CpuInstruction::Pop) => self.pop_any(),
            Instruction::Cpu(CpuInstruction::Dup) => self.sim_dup(),
            Instruction::Cpu(CpuInstruction::Swap) => self.sim_swap(),

            Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity })
            | Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity }) => {
                self.sim_fill(*arity);
            }
            Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity }) => {
                self.sim_move(*arity);
            }

            Instruction::QuantumGate(QuantumGateInstruction::LocalR { arity }) => {
                self.sim_local_r(*arity);
            }
            Instruction::QuantumGate(QuantumGateInstruction::LocalRz { arity }) => {
                self.sim_local_rz(*arity);
            }
            Instruction::QuantumGate(QuantumGateInstruction::GlobalR) => self.sim_global_r(),
            Instruction::QuantumGate(QuantumGateInstruction::GlobalRz) => self.sim_global_rz(),
            Instruction::QuantumGate(QuantumGateInstruction::CZ) => self.sim_cz(),

            Instruction::Measurement(MeasurementInstruction::Measure { arity }) => {
                self.sim_measure(*arity);
            }
            Instruction::Measurement(MeasurementInstruction::AwaitMeasure) => {
                self.sim_await_measure()
            }

            Instruction::Array(ArrayInstruction::NewArray {
                type_tag: _,
                dim0,
                dim1,
            }) => {
                self.sim_new_array(*dim0, *dim1);
            }
            Instruction::Array(ArrayInstruction::GetItem { ndims }) => {
                self.sim_get_item(*ndims);
            }
            Instruction::DetectorObservable(DetectorObservableInstruction::SetDetector) => {
                self.sim_set_ref(TAG_DETECTOR_REF);
            }
            Instruction::DetectorObservable(DetectorObservableInstruction::SetObservable) => {
                self.sim_set_ref(TAG_OBSERVABLE_REF);
            }

            Instruction::Cpu(CpuInstruction::Return) => self.sim_return(),
            Instruction::Cpu(CpuInstruction::Halt) => {}
        }
    }

    /// Run the stack simulation over the entire instruction stream.
    /// Collects type errors, stack underflow errors, and (when an `ArchSpec` is
    /// available) lane group validation errors.
    fn run(mut self, program: &Program) -> Vec<ValidationError> {
        for (pc, inst) in program.instructions.iter().enumerate() {
            self.pc = pc;
            self.dispatch(inst);
        }
        self.errors
    }
}

/// Simulate the type-level stack through the instruction stream.
/// Collects type errors and stack underflow errors.
/// If an `ArchSpec` is provided, also validates lane groups at `Move` instructions.
pub fn simulate_stack(program: &Program, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
    let sim = StackSimulator::new(arch);
    sim.run(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::Version;

    // --- Structural validation tests ---

    #[test]
    fn test_valid_program_no_errors() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0)),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 1 }),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };
        assert!(validate_structure(&program).is_empty());
    }

    #[test]
    fn test_initial_fill_not_first() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::Cpu(CpuInstruction::Halt),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0)),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 1 }),
            ],
        };
        let errors = validate_structure(&program);
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ValidationError::InitialFillNotFirst { pc: 2 }
        ));
    }

    #[test]
    fn test_initial_fill_after_constants_ok() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::Cpu(CpuInstruction::ConstFloat(1.0)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0)),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 1 }),
            ],
        };
        assert!(validate_structure(&program).is_empty());
    }

    #[test]
    fn test_new_array_zero_dim0() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::Array(ArrayInstruction::NewArray {
                type_tag: 0,
                dim0: 0,
                dim1: 0,
            })],
        };
        let errors = validate_structure(&program);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::NewArrayZeroDim0 { pc: 0 }))
        );
    }

    #[test]
    fn test_new_array_invalid_type_tag() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::Array(ArrayInstruction::NewArray {
                type_tag: 0xF,
                dim0: 1,
                dim1: 0,
            })],
        };
        let errors = validate_structure(&program);
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::NewArrayInvalidTypeTag {
                pc: 0,
                type_tag: 0xF
            }
        )));
    }

    // --- Stack simulation tests ---

    #[test]
    fn test_stack_sim_valid_program() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0)),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 1 }),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };
        assert!(simulate_stack(&program, None).is_empty());
    }

    #[test]
    fn test_stack_sim_underflow() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::Cpu(CpuInstruction::Pop)],
        };
        let errors = simulate_stack(&program, None);
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ValidationError::StackUnderflow { pc: 0 }
        ));
    }

    #[test]
    fn test_stack_sim_type_mismatch() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                // Push a float, try to use it as a location for InitialFill
                Instruction::Cpu(CpuInstruction::ConstFloat(1.0)),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 1 }),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ValidationError::TypeMismatch {
                pc: 1,
                expected,
                got
            } if expected == TAG_LOCATION && got == TAG_FLOAT
        ));
    }

    #[test]
    fn test_stack_sim_move() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(0x100, 0)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(0x200, 0)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 2 }),
            ],
        };
        assert!(simulate_stack(&program, None).is_empty());
    }

    #[test]
    fn test_stack_sim_local_r() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0)),
                Instruction::Cpu(CpuInstruction::ConstFloat(0.5)),
                Instruction::Cpu(CpuInstruction::ConstFloat(1.0)),
                Instruction::QuantumGate(QuantumGateInstruction::LocalR { arity: 1 }),
            ],
        };
        assert!(simulate_stack(&program, None).is_empty());
    }

    #[test]
    fn test_stack_sim_measure_and_await() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstZone(0)),
                Instruction::Measurement(MeasurementInstruction::Measure { arity: 1 }),
                Instruction::Measurement(MeasurementInstruction::AwaitMeasure),
                Instruction::Cpu(CpuInstruction::Return),
            ],
        };
        assert!(simulate_stack(&program, None).is_empty());
    }

    #[test]
    fn test_stack_sim_dup_and_swap() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::Cpu(CpuInstruction::ConstFloat(1.0)),
                Instruction::Cpu(CpuInstruction::Dup),
                Instruction::QuantumGate(QuantumGateInstruction::GlobalR),
            ],
        };
        assert!(simulate_stack(&program, None).is_empty());
    }

    #[test]
    fn test_stack_sim_cz_type_mismatch() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::Cpu(CpuInstruction::ConstFloat(1.0)),
                Instruction::QuantumGate(QuantumGateInstruction::CZ),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ValidationError::TypeMismatch {
                pc: 1,
                expected,
                got
            } if expected == TAG_ZONE && got == TAG_FLOAT
        ));
    }

    #[test]
    fn test_stack_sim_set_detector() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstZone(0)),
                Instruction::Measurement(MeasurementInstruction::Measure { arity: 1 }),
                Instruction::Measurement(MeasurementInstruction::AwaitMeasure),
                Instruction::DetectorObservable(DetectorObservableInstruction::SetDetector),
                Instruction::Cpu(CpuInstruction::Return),
            ],
        };
        assert!(simulate_stack(&program, None).is_empty());
    }

    // --- Address validation tests ---

    fn test_arch_spec() -> ArchSpec {
        let json = r#"{
            "version": 1,
            "geometry": {
                "sites_per_word": 2,
                "words": [
                    {
                        "positions": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0], "y_spacing": [] },
                        "site_indices": [[0, 0], [1, 0]]
                    }
                ]
            },
            "buses": {
                "site_buses": [
                    { "src": [0], "dst": [1] }
                ],
                "word_buses": []
            },
            "words_with_site_buses": [0],
            "sites_with_word_buses": [],
            "zones": [
                { "words": [0] }
            ],
            "entangling_zones": [0],
            "measurement_mode_zones": [0]
        }"#;
        ArchSpec::from_json(json).unwrap()
    }

    #[test]
    fn test_addr_valid_location() {
        let arch = test_arch_spec();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::LaneConst(LaneConstInstruction::ConstLoc(
                0x0001,
            ))],
        };
        assert!(validate_addresses(&program, &arch).is_empty());
    }

    #[test]
    fn test_addr_invalid_location() {
        let arch = test_arch_spec();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::LaneConst(LaneConstInstruction::ConstLoc(
                0x0005,
            ))],
        };
        let errors = validate_addresses(&program, &arch);
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ValidationError::LocationGroupValidation {
                pc: 0,
                error: LocationGroupError::InvalidAddress {
                    word_id: 0,
                    site_id: 5
                }
            }
        ));
    }

    #[test]
    fn test_addr_invalid_zone() {
        let arch = test_arch_spec();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::LaneConst(LaneConstInstruction::ConstZone(99))],
        };
        let errors = validate_addresses(&program, &arch);
        assert_eq!(errors.len(), 1);
        assert!(matches!(
            errors[0],
            ValidationError::InvalidZone { pc: 0, zone_id: 99 }
        ));
    }

    #[test]
    fn test_addr_valid_lane() {
        let arch = test_arch_spec();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::LaneConst(LaneConstInstruction::ConstLane(
                0x00000000, 0x00000000,
            ))],
        };
        assert!(validate_addresses(&program, &arch).is_empty());
    }

    #[test]
    fn test_addr_invalid_lane_bus() {
        let arch = test_arch_spec();
        // data0=0x00000000 (word_id=0, site_id=0), data1=0x00000005 (bus_id=5)
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::LaneConst(LaneConstInstruction::ConstLane(
                0x00000000, 0x00000005,
            ))],
        };
        let errors = validate_addresses(&program, &arch);
        assert!(!errors.is_empty());
    }

    // --- Lane group validation tests ---

    use crate::arch::addr::{Direction, LaneAddr, MoveType};

    fn lane_group_arch_spec() -> ArchSpec {
        crate::arch::example_arch_spec()
    }

    fn make_lane(
        dir: Direction,
        mt: MoveType,
        word_id: u32,
        site_id: u32,
        bus_id: u32,
    ) -> (u32, u32) {
        LaneAddr {
            direction: dir,
            move_type: mt,
            word_id,
            site_id,
            bus_id,
        }
        .encode()
    }

    #[test]
    fn test_lane_group_consistent_passes() {
        let arch = lane_group_arch_spec();
        let lane0 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 0, 0);
        let lane1 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 1, 0);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane0.0, lane0.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane1.0, lane1.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 2 }),
            ],
        };
        let errors = simulate_stack(&program, Some(&arch));
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_lane_group_inconsistent_bus_id() {
        let arch = lane_group_arch_spec();
        let lane0 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 0, 0);
        let lane1 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 1, 1);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane0.0, lane0.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane1.0, lane1.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 2 }),
            ],
        };
        let errors = simulate_stack(&program, Some(&arch));
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::LaneGroupValidation {
                pc: 2,
                error: LaneGroupError::Inconsistent { .. }
            }
        )));
    }

    #[test]
    fn test_lane_group_inconsistent_direction() {
        let arch = lane_group_arch_spec();
        let lane0 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 0, 0);
        let lane1 = make_lane(Direction::Backward, MoveType::SiteBus, 0, 1, 0);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane0.0, lane0.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane1.0, lane1.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 2 }),
            ],
        };
        let errors = simulate_stack(&program, Some(&arch));
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::LaneGroupValidation {
                pc: 2,
                error: LaneGroupError::Inconsistent { .. }
            }
        )));
    }

    #[test]
    fn test_lane_group_word_not_in_site_bus_list() {
        let json = r#"{
            "version": 1,
            "geometry": {
                "sites_per_word": 2,
                "words": [
                    {
                        "positions": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0], "y_spacing": [] },
                        "site_indices": [[0, 0], [1, 0]]
                    },
                    {
                        "positions": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0], "y_spacing": [] },
                        "site_indices": [[0, 0], [1, 0]]
                    }
                ]
            },
            "buses": {
                "site_buses": [{ "src": [0], "dst": [1] }],
                "word_buses": []
            },
            "words_with_site_buses": [0],
            "sites_with_word_buses": [],
            "zones": [{ "words": [0, 1] }],
            "entangling_zones": [0],
            "measurement_mode_zones": [0]
        }"#;
        let arch = ArchSpec::from_json(json).unwrap();

        let lane0 = make_lane(Direction::Forward, MoveType::SiteBus, 1, 0, 0);
        let lane1 = make_lane(Direction::Forward, MoveType::SiteBus, 1, 1, 0);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane0.0, lane0.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane1.0, lane1.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 2 }),
            ],
        };
        let errors = simulate_stack(&program, Some(&arch));
        assert!(errors.iter().any(|e| matches!(
            e,
            ValidationError::LaneGroupValidation {
                pc: 2,
                error: LaneGroupError::WordNotInSiteBusList { word_id: 1 }
            }
        )));
    }

    #[test]
    fn test_lane_group_aod_constraint_rectangle_passes() {
        let arch = lane_group_arch_spec();
        let lanes: Vec<(u32, u32)> = [0, 1, 5, 6]
            .iter()
            .map(|&s| make_lane(Direction::Forward, MoveType::SiteBus, 0, s, 0))
            .collect();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lanes[0].0, lanes[0].1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lanes[1].0, lanes[1].1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lanes[2].0, lanes[2].1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lanes[3].0, lanes[3].1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 4 }),
            ],
        };
        let errors = simulate_stack(&program, Some(&arch));
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_lane_group_aod_constraint_not_rectangle() {
        let arch = lane_group_arch_spec();
        let lanes: Vec<(u32, u32)> = [0, 1, 5]
            .iter()
            .map(|&s| make_lane(Direction::Forward, MoveType::SiteBus, 0, s, 0))
            .collect();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lanes[0].0, lanes[0].1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lanes[1].0, lanes[1].1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lanes[2].0, lanes[2].1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 3 }),
            ],
        };
        let errors = simulate_stack(&program, Some(&arch));
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::LaneGroupValidation {
                    pc: 3,
                    error: LaneGroupError::AODConstraintViolation { .. }
                }
            )),
            "expected AOD constraint violation error, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_lane_group_no_arch_skips_validation() {
        let lane0 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 0, 0);
        let lane1 = make_lane(Direction::Backward, MoveType::WordBus, 1, 1, 5);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane0.0, lane0.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane1.0, lane1.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 2 }),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(errors.is_empty());
    }

    // --- Duplicate address validation tests ---

    #[test]
    fn test_duplicate_location_in_initial_fill() {
        let addr = LocationAddr {
            word_id: 0,
            site_id: 9,
        }
        .encode();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0x0007)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 3 }),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LocationGroupValidation { pc: 3, error: LocationGroupError::DuplicateAddress { address } } if *address == addr)),
            "expected DuplicateLocationAddress, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_duplicate_location_in_fill() {
        let addr = LocationAddr {
            word_id: 1,
            site_id: 2,
        }
        .encode();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity: 2 }),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::LocationGroupValidation {
                    pc: 2,
                    error: LocationGroupError::DuplicateAddress { .. }
                }
            )),
            "expected DuplicateLocationAddress, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_duplicate_location_in_local_r() {
        let addr = LocationAddr {
            word_id: 0,
            site_id: 3,
        }
        .encode();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::Cpu(CpuInstruction::ConstFloat(0.5)),
                Instruction::Cpu(CpuInstruction::ConstFloat(1.0)),
                Instruction::QuantumGate(QuantumGateInstruction::LocalR { arity: 2 }),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::LocationGroupValidation {
                    pc: 4,
                    error: LocationGroupError::DuplicateAddress { .. }
                }
            )),
            "expected DuplicateLocationAddress, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_duplicate_location_in_local_rz() {
        let addr = LocationAddr {
            word_id: 0,
            site_id: 5,
        }
        .encode();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::Cpu(CpuInstruction::ConstFloat(0.5)),
                Instruction::QuantumGate(QuantumGateInstruction::LocalRz { arity: 2 }),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::LocationGroupValidation {
                    pc: 3,
                    error: LocationGroupError::DuplicateAddress { .. }
                }
            )),
            "expected DuplicateLocationAddress, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_duplicate_lane_in_move() {
        let lane = make_lane(Direction::Forward, MoveType::SiteBus, 0, 9, 0);
        let lane_other = make_lane(Direction::Forward, MoveType::SiteBus, 0, 7, 0);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane.0, lane.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane_other.0, lane_other.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane.0, lane.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 3 }),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LaneGroupValidation { pc: 3, error: LaneGroupError::DuplicateAddress { address } } if *address == lane)),

            "expected DuplicateLaneAddress, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_duplicate_location_reported_once() {
        let addr = LocationAddr {
            word_id: 1,
            site_id: 2,
        }
        .encode();
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(addr)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity: 3 }),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };
        let errors = simulate_stack(&program, None);
        let dup_count = errors
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    ValidationError::LocationGroupValidation {
                        error: LocationGroupError::DuplicateAddress { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            dup_count, 1,
            "expected exactly 1 DuplicateAddress, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_duplicate_lane_reported_once() {
        let lane = make_lane(Direction::Forward, MoveType::SiteBus, 0, 9, 0);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane.0, lane.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane.0, lane.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane.0, lane.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 3 }),
            ],
        };
        let errors = simulate_stack(&program, None);
        let dup_count = errors
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    ValidationError::LaneGroupValidation {
                        error: LaneGroupError::DuplicateAddress { .. },
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            dup_count, 1,
            "expected exactly 1 DuplicateAddress, got: {:?}",
            errors
        );
    }

    #[test]
    fn test_no_duplicate_location_passes() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0x0000)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0x0001)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0x0002)),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 3 }),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }

    #[test]
    fn test_no_duplicate_lane_passes() {
        let lane0 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 0, 0);
        let lane1 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 1, 0);
        let lane2 = make_lane(Direction::Forward, MoveType::SiteBus, 0, 2, 0);
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane0.0, lane0.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane1.0, lane1.1)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(lane2.0, lane2.1)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 3 }),
            ],
        };
        let errors = simulate_stack(&program, None);
        assert!(errors.is_empty(), "expected no errors, got: {:?}", errors);
    }
}
