use super::opcode::{
    ArrayInstCode, AtomArrangementInstCode, CpuInstCode, DetectorObservableInstCode, DeviceCode,
    LaneConstInstCode, MeasurementInstCode, QuantumGateInstCode, pack_opcode,
};

/// CPU/stack instructions (device code `0x00`, FLAIR-aligned).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CpuInstruction {
    /// Push a 64-bit float constant. Opcode `0x0300`.
    ConstFloat(f64),
    /// Push a 64-bit signed integer constant. Opcode `0x0200`.
    ConstInt(i64),
    /// Pop and discard the top stack value. Opcode `0x0500`.
    Pop,
    /// Duplicate the top stack value. Opcode `0x0400`.
    Dup,
    /// Swap the top two stack values. Opcode `0x0600`.
    Swap,
    /// Return from program. Opcode `0x6400`.
    Return,
    /// Halt execution. Opcode `0xFF00`.
    Halt,
}

/// Lane-specific constant instructions (device code `0x0F`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneConstInstruction {
    /// Push a packed [`LocationAddr`](crate::arch::addr::LocationAddr). Opcode `0x000F`.
    ConstLoc(u32),
    /// Push a packed [`LaneAddr`](crate::arch::addr::LaneAddr) as (data0, data1). Opcode `0x010F`.
    ConstLane(u32, u32),
    /// Push a packed [`ZoneAddr`](crate::arch::addr::ZoneAddr). Opcode `0x020F`.
    ConstZone(u32),
}

/// Atom arrangement instructions (device code `0x10`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomArrangementInstruction {
    /// Initial atom fill. Must be the first non-constant instruction. Opcode `0x0010`.
    InitialFill { arity: u32 },
    /// Refill atoms at locations. Opcode `0x0110`.
    Fill { arity: u32 },
    /// Move atoms along lanes. Opcode `0x0210`.
    Move { arity: u32 },
}

/// Quantum gate instructions (device code `0x11`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantumGateInstruction {
    /// Local R rotation on `arity` locations. Opcode `0x0011`.
    LocalR { arity: u32 },
    /// Local Rz rotation on `arity` locations. Opcode `0x0111`.
    LocalRz { arity: u32 },
    /// Global R rotation. Opcode `0x0211`.
    GlobalR,
    /// Global Rz rotation. Opcode `0x0311`.
    GlobalRz,
    /// Controlled-Z gate on a zone. Opcode `0x0411`.
    CZ,
}

/// Measurement instructions (device code `0x12`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementInstruction {
    /// Initiate measurement on `arity` zones. Opcode `0x0012`.
    Measure { arity: u32 },
    /// Block until measurement completes. Opcode `0x0112`.
    AwaitMeasure,
}

/// Array construction and indexing instructions (device code `0x13`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayInstruction {
    /// Construct an array. Opcode `0x0013`.
    NewArray { type_tag: u8, dim0: u16, dim1: u16 },
    /// Index into an array. Opcode `0x0113`.
    GetItem { ndims: u16 },
}

/// Detector and observable setup instructions (device code `0x14`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorObservableInstruction {
    /// Build a detector record from an array. Opcode `0x0014`.
    SetDetector,
    /// Build an observable record from an array. Opcode `0x0114`.
    SetObservable,
}

/// A single bytecode instruction.
///
/// Each variant wraps a device-specific instruction enum. The instruction
/// can be encoded to a 16-byte binary word or printed as SST text assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    /// CPU/stack operations (device `0x00`).
    Cpu(CpuInstruction),
    /// Lane address constants (device `0x0F`).
    LaneConst(LaneConstInstruction),
    /// Atom arrangement operations (device `0x10`).
    AtomArrangement(AtomArrangementInstruction),
    /// Quantum gate operations (device `0x11`).
    QuantumGate(QuantumGateInstruction),
    /// Measurement operations (device `0x12`).
    Measurement(MeasurementInstruction),
    /// Array operations (device `0x13`).
    Array(ArrayInstruction),
    /// Detector/observable setup (device `0x14`).
    DetectorObservable(DetectorObservableInstruction),
}

impl Instruction {
    /// Returns the packed opcode as a u16: `(instruction_code << 8) | device_code`.
    pub fn opcode(&self) -> u16 {
        match self {
            Instruction::Cpu(cpu) => {
                let inst = match cpu {
                    CpuInstruction::ConstFloat(_) => CpuInstCode::ConstFloat as u8,
                    CpuInstruction::ConstInt(_) => CpuInstCode::ConstInt as u8,
                    CpuInstruction::Pop => CpuInstCode::Pop as u8,
                    CpuInstruction::Dup => CpuInstCode::Dup as u8,
                    CpuInstruction::Swap => CpuInstCode::Swap as u8,
                    CpuInstruction::Return => CpuInstCode::Return as u8,
                    CpuInstruction::Halt => CpuInstCode::Halt as u8,
                };
                pack_opcode(DeviceCode::Cpu as u8, inst)
            }
            Instruction::LaneConst(lc) => {
                let inst = match lc {
                    LaneConstInstruction::ConstLoc(_) => LaneConstInstCode::ConstLoc as u8,
                    LaneConstInstruction::ConstLane(_, _) => LaneConstInstCode::ConstLane as u8,
                    LaneConstInstruction::ConstZone(_) => LaneConstInstCode::ConstZone as u8,
                };
                pack_opcode(DeviceCode::LaneConstants as u8, inst)
            }
            Instruction::AtomArrangement(aa) => {
                let inst = match aa {
                    AtomArrangementInstruction::InitialFill { .. } => {
                        AtomArrangementInstCode::InitialFill as u8
                    }
                    AtomArrangementInstruction::Fill { .. } => AtomArrangementInstCode::Fill as u8,
                    AtomArrangementInstruction::Move { .. } => AtomArrangementInstCode::Move as u8,
                };
                pack_opcode(DeviceCode::AtomArrangement as u8, inst)
            }
            Instruction::QuantumGate(qg) => {
                let inst = match qg {
                    QuantumGateInstruction::LocalR { .. } => QuantumGateInstCode::LocalR as u8,
                    QuantumGateInstruction::LocalRz { .. } => QuantumGateInstCode::LocalRz as u8,
                    QuantumGateInstruction::GlobalR => QuantumGateInstCode::GlobalR as u8,
                    QuantumGateInstruction::GlobalRz => QuantumGateInstCode::GlobalRz as u8,
                    QuantumGateInstruction::CZ => QuantumGateInstCode::CZ as u8,
                };
                pack_opcode(DeviceCode::QuantumGate as u8, inst)
            }
            Instruction::Measurement(m) => {
                let inst = match m {
                    MeasurementInstruction::Measure { .. } => MeasurementInstCode::Measure as u8,
                    MeasurementInstruction::AwaitMeasure => MeasurementInstCode::AwaitMeasure as u8,
                };
                pack_opcode(DeviceCode::Measurement as u8, inst)
            }
            Instruction::Array(arr) => {
                let inst = match arr {
                    ArrayInstruction::NewArray { .. } => ArrayInstCode::NewArray as u8,
                    ArrayInstruction::GetItem { .. } => ArrayInstCode::GetItem as u8,
                };
                pack_opcode(DeviceCode::Array as u8, inst)
            }
            Instruction::DetectorObservable(dob) => {
                let inst = match dob {
                    DetectorObservableInstruction::SetDetector => {
                        DetectorObservableInstCode::SetDetector as u8
                    }
                    DetectorObservableInstruction::SetObservable => {
                        DetectorObservableInstCode::SetObservable as u8
                    }
                };
                pack_opcode(DeviceCode::DetectorObservable as u8, inst)
            }
        }
    }

    /// Returns the device code for this instruction.
    pub fn device_code(&self) -> DeviceCode {
        match self {
            Instruction::Cpu(_) => DeviceCode::Cpu,
            Instruction::LaneConst(_) => DeviceCode::LaneConstants,
            Instruction::AtomArrangement(_) => DeviceCode::AtomArrangement,
            Instruction::QuantumGate(_) => DeviceCode::QuantumGate,
            Instruction::Measurement(_) => DeviceCode::Measurement,
            Instruction::Array(_) => DeviceCode::Array,
            Instruction::DetectorObservable(_) => DeviceCode::DetectorObservable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_cpu() {
        assert_eq!(
            Instruction::Cpu(CpuInstruction::ConstFloat(1.0)).opcode(),
            0x0300
        );
        assert_eq!(
            Instruction::Cpu(CpuInstruction::ConstInt(1)).opcode(),
            0x0200
        );
        assert_eq!(Instruction::Cpu(CpuInstruction::Pop).opcode(), 0x0500);
        assert_eq!(Instruction::Cpu(CpuInstruction::Dup).opcode(), 0x0400);
        assert_eq!(Instruction::Cpu(CpuInstruction::Swap).opcode(), 0x0600);
        assert_eq!(Instruction::Cpu(CpuInstruction::Return).opcode(), 0x6400);
        assert_eq!(Instruction::Cpu(CpuInstruction::Halt).opcode(), 0xFF00);
    }

    #[test]
    fn test_opcode_lane_constants() {
        assert_eq!(
            Instruction::LaneConst(LaneConstInstruction::ConstLoc(0)).opcode(),
            0x000F
        );
        assert_eq!(
            Instruction::LaneConst(LaneConstInstruction::ConstLane(0, 0)).opcode(),
            0x010F
        );
        assert_eq!(
            Instruction::LaneConst(LaneConstInstruction::ConstZone(0)).opcode(),
            0x020F
        );
    }

    #[test]
    fn test_opcode_atom_arrangement() {
        assert_eq!(
            Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 1 })
                .opcode(),
            0x0010
        );
        assert_eq!(
            Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity: 1 }).opcode(),
            0x0110
        );
        assert_eq!(
            Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 1 }).opcode(),
            0x0210
        );
    }

    #[test]
    fn test_opcode_quantum_gate() {
        assert_eq!(
            Instruction::QuantumGate(QuantumGateInstruction::LocalR { arity: 1 }).opcode(),
            0x0011
        );
        assert_eq!(
            Instruction::QuantumGate(QuantumGateInstruction::LocalRz { arity: 1 }).opcode(),
            0x0111
        );
        assert_eq!(
            Instruction::QuantumGate(QuantumGateInstruction::GlobalR).opcode(),
            0x0211
        );
        assert_eq!(
            Instruction::QuantumGate(QuantumGateInstruction::GlobalRz).opcode(),
            0x0311
        );
        assert_eq!(
            Instruction::QuantumGate(QuantumGateInstruction::CZ).opcode(),
            0x0411
        );
    }

    #[test]
    fn test_opcode_measurement() {
        assert_eq!(
            Instruction::Measurement(MeasurementInstruction::Measure { arity: 1 }).opcode(),
            0x0012
        );
        assert_eq!(
            Instruction::Measurement(MeasurementInstruction::AwaitMeasure).opcode(),
            0x0112
        );
    }

    #[test]
    fn test_opcode_array() {
        assert_eq!(
            Instruction::Array(ArrayInstruction::NewArray {
                type_tag: 0,
                dim0: 10,
                dim1: 20
            })
            .opcode(),
            0x0013
        );
        assert_eq!(
            Instruction::Array(ArrayInstruction::GetItem { ndims: 2 }).opcode(),
            0x0113
        );
    }

    #[test]
    fn test_opcode_detector_observable() {
        assert_eq!(
            Instruction::DetectorObservable(DetectorObservableInstruction::SetDetector).opcode(),
            0x0014
        );
        assert_eq!(
            Instruction::DetectorObservable(DetectorObservableInstruction::SetObservable).opcode(),
            0x0114
        );
    }

    #[test]
    fn test_device_code() {
        assert_eq!(
            Instruction::Cpu(CpuInstruction::Halt).device_code(),
            DeviceCode::Cpu
        );
        assert_eq!(
            Instruction::LaneConst(LaneConstInstruction::ConstLoc(0)).device_code(),
            DeviceCode::LaneConstants
        );
        assert_eq!(
            Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity: 1 })
                .device_code(),
            DeviceCode::AtomArrangement
        );
        assert_eq!(
            Instruction::QuantumGate(QuantumGateInstruction::CZ).device_code(),
            DeviceCode::QuantumGate
        );
        assert_eq!(
            Instruction::Measurement(MeasurementInstruction::AwaitMeasure).device_code(),
            DeviceCode::Measurement
        );
        assert_eq!(
            Instruction::Array(ArrayInstruction::GetItem { ndims: 1 }).device_code(),
            DeviceCode::Array
        );
        assert_eq!(
            Instruction::DetectorObservable(DetectorObservableInstruction::SetDetector)
                .device_code(),
            DeviceCode::DetectorObservable
        );
    }
}
