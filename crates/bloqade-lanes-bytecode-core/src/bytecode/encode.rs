use std::fmt;

use super::instruction::{
    ArrayInstruction, AtomArrangementInstruction, CpuInstruction, DetectorObservableInstruction,
    Instruction, LaneConstInstruction, MeasurementInstruction, QuantumGateInstruction,
};
use super::opcode::{OpCodeCategory, decode_opcode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    UnknownOpcode(u16),
    UnknownDeviceCode(u8),
    InvalidOperand { opcode: u16, message: String },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::UnknownOpcode(op) => write!(f, "unknown opcode: 0x{:04x}", op),
            DecodeError::UnknownDeviceCode(dc) => {
                write!(f, "unknown device code: 0x{:02x}", dc)
            }
            DecodeError::InvalidOperand { opcode, message } => {
                write!(
                    f,
                    "invalid operand for opcode 0x{:04x}: {}",
                    opcode, message
                )
            }
        }
    }
}

impl std::error::Error for DecodeError {}

impl Instruction {
    /// Encode instruction into (opcode, data0, data1, data2).
    ///
    /// Opcode word layout: `[unused:16][instruction_code:8][device_code:8]`
    pub fn encode(&self) -> (u32, u32, u32, u32) {
        let opcode = self.opcode() as u32;

        match self {
            // ConstFloat: f64 LE across data0 (low) and data1 (high)
            Instruction::Cpu(CpuInstruction::ConstFloat(f)) => {
                let bits = f.to_bits();
                let data0 = bits as u32;
                let data1 = (bits >> 32) as u32;
                (opcode, data0, data1, 0)
            }
            // ConstInt: i64 LE across data0 (low) and data1 (high)
            Instruction::Cpu(CpuInstruction::ConstInt(v)) => {
                let bits = *v as u64;
                let data0 = bits as u32;
                let data1 = (bits >> 32) as u32;
                (opcode, data0, data1, 0)
            }
            // ConstLoc: LocationAddr in data0
            Instruction::LaneConst(LaneConstInstruction::ConstLoc(v)) => (opcode, *v, 0, 0),
            // ConstLane: two u32 words (data0, data1)
            Instruction::LaneConst(LaneConstInstruction::ConstLane(d0, d1)) => {
                (opcode, *d0, *d1, 0)
            }
            // ConstZone: ZoneAddr in data0
            Instruction::LaneConst(LaneConstInstruction::ConstZone(v)) => (opcode, *v, 0, 0),

            // Simple CPU ops: no data
            Instruction::Cpu(CpuInstruction::Pop)
            | Instruction::Cpu(CpuInstruction::Dup)
            | Instruction::Cpu(CpuInstruction::Swap)
            | Instruction::Cpu(CpuInstruction::Return)
            | Instruction::Cpu(CpuInstruction::Halt) => (opcode, 0, 0, 0),

            // NewArray: data0 = [tag:8][pad:8][dim0:16], data1 = [pad:16][dim1:16]
            Instruction::Array(ArrayInstruction::NewArray {
                type_tag,
                dim0,
                dim1,
            }) => {
                let data0 = ((*type_tag as u32) << 24) | (*dim0 as u32);
                let data1 = *dim1 as u32;
                (opcode, data0, data1, 0)
            }

            // GetItem: data0 = ndims
            Instruction::Array(ArrayInstruction::GetItem { ndims }) => {
                (opcode, *ndims as u32, 0, 0)
            }

            // Device ops with arity: data0 = arity
            Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity })
            | Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity })
            | Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity })
            | Instruction::QuantumGate(QuantumGateInstruction::LocalR { arity })
            | Instruction::QuantumGate(QuantumGateInstruction::LocalRz { arity })
            | Instruction::Measurement(MeasurementInstruction::Measure { arity }) => {
                (opcode, *arity, 0, 0)
            }

            // Simple device ops: no data
            Instruction::QuantumGate(QuantumGateInstruction::GlobalR)
            | Instruction::QuantumGate(QuantumGateInstruction::GlobalRz)
            | Instruction::QuantumGate(QuantumGateInstruction::CZ)
            | Instruction::Measurement(MeasurementInstruction::AwaitMeasure)
            | Instruction::DetectorObservable(DetectorObservableInstruction::SetDetector)
            | Instruction::DetectorObservable(DetectorObservableInstruction::SetObservable) => {
                (opcode, 0, 0, 0)
            }
        }
    }

    /// Decode an instruction from (opcode_word, data0, data1, data2).
    pub fn decode(word: u32, data0: u32, data1: u32, _data2: u32) -> Result<Self, DecodeError> {
        let category = decode_opcode(word)?;

        match category {
            OpCodeCategory::Cpu(cpu_op) => {
                use super::opcode::CpuInstCode::*;
                match cpu_op {
                    ConstFloat => {
                        let bits = (data0 as u64) | ((data1 as u64) << 32);
                        Ok(Instruction::Cpu(CpuInstruction::ConstFloat(
                            f64::from_bits(bits),
                        )))
                    }
                    ConstInt => {
                        let bits = (data0 as u64) | ((data1 as u64) << 32);
                        Ok(Instruction::Cpu(CpuInstruction::ConstInt(bits as i64)))
                    }
                    Pop => Ok(Instruction::Cpu(CpuInstruction::Pop)),
                    Dup => Ok(Instruction::Cpu(CpuInstruction::Dup)),
                    Swap => Ok(Instruction::Cpu(CpuInstruction::Swap)),
                    Return => Ok(Instruction::Cpu(CpuInstruction::Return)),
                    Halt => Ok(Instruction::Cpu(CpuInstruction::Halt)),
                }
            }
            OpCodeCategory::LaneConstants(lc_op) => {
                use super::opcode::LaneConstInstCode::*;
                match lc_op {
                    ConstLoc => Ok(Instruction::LaneConst(LaneConstInstruction::ConstLoc(
                        data0,
                    ))),
                    ConstLane => Ok(Instruction::LaneConst(LaneConstInstruction::ConstLane(
                        data0, data1,
                    ))),
                    ConstZone => Ok(Instruction::LaneConst(LaneConstInstruction::ConstZone(
                        data0,
                    ))),
                }
            }
            OpCodeCategory::AtomArrangement(aa_op) => {
                use super::opcode::AtomArrangementInstCode::*;
                let arity = data0;
                match aa_op {
                    InitialFill => Ok(Instruction::AtomArrangement(
                        AtomArrangementInstruction::InitialFill { arity },
                    )),
                    Fill => Ok(Instruction::AtomArrangement(
                        AtomArrangementInstruction::Fill { arity },
                    )),
                    Move => Ok(Instruction::AtomArrangement(
                        AtomArrangementInstruction::Move { arity },
                    )),
                }
            }
            OpCodeCategory::QuantumGate(qg_op) => {
                use super::opcode::QuantumGateInstCode::*;
                match qg_op {
                    LocalR => Ok(Instruction::QuantumGate(QuantumGateInstruction::LocalR {
                        arity: data0,
                    })),
                    LocalRz => Ok(Instruction::QuantumGate(QuantumGateInstruction::LocalRz {
                        arity: data0,
                    })),
                    GlobalR => Ok(Instruction::QuantumGate(QuantumGateInstruction::GlobalR)),
                    GlobalRz => Ok(Instruction::QuantumGate(QuantumGateInstruction::GlobalRz)),
                    CZ => Ok(Instruction::QuantumGate(QuantumGateInstruction::CZ)),
                }
            }
            OpCodeCategory::Measurement(m_op) => {
                use super::opcode::MeasurementInstCode::*;
                match m_op {
                    Measure => Ok(Instruction::Measurement(MeasurementInstruction::Measure {
                        arity: data0,
                    })),
                    AwaitMeasure => Ok(Instruction::Measurement(
                        MeasurementInstruction::AwaitMeasure,
                    )),
                }
            }
            OpCodeCategory::Array(arr_op) => {
                use super::opcode::ArrayInstCode::*;
                match arr_op {
                    NewArray => {
                        let type_tag = ((data0 >> 24) & 0xFF) as u8;
                        let dim0 = (data0 & 0xFFFF) as u16;
                        let dim1 = (data1 & 0xFFFF) as u16;
                        Ok(Instruction::Array(ArrayInstruction::NewArray {
                            type_tag,
                            dim0,
                            dim1,
                        }))
                    }
                    GetItem => Ok(Instruction::Array(ArrayInstruction::GetItem {
                        ndims: data0 as u16,
                    })),
                }
            }
            OpCodeCategory::DetectorObservable(dob_op) => {
                use super::opcode::DetectorObservableInstCode::*;
                match dob_op {
                    SetDetector => Ok(Instruction::DetectorObservable(
                        DetectorObservableInstruction::SetDetector,
                    )),
                    SetObservable => Ok(Instruction::DetectorObservable(
                        DetectorObservableInstruction::SetObservable,
                    )),
                }
            }
        }
    }

    /// Encode to 16 little-endian bytes.
    pub fn to_bytes(&self) -> [u8; 16] {
        let (word, data0, data1, data2) = self.encode();
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&word.to_le_bytes());
        bytes[4..8].copy_from_slice(&data0.to_le_bytes());
        bytes[8..12].copy_from_slice(&data1.to_le_bytes());
        bytes[12..16].copy_from_slice(&data2.to_le_bytes());
        bytes
    }

    /// Decode from 16 little-endian bytes.
    pub fn from_bytes(bytes: &[u8; 16]) -> Result<Self, DecodeError> {
        let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let data0 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let data1 = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let data2 = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        Self::decode(word, data0, data1, data2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(instr: Instruction) {
        let (word, data0, data1, data2) = instr.encode();
        let decoded = Instruction::decode(word, data0, data1, data2).unwrap();
        assert_eq!(instr, decoded, "round-trip failed for {:?}", instr);
    }

    fn round_trip_bytes(instr: Instruction) {
        let bytes = instr.to_bytes();
        let decoded = Instruction::from_bytes(&bytes).unwrap();
        assert_eq!(instr, decoded, "bytes round-trip failed for {:?}", instr);
    }

    #[test]
    fn test_round_trip_all_instructions() {
        let instructions = vec![
            Instruction::Cpu(CpuInstruction::ConstFloat(1.5)),
            Instruction::Cpu(CpuInstruction::ConstInt(-42)),
            Instruction::Cpu(CpuInstruction::Pop),
            Instruction::Cpu(CpuInstruction::Dup),
            Instruction::Cpu(CpuInstruction::Swap),
            Instruction::Cpu(CpuInstruction::Return),
            Instruction::Cpu(CpuInstruction::Halt),
            Instruction::LaneConst(LaneConstInstruction::ConstLoc(0x1234)),
            Instruction::LaneConst(LaneConstInstruction::ConstLane(0x00125678, 0xC0009ABC)),
            Instruction::LaneConst(LaneConstInstruction::ConstZone(0x9ABC)),
            Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 4 }),
            Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity: 3 }),
            Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 2 }),
            Instruction::QuantumGate(QuantumGateInstruction::LocalR { arity: 1 }),
            Instruction::QuantumGate(QuantumGateInstruction::LocalRz { arity: 5 }),
            Instruction::QuantumGate(QuantumGateInstruction::GlobalR),
            Instruction::QuantumGate(QuantumGateInstruction::GlobalRz),
            Instruction::QuantumGate(QuantumGateInstruction::CZ),
            Instruction::Measurement(MeasurementInstruction::Measure { arity: 10 }),
            Instruction::Measurement(MeasurementInstruction::AwaitMeasure),
            Instruction::Array(ArrayInstruction::NewArray {
                type_tag: 0x2,
                dim0: 100,
                dim1: 200,
            }),
            Instruction::Array(ArrayInstruction::GetItem { ndims: 2 }),
            Instruction::DetectorObservable(DetectorObservableInstruction::SetDetector),
            Instruction::DetectorObservable(DetectorObservableInstruction::SetObservable),
        ];

        for instr in &instructions {
            round_trip(*instr);
            round_trip_bytes(*instr);
        }
    }

    #[test]
    fn test_const_float_bit_pattern() {
        let instr = Instruction::Cpu(CpuInstruction::ConstFloat(1.0));
        let (word, data0, data1, _data2) = instr.encode();
        assert_eq!(word, 0x0300); // opcode: device 0x00, inst 0x03
        let bits = 1.0_f64.to_bits();
        assert_eq!(data0, bits as u32);
        assert_eq!(data1, (bits >> 32) as u32);
    }

    #[test]
    fn test_const_int_bit_pattern() {
        let instr = Instruction::Cpu(CpuInstruction::ConstInt(42));
        let (word, data0, data1, _data2) = instr.encode();
        assert_eq!(word, 0x0200); // opcode: device 0x00, inst 0x02
        assert_eq!(data0, 42);
        assert_eq!(data1, 0);
    }

    #[test]
    fn test_const_int_negative() {
        let instr = Instruction::Cpu(CpuInstruction::ConstInt(-1));
        let (word, data0, data1, _data2) = instr.encode();
        assert_eq!(word, 0x0200);
        assert_eq!(data0, 0xFFFF_FFFF); // low 32 bits of -1i64
        assert_eq!(data1, 0xFFFF_FFFF); // high 32 bits of -1i64
        let decoded = Instruction::decode(word, data0, data1, 0).unwrap();
        assert_eq!(decoded, instr);
    }

    #[test]
    fn test_new_array_bit_pattern() {
        let instr = Instruction::Array(ArrayInstruction::NewArray {
            type_tag: 0x2,
            dim0: 10,
            dim1: 20,
        });
        let (word, data0, data1, _data2) = instr.encode();
        assert_eq!(word, 0x0013); // opcode: device 0x13, inst 0x00
        assert_eq!((data0 >> 24) & 0xFF, 0x2); // type_tag
        assert_eq!(data0 & 0xFFFF, 10); // dim0
        assert_eq!(data1 & 0xFFFF, 20); // dim1
    }

    #[test]
    fn test_unknown_opcode_error() {
        // device 0x01 is reserved/unknown
        let result = Instruction::decode(0x0001, 0, 0, 0);
        assert!(matches!(result, Err(DecodeError::UnknownDeviceCode(0x01))));
    }

    #[test]
    fn test_to_bytes_little_endian() {
        let instr = Instruction::Cpu(CpuInstruction::Halt);
        let bytes = instr.to_bytes();
        assert_eq!(bytes.len(), 16);
        // opcode 0xFF00 in LE: byte[0]=0x00 (device), byte[1]=0xFF (inst)
        assert_eq!(bytes[0], 0x00); // device code (Cpu)
        assert_eq!(bytes[1], 0xFF); // instruction code (Halt)
        assert_eq!(bytes[2], 0);
        assert_eq!(bytes[3], 0);
        assert_eq!(&bytes[4..], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_to_bytes_device_instruction() {
        let instr =
            Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 3 });
        let bytes = instr.to_bytes();
        assert_eq!(bytes.len(), 16);
        // opcode 0x0010 in LE: byte[0]=0x10 (device), byte[1]=0x00 (inst)
        assert_eq!(bytes[0], 0x10); // device code (AtomArrangement)
        assert_eq!(bytes[1], 0x00); // instruction code (InitialFill)
        assert_eq!(bytes[2], 0);
        assert_eq!(bytes[3], 0);
        // data0: arity 3 in LE
        assert_eq!(bytes[4], 3);
        assert_eq!(bytes[5], 0);
    }
}
