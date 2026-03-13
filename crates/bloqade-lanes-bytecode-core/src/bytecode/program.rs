use std::fmt;

use super::encode::DecodeError;
use super::instruction::Instruction;
use crate::version::Version;

const MAGIC: &[u8; 4] = b"BLQD";
const SECTION_TYPE_METADATA: u32 = 0;
const SECTION_TYPE_CODE: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub version: Version,
    pub instructions: Vec<Instruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramError {
    BadMagic,
    Truncated { expected: usize, got: usize },
    UnknownSectionType(u32),
    InvalidCodeSectionLength(usize),
    MissingMetadataSection,
    MissingCodeSection,
    Decode(DecodeError),
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProgramError::BadMagic => write!(f, "bad magic bytes (expected BLQD)"),
            ProgramError::Truncated { expected, got } => {
                write!(f, "truncated: expected {} bytes, got {}", expected, got)
            }
            ProgramError::UnknownSectionType(t) => write!(f, "unknown section type: {}", t),
            ProgramError::InvalidCodeSectionLength(len) => {
                write!(f, "code section length {} is not a multiple of 16", len)
            }
            ProgramError::MissingMetadataSection => write!(f, "missing metadata section"),
            ProgramError::MissingCodeSection => write!(f, "missing code section"),
            ProgramError::Decode(e) => write!(f, "decode error: {}", e),
        }
    }
}

impl std::error::Error for ProgramError {}

impl From<DecodeError> for ProgramError {
    fn from(e: DecodeError) -> Self {
        ProgramError::Decode(e)
    }
}

impl Program {
    /// Serialize the program to the BLQD binary format.
    pub fn to_binary(&self) -> Vec<u8> {
        let code_payload_len = self.instructions.len() * 16;
        // Header (8) + metadata section header (8) + metadata payload (4)
        // + code section header (8) + code payload
        let total = 8 + 8 + 4 + 8 + code_payload_len;
        let mut buf = Vec::with_capacity(total);

        // Header: magic + section_count
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&2u32.to_le_bytes());

        // Metadata section: type=0, length=4, version
        buf.extend_from_slice(&SECTION_TYPE_METADATA.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        let version_u32: u32 = self.version.into();
        buf.extend_from_slice(&version_u32.to_le_bytes());

        // Code section: type=1, length, instructions
        buf.extend_from_slice(&SECTION_TYPE_CODE.to_le_bytes());
        buf.extend_from_slice(&(code_payload_len as u32).to_le_bytes());
        for instr in &self.instructions {
            buf.extend_from_slice(&instr.to_bytes());
        }

        buf
    }

    /// Deserialize a program from the BLQD binary format.
    pub fn from_binary(bytes: &[u8]) -> Result<Self, ProgramError> {
        if bytes.len() < 8 {
            return Err(ProgramError::Truncated {
                expected: 8,
                got: bytes.len(),
            });
        }

        // Verify magic
        if &bytes[0..4] != MAGIC {
            return Err(ProgramError::BadMagic);
        }

        let section_count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

        let mut offset = 8usize;
        let mut version: Option<Version> = None;
        let mut instructions: Option<Vec<Instruction>> = None;

        for _ in 0..section_count {
            // Need at least 8 bytes for section header
            if offset + 8 > bytes.len() {
                return Err(ProgramError::Truncated {
                    expected: offset + 8,
                    got: bytes.len(),
                });
            }

            let section_type = u32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            let section_length = u32::from_le_bytes([
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]) as usize;
            offset += 8;

            if offset + section_length > bytes.len() {
                return Err(ProgramError::Truncated {
                    expected: offset + section_length,
                    got: bytes.len(),
                });
            }

            match section_type {
                SECTION_TYPE_METADATA => {
                    if section_length < 4 {
                        return Err(ProgramError::Truncated {
                            expected: 4,
                            got: section_length,
                        });
                    }
                    let v = u32::from_le_bytes([
                        bytes[offset],
                        bytes[offset + 1],
                        bytes[offset + 2],
                        bytes[offset + 3],
                    ]);
                    version = Some(Version::from(v));
                }
                SECTION_TYPE_CODE => {
                    if !section_length.is_multiple_of(16) {
                        return Err(ProgramError::InvalidCodeSectionLength(section_length));
                    }
                    let count = section_length / 16;
                    let mut instrs = Vec::with_capacity(count);
                    for i in 0..count {
                        let start = offset + i * 16;
                        let chunk: &[u8] = &bytes[start..start + 16];
                        let arr: [u8; 16] = chunk.try_into().unwrap();
                        instrs.push(Instruction::from_bytes(&arr)?);
                    }
                    instructions = Some(instrs);
                }
                other => return Err(ProgramError::UnknownSectionType(other)),
            }

            offset += section_length;
        }

        let version = version.ok_or(ProgramError::MissingMetadataSection)?;
        let instructions = instructions.ok_or(ProgramError::MissingCodeSection)?;

        Ok(Program {
            version,
            instructions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::instruction::{
        ArrayInstruction, AtomArrangementInstruction, CpuInstruction, LaneConstInstruction,
        QuantumGateInstruction,
    };
    use crate::version::Version;

    #[test]
    fn test_round_trip_empty_program() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![],
        };
        let binary = program.to_binary();
        let decoded = Program::from_binary(&binary).unwrap();
        assert_eq!(program, decoded);
    }

    #[test]
    fn test_round_trip_with_instructions() {
        let program = Program {
            version: Version::new(2, 0),
            instructions: vec![
                Instruction::Cpu(CpuInstruction::ConstFloat(1.5)),
                Instruction::Cpu(CpuInstruction::ConstInt(42)),
                Instruction::Cpu(CpuInstruction::Dup),
                Instruction::Array(ArrayInstruction::NewArray {
                    type_tag: 1,
                    dim0: 10,
                    dim1: 20,
                }),
                Instruction::Array(ArrayInstruction::GetItem { ndims: 2 }),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0x1234)),
                Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity: 5 }),
                Instruction::QuantumGate(QuantumGateInstruction::GlobalR),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };
        let binary = program.to_binary();
        let decoded = Program::from_binary(&binary).unwrap();
        assert_eq!(program, decoded);
    }

    #[test]
    fn test_bad_magic() {
        let mut binary = Program {
            version: Version::new(1, 0),
            instructions: vec![],
        }
        .to_binary();
        binary[0] = b'X';
        assert_eq!(Program::from_binary(&binary), Err(ProgramError::BadMagic));
    }

    #[test]
    fn test_truncated_header() {
        let bytes = b"BLQD";
        assert_eq!(
            Program::from_binary(bytes),
            Err(ProgramError::Truncated {
                expected: 8,
                got: 4,
            })
        );
    }

    #[test]
    fn test_truncated_empty() {
        let bytes = b"BL";
        assert_eq!(
            Program::from_binary(bytes),
            Err(ProgramError::Truncated {
                expected: 8,
                got: 2,
            })
        );
    }

    #[test]
    fn test_invalid_code_section_length() {
        // Build a binary with a code section whose length is not a multiple of 16
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BLQD");
        buf.extend_from_slice(&2u32.to_le_bytes()); // section_count = 2

        // Metadata section
        buf.extend_from_slice(&0u32.to_le_bytes()); // type = 0
        buf.extend_from_slice(&4u32.to_le_bytes()); // length = 4
        buf.extend_from_slice(&1u32.to_le_bytes()); // version = 1

        // Code section with bad length (5 bytes, not multiple of 16)
        buf.extend_from_slice(&1u32.to_le_bytes()); // type = 1
        buf.extend_from_slice(&5u32.to_le_bytes()); // length = 5
        buf.extend_from_slice(&[0u8; 5]); // payload

        assert_eq!(
            Program::from_binary(&buf),
            Err(ProgramError::InvalidCodeSectionLength(5))
        );
    }

    #[test]
    fn test_missing_code_section() {
        // Binary with only a metadata section
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BLQD");
        buf.extend_from_slice(&1u32.to_le_bytes()); // section_count = 1

        // Metadata section only
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());

        assert_eq!(
            Program::from_binary(&buf),
            Err(ProgramError::MissingCodeSection)
        );
    }

    #[test]
    fn test_missing_metadata_section() {
        // Binary with only a code section
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BLQD");
        buf.extend_from_slice(&1u32.to_le_bytes()); // section_count = 1

        // Code section only (empty)
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());

        assert_eq!(
            Program::from_binary(&buf),
            Err(ProgramError::MissingMetadataSection)
        );
    }

    #[test]
    fn test_unknown_section_type() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"BLQD");
        buf.extend_from_slice(&1u32.to_le_bytes()); // section_count = 1

        // Unknown section type = 99
        buf.extend_from_slice(&99u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());

        assert_eq!(
            Program::from_binary(&buf),
            Err(ProgramError::UnknownSectionType(99))
        );
    }

    #[test]
    fn test_binary_header_structure() {
        let version = Version::new(1, 2);
        let program = Program {
            version,
            instructions: vec![Instruction::Cpu(CpuInstruction::Halt)],
        };
        let binary = program.to_binary();

        // Check magic
        assert_eq!(&binary[0..4], b"BLQD");
        // Check section count = 2
        assert_eq!(u32::from_le_bytes(binary[4..8].try_into().unwrap()), 2);
        // Metadata section type = 0
        assert_eq!(u32::from_le_bytes(binary[8..12].try_into().unwrap()), 0);
        // Metadata section length = 4
        assert_eq!(u32::from_le_bytes(binary[12..16].try_into().unwrap()), 4);
        // Version = packed u32
        let expected_packed: u32 = version.into();
        assert_eq!(
            u32::from_le_bytes(binary[16..20].try_into().unwrap()),
            expected_packed
        );
        // Code section type = 1
        assert_eq!(u32::from_le_bytes(binary[20..24].try_into().unwrap()), 1);
        // Code section length = 16 (one instruction)
        assert_eq!(u32::from_le_bytes(binary[24..28].try_into().unwrap()), 16);
    }
}
