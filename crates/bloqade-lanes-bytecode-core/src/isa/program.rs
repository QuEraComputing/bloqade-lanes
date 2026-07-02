//! Flat program container for the vihaco-backed ISA.
//!
//! A program is a [`Version`] plus a flat `Vec<`[`Instruction`]`>` — no
//! functions, labels, or string interner (our programs are a single flat
//! instruction list; see <https://github.com/QuEraComputing/bloqade-lanes/issues/769>).
//! vihaco's [`vihaco::module::Module`] / [`vihaco::ProgramLoader`] carry that
//! structured-language machinery, so we keep a thin container and delegate the
//! per-instruction work to vihaco's derived codec ([`WriteBytes`]/[`FromBytes`])
//! and text parser ([`vihaco_parser_core::Parse`]).
//!
//! ## Binary layout (native, breaking vs. legacy `BLQD`)
//!
//! ```text
//! magic    : 5 bytes  = b"LANES"
//! version  : u32 LE   = (major << 16) | minor
//! code     : N * INSTRUCTION_WIDTH bytes (vihaco fixed-width words)
//! ```
//!
//! There is no section container: instruction words are fixed-width and
//! self-delimiting, so the code section is simply the remaining bytes.

use std::io::Cursor;

use vihaco::instruction::{FromBytes, WriteBytes};
use vihaco::module::Module;
use vihaco::value::{Type, Value};

use super::{INSTRUCTION_WIDTH, Instruction};
use crate::version::Version;

/// 5-byte magic identifying a native vihaco-backed Bloqade Lanes program.
/// Distinct from the legacy `BLQD` container so the two can't be confused.
pub const MAGIC: &[u8; 5] = b"LANES";

/// Header length: [`MAGIC`] (5 bytes) followed by a packed u32 version.
const HEADER_LEN: usize = MAGIC.len() + 4;

/// Consumer metadata carried in `Module::extra` (vihaco has no version field).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LanesInfo {
    pub version: Version,
}

impl Default for LanesInfo {
    fn default() -> Self {
        Self {
            version: Version::new(0, 0),
        }
    }
}

impl std::fmt::Display for LanesInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "version {}", self.version)
    }
}

/// A Bloqade Lanes program: a vihaco `Module` specialised to our ISA. A single
/// `@main` function's worth of flat code plus the version in `extra`.
pub type Program = Module<Instruction, Value, Type, LanesInfo>;

/// Build a `Program` from a version + flat instruction list. This is the ONE
/// constructor used by both binary and text loading, so all `Program`s built
/// from the same (version, code) compare equal regardless of source.
#[allow(clippy::field_reassign_with_default)] // `Module` is a foreign type; struct-literal init is not possible
pub fn from_code(version: Version, code: Vec<Instruction>) -> Program {
    let mut m = Program::default();
    m.code = code;
    m.extra = LanesInfo { version };
    m
}

/// Error from binary (de)serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryError {
    /// First five bytes were not [`MAGIC`].
    BadMagic,
    /// Buffer ended before a complete header or instruction word.
    Truncated { expected: usize, got: usize },
    /// The code region length is not a multiple of [`INSTRUCTION_WIDTH`].
    UnalignedCode { len: usize },
    /// A word held an opcode/payload vihaco could not decode.
    Decode { pc: usize, message: String },
}

impl std::fmt::Display for BinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryError::BadMagic => write!(f, "bad magic bytes (expected LANES)"),
            BinaryError::Truncated { expected, got } => {
                write!(f, "truncated: expected {expected} bytes, got {got}")
            }
            BinaryError::UnalignedCode { len } => write!(
                f,
                "code length {len} is not a multiple of {INSTRUCTION_WIDTH}"
            ),
            BinaryError::Decode { pc, message } => {
                write!(f, "decode error at instruction {pc}: {message}")
            }
        }
    }
}

impl std::error::Error for BinaryError {}

/// Error from text (`.sst`) parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextError {
    /// No `.version` directive was found before the first instruction.
    MissingVersion,
    /// The `.version` directive's value could not be parsed.
    InvalidVersion { line: usize, value: String },
    /// A line could not be parsed as an instruction.
    BadInstruction { line: usize, text: String },
}

impl std::fmt::Display for TextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextError::MissingVersion => write!(f, "missing .version directive"),
            TextError::InvalidVersion { line, value } => {
                write!(f, "line {line}: invalid version '{value}'")
            }
            TextError::BadInstruction { line, text } => {
                write!(f, "line {line}: cannot parse instruction '{text}'")
            }
        }
    }
}

impl std::error::Error for TextError {}

/// Serialize to the native binary format (see module docs).
pub fn to_binary(program: &Program) -> Vec<u8> {
    let mut buf = Vec::with_capacity(HEADER_LEN + program.code.len() * INSTRUCTION_WIDTH as usize);
    buf.extend_from_slice(MAGIC);
    let packed: u32 = program.extra.version.into();
    buf.extend_from_slice(&packed.to_le_bytes());
    for inst in &program.code {
        // WriteBytes into a Vec is infallible.
        inst.write_bytes(&mut buf)
            .expect("writing instruction bytes to a Vec cannot fail");
    }
    buf
}

/// Deserialize from the native binary format.
pub fn from_binary(bytes: &[u8]) -> Result<Program, BinaryError> {
    if bytes.len() < HEADER_LEN {
        return Err(BinaryError::Truncated {
            expected: HEADER_LEN,
            got: bytes.len(),
        });
    }
    if &bytes[0..MAGIC.len()] != MAGIC {
        return Err(BinaryError::BadMagic);
    }
    let packed = u32::from_le_bytes([
        bytes[MAGIC.len()],
        bytes[MAGIC.len() + 1],
        bytes[MAGIC.len() + 2],
        bytes[MAGIC.len() + 3],
    ]);
    let version = Version::from(packed);

    let code = &bytes[HEADER_LEN..];
    let width = INSTRUCTION_WIDTH as usize;
    if !code.len().is_multiple_of(width) {
        return Err(BinaryError::UnalignedCode { len: code.len() });
    }

    let count = code.len() / width;
    let mut instructions = Vec::with_capacity(count);
    let mut cursor = Cursor::new(code);
    for pc in 0..count {
        let inst = Instruction::from_bytes(&mut cursor).map_err(|e| BinaryError::Decode {
            pc,
            message: e.to_string(),
        })?;
        instructions.push(inst);
    }

    Ok(from_code(version, instructions))
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // illustrative sample floats, not math constants
mod tests {
    use super::*;

    fn sample() -> Program {
        use vihaco::value::Value;
        use vihaco_cpu::Instruction as Cpu;
        from_code(
            Version::new(1, 2),
            vec![
                Instruction::Cpu(Cpu::Const(Value::F64(1.5))),
                Instruction::Cpu(Cpu::Const(Value::I64(-42))),
                Instruction::Cpu(Cpu::Dup),
                Instruction::ConstLoc(0x0000_0000_0100_0000),
                Instruction::ConstLane(0x0000_0000_0000_0001),
                Instruction::ConstZone(0x0000_0003),
                Instruction::InitialFill(2),
                Instruction::Move(1),
                Instruction::LocalRz(1),
                Instruction::LocalR(3),
                Instruction::GlobalRz,
                Instruction::Cz,
                Instruction::Measure(1),
                Instruction::AwaitMeasure,
                Instruction::NewArray(2, 10, 20),
                Instruction::GetItem(2),
                Instruction::SetDetector,
                Instruction::Cpu(Cpu::Halt),
                Instruction::Return,
            ],
        )
    }

    #[test]
    fn binary_round_trips() {
        let program = sample();
        let bytes = to_binary(&program);
        assert_eq!(&bytes[0..MAGIC.len()], MAGIC);
        assert_eq!(
            bytes.len(),
            HEADER_LEN + program.code.len() * INSTRUCTION_WIDTH as usize
        );
        assert_eq!(from_binary(&bytes).unwrap(), program);
    }

    #[test]
    fn binary_preserves_version() {
        let bytes = to_binary(&sample());
        assert_eq!(
            from_binary(&bytes).unwrap().extra.version,
            Version::new(1, 2)
        );
    }

    #[test]
    fn empty_program_round_trips() {
        let program = from_code(Version::new(1, 0), vec![]);
        let bytes = to_binary(&program);
        assert_eq!(bytes.len(), HEADER_LEN);
        assert_eq!(from_binary(&bytes).unwrap(), program);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = to_binary(&sample());
        bytes[0] = b'X';
        assert_eq!(from_binary(&bytes), Err(BinaryError::BadMagic));
    }

    #[test]
    fn short_buffer_rejected() {
        assert_eq!(
            from_binary(b"LAN"),
            Err(BinaryError::Truncated {
                expected: HEADER_LEN,
                got: 3
            })
        );
    }

    #[test]
    fn unaligned_code_rejected() {
        let mut bytes = to_binary(&sample());
        bytes.push(0); // one stray byte past the last full word
        assert!(matches!(
            from_binary(&bytes),
            Err(BinaryError::UnalignedCode { .. })
        ));
    }
}
