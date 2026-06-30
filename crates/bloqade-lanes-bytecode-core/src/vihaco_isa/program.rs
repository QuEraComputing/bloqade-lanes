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

use chumsky::Parser as _;
use vihaco::instruction::{FromBytes, WriteBytes};
use vihaco_parser_core::Parse;

use super::{INSTRUCTION_WIDTH, Instruction};
use crate::version::Version;

/// 5-byte magic identifying a native vihaco-backed Bloqade Lanes program.
/// Distinct from the legacy `BLQD` container so the two can't be confused.
pub const MAGIC: &[u8; 5] = b"LANES";

/// Header length: [`MAGIC`] (5 bytes) followed by a packed u32 version.
const HEADER_LEN: usize = MAGIC.len() + 4;

/// A bytecode program: a version and a flat instruction sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Program version.
    pub version: Version,
    /// Instructions in execution order.
    pub instructions: Vec<Instruction>,
}

/// Error from binary (de)serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryError {
    /// First four bytes were not [`MAGIC`].
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

impl Program {
    /// Serialize to the native binary format (see module docs).
    pub fn to_binary(&self) -> Vec<u8> {
        let mut buf =
            Vec::with_capacity(HEADER_LEN + self.instructions.len() * INSTRUCTION_WIDTH as usize);
        buf.extend_from_slice(MAGIC);
        let packed: u32 = self.version.into();
        buf.extend_from_slice(&packed.to_le_bytes());
        for inst in &self.instructions {
            // WriteBytes into a Vec is infallible.
            inst.write_bytes(&mut buf)
                .expect("writing instruction bytes to a Vec cannot fail");
        }
        buf
    }

    /// Deserialize from the native binary format.
    pub fn from_binary(bytes: &[u8]) -> Result<Self, BinaryError> {
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

        Ok(Program {
            version,
            instructions,
        })
    }

    /// Parse assembly text. Requires a leading `.version` directive; `;`
    /// begins a line comment; blank lines are ignored. Each remaining line is
    /// delegated to the vihaco-derived [`Instruction`] parser.
    pub fn parse_text(source: &str) -> Result<Self, TextError> {
        let mut version: Option<Version> = None;
        let mut instructions = Vec::new();

        for (idx, raw) in source.lines().enumerate() {
            let line_num = idx + 1;
            let line = match raw.find(';') {
                Some(pos) => &raw[..pos],
                None => raw,
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some(rest) = line.strip_prefix('.') {
                let mut parts = rest.split_whitespace();
                if parts.next() == Some("version") {
                    let value = parts.next().unwrap_or("");
                    version =
                        Some(
                            parse_version(value).ok_or_else(|| TextError::InvalidVersion {
                                line: line_num,
                                value: value.to_string(),
                            })?,
                        );
                }
                continue;
            }

            let inst = Instruction::parser()
                .parse(line)
                .into_result()
                .map_err(|_| TextError::BadInstruction {
                    line: line_num,
                    text: line.to_string(),
                })?;
            instructions.push(inst);
        }

        Ok(Program {
            version: version.ok_or(TextError::MissingVersion)?,
            instructions,
        })
    }

    /// Render the program as assembly text that [`parse_text`](Self::parse_text)
    /// round-trips.
    pub fn to_text(&self) -> String {
        let mut out = format!(".version {}.{}\n", self.version.major, self.version.minor);
        for inst in &self.instructions {
            out.push_str(&instruction_to_text(inst));
            out.push('\n');
        }
        out
    }
}

/// Parse a `major.minor` (or bare `major`, minor defaulting to 0) version.
fn parse_version(s: &str) -> Option<Version> {
    if s.is_empty() {
        return None;
    }
    match s.split_once('.') {
        Some((major, minor)) => Some(Version::new(major.parse().ok()?, minor.parse().ok()?)),
        None => Some(Version::new(s.parse().ok()?, 0)),
    }
}

/// Render one instruction as its canonical text line.
fn instruction_to_text(inst: &Instruction) -> String {
    use Instruction::*;
    match inst {
        Pop => "pop".to_string(),
        Swap => "swap".to_string(),
        Return => "return".to_string(),
        ConstLoc(v) => format!("const_loc 0x{v:016x}"),
        ConstLane(v) => format!("const_lane 0x{v:016x}"),
        ConstZone(v) => format!("const_zone 0x{v:08x}"),
        InitialFill(arity) => format!("initial_fill {arity}"),
        Fill(arity) => format!("fill {arity}"),
        Move(arity) => format!("move {arity}"),
        LocalRz(arity) => format!("local_rz {arity}"),
        LocalR(arity) => format!("local_r {arity}"),
        GlobalRz => "global_rz".to_string(),
        GlobalR => "global_r".to_string(),
        Cz => "cz".to_string(),
        Measure(arity) => format!("measure {arity}"),
        AwaitMeasure => "await_measure".to_string(),
        // Always three operands — the native parser requires explicit dim1
        // (use 0 for 1-D arrays), unlike the legacy 1-D shorthand.
        NewArray(type_tag, dim0, dim1) => format!("new_array {type_tag} {dim0} {dim1}"),
        GetItem(ndims) => format!("get_item {ndims}"),
        SetDetector => "set_detector".to_string(),
        SetObservable => "set_observable".to_string(),
        // CPU ops render via vihaco-cpu's own `Display` (e.g. `const.i64 42`,
        // `dup`, `halt`), which the delegated parser reads back.
        Cpu(inner) => inner.to_string(),
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // illustrative sample floats, not math constants
mod tests {
    use super::*;

    fn sample() -> Program {
        use vihaco::value::Value;
        use vihaco_cpu::Instruction as Cpu;
        Program {
            version: Version::new(1, 2),
            instructions: vec![
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
        }
    }

    #[test]
    fn binary_round_trips() {
        let program = sample();
        let bytes = program.to_binary();
        assert_eq!(&bytes[0..MAGIC.len()], MAGIC);
        assert_eq!(
            bytes.len(),
            HEADER_LEN + program.instructions.len() * INSTRUCTION_WIDTH as usize
        );
        assert_eq!(Program::from_binary(&bytes).unwrap(), program);
    }

    #[test]
    fn binary_preserves_version() {
        let bytes = sample().to_binary();
        assert_eq!(
            Program::from_binary(&bytes).unwrap().version,
            Version::new(1, 2)
        );
    }

    #[test]
    fn empty_program_round_trips() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![],
        };
        let bytes = program.to_binary();
        assert_eq!(bytes.len(), HEADER_LEN);
        assert_eq!(Program::from_binary(&bytes).unwrap(), program);
    }

    #[test]
    fn text_round_trips() {
        let program = sample();
        let text = program.to_text();
        assert_eq!(Program::parse_text(&text).unwrap(), program);
    }

    #[test]
    fn text_then_binary_agree() {
        let program = sample();
        let from_text = Program::parse_text(&program.to_text()).unwrap();
        let from_binary = Program::from_binary(&from_text.to_binary()).unwrap();
        assert_eq!(from_binary, program);
    }

    #[test]
    fn parses_comments_and_blanks() {
        use vihaco::value::Value;
        use vihaco_cpu::Instruction as Cpu;
        let source = "\n; header comment\n.version 2\n\nconst.i64 42  ; inline\nhalt\n";
        let program = Program::parse_text(source).unwrap();
        assert_eq!(program.version, Version::new(2, 0));
        assert_eq!(
            program.instructions,
            vec![
                Instruction::Cpu(Cpu::Const(Value::I64(42))),
                Instruction::Cpu(Cpu::Halt)
            ]
        );
    }

    #[test]
    fn version_directive_required() {
        assert_eq!(
            Program::parse_text("halt\n"),
            Err(TextError::MissingVersion)
        );
    }

    #[test]
    fn major_minor_and_bare_versions() {
        assert_eq!(
            Program::parse_text(".version 2.3\nhalt\n").unwrap().version,
            Version::new(2, 3)
        );
        assert_eq!(
            Program::parse_text(".version 5\nhalt\n").unwrap().version,
            Version::new(5, 0)
        );
    }

    #[test]
    fn bad_instruction_reports_line() {
        let err = Program::parse_text(".version 1\nhalt\nnope_nope\n").unwrap_err();
        assert_eq!(
            err,
            TextError::BadInstruction {
                line: 3,
                text: "nope_nope".to_string()
            }
        );
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = sample().to_binary();
        bytes[0] = b'X';
        assert_eq!(Program::from_binary(&bytes), Err(BinaryError::BadMagic));
    }

    #[test]
    fn short_buffer_rejected() {
        assert_eq!(
            Program::from_binary(b"LAN"),
            Err(BinaryError::Truncated {
                expected: HEADER_LEN,
                got: 3
            })
        );
    }

    #[test]
    fn unaligned_code_rejected() {
        let mut bytes = sample().to_binary();
        bytes.push(0); // one stray byte past the last full word
        assert!(matches!(
            Program::from_binary(&bytes),
            Err(BinaryError::UnalignedCode { .. })
        ));
    }
}
