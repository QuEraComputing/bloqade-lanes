//! Flat program container for the vihaco-backed ISA.
//!
//! A program is a [`Version`] plus a flat `Vec<`[`Instruction`]`>` — no
//! functions, labels, or string interner (our programs are a single flat
//! instruction list; see <https://github.com/QuEraComputing/bloqade-lanes/issues/769>).
//! vihaco's [`LocalModule`] / loader machinery carries that structured-language
//! support, so we keep a thin container and delegate the per-instruction work to
//! vihaco's derived codec ([`WriteBytes`]/[`FromBytes`]) and to the text parser
//! in [`super::syntax`].
//!
//! ## Container
//!
//! Both serialized forms are vihaco's own section-based containers — binary
//! `VHBC` and text `sst v1` — carrying one root section whose header is
//! [`LanesInfo`]. Reading goes through [`vihaco::BytecodeFile`]; writing goes
//! through [`super::container`], because vihaco ships no emitters. See that
//! module for the byte layout.

use vihaco::BytecodeFile;
use vihaco::instruction::{FromBytes, WriteBytes};
use vihaco::module::LocalModule;
use vihaco::value::{Type, Value};

use super::container::LanesContext;
use super::{INSTRUCTION_WIDTH, Instruction};
use crate::version::Version;

/// The section header of a lanes program: everything the container carries
/// about the program besides its code.
///
/// vihaco's header traits are blanket impls over `FromBytes` / `WriteBytes` /
/// `FromText`, so implementing those three makes this usable as both a
/// [`vihaco::BytecodeHeader`] and a [`vihaco::SstHeader`].
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

/// Binary header: the version packed into a `u32` LE.
impl WriteBytes for LanesInfo {
    fn write_bytes<W: std::io::Write>(&self, io: &mut W) -> eyre::Result<()> {
        let packed: u32 = self.version.into();
        io.write_all(&packed.to_le_bytes())?;
        Ok(())
    }
}

impl FromBytes for LanesInfo {
    fn from_bytes<R: std::io::Read>(bytes: &mut R) -> eyre::Result<Self> {
        let mut buf = [0u8; 4];
        bytes.read_exact(&mut buf)?;
        Ok(Self {
            version: Version::from(u32::from_le_bytes(buf)),
        })
    }
}

/// Text header: the `version <major>.<minor>` directive.
impl vihaco::traits::FromText for LanesInfo {
    fn from_text(text: &str) -> eyre::Result<Self> {
        let rest = text
            .trim()
            .strip_prefix("version")
            .ok_or_else(|| eyre::eyre!("missing version header"))?;
        let (major, minor) = rest
            .trim()
            .split_once('.')
            .ok_or_else(|| eyre::eyre!("expected `version <major>.<minor>`"))?;
        Ok(Self {
            version: Version::new(major.trim().parse()?, minor.trim().parse()?),
        })
    }
}

impl vihaco::SstHeader for LanesInfo {}

/// A Bloqade Lanes program: a vihaco `LocalModule` specialised to our ISA. A
/// single `@main` function's worth of flat code plus the version in `extra`.
pub type Program = LocalModule<Instruction, Value, Type, LanesInfo>;

/// Build a `Program` from a version + flat instruction list. This is the ONE
/// constructor used by both binary and text loading, so all `Program`s built
/// from the same (version, code) compare equal regardless of source.
#[allow(clippy::field_reassign_with_default)] // `LocalModule` is a foreign type; struct-literal init is not possible
pub fn from_code(version: Version, code: Vec<Instruction>) -> Program {
    let mut m = Program::default();
    m.code = code;
    m.extra = LanesInfo { version };
    m
}

/// Error from binary (de)serialization.
///
/// The variants are unchanged from the pre-container format because each maps to
/// a Python exception class; what changed is where they come from. vihaco's
/// container parser reports failures as `eyre::Report`, so
/// [`classify`] sorts those messages back into these variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryError {
    /// The file did not start with vihaco's `VHBC` magic.
    BadMagic,
    /// Buffer ended before a complete header, section or instruction word.
    Truncated { expected: usize, got: usize },
    /// The code region length is not a multiple of [`INSTRUCTION_WIDTH`].
    UnalignedCode { len: usize },
    /// A word held an opcode/payload vihaco could not decode, or the container
    /// was otherwise malformed.
    Decode { pc: usize, message: String },
}

impl std::fmt::Display for BinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryError::BadMagic => write!(f, "bad magic bytes (expected VHBC)"),
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

/// Length of vihaco's fixed file header: magic + version + flags + context_len.
const FILE_HEADER_LEN: usize = 4 + 2 + 2 + 8;

/// Sort a vihaco container/decode error into a [`BinaryError`].
///
/// vihaco reports these as free-form `eyre` messages, so this matches on their
/// text. A message we do not recognise still surfaces in full via
/// [`BinaryError::Decode`] — the classification only decides which Python
/// exception class is raised, never whether the error is reported.
fn classify(err: &eyre::Report) -> BinaryError {
    let msg = err.to_string();
    if msg.contains("invalid bytecode magic") {
        BinaryError::BadMagic
    } else if let Some(len) = unaligned_len(&msg) {
        BinaryError::UnalignedCode { len }
    } else {
        // vihaco's structural complaints (section extends past end, header out
        // of bounds, length mismatch, …) carry no byte counts we could put in
        // `Truncated`, so they surface with their message intact rather than
        // with two invented zeroes. `Truncated` is reserved for the one case we
        // measure ourselves: a buffer too short to hold the file header.
        BinaryError::Decode {
            pc: 0,
            message: msg,
        }
    }
}

/// Recover the offending length from vihaco's
/// "bytecode length {len} is not a multiple of instruction width {w}" message.
fn unaligned_len(msg: &str) -> Option<usize> {
    let rest = msg.strip_prefix("bytecode length ")?;
    if !rest.contains("is not a multiple of instruction width") {
        return None;
    }
    rest.split_whitespace().next()?.parse().ok()
}

/// Serialize into vihaco's `VHBC` container (see [`super::container`]).
pub fn to_binary(program: &Program) -> Vec<u8> {
    super::container::to_binary(program)
}

/// Deserialize from vihaco's `VHBC` container.
pub fn from_binary(bytes: &[u8]) -> Result<Program, BinaryError> {
    // Check the magic up front: vihaco needs a full file header before it can
    // report anything, so a short-but-wrong-magic buffer would otherwise come
    // back as a truncation rather than the more useful `BadMagic`.
    if bytes.len() >= vihaco::MAGIC.len() && &bytes[..vihaco::MAGIC.len()] != vihaco::MAGIC {
        return Err(BinaryError::BadMagic);
    }
    if bytes.len() < FILE_HEADER_LEN {
        return Err(BinaryError::Truncated {
            expected: FILE_HEADER_LEN,
            got: bytes.len(),
        });
    }

    let file =
        BytecodeFile::<LanesContext>::from_bytes(bytes.to_vec()).map_err(|e| classify(&e))?;
    let root = file.root();
    let info: LanesInfo = root.decode_header().map_err(|e| classify(&e))?;
    let code = root
        .decode_instructions::<Instruction>()
        .map_err(|e| classify(&e))?;
    Ok(from_code(info.version, code))
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // illustrative sample floats, not math constants
mod tests {
    use super::*;

    fn sample() -> Program {
        from_code(
            Version::new(1, 2),
            vec![
                Instruction::ConstFloat(1.5),
                Instruction::ConstInt(-42),
                Instruction::Dup,
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
                Instruction::Halt,
                Instruction::Return,
            ],
        )
    }

    #[test]
    fn binary_round_trips() {
        let program = sample();
        let bytes = to_binary(&program);
        assert_eq!(&bytes[0..vihaco::MAGIC.len()], vihaco::MAGIC);
        assert_eq!(from_binary(&bytes).unwrap(), program);
    }

    #[test]
    fn binary_is_readable_by_vihacos_own_parser() {
        // We emit the container by hand (vihaco ships no writer), so the thing
        // worth pinning is that vihaco can read back what we wrote — not that
        // our writer agrees with our reader.
        let program = sample();
        let file = vihaco::BytecodeFile::<LanesContext>::from_bytes(to_binary(&program)).unwrap();
        let root = file.root();
        assert!(root.path().is_root());
        assert_eq!(root.children().count(), 0);
        assert_eq!(
            root.decode_instructions::<Instruction>().unwrap(),
            program.code
        );
        assert_eq!(
            root.decode_header::<LanesInfo>().unwrap().version,
            program.extra.version
        );
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
        // Shorter than the file header, which is the one truncation we measure
        // ourselves, so it carries real byte counts.
        assert_eq!(
            from_binary(b"VH"),
            Err(BinaryError::Truncated {
                expected: FILE_HEADER_LEN,
                got: 2
            })
        );
    }

    #[test]
    fn truncated_section_reports_vihacos_message() {
        // A complete file header, then a section that claims more bytes than
        // the file holds.
        let mut bytes = to_binary(&sample());
        bytes.truncate(bytes.len() - 20);
        // vihaco reports this structurally, with no byte counts to report, so
        // it surfaces as `Decode` carrying vihaco's own message.
        assert!(
            matches!(from_binary(&bytes), Err(BinaryError::Decode { .. })),
            "got {:?}",
            from_binary(&bytes)
        );
    }

    #[test]
    fn unaligned_code_rejected() {
        // Grow the bytecode region by one byte — and the section that holds it,
        // so the container itself stays well-formed. The only thing wrong is
        // that the region is no longer a whole number of instruction words.
        //
        // Offsets follow the layout in `super::super::container`: section_len at
        // 16, composite_header_len at 24, the 4-byte header at 32, bytecode_len
        // at 36, bytecode from 44.
        const SECTION_LEN: usize = 16;
        const BYTECODE_LEN: usize = 36;
        const BYTECODE: usize = 44;

        let mut bytes = to_binary(&from_code(Version::new(1, 0), vec![Instruction::Halt]));
        let bump = |buf: &mut Vec<u8>, at: usize| {
            let v = u64::from_le_bytes(buf[at..at + 8].try_into().unwrap()) + 1;
            buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
        };
        bump(&mut bytes, SECTION_LEN);
        bump(&mut bytes, BYTECODE_LEN);
        bytes.insert(BYTECODE + INSTRUCTION_WIDTH as usize, 0);

        assert!(
            matches!(from_binary(&bytes), Err(BinaryError::UnalignedCode { len }) if len
                == INSTRUCTION_WIDTH as usize + 1),
            "got {:?}",
            from_binary(&bytes)
        );
    }

    #[test]
    fn decode_error_on_bad_opcode() {
        // A well-formed container holding one aligned word whose opcode byte
        // (0xFF) names no instruction: every length check passes, so the
        // failure must come from per-word decoding.
        let good = to_binary(&from_code(Version::new(1, 0), vec![Instruction::Halt; 1]));
        let mut bytes = good.clone();
        let last_word = bytes.len() - 4 - INSTRUCTION_WIDTH as usize;
        bytes[last_word] = 0xFF;
        assert!(
            matches!(from_binary(&bytes), Err(BinaryError::Decode { .. })),
            "got {:?}",
            from_binary(&bytes)
        );
    }

    #[test]
    fn binary_error_display_strings() {
        assert_eq!(
            BinaryError::BadMagic.to_string(),
            "bad magic bytes (expected VHBC)"
        );
        assert_eq!(
            BinaryError::Truncated {
                expected: 9,
                got: 3
            }
            .to_string(),
            "truncated: expected 9 bytes, got 3"
        );
        assert_eq!(
            BinaryError::UnalignedCode { len: 5 }.to_string(),
            format!("code length 5 is not a multiple of {INSTRUCTION_WIDTH}")
        );
        assert_eq!(
            BinaryError::Decode {
                pc: 2,
                message: "boom".into()
            }
            .to_string(),
            "decode error at instruction 2: boom"
        );
    }

    #[test]
    fn lanes_info_display() {
        let info = LanesInfo {
            version: Version::new(1, 4),
        };
        assert_eq!(info.to_string(), "version 1.4");
    }
}
