//! Flat program container for the vihaco-backed ISA.
//!
//! A program is a [`Version`] plus a `Vec<`[`MachineInstruction`]`>` whose
//! functions delimit themselves with `func_start`/`func_end` —
//! no functions, labels, or string interner (our programs are a single flat
//! instruction list; see
//! <https://github.com/QuEraComputing/bloqade-lanes/issues/769>). vihaco's
//! [`LocalModule`] / loader machinery carries that structured-language
//! support, so we keep a thin container and delegate the per-instruction
//! work to the mirror ISA's derived codec ([`WriteBytes`]/[`FromBytes`], see
//! [`super::bytecode`]) and to the text parser in [`super::text`].
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
use vihaco::module::{FunctionInfo, LocalModule, Signature};
use vihaco::value::{Type, Value};

use super::bytecode::{self, BytecodeInstruction};
use super::container::LanesContext;
use super::machine::MachineInstruction;
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
pub type Program = LocalModule<MachineInstruction, Value, Type, LanesInfo>;

/// Build a `Program` from a version + flat instruction list, wrapped in a
/// single `@main`.
///
/// This is the ONE constructor used by both binary and text loading, so all
/// `Program`s built from the same (version, code) compare equal regardless of
/// source. It declares the `@main` function table entry that rendering and
/// execution both need — a `Program` with code but no functions would emit
/// nothing and have no entry point.
///
/// Programs with several functions or with labels come from
/// [`super::resolve::resolve`] instead; this is the flat case.
#[allow(clippy::field_reassign_with_default)] // `LocalModule` is a foreign type; struct-literal init is not possible
pub fn from_code(version: Version, code: Vec<MachineInstruction>) -> eyre::Result<Program> {
    // The caller supplies a function *body*; the markers are this function's
    // job to add. Accepting a list that already has them wrapped it twice —
    // nested `fn` blocks with one closing brace, unparseable text, and a
    // binary that does not compare equal to what produced it. The migration
    // guide documents the markers as visible in `Program.instructions`, so
    // `Program(v, list(p.instructions))` is the obvious thing to try; it now
    // says what to do instead.
    if code.iter().any(|i| {
        matches!(
            i,
            MachineInstruction::Cpu(
                vihaco_cpu::RuntimeInstruction::FunctionStart
                    | vihaco_cpu::RuntimeInstruction::FunctionEnd
            )
        )
    }) {
        eyre::bail!(
            "instruction list already carries func_start/func_end; these delimit a \
             function and are added here. To copy a program, use from_binary/to_binary \
             or from_text/to_text."
        );
    }

    // Wrap the body in the function markers, so a program built from a bare
    // instruction list is in the same shape as one resolved from text. The
    // format has one layout, not two: a reader can rely on `func_start`
    // delimiting every function because there is no way to build a program
    // without it.
    let mut wrapped = Vec::with_capacity(code.len() + 2);
    wrapped.push(MachineInstruction::Cpu(
        vihaco_cpu::RuntimeInstruction::FunctionStart,
    ));
    wrapped.extend(code);
    wrapped.push(MachineInstruction::Cpu(
        vihaco_cpu::RuntimeInstruction::FunctionEnd,
    ));
    let code = wrapped;

    let end_address = code.len() as u32;
    let mut m = Program::default();
    m.code = code;
    m.strings = vec!["main".to_owned()];
    m.functions = vec![FunctionInfo {
        name: 0,
        signature: Signature {
            params: Vec::new(),
            ret: Vec::new(),
        },
        local_count: 0,
        start_address: 0,
        end_address,
        file: 0,
    }];
    m.main_function = Some(0);
    m.extra = LanesInfo { version };
    Ok(m)
}

/// Error from binary (de)serialization.
///
/// The variants are unchanged from the pre-container format because each maps to
/// a Python exception class; what changed is where they come from. vihaco's
/// container parser reports failures as `eyre::Report`, so `classify` sorts
/// those messages back into these variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryError {
    /// The file did not start with vihaco's `VHBC` magic.
    BadMagic,
    /// Buffer ended before a complete header, section or instruction word.
    Truncated { expected: usize, got: usize },
    /// The code region length is not a multiple of the instruction width.
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
                "code length {len} is not a multiple of {}",
                bytecode::instruction_width()
            ),
            BinaryError::Decode { pc, message } => {
                write!(f, "decode error at instruction {pc}: {message}")
            }
        }
    }
}

impl std::error::Error for BinaryError {}

/// Length of vihaco's fixed file header, taken from the writer's own layout so
/// the reader's truncation guard cannot drift away from what is emitted.
use super::container::len::FILE_HEADER as FILE_HEADER_LEN;

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
///
/// Fails only if an instruction has no encodable form — today just a runtime
/// label, whose identifier is meaningless outside its parse.
pub fn to_binary(program: &Program) -> Result<Vec<u8>, BinaryError> {
    super::container::to_binary(program).map_err(|e| BinaryError::Decode {
        pc: 0,
        message: e.to_string(),
    })
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
        .decode_instructions::<BytecodeInstruction>()
        .map_err(|e| classify(&e))?
        .into_iter()
        .map(bytecode::decode)
        .collect();

    // Built directly rather than through `from_code`, which wraps its input
    // in the function markers — the decoded stream already carries them.
    // `reconcile_function_spans` then derives the table from those markers.
    // `LocalModule` is a foreign type, so it is built field by field.
    let mut program = Program {
        code,
        extra: LanesInfo {
            version: info.version,
        },
        ..Default::default()
    };
    super::container::read_tables(&root, &mut program).map_err(|e| classify(&e))?;
    reconcile_function_spans(&mut program);
    if program.functions.is_empty() {
        return Err(BinaryError::Decode {
            pc: 0,
            message: "no func_start/func_end markers: this container predates the \
                      function layout and cannot be loaded. Re-assemble it from source."
                .to_owned(),
        });
    }
    resolve_entry_point(&mut program)?;
    Ok(program)
}

/// Re-derive every function's span from the `func_start`/`func_end` markers in
/// the code, keeping only the names the table supplied.
///
/// The code stream is the authority on where a function begins and ends; the
/// table is an index over it. Trusting the decoded spans instead let a
/// well-formed container carry a table that disagreed with its own code — an
/// `end_address` past the end of `code` panicked the disassembler, and a span
/// that under-covered the code made instructions disappear from the rendered
/// text with no error. Re-deriving makes both unrepresentable.
///
/// A stream with no markers is left alone: that is the flat single-`@main`
/// shape [`from_code`] builds, and its span is derived from the code already.
fn reconcile_function_spans(program: &mut Program) {
    use super::machine::MachineInstruction as M;
    use vihaco_cpu::RuntimeInstruction as C;

    let mut spans: Vec<(u32, u32)> = Vec::new();
    let mut open: Option<u32> = None;
    for (address, inst) in program.code.iter().enumerate() {
        match inst {
            M::Cpu(C::FunctionStart) => open = Some(address as u32),
            M::Cpu(C::FunctionEnd) => {
                if let Some(start) = open.take() {
                    spans.push((start, address as u32 + 1));
                }
            }
            _ => {}
        }
    }
    // An unterminated final function still owns the rest of the code.
    if let Some(start) = open {
        spans.push((start, program.code.len() as u32));
    }

    // Names come from the table, matched on `start_address` — the field the
    // emitter already writes and the one thing the table and the code agree
    // about. Matching positionally assumed the table was in layout order:
    // a well-formed table listing `[helper@3, main@0]` silently renamed both,
    // and `LanesMachine::run` then entered the wrong function reporting
    // success. A short table left `name: u32::MAX`, which renders as `fn @?()`
    // and re-parses as a duplicate once two functions have it.
    let named = std::mem::take(&mut program.functions);
    program.functions = spans
        .into_iter()
        .map(|(start_address, end_address)| {
            let source = named.iter().find(|f| f.start_address == start_address);
            FunctionInfo {
                name: source.map_or(u32::MAX, |f| f.name),
                signature: Signature {
                    params: Vec::new(),
                    ret: Vec::new(),
                },
                local_count: source.map_or(0, |f| f.local_count),
                start_address,
                end_address,
                file: source.map_or(0, |f| f.file),
            }
        })
        .collect();
}

/// Point `main_function` at the function actually named `main`.
///
/// The container records the function table but not which entry is the entry
/// point, so `from_code`'s default of `Some(0)` survived `read_tables` and was
/// right only when `@main` happened to be laid out first. A program whose text
/// declared `@helper` before `@main` came back reporting function 0 as the
/// entry — the same program, unequal across a round-trip.
///
/// Recovered by name rather than recorded as a new field: the name is already
/// in the string table, `super::resolve` derives it the same way from text,
/// and deriving it in both places is what makes the two agree. (PPVM does
/// serialize the index, because a PPVM module may legitimately have no `main`;
/// `resolve` makes ours mandatory, so there is no absent case to encode.)
fn resolve_entry_point(program: &mut Program) -> Result<(), BinaryError> {
    let main = program.functions.iter().position(|f| {
        program
            .strings
            .get(f.name as usize)
            .is_some_and(|name| name == "main")
    });
    match main {
        Some(index) => {
            program.main_function = Some(index as u32);
            Ok(())
        }
        None => Err(BinaryError::Decode {
            pc: 0,
            message: "function table declares no @main".to_owned(),
        }),
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // illustrative sample floats, not math constants
mod tests {
    use super::*;
    use crate::isa::device::LanesInstruction as L;
    use crate::isa::machine::MachineInstruction as M;
    use vihaco_cpu::RuntimeInstruction as C;

    fn sample() -> Program {
        from_code(
            Version::new(1, 2),
            vec![
                M::Cpu(C::Const(Type::F64, Value::F64(1.5))),
                M::Cpu(C::Const(Type::I64, Value::I64(-42))),
                M::Cpu(C::Dup),
                M::Lanes(L::ConstLoc(0x0000_0000_0100_0000)),
                M::Lanes(L::ConstLane(0x0000_0000_0000_0001)),
                M::Lanes(L::ConstZone(0x0000_0003)),
                M::Lanes(L::InitialFill(2)),
                M::Lanes(L::Move(1)),
                M::Lanes(L::LocalRz(1)),
                M::Lanes(L::LocalR(3)),
                M::Lanes(L::GlobalRz),
                M::Lanes(L::Cz),
                M::Lanes(L::Measure(1)),
                M::Lanes(L::AwaitMeasure),
                M::Lanes(L::NewArray(2, 10, 20)),
                M::Lanes(L::GetItem(2)),
                M::Lanes(L::SetDetector),
                M::Cpu(C::Halt),
                M::Cpu(C::Return(0)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn binary_round_trips() {
        let program = sample();
        let bytes = to_binary(&program).unwrap();
        assert_eq!(&bytes[0..vihaco::MAGIC.len()], vihaco::MAGIC);
        assert_eq!(from_binary(&bytes).unwrap(), program);
    }

    #[test]
    fn binary_is_readable_by_vihacos_own_parser() {
        // We emit the container by hand (vihaco ships no writer), so the thing
        // worth pinning is that vihaco can read back what we wrote — not that
        // our writer agrees with our reader.
        let program = sample();
        let file =
            vihaco::BytecodeFile::<LanesContext>::from_bytes(to_binary(&program).unwrap()).unwrap();
        let root = file.root();
        assert!(root.path().is_root());
        // The symbol tables ride along as named child sections.
        let mut names: Vec<String> = root
            .children()
            .filter_map(|c| c.local_name().map(str::to_owned))
            .collect();
        names.sort_unstable();
        assert_eq!(names, ["functions", "labels", "strings"]);
        assert_eq!(
            root.decode_instructions::<BytecodeInstruction>()
                .unwrap()
                .into_iter()
                .map(bytecode::decode)
                .collect::<Vec<_>>(),
            program.code
        );
        assert_eq!(
            root.decode_header::<LanesInfo>().unwrap().version,
            program.extra.version
        );
    }

    #[test]
    fn binary_preserves_functions_labels_and_names() {
        // The point of the child sections: a program's symbol tables have to
        // survive a binary round-trip, or function and label names are lost and
        // the disassembly is unreadable.
        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n\
                   .text(root):\n\
                   fn @main() {\n  cpu::cpu.call 0, helper\n  cpu::cpu.br @done\n  \
                   lanes::lanes.cz\n  cpu::cpu.label @done\n  cpu::cpu.halt\n}\n\
                   fn @helper() {\n  cpu::cpu.ret 0\n}\n\
                   .text(root).\n.section(root).\n";
        let original = crate::isa::text::parse_text(src).unwrap();
        let restored = from_binary(&to_binary(&original).unwrap()).unwrap();

        assert_eq!(restored, original);
        assert_eq!(restored.functions.len(), 2);
        assert_eq!(restored.labels.len(), 1);
        assert_eq!(restored.strings, original.strings);
        assert_eq!(restored.main_function, Some(0));

        // And the names come back, so the text form is identical.
        assert_eq!(
            crate::isa::text::to_text(&restored),
            crate::isa::text::to_text(&original)
        );
    }

    /// The entry point survives the binary, which it did not.
    ///
    /// The container records the function table but not which entry is main,
    /// so `from_code`'s `Some(0)` default came back regardless of what the
    /// text declared — making the same program compare unequal across a
    /// round-trip, since `LocalModule`'s `PartialEq` covers the field.
    /// The markers delimit a function; a caller supplies the body.
    ///
    /// The migration guide documents them as visible in
    /// `Program.instructions`, so feeding that list back is the obvious thing
    /// to try — and it used to wrap them twice, producing nested `fn` blocks.
    #[test]
    fn from_code_rejects_a_body_that_already_has_markers() {
        let p = from_code(Version::new(1, 0), vec![M::Cpu(C::Halt)]).unwrap();
        let err = from_code(Version::new(1, 0), p.code.clone())
            .expect_err("a wrapped body should be refused")
            .to_string();
        assert!(err.contains("already carries"), "got {err}");
    }

    /// Names are matched to spans by address, not by position.
    ///
    /// A well-formed table not in layout order used to rename every function,
    /// and `run` then entered the wrong one reporting success.
    #[test]
    fn a_table_out_of_layout_order_still_names_the_right_spans() {
        use crate::isa::text::parse_text;

        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n                   .text(root):\nfn @helper() {\n  cpu::cpu.ret 0\n}\n                   fn @main() {\n  cpu::cpu.halt\n}\n.text(root).\n.section(root).\n";
        let mut p = parse_text(src).unwrap();
        p.functions.reverse();

        let back = from_binary(&to_binary(&p).unwrap()).unwrap();
        let name_of = |i: usize| back.strings[back.functions[i].name as usize].as_str();
        assert_eq!((name_of(0), name_of(1)), ("helper", "main"));
        assert_eq!(back.main_function, Some(1), "@main is the second span");
    }

    /// A container predating the function markers is refused, not guessed at.
    #[test]
    fn a_binary_without_markers_is_rejected() {
        let mut p = from_code(Version::new(1, 0), vec![M::Cpu(C::Halt)]).unwrap();
        p.code
            .retain(|i| !matches!(i, M::Cpu(C::FunctionStart) | M::Cpu(C::FunctionEnd)));
        let err = from_binary(&to_binary(&p).unwrap()).unwrap_err();
        assert!(
            matches!(&err, BinaryError::Decode { message, .. } if message.contains("predates")),
            "got {err:?}"
        );
    }

    #[test]
    fn binary_preserves_the_entry_point() {
        use crate::isa::text::parse_text;

        let from_text = parse_text(
            "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n             .text(root):\nfn @helper() {\n  cpu::cpu.ret 0\n}\n             fn @main() {\n  cpu::cpu.halt\n}\n.text(root).\n.section(root).\n",
        )
        .unwrap();
        assert_eq!(from_text.main_function, Some(1), "@main is declared second");

        let restored = from_binary(&to_binary(&from_text).unwrap()).unwrap();
        assert_eq!(restored.main_function, Some(1));
        assert_eq!(restored, from_text, "the round-trip should be lossless");
    }

    /// A function table with no `@main` is a malformed executable.
    ///
    /// `resolve` makes `@main` mandatory on the text path; a hand-crafted
    /// container could otherwise load without one and be executed from
    /// wherever address 0 happened to land.
    #[test]
    fn a_binary_without_main_is_rejected() {
        let mut program = from_code(Version::new(1, 0), vec![M::Cpu(C::Halt)]).unwrap();
        program.strings = vec!["helper".to_owned()];

        let err = from_binary(&to_binary(&program).unwrap()).unwrap_err();
        assert!(
            matches!(&err, BinaryError::Decode { message, .. } if message.contains("no @main")),
            "got {err:?}"
        );
    }

    #[test]
    fn binary_preserves_version() {
        let bytes = to_binary(&sample()).unwrap();
        assert_eq!(
            from_binary(&bytes).unwrap().extra.version,
            Version::new(1, 2)
        );
    }

    #[test]
    fn empty_program_round_trips() {
        let program = from_code(Version::new(1, 0), vec![]).unwrap();
        let bytes = to_binary(&program).unwrap();
        assert_eq!(from_binary(&bytes).unwrap(), program);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut bytes = to_binary(&sample()).unwrap();
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
        let mut bytes = to_binary(&sample()).unwrap();
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
        // so the container stays well-formed. The only fault is that the region
        // is no longer a whole number of instruction words.
        //
        // Offsets are read out of the file rather than hard-coded, so this keeps
        // working when the layout around them changes.
        let mut bytes =
            to_binary(&from_code(Version::new(1, 0), vec![M::Cpu(C::Halt)]).unwrap()).unwrap();

        let read_u64 =
            |b: &[u8], at: usize| u64::from_le_bytes(b[at..at + 8].try_into().unwrap()) as usize;
        let context_len = read_u64(&bytes, 8);
        let section = FILE_HEADER_LEN + context_len;
        let composite_header_len = read_u64(&bytes, section + 8);
        let bytecode_len_at = section + 16 + composite_header_len;
        let bytecode_at = bytecode_len_at + 8;
        let code_len = read_u64(&bytes, bytecode_len_at);

        let bump = |buf: &mut Vec<u8>, at: usize| {
            let v = read_u64(buf, at) + 1;
            buf[at..at + 8].copy_from_slice(&(v as u64).to_le_bytes());
        };
        bump(&mut bytes, section); // section_len
        bump(&mut bytes, bytecode_len_at);
        bytes.insert(bytecode_at + code_len, 0);

        // The inserted byte shifts everything after the code, so the child
        // sections' recorded offsets move with it. Without this the container
        // rejects the file for a structural reason and never reaches the
        // alignment check.
        let child_table = bytecode_at + code_len + 1;
        let child_count =
            u32::from_le_bytes(bytes[child_table..child_table + 4].try_into().unwrap()) as usize;
        for i in 0..child_count {
            bump(&mut bytes, child_table + 4 + i * 12 + 4);
        }

        assert!(
            matches!(from_binary(&bytes), Err(BinaryError::UnalignedCode { len }) if len
                == code_len + 1),
            "got {:?}",
            from_binary(&bytes)
        );
    }

    #[test]
    fn decode_error_on_bad_opcode() {
        // A well-formed container holding one aligned word whose opcode byte
        // (0xFF) names no instruction: every length check passes, so the
        // failure must come from per-word decoding.
        let good =
            to_binary(&from_code(Version::new(1, 0), vec![M::Cpu(C::Halt)]).unwrap()).unwrap();
        let mut bytes = good.clone();
        let last_word = bytes.len() - 4 - bytecode::instruction_width() as usize;
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
            format!(
                "code length 5 is not a multiple of {}",
                bytecode::instruction_width()
            )
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
