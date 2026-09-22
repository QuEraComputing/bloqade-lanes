//! Writers for vihaco's section-based container formats.
//!
//! vihaco 0.4 ships **readers only** — [`vihaco::BytecodeFile::from_bytes`] and
//! [`vihaco::SstFile::from_text`] parse its container, but nothing emits one. So
//! we own the two emitters here, matching vihaco's layout byte for byte.
//!
//! The round-trip tests are what keep these honest: everything written here is
//! read back through vihaco's own parser, so a layout mistake fails the build
//! rather than producing a file only we can read.
//!
//! ## Shape of a lanes program
//!
//! Both emitters write one root section named `root`, whose header carries the
//! [`LanesInfo`] version and whose body is the code stream. A program may
//! declare any number of functions; they delimit themselves in that stream with
//! `func_start`/`func_end`, so the root section needs no per-function framing.
//!
//! The two containers differ in how the symbol tables ride along. The binary
//! form appends them as child sections ([`TABLE_SECTIONS`]), whose names resolve
//! through the global context — which is why [`LanesContext`] is not empty and
//! [`vihaco::NoContext`] will not do. The text form needs none of it: functions
//! and labels are written syntactically (`fn @name`, `cpu::cpu.label @x`), so
//! `.sst` has no child sections and an empty `.global:` block.
//!
//! ## Binary layout (vihaco `VHBC`)
//!
//! ```text
//! magic                : 4 bytes = b"VHBC"
//! version              : u16 LE  = 1
//! flags                : u16 LE  = 0
//! context_len          : u64 LE             (child-section name table)
//! ── root section ──
//! section_len          : u64 LE             (total, including this frame)
//! composite_header_len : u64 LE
//! composite header     : the LanesInfo header
//! bytecode_len         : u64 LE
//! bytecode             : N × INSTRUCTION_WIDTH bytes
//! child_count          : u32 LE             (functions, labels, strings)
//! ```
//!
//! ## Text layout (vihaco `sst v1`)
//!
//! ```text
//! sst v1
//!
//! .section(root):
//! .header(root):
//! version 1.0
//! .header(root).
//! .text(root):
//! fn @main() {
//!   …
//! }
//! .text(root).
//! .section(root).
//! ```
//!
//! The global section is omitted entirely, which vihaco reads as an empty
//! context.

use vihaco::instruction::WriteBytes;

use super::bytecode;
use super::program::{LanesInfo, Program};

/// The one section name vihaco accepts for a file's root.
pub const ROOT_SECTION: &str = "root";

/// Child section carrying [`Program::functions`].
pub const FUNCTIONS_SECTION: &str = "functions";
/// Child section carrying [`Program::labels`].
pub const LABELS_SECTION: &str = "labels";
/// Child section carrying [`Program::strings`].
pub const STRINGS_SECTION: &str = "strings";

/// Names of the child sections carrying a program's symbol tables, in the order
/// they are emitted.
///
/// The single source for all three uses — the context name table, the writer in
/// [`to_binary`], and the reader in [`read_tables`] — so adding a table cannot
/// leave one of them spelling a name the others do not know.
/// `emitted_sections_are_exactly_the_declared_tables` pins that.
pub const TABLE_SECTIONS: [&str; 3] = [FUNCTIONS_SECTION, LABELS_SECTION, STRINGS_SECTION];

/// The global context of a lanes file: the child-section name table.
///
/// vihaco's binary child entries store a section's name as an *index*, resolved
/// through the global context — so the moment a program has child sections, the
/// context stops being empty. Ours carries exactly [`TABLE_SECTIONS`].
///
/// The text container needs none of this: functions and labels are written
/// syntactically (`fn @name`, `cpu::cpu.label @x`), so `.sst` still has no child
/// sections and an empty `.global:` block.
///
/// vihaco's own [`vihaco::NoContext`] cannot be used either way: it implements
/// only the SST half of the pair, not [`vihaco::BytecodeGlobalContext`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LanesContext {
    names: Vec<String>,
}

impl LanesContext {
    /// The context a binary program carries.
    pub fn with_tables() -> Self {
        Self {
            names: TABLE_SECTIONS.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    /// Serialized form: one name per line, matching the
    /// [`BytecodeGlobalContext::from_bytes`](vihaco::BytecodeGlobalContext::from_bytes)
    /// impl below.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.names.join("\n").into_bytes()
    }

    /// The index a child-section name is stored under.
    pub fn index_of(&self, name: &str) -> Option<u32> {
        self.names.iter().position(|n| n == name).map(|i| i as u32)
    }
}

impl vihaco::SectionNameResolver for LanesContext {
    fn section_name(&self, index: u32) -> Option<&str> {
        self.names.get(index as usize).map(String::as_str)
    }
}

impl vihaco::BytecodeGlobalContext for LanesContext {
    fn from_bytes(bytes: &[u8]) -> eyre::Result<Self> {
        let text = std::str::from_utf8(bytes)?;
        Ok(Self {
            names: text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(ToOwned::to_owned)
                .collect(),
        })
    }
}

impl vihaco::SstGlobalContext for LanesContext {
    fn from_text(text: &str) -> eyre::Result<Self> {
        if text.trim().is_empty() {
            Ok(Self::default())
        } else {
            Err(eyre::eyre!(
                "a lanes `.sst` program carries no global context, found {:?}",
                text.trim()
            ))
        }
    }
}

/// Byte lengths of the fixed fields in vihaco's binary container. Named here so
/// the emitter's arithmetic reads against the layout in the module docs.
pub(super) mod len {
    /// `magic` + `version` + `flags` + `context_len`.
    ///
    /// The reader's truncation guard in [`super::super::program::from_binary`]
    /// uses this too: a buffer shorter than the file header is the one
    /// truncation we measure ourselves, and it has to agree with what the
    /// writer lays down.
    pub const FILE_HEADER: usize = 4 + 2 + 2 + 8;
    /// `section_len` + `composite_header_len`.
    pub const SECTION_FRAME: usize = 8 + 8;
    /// `bytecode_len`.
    pub const BYTECODE_HEADER: usize = 8;
    /// `child_count`.
    pub const CHILD_TABLE_HEADER: usize = 4;
    /// `local_name_string` + `section_offset`, per child.
    pub const CHILD_TABLE_ENTRY: usize = 4 + 8;
}

/// A child section holding one symbol table.
///
/// Child sections are ordinary sections nested inside the root, so each needs
/// the same framing. The table's bytes go in its bytecode region — vihaco only
/// interprets that as instructions if you ask it to.
struct ChildSection {
    name: &'static str,
    payload: Vec<u8>,
}

impl ChildSection {
    /// Total encoded length: frame + empty header + payload + empty child table.
    fn encoded_len(&self) -> usize {
        len::SECTION_FRAME + len::BYTECODE_HEADER + self.payload.len() + len::CHILD_TABLE_HEADER
    }

    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.encoded_len() as u64).to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // no composite header
        out.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out.extend_from_slice(&0u32.to_le_bytes()); // no grandchildren
    }
}

// ── Symbol tables ─────────────────────────────────────────────────────────────
//
// Each table is a `u32` count followed by fixed-size records, little-endian
// throughout, matching the rest of the container. Strings are length-prefixed
// bytes. These live in child sections because the root section has exactly one
// payload slot and the code occupies it.

fn encode_functions(program: &Program) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(program.functions.len() as u32).to_le_bytes());
    for f in &program.functions {
        out.extend_from_slice(&f.name.to_le_bytes());
        out.extend_from_slice(&f.local_count.to_le_bytes());
        out.extend_from_slice(&f.start_address.to_le_bytes());
        out.extend_from_slice(&f.end_address.to_le_bytes());
        out.extend_from_slice(&f.file.to_le_bytes());

        // The signature follows, length-prefixed. A function's arity is
        // declared here and nowhere else: `call <arity>` carries the caller's
        // claim, and this is what that claim is checked against.
        out.extend_from_slice(&(f.signature.params.len() as u32).to_le_bytes());
        for p in &f.signature.params {
            out.extend_from_slice(&p.name.to_le_bytes());
            out.extend_from_slice(&type_code(p.ty).to_le_bytes());
        }
        out.extend_from_slice(&(f.signature.ret.len() as u32).to_le_bytes());
        for ty in &f.signature.ret {
            out.extend_from_slice(&type_code(*ty).to_le_bytes());
        }
    }
    out
}

/// Capacity to reserve for a table of `count` records of `record` bytes each.
///
/// `count` is the first word of the payload, so it is whatever the file says —
/// and a capacity taken straight from it is a denial of service: `u32::MAX`
/// `FunctionInfo`s is 288 GiB, which Linux's allocator refuses and Rust turns
/// into an abort rather than a decode error. (macOS commits lazily and hands
/// the reservation back, so this only shows up on one platform.)
///
/// After the 4-byte count the payload can hold at most `(len - 4) / record`
/// entries, which is both a safe bound and an exact one: a well-formed table
/// still gets its single up-front allocation, and a malformed count is caught
/// by the per-field reads that follow.
fn table_capacity(bytes: &[u8], count: usize, record: usize) -> usize {
    count.min(bytes.len().saturating_sub(4) / record.max(1))
}

/// Smallest a function record can be: the five fixed words plus the two
/// counts, with an empty signature. Only a *lower* bound now that records vary
/// in length, which is all [`table_capacity`] needs it to be — it bounds the
/// reservation, and the decode below still fails honestly on a short payload.
const FUNCTION_RECORD_MIN: usize = 4 * 7;

fn decode_functions(bytes: &[u8]) -> eyre::Result<Vec<vihaco::module::FunctionInfo<vihaco::Type>>> {
    let count = read_u32(bytes, 0)? as usize;
    let mut out = Vec::with_capacity(table_capacity(bytes, count, FUNCTION_RECORD_MIN));

    // A signature is variable-length, so records are walked with a cursor
    // rather than indexed — `4 + i * RECORD` stopped being an address the
    // moment a function could declare parameters.
    let mut at = 4;
    let next = |at: &mut usize| -> eyre::Result<u32> {
        let v = read_u32(bytes, *at)?;
        *at += 4;
        Ok(v)
    };

    for _ in 0..count {
        let name = next(&mut at)?;
        let local_count = next(&mut at)?;
        let start_address = next(&mut at)?;
        let end_address = next(&mut at)?;
        let file = next(&mut at)?;

        let param_count = next(&mut at)? as usize;
        let mut params = Vec::with_capacity(table_capacity(bytes, param_count, 4 * 2));
        for _ in 0..param_count {
            let name = next(&mut at)?;
            params.push(vihaco::module::Parameter {
                name,
                ty: decode_type_code(next(&mut at)?)?,
            });
        }

        let ret_count = next(&mut at)? as usize;
        let mut ret = Vec::with_capacity(table_capacity(bytes, ret_count, 4));
        for _ in 0..ret_count {
            ret.push(decode_type_code(next(&mut at)?)?);
        }

        out.push(vihaco::module::FunctionInfo {
            name,
            signature: vihaco::module::Signature { params, ret },
            local_count,
            start_address,
            end_address,
            file,
        });
    }
    Ok(out)
}

/// Stable wire codes for [`vihaco::Type`].
///
/// Spelled out rather than taken from the enum's declaration order, which is
/// what the *instruction* opcodes do — those are allowed to shift when the
/// instruction set gains a variant, and a type written into a file is not.
/// The match is exhaustive, so a new vihaco type is a compile error here
/// rather than a silently mis-encoded file.
fn type_code(ty: vihaco::Type) -> u32 {
    use vihaco::Type as T;
    match ty {
        T::Undefined => 0,
        T::String => 1,
        T::Bool => 2,
        T::I64 => 3,
        T::U32 => 4,
        T::U64 => 5,
        T::F64 => 6,
        T::FunctionRef => 7,
        T::HeapRef => 8,
    }
}

fn decode_type_code(code: u32) -> eyre::Result<vihaco::Type> {
    use vihaco::Type as T;
    Ok(match code {
        0 => T::Undefined,
        1 => T::String,
        2 => T::Bool,
        3 => T::I64,
        4 => T::U32,
        5 => T::U64,
        6 => T::F64,
        7 => T::FunctionRef,
        8 => T::HeapRef,
        other => eyre::bail!("unknown type code {other} in a function signature"),
    })
}

fn encode_labels(program: &Program) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(program.labels.len() as u32).to_le_bytes());
    for l in &program.labels {
        out.extend_from_slice(&l.address.to_le_bytes());
        out.extend_from_slice(&l.name.to_le_bytes());
    }
    out
}

fn decode_labels(bytes: &[u8]) -> eyre::Result<Vec<vihaco::module::LabelInfo>> {
    const RECORD: usize = 4 * 2;
    let count = read_u32(bytes, 0)? as usize;
    let mut out = Vec::with_capacity(table_capacity(bytes, count, RECORD));
    for i in 0..count {
        let at = 4 + i * RECORD;
        out.push(vihaco::module::LabelInfo {
            address: read_u32(bytes, at)?,
            name: read_u32(bytes, at + 4)?,
        });
    }
    Ok(out)
}

fn encode_strings(program: &Program) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(program.strings.len() as u32).to_le_bytes());
    for s in &program.strings {
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    out
}

fn decode_strings(bytes: &[u8]) -> eyre::Result<Vec<String>> {
    // Entries are variable-length, but each carries a 4-byte length prefix,
    // so four bytes is the smallest an entry can be.
    let count = read_u32(bytes, 0)? as usize;
    let mut out = Vec::with_capacity(table_capacity(bytes, count, 4));
    let mut at = 4;
    for _ in 0..count {
        let len = read_u32(bytes, at)? as usize;
        at += 4;
        let end = at
            .checked_add(len)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| eyre::eyre!("string table entry runs past the section"))?;
        out.push(String::from_utf8(bytes[at..end].to_vec())?);
        at = end;
    }
    Ok(out)
}

fn read_u32(bytes: &[u8], at: usize) -> eyre::Result<u32> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| eyre::eyre!("table section is truncated at byte {at}"))
}

/// Read the symbol tables back out of a parsed file's child sections.
///
/// An absent section leaves its table empty. That is not a compatibility
/// path: a container predating the tables also predates the `func_start` /
/// `func_end` markers, so it carries no functions to name and
/// [`super::program::from_binary`] rejects it outright.
pub fn read_tables(
    root: &vihaco::BytecodeSectionView<'_, LanesContext>,
    program: &mut Program,
) -> eyre::Result<()> {
    for child in root.children() {
        match child.local_name() {
            Some(FUNCTIONS_SECTION) => program.functions = decode_functions(child.bytecode())?,
            Some(LABELS_SECTION) => program.labels = decode_labels(child.bytecode())?,
            Some(STRINGS_SECTION) => program.strings = decode_strings(child.bytecode())?,
            Some(other) => return Err(eyre::eyre!("unknown child section `{other}`")),
            None => return Err(eyre::eyre!("child section has no name")),
        }
    }
    Ok(())
}

/// Serialize a program into vihaco's `VHBC` container.
///
/// Instructions go through [`bytecode::encode`] because neither half of the
/// composite carries a codec of its own. The symbol tables follow as child
/// sections, whose offsets are relative to the start of the root section.
pub fn to_binary(program: &Program) -> eyre::Result<Vec<u8>> {
    // Two fields of `LocalModule` still have no place in the container, and
    // neither is ever populated. Refusing to write a program that uses one
    // turns a future silent drop into a loud failure.
    //
    // `FunctionInfo.signature` used to be the third, with a note that it was
    // "the one to watch, since the moment the calling convention records
    // parameters, `encode_functions` would discard them". That moment came:
    // signatures are now declared, encoded below, and read back — so the
    // refusal is gone rather than the field being dropped.
    if !program.constants.is_empty() {
        eyre::bail!("the container cannot carry a constant pool yet");
    }
    if !program.source_symbols.is_empty() {
        eyre::bail!("the container cannot carry source symbols yet");
    }

    let context = LanesContext::with_tables();
    let context_bytes = context.to_bytes();

    let mut header = Vec::new();
    program
        .extra
        .write_bytes(&mut header)
        .expect("writing a header to a Vec cannot fail");

    let mut code = Vec::new();
    for inst in &program.code {
        bytecode::encode(inst)?
            .write_bytes(&mut code)
            .expect("writing instruction bytes to a Vec cannot fail");
    }

    let children = [
        ChildSection {
            name: FUNCTIONS_SECTION,
            payload: encode_functions(program),
        },
        ChildSection {
            name: LABELS_SECTION,
            payload: encode_labels(program),
        },
        ChildSection {
            name: STRINGS_SECTION,
            payload: encode_strings(program),
        },
    ];

    // The root's own extent, before the children are appended.
    let root_prefix = len::SECTION_FRAME
        + header.len()
        + len::BYTECODE_HEADER
        + code.len()
        + len::CHILD_TABLE_HEADER
        + children.len() * len::CHILD_TABLE_ENTRY;
    let section_len = root_prefix
        + children
            .iter()
            .map(ChildSection::encoded_len)
            .sum::<usize>();

    let mut out = Vec::with_capacity(len::FILE_HEADER + context_bytes.len() + section_len);
    // ── file header ──
    out.extend_from_slice(vihaco::MAGIC);
    out.extend_from_slice(&vihaco::VERSION.to_le_bytes());
    out.extend_from_slice(&vihaco::FLAGS.to_le_bytes());
    out.extend_from_slice(&(context_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&context_bytes);
    // ── root section ──
    out.extend_from_slice(&(section_len as u64).to_le_bytes());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&(code.len() as u64).to_le_bytes());
    out.extend_from_slice(&code);
    // ── child table: offsets are relative to the root section's start ──
    out.extend_from_slice(&(children.len() as u32).to_le_bytes());
    let mut offset = root_prefix;
    for child in &children {
        let index = context.index_of(child.name).ok_or_else(|| {
            eyre::eyre!("child section `{}` is missing from the context", child.name)
        })?;
        out.extend_from_slice(&index.to_le_bytes());
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        offset += child.encoded_len();
    }
    for child in &children {
        child.write_to(&mut out);
    }
    Ok(out)
}

/// Emit a program as vihaco's `sst v1` text container.
///
/// `body` is the already-rendered `fn @main() { … }` block.
pub fn to_sst(info: &LanesInfo, body: &str) -> String {
    let mut out = String::from("sst v1\n\n");
    out.push_str(&format!(".section({ROOT_SECTION}):\n"));
    out.push_str(&format!(".header({ROOT_SECTION}):\n"));
    out.push_str(&info.to_string());
    out.push('\n');
    out.push_str(&format!(".header({ROOT_SECTION}).\n"));
    out.push_str(&format!(".text({ROOT_SECTION}):\n"));
    out.push_str(body);
    out.push_str(&format!(".text({ROOT_SECTION}).\n"));
    out.push_str(&format!(".section({ROOT_SECTION}).\n"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fields the container cannot carry are refused, not dropped.
    ///
    /// `constants`, `source_symbols` and `FunctionInfo.signature` are part of
    /// `LocalModule`'s `PartialEq` surface but have no encoding here. Nothing
    /// populates them today, so the round-trip tests pass because they are
    /// empty — not because they survive. This makes the first program that
    /// uses one fail loudly instead of losing it.
    #[test]
    fn a_field_the_container_cannot_carry_is_refused() {
        use crate::isa::program::from_code;
        use crate::version::Version;
        use vihaco::module::{Parameter, Signature, SourceSymbolInfo};
        use vihaco::value::{Type, Value};
        use vihaco_cpu::RuntimeInstruction as C;

        let base = from_code(
            Version::new(1, 0),
            vec![crate::isa::machine::MachineInstruction::Cpu(C::Halt)],
        )
        .unwrap();
        assert!(to_binary(&base).is_ok(), "the baseline should still write");

        let mut p = base.clone();
        p.constants = vec![Value::I64(3)];
        assert!(
            to_binary(&p)
                .unwrap_err()
                .to_string()
                .contains("constant pool")
        );

        let mut p = base.clone();
        p.source_symbols = vec![SourceSymbolInfo {
            name: "x".into(),
            index: 1,
        }];
        assert!(
            to_binary(&p)
                .unwrap_err()
                .to_string()
                .contains("source symbols")
        );

        // `signature` used to be the third refusal, and was the one this test
        // existed to guard: the comment on the bail called it "the one to
        // watch, since the moment the calling convention records parameters,
        // `encode_functions` would discard them". It now carries them, so the
        // assertion is that the field survives rather than that it is refused.
        let mut p = base;
        p.functions[0].signature = Signature {
            params: vec![Parameter {
                name: 0,
                ty: Type::I64,
            }],
            ret: vec![Type::Bool],
        };
        let bytes = to_binary(&p).expect("a declared signature is written, not refused");
        let back = crate::isa::program::from_binary(&bytes).expect("and read back");
        assert_eq!(
            back.functions[0].signature, p.functions[0].signature,
            "the signature should survive the container"
        );
    }

    /// A table's declared count cannot be trusted as a capacity.
    ///
    /// It is the first word of the payload, so a malformed file can claim
    /// `u32::MAX` records — 288 GiB of `FunctionInfo` — which Linux's
    /// allocator refuses and Rust turns into a process abort rather than the
    /// decode error the caller is waiting for. macOS commits lazily and hands
    /// the reservation back, so the bound is asserted here rather than left to
    /// an allocator to object on one platform and not the other.
    #[test]
    fn a_table_count_cannot_reserve_more_than_the_payload_holds() {
        // A four-byte payload holding nothing but the count itself.
        let mut bytes = u32::MAX.to_le_bytes().to_vec();
        assert_eq!(table_capacity(&bytes, u32::MAX as usize, 20), 0);

        // One complete 20-byte record: room for exactly one.
        bytes.extend_from_slice(&[0u8; 20]);
        assert_eq!(table_capacity(&bytes, u32::MAX as usize, 20), 1);

        // A well-formed count still gets its exact capacity.
        assert_eq!(table_capacity(&bytes, 1, 20), 1);
    }

    /// And the decoders reject the malformed count rather than allocating for
    /// it — each reports the truncation the count implies.
    #[test]
    fn a_malformed_table_count_is_a_decode_error() {
        let bytes = u32::MAX.to_le_bytes().to_vec();
        assert!(decode_functions(&bytes).is_err());
        assert!(decode_labels(&bytes).is_err());
        assert!(decode_strings(&bytes).is_err());
    }
}
