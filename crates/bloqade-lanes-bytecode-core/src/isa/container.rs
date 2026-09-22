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
//! A lanes program is a single flat `@main`, so both emitters write the minimal
//! well-formed tree: an empty global context ([`vihaco::NoContext`]), one root
//! section named `root`, no child sections. The [`LanesInfo`] version lives in
//! that section's header.
//!
//! ## Binary layout (vihaco `VHBC`)
//!
//! ```text
//! magic                : 4 bytes = b"VHBC"
//! version              : u16 LE  = 1
//! flags                : u16 LE  = 0
//! context_len          : u64 LE  = 0        (empty global context)
//! ── root section ──
//! section_len          : u64 LE             (total, including this frame)
//! composite_header_len : u64 LE
//! composite header     : the LanesInfo header
//! bytecode_len         : u64 LE
//! bytecode             : N × INSTRUCTION_WIDTH bytes
//! child_count          : u32 LE  = 0        (no child sections)
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

/// Names of the child sections carrying a program's symbol tables, in the order
/// they are emitted.
pub const TABLE_SECTIONS: [&str; 3] = ["functions", "labels", "strings"];

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

fn decode_functions(bytes: &[u8]) -> eyre::Result<Vec<vihaco::module::FunctionInfo<vihaco::Type>>> {
    const RECORD: usize = 4 * 5;
    let count = read_u32(bytes, 0)? as usize;
    let mut out = Vec::with_capacity(table_capacity(bytes, count, RECORD));
    for i in 0..count {
        let at = 4 + i * RECORD;
        out.push(vihaco::module::FunctionInfo {
            name: read_u32(bytes, at)?,
            signature: vihaco::module::Signature {
                params: Vec::new(),
                ret: Vec::new(),
            },
            local_count: read_u32(bytes, at + 4)?,
            start_address: read_u32(bytes, at + 8)?,
            end_address: read_u32(bytes, at + 12)?,
            file: read_u32(bytes, at + 16)?,
        });
    }
    Ok(out)
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
/// A section that is absent leaves its table empty, so a program written before
/// the tables existed still loads.
pub fn read_tables(
    root: &vihaco::BytecodeSectionView<'_, LanesContext>,
    program: &mut Program,
) -> eyre::Result<()> {
    for child in root.children() {
        match child.local_name() {
            Some("functions") => program.functions = decode_functions(child.bytecode())?,
            Some("labels") => program.labels = decode_labels(child.bytecode())?,
            Some("strings") => program.strings = decode_strings(child.bytecode())?,
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
            name: "functions",
            payload: encode_functions(program),
        },
        ChildSection {
            name: "labels",
            payload: encode_labels(program),
        },
        ChildSection {
            name: "strings",
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
