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

/// The global context of a lanes file: empty.
///
/// The global context exists to resolve child-section *names*, and a lanes
/// program has no child sections — so there is nothing to carry and nothing to
/// resolve. vihaco's own [`vihaco::NoContext`] would say the same thing, but it
/// only implements the SST half of the pair ([`SstGlobalContext`], not
/// [`BytecodeGlobalContext`]), so it cannot be used on the binary path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LanesContext;

impl vihaco::SectionNameResolver for LanesContext {
    fn section_name(&self, _index: u32) -> Option<&str> {
        None
    }
}

impl vihaco::BytecodeGlobalContext for LanesContext {
    fn from_bytes(bytes: &[u8]) -> eyre::Result<Self> {
        if bytes.is_empty() {
            Ok(Self)
        } else {
            Err(eyre::eyre!(
                "a lanes program carries no global context, found {} byte(s)",
                bytes.len()
            ))
        }
    }
}

impl vihaco::SstGlobalContext for LanesContext {
    fn from_text(text: &str) -> eyre::Result<Self> {
        if text.trim().is_empty() {
            Ok(Self)
        } else {
            Err(eyre::eyre!(
                "a lanes program carries no global context, found {:?}",
                text.trim()
            ))
        }
    }
}

/// Byte lengths of the fixed fields in vihaco's binary container. Named here so
/// the emitter's arithmetic reads against the layout in the module docs.
mod len {
    /// `magic` + `version` + `flags` + `context_len`.
    pub const FILE_HEADER: usize = 4 + 2 + 2 + 8;
    /// `section_len` + `composite_header_len`.
    pub const SECTION_FRAME: usize = 8 + 8;
    /// `bytecode_len`.
    pub const BYTECODE_HEADER: usize = 8;
    /// `child_count`.
    pub const CHILD_TABLE_HEADER: usize = 4;
}

/// Serialize a program into vihaco's `VHBC` container.
///
/// Instructions go through [`bytecode::encode`] because neither half of the
/// composite carries a codec of its own.
pub fn to_binary(program: &Program) -> eyre::Result<Vec<u8>> {
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

    let section_len = len::SECTION_FRAME
        + header.len()
        + len::BYTECODE_HEADER
        + code.len()
        + len::CHILD_TABLE_HEADER;

    let mut out = Vec::with_capacity(len::FILE_HEADER + section_len);
    // ── file header ──
    out.extend_from_slice(vihaco::MAGIC);
    out.extend_from_slice(&vihaco::VERSION.to_le_bytes());
    out.extend_from_slice(&vihaco::FLAGS.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // empty global context
    // ── root section ──
    out.extend_from_slice(&(section_len as u64).to_le_bytes());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    out.extend_from_slice(&(code.len() as u64).to_le_bytes());
    out.extend_from_slice(&code);
    out.extend_from_slice(&0u32.to_le_bytes()); // no child sections
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
