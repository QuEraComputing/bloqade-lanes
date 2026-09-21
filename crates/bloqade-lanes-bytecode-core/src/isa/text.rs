//! Text (`.sst`) codec for the vihaco-backed ISA.
//!
//! A program is vihaco's `sst v1` section container holding one root section:
//! a `.header(root)` block carrying the version, and a `.text(root)` block
//! carrying `fn @main()`. [`parse_text`] shows the full shape.
//!
//! Both halves are vihaco's own grammar. [`SstFile`] frames the container and
//! [`ParsedModule::parse_section`] parses the functions inside it; this module
//! reads the version header itself (see `parse_version_header`) and lowers the
//! parsed surface instructions into a [`Program`] via [`machine::lower`].
//!
//! vihaco ships a reader for this container and no writer, so [`to_text`]
//! emits it via [`super::container::to_sst`].

use chumsky::prelude::*;
use vihaco::SstFile;
use vihaco::syntax::ParsedModule;
use vihaco::traits::FromText as _;
use vihaco_parser::Parse;

use super::container::LanesContext;
use super::machine::{self, MachineSurfaceInstruction};
use super::program::{LanesInfo, Program, from_code};

/// Error from text (`.sst`) parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextError {
    /// The root section has no `version <major>.<minor>` header.
    MissingVersion,
    /// The header is present but its value will not parse — `version 1`,
    /// `version abc`, `version 1.x`. Distinct from [`MissingVersion`], which
    /// it used to be reported as.
    ///
    /// [`MissingVersion`]: TextError::MissingVersion
    InvalidVersion { line: usize, value: String },
    /// A line could not be parsed as an instruction. `line` is currently always
    /// `0` for parse-level failures from the chumsky parser.
    BadInstruction { line: usize, text: String },
}

impl std::fmt::Display for TextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextError::MissingVersion => write!(f, "missing version header"),
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

// ── NoType ────────────────────────────────────────────────────────────────────

/// Source-type syntax for [`ParsedModule`]. A lanes `@main` takes no parameters
/// and returns nothing, so no type ever appears in the grammar; this parser
/// rejects everything, making `params` and `return_ty` unreachable.
#[derive(Debug, Clone, PartialEq)]
pub enum NoType {}

impl<'src> Parse<'src> for NoType {
    fn parser() -> impl chumsky::Parser<
        'src,
        &'src str,
        Self,
        chumsky::extra::Err<chumsky::error::Simple<'src, char>>,
    > {
        chumsky::primitive::empty().try_map(|(), span| Err(chumsky::error::Simple::new(None, span)))
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse vihaco's `sst v1` container into a [`Program`].
///
/// Expected format:
/// ```text
/// sst v1
///
/// .section(root):
/// .header(root):
/// version <major>.<minor>
/// .header(root).
/// .text(root):
/// fn @main() {
///   <instruction>
///   ...
/// }
/// .text(root).
/// .section(root).
/// ```
///
/// Note: `TextError::BadInstruction` errors have `line: 0`; the underlying
/// diagnostics are preserved in `text` even though a precise line number is not
/// yet available.
pub fn parse_text(src: &str) -> Result<Program, TextError> {
    let file = SstFile::<LanesContext>::from_text(src).map_err(|e| TextError::BadInstruction {
        line: 0,
        text: e.to_string(),
    })?;

    // Read the header before handing the section to vihaco. `parse_section`
    // parses it too, but reports every failure as an `eyre` message wrapped
    // around `LanesInfo::from_text`'s own — so telling a *missing* header from
    // a *malformed* one downstream meant matching on prose, and got it wrong:
    // `version 1` and `version abc` were both reported as missing, and
    // `version 1.x` as an unparseable instruction. Here the distinction is
    // still available.
    parse_version_header(src, file.root().header_text())?;

    let parsed =
        ParsedModule::<MachineSurfaceInstruction, NoType, LanesInfo>::parse_section(file.root())
            .map_err(|e| TextError::BadInstruction {
                line: 0,
                text: e.to_string(),
            })?;

    // Exactly one function, and it must be `@main`. The parser strips the
    // leading `@`, so the parsed name is bare `"main"`.
    let func = match parsed.functions.as_slice() {
        [f] => f,
        _ => {
            return Err(TextError::BadInstruction {
                line: 0,
                text: format!(
                    "expected exactly one function (@main), found {}",
                    parsed.functions.len()
                ),
            });
        }
    };
    if func.name.as_str() != "main" {
        return Err(TextError::BadInstruction {
            line: 0,
            text: format!("expected function @main, found @{}", func.name.as_str()),
        });
    }

    // The composite's surface and runtime enums are independent, so each
    // parsed instruction is lowered before it becomes program code.
    let code = func
        .body
        .iter()
        .cloned()
        .map(machine::lower)
        .collect::<eyre::Result<Vec<_>>>()
        .map_err(|e| TextError::BadInstruction {
            line: 0,
            text: e.to_string(),
        })?;

    Ok(from_code(parsed.header.version, code))
}

/// Read the root section's `version` directive.
///
/// The two failures are genuinely different mistakes and get different
/// errors: no `version` line at all, versus one whose value will not parse.
fn parse_version_header(src: &str, header_text: &str) -> Result<LanesInfo, TextError> {
    let trimmed = header_text.trim();
    let Some(value) = trimmed.strip_prefix("version") else {
        return Err(TextError::MissingVersion);
    };
    LanesInfo::from_text(trimmed).map_err(|_| TextError::InvalidVersion {
        line: line_of(src, trimmed),
        value: value.trim().to_owned(),
    })
}

/// 1-based line of `needle` in `src`, or `0` if it cannot be located.
///
/// The header arrives as a detached slice, so its position is recovered by
/// search rather than carried along — good enough to point an author at the
/// right line, and honest about failing (`line 0`) rather than guessing.
fn line_of(src: &str, needle: &str) -> usize {
    src.lines()
        .position(|line| line.trim() == needle)
        .map_or(0, |i| i + 1)
}

/// Emit the program as vihaco's `sst v1` container.
///
/// The output is accepted by [`parse_text`] and round-trips losslessly.
pub fn to_text(program: &Program) -> String {
    let mut body = String::from("fn @main() {\n");
    for inst in &program.code {
        body.push_str("  ");
        body.push_str(&machine::to_sst_text(inst));
        body.push('\n');
    }
    body.push_str("}\n");
    super::container::to_sst(&program.extra, &body)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::device::LanesInstruction as L;
    use crate::isa::machine::MachineInstruction as M;
    use crate::version::Version;
    use vihaco::{Type, Value};
    use vihaco_cpu::RuntimeInstruction as C;

    /// Wrap a `fn @main` body in the `sst v1` container, for tests that care
    /// about the instructions rather than the framing.
    fn sst(version: &str, body: &str) -> String {
        format!(
            "sst v1\n\n.section(root):\n.header(root):\nversion {version}\n.header(root).\n\
             .text(root):\n{body}.text(root).\n.section(root).\n"
        )
    }

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
    }

    #[test]
    fn text_round_trips_fn_main() {
        let src = sst(
            "1.2",
            "fn @main() {\n  lanes::lanes.const_loc 0x0000000000000000\n  lanes::lanes.initial_fill 1\n  cpu::cpu.halt\n}\n",
        );
        let p = parse_text(&src).unwrap();
        assert_eq!(p.extra.version, Version::new(1, 2));
        assert_eq!(p.code.len(), 3);
        assert_eq!(parse_text(&to_text(&p)).unwrap(), p);
    }

    #[test]
    fn text_round_trips_full_sample() {
        let program = sample();
        let text = to_text(&program);
        assert_eq!(parse_text(&text).unwrap(), program);
    }

    #[test]
    fn to_text_emits_the_sst_container() {
        let text = to_text(&sample());
        assert!(text.starts_with("sst v1\n"), "got {text}");
        for expected in [
            ".section(root):",
            ".header(root):",
            "version 1.2",
            ".header(root).",
            ".text(root):",
            "fn @main() {",
            ".text(root).",
            ".section(root).",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }

    #[test]
    fn to_text_is_readable_by_vihacos_own_parser() {
        // We emit the container by hand (vihaco ships no writer), so pin that
        // vihaco reads back what we wrote, not just that we round-trip.
        let text = to_text(&sample());
        let file = SstFile::<LanesContext>::from_text(&text).unwrap();
        let root = file.root();
        assert!(root.path().is_root());
        assert_eq!(root.children().count(), 0);
        assert_eq!(
            root.parse_header::<LanesInfo>().unwrap().version,
            Version::new(1, 2)
        );
        assert!(root.sst().contains("fn @main() {"));
    }

    #[test]
    fn to_text_indents_instructions() {
        let prog = from_code(Version::new(1, 0), vec![M::Cpu(C::Halt)]);
        assert!(to_text(&prog).contains("  cpu::cpu.halt\n"));
    }

    #[test]
    fn empty_program_round_trips() {
        let prog = from_code(Version::new(1, 0), vec![]);
        assert_eq!(parse_text(&to_text(&prog)).unwrap(), prog);
    }

    #[test]
    fn missing_version_header_returns_error() {
        // A well-formed container whose header section is absent.
        let src = "sst v1\n\n.section(root):\n.text(root):\nfn @main() {\n  cpu::cpu.halt\n}\n\
                   .text(root).\n.section(root).\n";
        assert_eq!(parse_text(src), Err(TextError::MissingVersion));
    }

    /// A header that is present but malformed is not a *missing* header.
    ///
    /// All three of these used to be misreported: `version 1` and
    /// `version abc` as `MissingVersion`, and `version 1.x` as an unparseable
    /// *instruction* — because the classification matched on the text of
    /// vihaco's wrapped `eyre` message rather than reading the header.
    #[test]
    fn a_malformed_version_is_not_a_missing_one() {
        for value in ["1", "abc", "1.x", "x.1", "1.2.3", ""] {
            let src = sst(value, "fn @main() {\n  cpu::cpu.halt\n}\n");
            assert_eq!(
                parse_text(&src),
                Err(TextError::InvalidVersion {
                    // `version <value>` is the fifth line of the container.
                    line: 5,
                    value: value.to_owned(),
                }),
                "version {value:?}"
            );
        }
    }

    #[test]
    fn bad_instruction_returns_error() {
        let src = sst("1.0", "fn @main() {\n  lanes::lanes.nope_nope\n}\n");
        assert!(matches!(
            parse_text(&src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn version_preserved() {
        let prog = from_code(Version::new(3, 7), vec![]);
        let text = to_text(&prog);
        assert!(text.contains("version 3.7"));
        assert_eq!(parse_text(&text).unwrap().extra.version, Version::new(3, 7));
    }

    #[test]
    fn text_error_display_strings() {
        assert_eq!(
            TextError::MissingVersion.to_string(),
            "missing version header"
        );
        assert_eq!(
            TextError::InvalidVersion {
                line: 2,
                value: "1.x".into()
            }
            .to_string(),
            "line 2: invalid version '1.x'"
        );
        assert_eq!(
            TextError::BadInstruction {
                line: 3,
                text: "nope".into()
            }
            .to_string(),
            "line 3: cannot parse instruction 'nope'"
        );
    }

    #[test]
    fn multiple_functions_rejected() {
        // The grammar admits several `fn` blocks, but a lanes program is a
        // single flat `@main`; the resolver rejects anything but exactly one.
        let src = sst(
            "1.0",
            "fn @main() {\n  cpu::cpu.halt\n}\nfn @extra() {\n  cpu::cpu.halt\n}\n",
        );
        assert!(matches!(
            parse_text(&src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn non_main_function_rejected() {
        let src = sst("1.0", "fn @extra() {\n  cpu::cpu.halt\n}\n");
        assert!(matches!(
            parse_text(&src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn syntactically_broken_source_is_a_parse_error() {
        // An unterminated function body fails the function grammar.
        let src = sst("1.0", "fn @main() {\n  cpu::cpu.halt\n");
        assert!(matches!(
            parse_text(&src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn broken_container_is_a_parse_error() {
        // Missing the `sst v1` line entirely: the container parser rejects it
        // before any instruction is seen.
        let src = ".section(root):\n.section(root).\n";
        assert!(matches!(
            parse_text(src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn parse_error_preserves_diagnostic_text() {
        let src = sst("1.0", "fn @main() {\n  cpu::cpu.halt\n");
        match parse_text(&src) {
            Err(TextError::BadInstruction { text, .. }) => {
                assert_ne!(text, "parse error");
                assert!(!text.is_empty(), "diagnostic text should be non-empty");
            }
            other => panic!("expected BadInstruction, got {other:?}"),
        }
    }

    #[test]
    fn global_context_must_be_empty() {
        // A lanes program has no child sections, so it carries no global
        // context; a non-empty one is a malformed file rather than ignored.
        let src = "sst v1\n.global:\nsomething\n.global.\n.section(root):\n.section(root).\n";
        assert!(matches!(
            parse_text(src),
            Err(TextError::BadInstruction { .. })
        ));
    }
}
