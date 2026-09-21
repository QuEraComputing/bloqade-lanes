//! Text (`.sst`) codec for the vihaco-backed ISA.
//!
//! The grammar is:
//! ```text
//! version <major>.<minor>;
//! fn @main() {
//!   <instruction>
//!   ...
//! }
//! ```
//!
//! `version_header` parses the `version <major>.<minor>;` directive; the
//! function body is vihaco's own grammar via
//! [`ParsedFunction::parser`](vihaco::syntax::ParsedFunction), and `resolve`
//! lowers the result into a `Program` via [`super::program::from_code`].
//!
//! ## Why not `ParsedModule`
//!
//! vihaco 0.4 reshaped `ParsedModule` around its multi-section `.sst`
//! container: it no longer implements `Parse`, only `parse_section(SstSectionView)`,
//! which requires adopting vihaco's section format wholesale. We keep our own
//! single-section header and compose vihaco's *function* grammar underneath —
//! adopting the vihaco container is tracked separately.

use chumsky::prelude::*;
use vihaco::syntax::ParsedFunction;
use vihaco_parser::Parse;

use super::Instruction;
use super::program::{Program, from_code};
use crate::version::Version;

/// Error from text (`.sst`) parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TextError {
    /// No `version M.N;` header was found before the first instruction.
    MissingVersion,
    /// The version header's value could not be parsed.
    /// Currently unreachable: a malformed version fails the whole parse via `BadInstruction`.
    /// Retained for API stability and potential future use.
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

// ── LanesHeader ──────────────────────────────────────────────────────────────

/// The only header we support: `version <major>.<minor>`.
#[derive(Debug, Clone, PartialEq)]
pub enum LanesHeader {
    Version(Version),
}

impl<'src> Parse<'src> for LanesHeader {
    fn parser() -> impl chumsky::Parser<
        'src,
        &'src str,
        Self,
        chumsky::extra::Err<chumsky::error::Simple<'src, char>>,
    > {
        let uint = || {
            any()
                .filter(|c: &char| c.is_ascii_digit())
                .repeated()
                .at_least(1)
                .collect::<String>()
        };
        just("version")
            .ignore_then(chumsky::text::whitespace())
            .ignore_then(uint().then_ignore(just('.')).then(uint()))
            .try_map(|(maj, min), span| {
                let major = maj
                    .parse::<u16>()
                    .map_err(|_| chumsky::error::Simple::new(None, span))?;
                let minor = min
                    .parse::<u16>()
                    .map_err(|_| chumsky::error::Simple::new(None, span))?;
                Ok(LanesHeader::Version(Version::new(major, minor)))
            })
    }
}

// ── NoType ────────────────────────────────────────────────────────────────────

/// Source-type syntax for [`ParsedFunction`]. A lanes `@main` takes no
/// parameters and returns nothing, so no type ever appears in the grammar; this
/// parser rejects everything, making `params` and `return_ty` unreachable.
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

// ── resolve ───────────────────────────────────────────────────────────────────

/// Lower the parsed header + functions into a [`Program`].
fn resolve(
    version: Version,
    functions: Vec<ParsedFunction<Instruction, NoType>>,
) -> Result<Program, TextError> {
    // Exactly one function, and it must be `@main`. The parser strips the
    // leading `@`, so the parsed name is bare `"main"`.
    let func = match functions.as_slice() {
        [f] => f,
        _ => {
            return Err(TextError::BadInstruction {
                line: 0,
                text: format!(
                    "expected exactly one function (@main), found {}",
                    functions.len()
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

    Ok(from_code(version, func.body.clone()))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse the vihaco `fn @main` grammar into a [`Program`].
///
/// Expected format:
/// ```text
/// version <major>.<minor>;
/// fn @main() {
///   <instruction>
///   ...
/// }
/// ```
///
/// Note: `TextError::BadInstruction` errors have `line: 0` for parse-level
/// failures; the underlying chumsky diagnostics are preserved in `text` even
/// though a precise line number is not yet available.
pub fn parse_text(src: &str) -> Result<Program, TextError> {
    let skip = vihaco::syntax::skip;

    let module = skip()
        .ignore_then(LanesHeader::parser())
        .then_ignore(skip())
        .then_ignore(just(';'))
        .then(
            skip()
                .ignore_then(ParsedFunction::<Instruction, NoType>::parser())
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then_ignore(skip());

    let (LanesHeader::Version(version), functions) =
        module.parse(src).into_result().map_err(|errs| {
            // Distinguish "no header at all" from a malformed one: the former
            // is the common authoring mistake and gets its own error.
            if !src.trim_start().starts_with("version") {
                return TextError::MissingVersion;
            }
            TextError::BadInstruction {
                line: 0,
                text: errs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            }
        })?;

    resolve(version, functions)
}

/// Emit the program as the vihaco `fn @main` grammar.
///
/// The output is accepted by [`parse_text`] and round-trips losslessly.
pub fn to_text(program: &Program) -> String {
    let mut out = format!(
        "version {}.{};\nfn @main() {{\n",
        program.extra.version.major, program.extra.version.minor
    );
    for inst in &program.code {
        out.push_str("  ");
        out.push_str(&inst.to_string());
        out.push('\n');
    }
    out.push_str("}\n");
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
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
    fn text_round_trips_fn_main() {
        let src = "version 1.2;\nfn @main() {\n  lanes.const_loc 0x0000000000000000\n  lanes.initial_fill 1\n  cpu.halt\n}\n";
        let p = parse_text(src).unwrap();
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
    fn to_text_contains_fn_main_header() {
        let text = to_text(&sample());
        assert!(text.starts_with("version 1.2;\n"));
        assert!(text.contains("fn @main() {"));
        assert!(text.ends_with("}\n"));
    }

    #[test]
    fn to_text_indents_instructions() {
        let prog = from_code(Version::new(1, 0), vec![Instruction::Halt]);
        let text = to_text(&prog);
        assert!(text.contains("  cpu.halt\n"));
    }

    #[test]
    fn empty_program_round_trips() {
        let prog = from_code(Version::new(1, 0), vec![]);
        assert_eq!(parse_text(&to_text(&prog)).unwrap(), prog);
    }

    #[test]
    fn missing_version_header_returns_error() {
        let src = "fn @main() {\n  cpu.halt\n}\n";
        assert_eq!(parse_text(src), Err(TextError::MissingVersion));
    }

    #[test]
    fn bad_instruction_returns_error() {
        let src = "version 1.0;\nfn @main() {\n  lanes.nope_nope\n}\n";
        assert!(matches!(
            parse_text(src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn version_preserved() {
        let prog = from_code(Version::new(3, 7), vec![]);
        let text = to_text(&prog);
        assert!(text.starts_with("version 3.7;\n"));
        let reparsed = parse_text(&text).unwrap();
        assert_eq!(reparsed.extra.version, Version::new(3, 7));
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
        let src = "version 1.0;\nfn @main() {\n  cpu.halt\n}\nfn @extra() {\n  cpu.halt\n}\n";
        assert!(matches!(
            parse_text(src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn syntactically_broken_source_is_a_parse_error() {
        // An unterminated function body fails the `ParsedModule` parser itself
        // (before resolution), so `parse_text` maps it to `BadInstruction`.
        let src = "version 1.0;\nfn @main() {\n  cpu.halt\n";
        assert!(matches!(
            parse_text(src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn parse_error_preserves_diagnostic_text() {
        // Parse-level failures surface the underlying chumsky diagnostic rather
        // than a bare "parse error" placeholder, so CLI/Python errors are
        // debuggable.
        let src = "version 1.0;\nfn @main() {\n  cpu.halt\n";
        match parse_text(src) {
            Err(TextError::BadInstruction { text, .. }) => {
                assert_ne!(text, "parse error");
                assert!(!text.is_empty(), "diagnostic text should be non-empty");
            }
            other => panic!("expected BadInstruction, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_version_headers_rejected() {
        // Two `version` directives are ambiguous; the resolver rejects them
        // rather than silently taking the first.
        let src = "version 1.0;\nversion 2.0;\nfn @main() {\n  cpu.halt\n}\n";
        assert!(matches!(
            parse_text(src),
            Err(TextError::BadInstruction { .. })
        ));
    }

    #[test]
    fn non_main_function_rejected() {
        // The single function must be named `@main`.
        let src = "version 1.0;\nfn @extra() {\n  cpu.halt\n}\n";
        assert!(matches!(
            parse_text(src),
            Err(TextError::BadInstruction { .. })
        ));
    }
}
