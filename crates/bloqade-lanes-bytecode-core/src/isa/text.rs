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

use vihaco::SstFile;
use vihaco::syntax::ParsedModule;
use vihaco::traits::FromText as _;
use vihaco_cpu::SurfaceType;

use super::container::LanesContext;
use super::machine::{self, MachineInstruction, MachineSurfaceInstruction};
use super::program::{LanesInfo, Program};
use super::resolve::resolve;

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

    // Read the header here rather than taking `parse_section`'s copy of it.
    // `parse_section` parses it too, but reports every failure as an `eyre`
    // message wrapped around `LanesInfo::from_text`'s own — so telling a
    // *missing* header from a *malformed* one downstream meant matching on
    // prose, and got it wrong: `version 1` and `version abc` were both
    // reported as missing, and `version 1.x` as an unparseable instruction.
    // Here the distinction is still available, so this is the value the
    // program is built from.
    let info = parse_version_header(src, file.root().header_text())?;

    let parsed = ParsedModule::<MachineSurfaceInstruction, SurfaceType, LanesInfo>::parse_section(
        file.root(),
    )
    .map_err(|e| TextError::BadInstruction {
        line: 0,
        text: e.to_string(),
    })?;

    // Any number of functions, resolved together: symbolic branch and call
    // targets need the whole module in view. See `super::resolve`.
    //
    // The header comes from `parse_version_header` above rather than
    // `parsed.header`: both parse the same text with the same `FromText` impl,
    // but only that one classifies its failures.
    resolve(parsed.functions, info).map_err(|e| TextError::BadInstruction {
        line: 0,
        text: e.to_string(),
    })
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

/// The structural errors that stop a program being written as text.
///
/// Rendering needs a *name* for every branch and call target, and a place to
/// put every instruction. Only two failures take those away, and everything
/// else `validate_structure` reports renders fine — a program with dead code,
/// a missing terminator or an out-of-range operand is exactly the kind you
/// disassemble in order to look at, so refusing to render it would be
/// backwards.
pub fn render_blockers(program: &Program) -> Vec<super::validate::ValidationError> {
    use super::validate::ValidationError as E;
    super::validate::validate_structure(program)
        .into_iter()
        .filter(|e| {
            matches!(
                e,
                // A target with no name: `to_text` would invent `@L99` / `@F1`
                // and emit a use with no definition.
                E::InvalidControlFlowTarget { .. }
                    // An instruction with no function to sit in: `to_text`
                    // would emit it after the closing brace.
                    | E::CodeOutsideFunction { .. }
            )
        })
        .collect()
}

/// Emit the program as vihaco's `sst v1` container.
///
/// Expects a program with no [`render_blockers`] — both callers
/// (`bloqade-bytecode disassemble` and `Program.to_text`) check first. Rendering needs a name for every branch and call target, and a
/// decoded program can carry ones that have none; the gate is what keeps this
/// function from having to invent text that will not read back.
///
/// The output is accepted by [`parse_text`], and the *code* round-trips
/// exactly. The label table may not: control flow is stored as addresses and
/// written as symbols, so a branch to an address the binary never named is
/// given a synthesised `L<addr>` and a defining `cpu::cpu.label` — the way a
/// disassembler emits `.L1:`. Re-reading that text therefore produces a
/// program with one more label than it started with, carrying the same code.
///
/// Targets that cannot be named at all — a branch past the end of the code, a
/// call to an address that begins no function — are rejected by
/// [`super::validate::validate_structure`] rather than rendered, since there
/// is no text that would read back.
pub fn to_text(program: &Program) -> String {
    use vihaco_cpu::RuntimeInstruction as C;

    let name_of = |index: u32| -> &str {
        program
            .strings
            .get(index as usize)
            .map(String::as_str)
            .unwrap_or("?")
    };

    // Control flow is stored as addresses but written as symbols, so rendering
    // needs a name for every branch target. A module resolved from text already
    // has one; a module built programmatically may not, so those get a
    // synthesised `L<address>` — emitted as a label too, so the text still
    // round-trips.
    let mut label_names: Vec<(u32, String)> = program
        .labels
        .iter()
        .map(|l| (l.address, name_of(l.name).to_owned()))
        .collect();
    for inst in &program.code {
        let targets: &[u32] = match inst {
            MachineInstruction::Cpu(C::Branch(t)) => &[*t],
            MachineInstruction::Cpu(C::ConditionalBranch(t, f)) => &[*t, *f],
            _ => &[],
        };
        for target in targets {
            if label_names.iter().any(|(a, _)| a == target) {
                continue;
            }
            // Deduping by address alone let a synthesised `L3` collide with a
            // *real* label named `L3` at a different address, so the text
            // defined the same name twice and would not re-read.
            let mut name = format!("L{target}");
            while label_names.iter().any(|(_, n)| *n == name) {
                name.push('_');
            }
            label_names.push((*target, name));
        }
    }
    let label_of = |address: u32| -> String {
        label_names
            .iter()
            .find(|(a, _)| *a == address)
            .map(|(_, n)| n.clone())
            .unwrap_or_else(|| format!("L{address}"))
    };
    // Both the `fn @name()` header and a `call`'s callee name the function
    // starting at an address, so both resolve it the same way: by the address
    // of its `func_start`, falling back to a synthesised `F<address>` for a
    // function the table does not name.
    let function_at = |address: u32| -> String {
        program
            .functions
            .iter()
            .find(|f| f.start_address == address)
            .map(|f| name_of(f.name).to_owned())
            .unwrap_or_else(|| format!("F{address}"))
    };

    // `fn @name(p: ty, ...) -> ty`. The declaration is what a `call`'s arity is
    // checked against, so dropping it here would lose the only record of what
    // the function takes — the code stream carries the boundary, not the shape.
    let signature_at = |address: u32| -> String {
        let Some(f) = program
            .functions
            .iter()
            .find(|f| f.start_address == address)
        else {
            return "()".to_owned();
        };
        let params: Vec<String> = f
            .signature
            .params
            .iter()
            .map(|p| format!("{}: {}", name_of(p.name), machine::cpu_type_text(p.ty)))
            .collect();
        let mut out = format!("({})", params.join(", "));
        // vihaco's grammar takes at most one return type (`-> Ty`), so a
        // multi-value signature has no syntax yet. Render the first and let
        // the round-trip test catch it if that ever stops being enough.
        if let Some(ty) = f.signature.ret.first() {
            out.push_str(&format!(" -> {}", machine::cpu_type_text(*ty)));
        }
        out
    };

    let render = |inst: &MachineInstruction| -> String {
        match inst {
            MachineInstruction::Cpu(C::Branch(t)) => {
                format!("cpu::cpu.br @{}", label_of(*t))
            }
            MachineInstruction::Cpu(C::ConditionalBranch(t, f)) => {
                format!("cpu::cpu.cond_br @{}, @{}", label_of(*t), label_of(*f))
            }
            // `br`/`cond_br` spell their target `@name` (their patterns include
            // the sigil); `call` does not — its generated pattern is
            // `'call $0 `,` $1`, so the callee is a bare identifier.
            MachineInstruction::Cpu(C::Call(arity, target)) => {
                format!("cpu::cpu.call {arity}, {}", function_at(*target))
            }
            other => machine::to_sst_text(other),
        }
    };

    // Walk the code stream, not the function table. `func_start`/`func_end`
    // delimit each body, so the boundaries come from the instructions being
    // rendered and cannot disagree with them. The table supplies only the
    // name, looked up by the address of its `func_start`.
    //
    // A stream with no markers at all is one implicit `@main` spanning the
    // whole program — the shape `from_code` builds for a caller that handed us
    // a bare instruction list.
    let emit_labels = |body: &mut String, address: u32| {
        for (_, name) in label_names.iter().filter(|(a, _)| *a == address) {
            body.push_str(&format!("  cpu::cpu.label @{name}\n"));
        }
    };

    let mut body = String::new();
    for (address, inst) in program.code.iter().enumerate() {
        let address = address as u32;
        match inst {
            // The markers are structure, not instructions: they open and close
            // the block rather than being rendered inside it.
            MachineInstruction::Cpu(C::FunctionStart) => {
                body.push_str(&format!(
                    "fn @{}{} {{\n",
                    function_at(address),
                    signature_at(address)
                ));
                emit_labels(&mut body, address);
                continue;
            }
            MachineInstruction::Cpu(C::FunctionEnd) => {
                emit_labels(&mut body, address);
                body.push_str("}\n");
                continue;
            }
            _ => {}
        }
        emit_labels(&mut body, address);
        body.push_str("  ");
        body.push_str(&render(inst));
        body.push('\n');
    }

    super::container::to_sst(&program.extra, &body)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::device::LanesInstruction as L;
    use crate::isa::machine::MachineInstruction as M;
    use crate::isa::program::from_code;
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
        .unwrap()
    }

    #[test]
    fn text_round_trips_fn_main() {
        let src = sst(
            "1.2",
            "fn @main() {\n  lanes::lanes.const_loc 0x0000000000000000\n  lanes::lanes.initial_fill 1\n  cpu::cpu.halt\n}\n",
        );
        let p = parse_text(&src).unwrap();
        assert_eq!(p.extra.version, Version::new(1, 2));
        // Three instructions wrapped in the function's `func_start`/`func_end`.
        assert_eq!(p.code.len(), 5);
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
        let prog = from_code(Version::new(1, 0), vec![M::Cpu(C::Halt)]).unwrap();
        assert!(to_text(&prog).contains("  cpu::cpu.halt\n"));
    }

    #[test]
    fn empty_program_round_trips() {
        let prog = from_code(Version::new(1, 0), vec![]).unwrap();
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

    /// `render_blockers` is empty exactly when the rendered text re-parses.
    ///
    /// That is the whole contract, so it is tested as one: render every
    /// program, try to read it back, and assert the predicate agreed. Gating
    /// on all of `validate_structure` failed this — dead code and a missing
    /// terminator render and re-parse perfectly well, and refusing them is
    /// backwards, since those are the programs you disassemble to look at.
    #[test]
    fn rendering_is_blocked_exactly_when_the_text_would_not_re_parse() {
        use crate::isa::program::from_code;
        use crate::isa::validate::validate_structure;
        use crate::version::Version;

        let wrapped = |code: Vec<M>| from_code(Version::new(1, 0), code).unwrap();

        // Programs the validator objects to, which nonetheless render and
        // read back: dead code, no terminator, a bad operand, a wild local.
        let renderable = vec![
            wrapped(vec![M::Cpu(C::Halt), M::Lanes(L::Cz)]),
            wrapped(vec![M::Lanes(L::Cz)]),
            wrapped(vec![M::Lanes(L::NewArray(0, 0, 0)), M::Cpu(C::Halt)]),
            wrapped(vec![M::Cpu(C::Store(Type::U64, 999_999)), M::Cpu(C::Halt)]),
        ];

        // A branch with no nameable target, and an instruction outside every
        // function — the two the renderer cannot write down.
        let mut orphan = wrapped(vec![M::Cpu(C::Halt)]);
        orphan.code.push(M::Lanes(L::Cz));
        let unrenderable = vec![
            wrapped(vec![M::Cpu(C::Branch(99)), M::Cpu(C::Halt)]),
            orphan,
        ];

        for p in renderable {
            assert!(
                !validate_structure(&p).is_empty(),
                "this case is meant to fail validation: {:?}",
                p.code
            );
            assert_eq!(render_blockers(&p), vec![], "should render: {:?}", p.code);
            parse_text(&to_text(&p))
                .unwrap_or_else(|e| panic!("rendered text should re-parse: {e}\n{:?}", p.code));
        }
        for p in unrenderable {
            assert!(
                !render_blockers(&p).is_empty(),
                "should be blocked: {:?}",
                p.code
            );
            assert!(
                parse_text(&to_text(&p)).is_err(),
                "the text really would not re-parse: {:?}",
                p.code
            );
        }
    }

    /// A label opening a non-entry function lands inside that function's
    /// braces and survives the round trip.
    ///
    /// This began as `a_label_at_a_function_boundary_is_not_emitted_twice`,
    /// guarding a bug from the span-recorded layout: functions were laid out
    /// contiguously, so one function's `end_address` *was* the next one's
    /// `start_address`, and a label there was emitted by both — re-reading the
    /// output then failed with "duplicate label".
    ///
    /// The markers made that unrepresentable rather than merely fixed. A
    /// function now starts at its own `func_start` and its body begins one
    /// address later, so no label can share an address with a boundary, and
    /// `to_text` visits each address exactly once across three mutually
    /// exclusive arms. The count assertion below can no longer reach 2. It is
    /// kept because the *placement* is still worth pinning — the label has to
    /// render after `fn @helper() {`, not before it — and renamed because a
    /// test named for a property nothing can violate reads like coverage it
    /// does not provide.
    #[test]
    fn a_label_opening_a_non_entry_function_round_trips() {
        let src = sst(
            "1.0",
            "fn @main() {\n  cpu::cpu.call 0, helper\n  cpu::cpu.halt\n}\n             fn @helper() {\n  cpu::cpu.label @entry\n  cpu::cpu.ret 0\n}\n",
        );
        let program = parse_text(&src).expect("the module should parse");

        let rendered = to_text(&program);
        assert_eq!(
            rendered.matches("label @entry").count(),
            1,
            "the label should appear once:\n{rendered}"
        );
        // The placement is the part still worth asserting: inside `@helper`'s
        // braces, after its header, not hoisted above it into `@main`.
        let header = rendered
            .find("fn @helper()")
            .expect("`@helper` should be rendered");
        let label = rendered
            .find("label @entry")
            .expect("the label should be rendered");
        assert!(
            header < label,
            "the label belongs inside `@helper`:\n{rendered}"
        );
        assert_eq!(
            parse_text(&rendered).expect("the rendered text should re-parse"),
            program,
            "the round-trip should be lossless"
        );
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
        let prog = from_code(Version::new(3, 7), vec![]).unwrap();
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
    fn several_functions_resolve_together() {
        let src = sst(
            "1.0",
            "fn @main() {\n  cpu::cpu.call 0, helper\n  cpu::cpu.halt\n}\n\
             fn @helper() {\n  cpu::cpu.ret 0\n}\n",
        );
        let p = parse_text(&src).unwrap();

        assert_eq!(p.functions.len(), 2);
        assert_eq!(p.main_function, Some(0));
        // `@main` is `func_start, call, halt, func_end`, so `@helper`'s own
        // `func_start` is at 4 — and that is where the call was patched to,
        // since entering a function means entering at its marker.
        assert_eq!(p.functions[1].start_address, 4);
        assert_eq!(
            p.code[1],
            M::Cpu(C::Call(0, 4)),
            "call target should be resolved to @helper's address"
        );
        assert_eq!(parse_text(&to_text(&p)).unwrap(), p);
    }

    #[test]
    fn labels_resolve_and_leave_the_code_stream() {
        let src = sst(
            "1.0",
            "fn @main() {\n  cpu::cpu.br @done\n  lanes::lanes.cz\n  \
             cpu::cpu.label @done\n  cpu::cpu.halt\n}\n",
        );
        let p = parse_text(&src).unwrap();

        // Three instructions plus the two function markers: the label is
        // metadata, not code.
        assert_eq!(p.code.len(), 5);
        assert_eq!(p.labels.len(), 1);
        assert_eq!(p.labels[0].address, 3, "@done marks the halt");
        assert_eq!(p.code[1], M::Cpu(C::Branch(3)));
        assert_eq!(parse_text(&to_text(&p)).unwrap(), p);
    }

    #[test]
    fn conditional_branches_resolve_both_arms() {
        let src = sst(
            "1.0",
            "fn @main() {\n  cpu::cpu.cond_br @yes, @no\n  \
             cpu::cpu.label @yes\n  lanes::lanes.cz\n  \
             cpu::cpu.label @no\n  cpu::cpu.halt\n}\n",
        );
        let p = parse_text(&src).unwrap();
        // Addresses account for the leading `func_start`.
        assert_eq!(p.code[1], M::Cpu(C::ConditionalBranch(2, 3)));
        assert_eq!(parse_text(&to_text(&p)).unwrap(), p);
    }

    #[test]
    fn unresolvable_symbols_are_reported_by_name() {
        for (src, needle) in [
            ("fn @main() {\n  cpu::cpu.br @nowhere\n}\n", "@nowhere"),
            ("fn @main() {\n  cpu::cpu.call 0, missing\n}\n", "@missing"),
        ] {
            let err = parse_text(&sst("1.0", src)).unwrap_err().to_string();
            assert!(err.contains(needle), "got {err}");
        }
    }

    #[test]
    fn duplicate_symbols_are_rejected() {
        let dup_label = sst(
            "1.0",
            "fn @main() {\n  cpu::cpu.label @x\n  cpu::cpu.label @x\n  cpu::cpu.halt\n}\n",
        );
        assert!(
            parse_text(&dup_label)
                .unwrap_err()
                .to_string()
                .contains("duplicate label")
        );

        let dup_fn = sst(
            "1.0",
            "fn @main() {\n  cpu::cpu.halt\n}\nfn @main() {\n  cpu::cpu.halt\n}\n",
        );
        assert!(
            parse_text(&dup_fn)
                .unwrap_err()
                .to_string()
                .contains("duplicate function")
        );
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
