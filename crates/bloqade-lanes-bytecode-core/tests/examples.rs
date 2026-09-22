//! The shipped example programs are exercised, not just shipped.
//!
//! Nothing in the repository referenced `examples/programs/` — no test, no
//! script, no CI step — so the 26 fixtures there rotted silently through two
//! format changes, and one of them (`valid/stack_full_pipeline.sst`) shipped
//! failing the very validation its directory name claims it passes.
//!
//! ## The directives
//!
//! A fixture cannot be checked without knowing what it is *supposed* to do,
//! and that was nowhere written down: half the `invalid/` programs are only
//! invalid against a particular architecture, so validating them against the
//! wrong one reports `valid` and looks like a pass. Each file now says so
//! itself, in comments inside its `.text(root)` block:
//!
//! ```text
//! // arch: simple
//! // expect: FillRequiresAtomReloading
//! ```
//!
//! - `arch:` names a spec in `examples/arch/` (without the `.json`), or
//!   `none` for a program that consults no architecture.
//! - `expect:` names a [`ValidationError`] variant. Zero or more; the set must
//!   match exactly, so a fixture cannot quietly acquire a second error — and a
//!   misspelled name shows up as one the fixture declared but nothing
//!   reported, which is why no list of valid names is maintained here.
//!
//! A `valid/` program declares no `expect:` and must validate clean.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_bytecode_core::isa::Program;
use bloqade_lanes_bytecode_core::isa::text::parse_text;
use bloqade_lanes_bytecode_core::isa::validate::{
    ValidationError, simulate_stack, validate, validate_structure,
};

/// The workspace root, resolved once. Every fixture needs it to find its arch
/// spec, and `canonicalize` hits the filesystem.
static REPO_ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root should resolve")
});

/// Which [`ValidationError`] this is, as the variant's own name.
///
/// Spelled out rather than scraped from `{:?}`, so a rename is a compile
/// error here and not a silently unmatched fixture directive. There is no
/// companion list of valid names: a directive naming something `kind` never
/// returns simply shows up as a declared-but-not-reported entry when the sets
/// are compared, which is the same failure and needs nothing to maintain.
fn kind(error: &ValidationError) -> &'static str {
    use ValidationError as E;
    match error {
        E::ControlFlowRequiresFeedForward { .. } => "ControlFlowRequiresFeedForward",
        E::MultipleMeasuresRequireFeedForward { .. } => "MultipleMeasuresRequireFeedForward",
        E::FillRequiresAtomReloading { .. } => "FillRequiresAtomReloading",
        E::InvalidLocation { .. } => "InvalidLocation",
        E::InvalidLane { .. } => "InvalidLane",
        E::InvalidZone { .. } => "InvalidZone",
        E::NewArrayZeroDim0 { .. } => "NewArrayZeroDim0",
        E::NewArrayInvalidTypeTag { .. } => "NewArrayInvalidTypeTag",
        E::NewArrayTooManyElements { .. } => "NewArrayTooManyElements",
        E::GetItemInvalidDims { .. } => "GetItemInvalidDims",
        E::LocalIndexOutOfRange { .. } => "LocalIndexOutOfRange",
        E::InvalidControlFlowTarget { .. } => "InvalidControlFlowTarget",
        E::CodeOutsideFunction { .. } => "CodeOutsideFunction",
        E::InitialFillNotFirst { .. } => "InitialFillNotFirst",
        E::EmptyProgram => "EmptyProgram",
        E::MissingTerminator { .. } => "MissingTerminator",
        E::UnreachableInstruction { .. } => "UnreachableInstruction",
        E::StackUnderflow { .. } => "StackUnderflow",
        E::TypeMismatch { .. } => "TypeMismatch",
        E::LocationGroupValidation { .. } => "LocationGroupValidation",
        E::LaneGroupValidation { .. } => "LaneGroupValidation",
    }
}

/// A `// <key>: <value>` directive line, if this line is one.
fn directive<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.trim()
        .strip_prefix("//")?
        .trim()
        .strip_prefix(key)?
        .strip_prefix(':')
        .map(str::trim)
}

struct Fixture {
    name: String,
    arch: Option<ArchSpec>,
    expected: BTreeSet<String>,
    /// Parsed once, here: every test needs it, and parsing at load time is
    /// also how "every fixture parses" is asserted.
    program: Program,
}

/// Read a fixture and its directives, failing loudly on anything missing —
/// an unannotated fixture is one this test would otherwise skip.
fn load(path: &Path) -> Fixture {
    let name = path.file_name().unwrap().to_string_lossy().into_owned();
    let src = fs::read_to_string(path).unwrap_or_else(|e| panic!("{name}: {e}"));

    let arch_name = src
        .lines()
        .find_map(|l| directive(l, "arch"))
        .unwrap_or_else(|| panic!("{name}: no `// arch: <spec>` directive"));
    let arch = (arch_name != "none").then(|| {
        let json = fs::read_to_string(
            REPO_ROOT
                .join("examples/arch")
                .join(arch_name)
                .with_extension("json"),
        )
        .unwrap_or_else(|e| panic!("{name}: arch {arch_name}: {e}"));
        ArchSpec::from_json(&json).unwrap_or_else(|e| panic!("{name}: arch {arch_name}: {e:?}"))
    });

    let expected = src
        .lines()
        .filter_map(|l| directive(l, "expect"))
        .map(str::to_owned)
        .collect();

    let program = parse_text(&src).unwrap_or_else(|e| panic!("{name}: {e}"));

    Fixture {
        name,
        arch,
        expected,
        program,
    }
}

fn fixtures(dir: &str) -> Vec<Fixture> {
    let root = REPO_ROOT.join("examples/programs").join(dir);
    let mut paths: Vec<PathBuf> = fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{}: {e}", root.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "sst"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no fixtures in {}", root.display());
    paths.iter().map(|p| load(p)).collect()
}

/// Every error a fixture provokes, from all three validation passes.
fn all_errors(fixture: &Fixture) -> Vec<ValidationError> {
    let program = &fixture.program;
    let arch = fixture.arch.as_ref();
    let mut errors = validate_structure(program);
    errors.extend(validate(program, arch));
    errors.extend(simulate_stack(program, arch));
    errors
}

/// Report every mismatched fixture, not just the first.
///
/// A format change tends to break many at once, and fixing them one failed
/// run at a time is exactly the loop that let these rot.
fn report(mismatches: Vec<String>) {
    assert!(
        mismatches.is_empty(),
        "{} fixture(s) do not match their directives:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
fn every_valid_example_validates_clean() {
    let mut bad = Vec::new();
    for fixture in fixtures("valid") {
        if !fixture.expected.is_empty() {
            bad.push(format!(
                "  {}: a valid/ fixture must declare no `// expect:`",
                fixture.name
            ));
            continue;
        }
        let errors = all_errors(&fixture);
        if !errors.is_empty() {
            bad.push(format!(
                "  {} is filed under valid/ but reports {errors:?}",
                fixture.name
            ));
        }
    }
    report(bad);
}

#[test]
fn every_invalid_example_reports_exactly_what_it_declares() {
    let mut bad = Vec::new();
    for fixture in fixtures("invalid") {
        if fixture.expected.is_empty() {
            bad.push(format!(
                "  {}: an invalid/ fixture must declare at least one `// expect:`",
                fixture.name
            ));
            continue;
        }
        let got: BTreeSet<String> = all_errors(&fixture)
            .iter()
            .map(|e| kind(e).to_owned())
            .collect();
        if got != fixture.expected {
            bad.push(format!(
                "  {}: declared {:?}, reports {:?}",
                fixture.name, fixture.expected, got
            ));
        }
    }
    report(bad);
}

/// A program that validates must also *run*.
///
/// Validation cannot see everything: it type-checks the stack and checks each
/// address against the architecture, but it does not track which sites are
/// occupied. `stack_full_pipeline.sst` validated clean while trying to refill
/// the two sites its own `move` had just filled — only executing it says so.
#[test]
fn every_valid_example_runs_to_completion() {
    use bloqade_lanes_bytecode_core::isa::machine::{LanesMachine, Stopped};

    let mut bad = Vec::new();
    for fixture in fixtures("valid") {
        let mut machine = LanesMachine::new();
        if let Some(arch) = fixture.arch.clone() {
            machine = machine.with_arch(arch);
        }
        match machine.run(&fixture.program, 10_000) {
            Err(e) => bad.push(format!("  {}: {e}", fixture.name)),
            Ok(run) if !matches!(run.stopped, Stopped::Halted | Stopped::Returned) => {
                bad.push(format!("  {}: stopped as {:?}", fixture.name, run.stopped));
            }
            Ok(_) => {}
        }
    }
    report(bad);
}

/// Round-tripping is the other thing the fixtures are for: every one of them
/// must survive text -> binary -> text and come back identical.
#[test]
fn every_example_round_trips() {
    use bloqade_lanes_bytecode_core::isa::program::{from_binary, to_binary};
    use bloqade_lanes_bytecode_core::isa::text::to_text;

    for fixture in fixtures("valid").into_iter().chain(fixtures("invalid")) {
        let program = &fixture.program;

        let bytes = to_binary(program).unwrap_or_else(|e| panic!("{}: {e}", fixture.name));
        assert_eq!(
            &from_binary(&bytes).unwrap_or_else(|e| panic!("{}: {e}", fixture.name)),
            program,
            "{}: binary round-trip changed the program",
            fixture.name
        );

        let rendered = to_text(program);
        assert_eq!(
            &parse_text(&rendered).unwrap_or_else(|e| panic!("{}: re-parse: {e}", fixture.name)),
            program,
            "{}: text round-trip changed the program",
            fixture.name
        );
    }
}
