use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

mod common;
use common::{sst, sst_version};

fn cmd() -> Command {
    assert_cmd::cargo_bin_cmd!("bloqade-bytecode")
}

/// A small program for basic command tests.
fn sample_program() -> String {
    sst("\
fn @main() {
  lanes::lanes.const_loc 0x00000102
  lanes::lanes.const_lane 0x0000000100030002
  cpu::cpu.halt
}
")
}

/// All 23 distinct instructions exercised in a single program
/// (25 code slots, counting `@main`'s two function markers).
/// Ordered so that initial_fill comes right after constants (structurally valid).
fn all_instructions_program() -> String {
    sst("\
fn @main() {
  cpu::cpu.const f64, 1.5
  cpu::cpu.const i64, 42
  lanes::lanes.const_loc 0x00010002
  lanes::lanes.const_lane 0x8000000000010002
  lanes::lanes.const_zone 0x00000003
  lanes::lanes.initial_fill 3
  lanes::lanes.pop
  cpu::cpu.dup
  lanes::lanes.swap
  lanes::lanes.fill 2
  lanes::lanes.move 1
  lanes::lanes.local_r 4
  lanes::lanes.local_rz 2
  lanes::lanes.global_r
  lanes::lanes.global_rz
  lanes::lanes.cz
  lanes::lanes.measure 1
  lanes::lanes.await_measure
  lanes::lanes.new_array 2 10 20
  lanes::lanes.get_item 2
  lanes::lanes.set_detector
  lanes::lanes.set_observable
  cpu::cpu.halt
}
")
}

/// A program with addresses valid for the test arch spec (word_id=0, site_id in 0..5, bus_id=0).
fn arch_valid_program() -> String {
    sst("\
fn @main() {
  lanes::lanes.const_loc 0x00000001
  lanes::lanes.const_zone 0x00000000
  cpu::cpu.halt
}
")
}

#[test]
fn test_assemble_creates_binary() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    let output = dir.path().join("prog.bin");
    fs::write(&input, all_instructions_program()).unwrap();

    cmd()
        .args([
            "assemble",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("assembled 25 instructions"));

    let bytes = fs::read(&output).unwrap();
    // Should start with vihaco's container magic
    assert_eq!(&bytes[..4], b"VHBC");
}

#[test]
fn test_disassemble_to_stdout() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    fs::write(&input_txt, all_instructions_program()).unwrap();

    // First assemble
    cmd()
        .args([
            "assemble",
            input_txt.to_str().unwrap(),
            "-o",
            binary.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Then disassemble to stdout — verify representative instructions survive round-trip
    cmd()
        .args(["disassemble", binary.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("cpu::cpu.const f64, 1.5"))
        .stdout(predicate::str::contains("cpu::cpu.const i64, 42"))
        .stdout(predicate::str::contains("lanes::lanes.const_loc"))
        .stdout(predicate::str::contains("lanes::lanes.const_lane"))
        .stdout(predicate::str::contains("lanes::lanes.const_zone"))
        .stdout(predicate::str::contains("new_array 2 10 20"))
        .stdout(predicate::str::contains("halt"));
}

#[test]
fn test_disassemble_to_file() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    let output_txt = dir.path().join("out.sst");
    fs::write(&input_txt, all_instructions_program()).unwrap();

    cmd()
        .args([
            "assemble",
            input_txt.to_str().unwrap(),
            "-o",
            binary.to_str().unwrap(),
        ])
        .assert()
        .success();

    cmd()
        .args([
            "disassemble",
            binary.to_str().unwrap(),
            "-o",
            output_txt.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("disassembled 25 instructions"));

    let text = fs::read_to_string(&output_txt).unwrap();
    // Spot-check all instruction categories are present. Each is asserted with
    // its dialect head, so the check also pins which dialect an op belongs to.
    for expected in [
        "cpu::cpu.const f64,",
        "cpu::cpu.const i64,",
        "lanes::lanes.const_loc",
        "lanes::lanes.const_lane",
        "lanes::lanes.const_zone",
        "lanes::lanes.pop",
        "cpu::cpu.dup",
        "lanes::lanes.swap",
        "lanes::lanes.initial_fill 3",
        "lanes::lanes.fill 2",
        "lanes::lanes.move 1",
        "lanes::lanes.local_r 4",
        "lanes::lanes.local_rz 2",
        "lanes::lanes.global_r",
        "lanes::lanes.global_rz",
        "lanes::lanes.cz",
        "lanes::lanes.measure 1",
        "lanes::lanes.await_measure",
        "lanes::lanes.new_array 2 10 20",
        "lanes::lanes.get_item 2",
        "lanes::lanes.set_detector",
        "lanes::lanes.set_observable",
        "cpu::cpu.halt",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
}

#[test]
fn test_round_trip_assemble_disassemble() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    let output_txt = dir.path().join("out.sst");
    fs::write(&input_txt, all_instructions_program()).unwrap();

    // Assemble
    cmd()
        .args([
            "assemble",
            input_txt.to_str().unwrap(),
            "-o",
            binary.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Disassemble
    cmd()
        .args([
            "disassemble",
            binary.to_str().unwrap(),
            "-o",
            output_txt.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Re-assemble from disassembled output
    let binary2 = dir.path().join("prog2.bin");
    cmd()
        .args([
            "assemble",
            output_txt.to_str().unwrap(),
            "-o",
            binary2.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Both binaries should be identical
    let b1 = fs::read(&binary).unwrap();
    let b2 = fs::read(&binary2).unwrap();
    assert_eq!(b1, b2, "round-trip binary mismatch");
}

/// The same round trip for a program with more than one function.
///
/// `test_round_trip_assemble_disassemble` above uses a single flat `@main`,
/// which is what every CLI test used to do — so the whole symbol-table path
/// (the `functions`, `labels` and `strings` child sections, the `call` target
/// patched to an address and named back, the labels resolved and re-emitted)
/// went through the CLI untested. Byte equality across the second assemble is
/// what pins it: a name resolved to the wrong address, or a label dropped,
/// diverges here even when both halves parse.
#[test]
fn test_round_trip_preserves_functions_and_labels() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    let output_txt = dir.path().join("out.sst");
    let binary2 = dir.path().join("prog2.bin");

    fs::write(
        &input_txt,
        sst("fn @main() {\n  \
             cpu::cpu.call 0, helper\n  \
             cpu::cpu.halt\n\
             }\n\n\
             fn @helper() {\n  \
             cpu::cpu.label @spin\n  \
             cpu::cpu.br @spin\n\
             }\n"),
    )
    .unwrap();

    cmd()
        .args([
            "assemble",
            input_txt.to_str().unwrap(),
            "-o",
            binary.to_str().unwrap(),
        ])
        .assert()
        .success();

    cmd()
        .args([
            "disassemble",
            binary.to_str().unwrap(),
            "-o",
            output_txt.to_str().unwrap(),
        ])
        .assert()
        .success();

    cmd()
        .args([
            "assemble",
            output_txt.to_str().unwrap(),
            "-o",
            binary2.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Both function names survive the trip through binary, where they live in
    // the `strings` table rather than in the text.
    let rendered = fs::read_to_string(&output_txt).unwrap();
    assert!(
        rendered.contains("fn @main()") && rendered.contains("fn @helper()"),
        "both functions should be named, not synthesised as F<address>:\n{rendered}"
    );
    assert!(
        rendered.contains("cpu::cpu.call 0, helper"),
        "the call should name its callee:\n{rendered}"
    );
    assert!(
        rendered.contains("label @spin"),
        "the label should survive:\n{rendered}"
    );

    assert_eq!(
        fs::read(&binary).unwrap(),
        fs::read(&binary2).unwrap(),
        "round-trip binary mismatch for a multi-function program"
    );
}

#[test]
fn test_validate_text_file() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(&input, all_instructions_program()).unwrap();

    cmd()
        .args(["validate", input.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("valid (25 instructions)"));
}

#[test]
fn test_validate_binary_file() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    fs::write(&input_txt, all_instructions_program()).unwrap();

    cmd()
        .args([
            "assemble",
            input_txt.to_str().unwrap(),
            "-o",
            binary.to_str().unwrap(),
        ])
        .assert()
        .success();

    cmd()
        .args(["validate", binary.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("valid (25 instructions)"));
}

#[test]
fn test_validate_with_arch_spec() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(&input, arch_valid_program()).unwrap();

    let arch_json = r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]] }
        ],
        "zones": [
            {
                "grid": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [] },
                "site_buses": [
                    { "src": [0, 1], "dst": [3, 4] }
                ],
                "word_buses": [],
                "words_with_site_buses": [0],
                "sites_with_word_buses": []
            }
        ],
        "zone_buses": [],

        "modes": [
            { "name": "default", "zones": [0], "bitstring_order": [] }
        ]
    }"#;
    let arch_path = dir.path().join("arch.json");
    fs::write(&arch_path, arch_json).unwrap();

    cmd()
        .args([
            "validate",
            input.to_str().unwrap(),
            "--arch",
            arch_path.to_str().unwrap(),
        ])
        .assert()
        .success();
}

#[test]
fn test_validate_with_simulate_stack() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(&input, sample_program()).unwrap();

    cmd()
        .args(["validate", input.to_str().unwrap(), "--simulate-stack"])
        .assert()
        .success();
}

#[test]
fn test_assemble_missing_input() {
    cmd()
        .args(["assemble", "/nonexistent/file.txt", "-o", "/tmp/out.bin"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_no_subcommand_shows_help() {
    cmd()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn test_validate_detects_invalid_arch_addresses() {
    let dir = TempDir::new().unwrap();
    // This program references word_id=1, site_id=2 which doesn't exist in the arch
    let input = dir.path().join("prog.sst");
    fs::write(&input, sample_program()).unwrap();

    let arch_json = r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [0, 1]] }
        ],
        "zones": [
            {
                "grid": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [], "y_spacing": [2.0] },
                "site_buses": [],
                "word_buses": [],
                "words_with_site_buses": [],
                "sites_with_word_buses": []
            }
        ],
        "zone_buses": [],

        "modes": [
            { "name": "default", "zones": [0], "bitstring_order": [] }
        ]
    }"#;
    let arch_path = dir.path().join("arch.json");
    fs::write(&arch_path, arch_json).unwrap();

    cmd()
        .args([
            "validate",
            input.to_str().unwrap(),
            "--arch",
            arch_path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("validation error"));
}

#[test]
fn test_assemble_invalid_syntax() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("bad.sst");
    let output = dir.path().join("out.bin");
    fs::write(&input, sst("fn @main() {\n  foobar_invalid}\n")).unwrap();

    cmd()
        .args([
            "assemble",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_disassemble_invalid_binary() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("bad.bin");
    fs::write(&input, b"not a valid binary").unwrap();

    cmd()
        .args(["disassemble", input.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_arch_pretty_print() {
    let dir = TempDir::new().unwrap();
    let arch_json = r#"{"version":"2.0","words":[{"sites":[[0,0],[0,1]]}],"zones":[{"grid":{"x_start":1.0,"y_start":2.0,"x_spacing":[],"y_spacing":[2.0]},"site_buses":[],"word_buses":[],"words_with_site_buses":[],"sites_with_word_buses":[]}],"zone_buses":[],"modes":[{"name":"default","zones":[0],"bitstring_order":[]}]}"#;
    let arch_path = dir.path().join("arch.json");
    fs::write(&arch_path, arch_json).unwrap();

    cmd()
        .args(["arch", arch_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("ArchSpec v2.0"))
        .stdout(predicate::str::contains("1 word(s), 2 sites/word"))
        .stdout(predicate::str::contains("Word 0: sites=[(0,0) (0,1)]"))
        .stdout(predicate::str::contains("Zone 0: 1x2 grid"));
}

#[test]
fn test_arch_pretty_print_invalid_json() {
    let dir = TempDir::new().unwrap();
    let arch_path = dir.path().join("bad.json");
    fs::write(&arch_path, "not json").unwrap();

    cmd()
        .args(["arch", arch_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_validate_arch_spec_valid() {
    let dir = TempDir::new().unwrap();
    let arch_json = r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]] }
        ],
        "zones": [
            {
                "grid": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [] },
                "site_buses": [
                    { "src": [0, 1], "dst": [3, 4] }
                ],
                "word_buses": [],
                "words_with_site_buses": [0],
                "sites_with_word_buses": []
            }
        ],
        "zone_buses": [],

        "modes": [
            { "name": "default", "zones": [0], "bitstring_order": [] }
        ]
    }"#;
    let arch_path = dir.path().join("arch.json");
    fs::write(&arch_path, arch_json).unwrap();

    cmd()
        .args(["arch", "validate", arch_path.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("arch spec is valid"));
}

#[test]
fn test_validate_arch_spec_invalid() {
    let dir = TempDir::new().unwrap();
    // Site bus 0 is cyclic (0→1, 1→0 — a rotation): parses fine, but
    // fails the bus well-formedness validation.
    let arch_json = r#"{
        "version": "2.0",
        "words": [
            { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]] }
        ],
        "zones": [
            {
                "grid": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [] },
                "site_buses": [
                    { "src": [0, 1], "dst": [1, 0] }
                ],
                "word_buses": [],
                "words_with_site_buses": [0],
                "sites_with_word_buses": []
            }
        ],
        "zone_buses": [],

        "modes": [
            { "name": "default", "zones": [0], "bitstring_order": [] }
        ]
    }"#;
    let arch_path = dir.path().join("bad_arch.json");
    fs::write(&arch_path, arch_json).unwrap();

    cmd()
        .args(["arch", "validate", arch_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cycle"));
}

#[test]
fn test_validate_arch_spec_bad_json() {
    let dir = TempDir::new().unwrap();
    let arch_path = dir.path().join("bad.json");
    fs::write(&arch_path, "not valid json").unwrap();

    cmd()
        .args(["arch", "validate", arch_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_round_trip_preserves_version() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    fs::write(
        &input,
        sst_version(
            "2.3",
            "fn @main() {\n  cpu::cpu.const i64, 99\n  cpu::cpu.halt}\n",
        ),
    )
    .unwrap();

    cmd()
        .args([
            "assemble",
            input.to_str().unwrap(),
            "-o",
            binary.to_str().unwrap(),
        ])
        .assert()
        .success();

    cmd()
        .args(["disassemble", binary.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("version 2.3"))
        .stdout(predicate::str::contains("fn @main()"));
}

/// A malformed `new_array` is a diagnosis, not a denial of service.
///
/// `dim0` and `dim1` are read straight out of the instruction word, and their
/// product drove a pop loop that emitted one `stack underflow` per missing
/// element. This 28-byte program produced **5,000,001** lines in 8.5 s and
/// 405 MB; the count is now bounded and the repeats collapsed, so it produces
/// two. Asserting the error count is what pins that — the unit tests reach
/// `simulate_stack` directly and never see the CLI's output.
#[test]
fn test_validate_bounds_an_oversized_new_array() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("bigarray.sst");
    fs::write(
        &input,
        sst("fn @main() {\n  lanes::lanes.new_array 0 5000000 0\n  cpu::cpu.halt\n}\n"),
    )
    .unwrap();

    cmd()
        .args(["validate", input.to_str().unwrap(), "--simulate-stack"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "new_array declares 5000000 elements, more than the maximum of 1048576",
        ))
        .stderr(predicate::str::contains("error: 2 validation error(s)"));
}

/// The same operands one step further out: `65536 * 65536` is `2^32`, which
/// wrapped to zero in the old `u32` arithmetic — so the CLI reported the
/// program *valid* without examining an operand.
#[test]
fn test_validate_rejects_a_wrapping_new_array() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("wrap.sst");
    fs::write(
        &input,
        sst("fn @main() {\n  lanes::lanes.new_array 0 65536 65536\n  cpu::cpu.halt\n}\n"),
    )
    .unwrap();

    cmd()
        .args(["validate", input.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("declares 4294967296 elements"));
}

// --- run ---

/// A five-site word with one site bus, written to `dir` — the same shape the
/// validate tests use, kept here so the run tests do not depend on the
/// repository layout.
fn test_arch(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("arch.json");
    fs::write(
        &path,
        r#"{
            "version": "2.0",
            "words": [
                { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]] }
            ],
            "zones": [
                {
                    "grid": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [] },
                    "site_buses": [ { "src": [0, 1], "dst": [3, 4] } ],
                    "word_buses": [],
                    "words_with_site_buses": [0],
                    "sites_with_word_buses": []
                }
            ],
            "zone_buses": [],
            "modes": [ { "name": "default", "zones": [0], "bitstring_order": [] } ]
        }"#,
    )
    .unwrap();
    path
}

/// The execution layer had no caller before this: nothing in the CLI, the C
/// FFI or the PyO3 bindings constructed a `LanesMachine`, so "lanes programs
/// now run" was not something a user could do.
#[test]
fn test_run_places_atoms() {
    let dir = TempDir::new().unwrap();
    let arch = test_arch(&dir);
    let input = dir.path().join("prog.sst");
    fs::write(
        &input,
        sst(
            "fn @main() {\n  lanes::lanes.const_loc 0x0000000000000000\n  \
             lanes::lanes.const_loc 0x0000000001000000\n  \
             lanes::lanes.initial_fill 2\n  cpu::cpu.halt\n}\n",
        ),
    )
    .unwrap();

    cmd()
        .args(["run", input.to_str().unwrap()])
        .args(["--arch", arch.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("halted"))
        .stderr(predicate::str::contains("2 atom(s) placed"));
}

/// A move drives the atom state, which is the part the machine really does
/// simulate — and the reason it needs the architecture.
#[test]
fn test_run_moves_atoms() {
    let dir = TempDir::new().unwrap();
    let arch = test_arch(&dir);
    let input = dir.path().join("prog.sst");
    fs::write(
        &input,
        sst(
            "fn @main() {\n  lanes::lanes.const_loc 0x0000000000000000\n  \
             lanes::lanes.initial_fill 1\n  \
             lanes::lanes.const_lane 0x0000000000000000\n  \
             lanes::lanes.move 1\n  cpu::cpu.halt\n}\n",
        ),
    )
    .unwrap();

    cmd()
        .args(["run", input.to_str().unwrap()])
        .args(["--arch", arch.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("1 atom(s) placed"));
}

/// `move` cannot resolve a lane into endpoints without an architecture, so
/// the failure has to name that rather than something generic.
#[test]
fn test_run_without_arch_reports_why_move_failed() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(
        &input,
        sst(
            "fn @main() {\n  lanes::lanes.const_lane 0x0000000000000000\n  \
             lanes::lanes.move 1\n  cpu::cpu.halt\n}\n",
        ),
    )
    .unwrap();

    cmd()
        .args(["run", input.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("arch spec"));
}

/// The budget turns a runaway program into a message rather than a hang.
///
/// A backward branch is the way to run forever, but this branch cannot lower
/// symbolic control flow from text yet, so the budget itself is exercised by
/// setting it below the program's length. `machine::tests` covers the actual
/// loop, where a `Branch` can be constructed directly.
#[test]
fn test_run_bounds_execution_by_max_steps() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(
        &input,
        sst("fn @main() {\n  lanes::lanes.const_zone 0x00000000\n  \
             lanes::lanes.cz\n  cpu::cpu.halt\n}\n"),
    )
    .unwrap();

    cmd()
        .args(["run", input.to_str().unwrap()])
        .args(["--max-steps", "1"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("budget"));

    // The same program finishes when the budget allows it.
    cmd()
        .args(["run", input.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("halted"));
}

/// The ops the machine does not simulate are still reported, which is what
/// `--effects` is for.
#[test]
fn test_run_effects_reports_unsimulated_ops() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(
        &input,
        sst("fn @main() {\n  lanes::lanes.const_zone 0x00000000\n  \
             lanes::lanes.cz\n  cpu::cpu.halt\n}\n"),
    )
    .unwrap();

    cmd()
        .args(["run", input.to_str().unwrap(), "--effects"])
        .assert()
        .success()
        .stderr(predicate::str::contains("NotSimulated"));
}
