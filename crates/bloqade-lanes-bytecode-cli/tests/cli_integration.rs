use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

fn cmd() -> Command {
    assert_cmd::cargo_bin_cmd!("bloqade-bytecode")
}

/// A small program for basic command tests.
const SAMPLE_PROGRAM: &str = "\
.version 1.0
const_loc 0x00000102
const_lane 0x0000000100030002
halt
";

/// All 24 instructions exercised in a single program.
/// Ordered so that initial_fill comes right after constants (structurally valid).
const ALL_INSTRUCTIONS_PROGRAM: &str = "\
.version 1.0
const_float 1.5
const_int 42
const_loc 0x00010002
const_lane 0x8000000000010002
const_zone 0x00000003
initial_fill 3
pop
dup
swap
fill 2
move 1
local_r 4
local_rz 2
global_r
global_rz
cz
measure 1
await_measure
new_array 2 10 20
get_item 2
set_detector
set_observable
return
halt
";

/// A program with addresses valid for the test arch spec (word_id=0, site_id in 0..5, bus_id=0).
const ARCH_VALID_PROGRAM: &str = "\
.version 1.0
const_loc 0x00000001
const_zone 0x00000000
halt
";

#[test]
fn test_assemble_creates_binary() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    let output = dir.path().join("prog.bin");
    fs::write(&input, ALL_INSTRUCTIONS_PROGRAM).unwrap();

    cmd()
        .args([
            "assemble",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stderr(predicate::str::contains("assembled 24 instructions"));

    let bytes = fs::read(&output).unwrap();
    // Should start with BLQD magic bytes
    assert_eq!(&bytes[..4], b"BLQD");
}

#[test]
fn test_disassemble_to_stdout() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    fs::write(&input_txt, ALL_INSTRUCTIONS_PROGRAM).unwrap();

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
        .stdout(predicate::str::contains("const_float 1.5"))
        .stdout(predicate::str::contains("const_int 42"))
        .stdout(predicate::str::contains("const_loc"))
        .stdout(predicate::str::contains("const_lane"))
        .stdout(predicate::str::contains("const_zone"))
        .stdout(predicate::str::contains("new_array 2 10 20"))
        .stdout(predicate::str::contains("halt"));
}

#[test]
fn test_disassemble_to_file() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    let output_txt = dir.path().join("out.sst");
    fs::write(&input_txt, ALL_INSTRUCTIONS_PROGRAM).unwrap();

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
        .stderr(predicate::str::contains("disassembled 24 instructions"));

    let text = fs::read_to_string(&output_txt).unwrap();
    // Spot-check all instruction categories are present
    assert!(text.contains("const_float"));
    assert!(text.contains("const_int"));
    assert!(text.contains("const_loc"));
    assert!(text.contains("const_lane"));
    assert!(text.contains("const_zone"));
    assert!(text.contains("pop"));
    assert!(text.contains("dup"));
    assert!(text.contains("swap"));
    assert!(text.contains("initial_fill 3"));
    assert!(text.contains("fill 2"));
    assert!(text.contains("move 1"));
    assert!(text.contains("local_r 4"));
    assert!(text.contains("local_rz 2"));
    assert!(text.contains("global_r"));
    assert!(text.contains("global_rz"));
    assert!(text.contains("cz"));
    assert!(text.contains("measure 1"));
    assert!(text.contains("await_measure"));
    assert!(text.contains("new_array 2 10 20"));
    assert!(text.contains("get_item 2"));
    assert!(text.contains("set_detector"));
    assert!(text.contains("set_observable"));
    assert!(text.contains("return"));
    assert!(text.contains("halt"));
}

#[test]
fn test_round_trip_assemble_disassemble() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    let output_txt = dir.path().join("out.sst");
    fs::write(&input_txt, ALL_INSTRUCTIONS_PROGRAM).unwrap();

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

#[test]
fn test_validate_text_file() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(&input, ALL_INSTRUCTIONS_PROGRAM).unwrap();

    cmd()
        .args(["validate", input.to_str().unwrap()])
        .assert()
        .success()
        .stderr(predicate::str::contains("valid (24 instructions)"));
}

#[test]
fn test_validate_binary_file() {
    let dir = TempDir::new().unwrap();
    let input_txt = dir.path().join("prog.sst");
    let binary = dir.path().join("prog.bin");
    fs::write(&input_txt, ALL_INSTRUCTIONS_PROGRAM).unwrap();

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
        .stderr(predicate::str::contains("valid (24 instructions)"));
}

#[test]
fn test_validate_with_arch_spec() {
    let dir = TempDir::new().unwrap();
    let input = dir.path().join("prog.sst");
    fs::write(&input, ARCH_VALID_PROGRAM).unwrap();

    let arch_json = r#"{
        "version": 1,
        "geometry": {
            "sites_per_word": 5,
            "words": [
                {
                    "positions": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [] },
                    "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
                }
            ]
        },
        "buses": {
            "site_buses": [
                { "src": [0, 1], "dst": [3, 4] }
            ],
            "word_buses": []
        },
        "words_with_site_buses": [0],
        "sites_with_word_buses": [],
        "zones": [
            { "words": [0] }
        ],
        "entangling_zones": [],
        "measurement_mode_zones": [0]
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
    fs::write(&input, SAMPLE_PROGRAM).unwrap();

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
    fs::write(&input, SAMPLE_PROGRAM).unwrap();

    let arch_json = r#"{
        "version": 1,
        "geometry": {
            "sites_per_word": 2,
            "words": [
                {
                    "positions": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [], "y_spacing": [2.0] },
                    "site_indices": [[0, 0], [0, 1]]
                }
            ]
        },
        "buses": { "site_buses": [], "word_buses": [] },
        "words_with_site_buses": [],
        "sites_with_word_buses": [],
        "zones": [{ "words": [0] }],
        "entangling_zones": [],
        "measurement_mode_zones": [0]
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
    fs::write(&input, ".version 1\nfoobar_invalid\n").unwrap();

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
    let arch_json = r#"{"version":1,"geometry":{"sites_per_word":2,"words":[{"positions":{"x_start":1.0,"y_start":2.0,"x_spacing":[],"y_spacing":[2.0]},"site_indices":[[0,0],[0,1]]}]},"buses":{"site_buses":[],"word_buses":[]},"words_with_site_buses":[],"sites_with_word_buses":[],"zones":[{"words":[0]}],"entangling_zones":[],"measurement_mode_zones":[0]}"#;
    let arch_path = dir.path().join("arch.json");
    fs::write(&arch_path, arch_json).unwrap();

    cmd()
        .args(["arch", arch_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("ArchSpec v1.0"))
        .stdout(predicate::str::contains("1 word(s), 2 sites/word"))
        .stdout(predicate::str::contains("site_indices: (0,0) (0,1)"))
        .stdout(predicate::str::contains("Zone 0: words=[0]"));
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
        "version": 1,
        "geometry": {
            "sites_per_word": 5,
            "words": [
                {
                    "positions": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [] },
                    "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
                }
            ]
        },
        "buses": {
            "site_buses": [
                { "src": [0, 1], "dst": [3, 4] }
            ],
            "word_buses": []
        },
        "words_with_site_buses": [0],
        "sites_with_word_buses": [],
        "zones": [
            { "words": [0] }
        ],
        "entangling_zones": [],
        "measurement_mode_zones": [0]
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
    // measurement_mode_zones[0] is not zone 0 — should fail validation
    let arch_json = r#"{
        "version": 1,
        "geometry": {
            "sites_per_word": 2,
            "words": [
                {
                    "positions": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [], "y_spacing": [2.0] },
                    "site_indices": [[0, 0], [0, 1]]
                }
            ]
        },
        "buses": { "site_buses": [], "word_buses": [] },
        "words_with_site_buses": [],
        "sites_with_word_buses": [],
        "zones": [{ "words": [0] }, { "words": [0] }],
        "entangling_zones": [],
        "measurement_mode_zones": [1]
    }"#;
    let arch_path = dir.path().join("bad_arch.json");
    fs::write(&arch_path, arch_json).unwrap();

    cmd()
        .args(["arch", "validate", arch_path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
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
    fs::write(&input, ".version 2.3\nconst_int 99\nhalt\n").unwrap();

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
        .stdout(predicate::str::contains(".version 2.3"));
}
