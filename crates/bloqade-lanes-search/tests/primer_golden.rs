//! Golden-file test for the primer generator.
//!
//! Feeds curated stub registration files through the same parsing
//! pipeline the binary uses, and compares to a committed expected.md.
//! The live `policies/primer.md` is *not* the test oracle — that file
//! tracks the real DSL surface, which evolves under separate review.

use assert_cmd::Command;
use std::path::PathBuf;

#[test]
fn primer_generator_matches_golden_for_curated_stubs() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let golden_input = manifest.join("tests/fixtures/primer/input");
    let expected_path = manifest.join("tests/fixtures/primer/expected.md");
    let expected = std::fs::read_to_string(&expected_path).expect("read expected.md");

    let out = Command::cargo_bin("policies-primer")
        .unwrap()
        .arg("--input-dir")
        .arg(&golden_input)
        .arg("--stdout")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let actual = String::from_utf8(out).unwrap();
    if actual.trim() != expected.trim() {
        panic!("primer-golden mismatch.\n--- expected ---\n{expected}\n--- actual ---\n{actual}");
    }
}
