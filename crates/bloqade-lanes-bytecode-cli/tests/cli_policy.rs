use assert_cmd::Command;
use predicates::prelude::*;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli_policy")
}

#[test]
fn eval_policy_move_human_output_succeeds() {
    let dir = fixture_dir();
    Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("eval-policy")
        .arg("--policy")
        .arg(dir.join("halt_now.star"))
        .arg("--problem")
        .arg(dir.join("move_problem.json"))
        .assert()
        .success()
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("Solved"));
}

#[test]
fn eval_policy_move_json_emits_envelope() {
    let dir = fixture_dir();
    let out = Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("eval-policy")
        .arg("--json")
        .arg("--policy")
        .arg(dir.join("halt_now.star"))
        .arg("--problem")
        .arg(dir.join("move_problem.json"))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out).expect("valid JSON");
    assert_eq!(v["v"], 1);
    assert_eq!(v["kind"], "move");
    assert_eq!(v["status"], "Solved");
}

#[test]
fn trace_policy_move_ndjson_lines_parse() {
    let dir = fixture_dir();
    let out = Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("trace-policy")
        .arg("--json")
        .arg("--policy")
        .arg(dir.join("halt_now.star"))
        .arg("--problem")
        .arg(dir.join("move_problem.json"))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = std::str::from_utf8(&out).unwrap();
    let lines: Vec<_> = s.lines().collect();
    assert!(!lines.is_empty(), "expected NDJSON output");
    for line in &lines {
        let v: serde_json::Value =
            serde_json::from_str(line).unwrap_or_else(|e| panic!("non-JSON line: {line}: {e}"));
        assert_eq!(v["v"], 1);
        assert!(
            ["init", "step", "builtin", "halt"].contains(&v["kind"].as_str().unwrap_or("")),
            "unexpected kind in line: {line}"
        );
    }
    let kinds: Vec<_> = lines
        .iter()
        .filter_map(|ln| {
            let v: serde_json::Value = serde_json::from_str(ln).ok()?;
            Some(v["kind"].as_str()?.to_string())
        })
        .collect();
    assert_eq!(kinds.first().unwrap(), "init");
    assert_eq!(kinds.last().unwrap(), "halt");
}

#[test]
fn eval_policy_unknown_kind_exits_one() {
    let tmp = tempfile::TempDir::new().unwrap();
    let arch = tmp.path().join("arch.json");
    std::fs::write(&arch, br#"{}"#).unwrap();
    let prob = tmp.path().join("p.json");
    std::fs::write(&prob, br#"{"v":1,"kind":"flavor","arch":"arch.json"}"#).unwrap();
    Command::cargo_bin("bloqade-bytecode")
        .unwrap()
        .arg("eval-policy")
        .arg("--policy")
        .arg("nonexistent.star")
        .arg("--problem")
        .arg(&prob)
        .assert()
        .failure()
        .code(1);
}
