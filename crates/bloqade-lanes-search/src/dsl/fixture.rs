//! Shared problem-fixture loader for the Move and Target Policy DSLs.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §6.
//!
//! Fixture files are self-contained JSON documents discriminated by a
//! top-level `"kind"` field (`"move"` or `"target"`). The `arch` field is
//! a path resolved relative to the fixture file, allowing one ArchSpec
//! JSON to back many fixtures.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum FixtureError {
    #[error("reading fixture {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing fixture {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("schema version mismatch in {path}: got {got}, expected {expected}")]
    SchemaVersion {
        path: PathBuf,
        got: u32,
        expected: u32,
    },
    #[error("resolving arch path '{arch}' relative to fixture {path}: {reason}")]
    ArchResolve {
        path: PathBuf,
        arch: String,
        reason: String,
    },
}

const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Problem {
    Move(MoveProblem),
    Target(TargetProblem),
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct MoveProblem {
    pub v: u32,
    pub arch: String,
    pub initial: Vec<(u32, [i32; 3])>,
    pub target: Vec<(u32, [i32; 3])>,
    #[serde(default)]
    pub blocked: Vec<[i32; 3]>,
    #[serde(default)]
    pub budget: Option<Budget>,
    #[serde(default)]
    pub policy_params: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct TargetProblem {
    pub v: u32,
    pub arch: String,
    pub current_placement: Vec<(u32, [i32; 3])>,
    pub controls: Vec<u32>,
    pub targets: Vec<u32>,
    #[serde(default)]
    pub lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
    #[serde(default)]
    pub cz_stage_index: u32,
    #[serde(default)]
    pub policy_params: serde_json::Value,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct Budget {
    pub max_expansions: u64,
    pub timeout_s: f64,
}

impl Problem {
    pub fn schema_version(&self) -> u32 {
        match self {
            Problem::Move(p) => p.v,
            Problem::Target(p) => p.v,
        }
    }

    pub fn arch_path_str(&self) -> &str {
        match self {
            Problem::Move(p) => &p.arch,
            Problem::Target(p) => &p.arch,
        }
    }
}

/// Load and validate a problem-fixture file.
///
/// Returns the parsed `Problem` and the resolved absolute path to the
/// referenced ArchSpec JSON. The arch path is resolved relative to the
/// fixture file's parent directory.
pub fn load(path: &Path) -> Result<(Problem, PathBuf), FixtureError> {
    let bytes = std::fs::read(path).map_err(|e| FixtureError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let problem: Problem = serde_json::from_slice(&bytes).map_err(|e| FixtureError::Parse {
        path: path.to_path_buf(),
        source: e,
    })?;
    if problem.schema_version() != SCHEMA_VERSION {
        return Err(FixtureError::SchemaVersion {
            path: path.to_path_buf(),
            got: problem.schema_version(),
            expected: SCHEMA_VERSION,
        });
    }
    let arch_str = problem.arch_path_str();
    let parent = path.parent().ok_or_else(|| FixtureError::ArchResolve {
        path: path.to_path_buf(),
        arch: arch_str.into(),
        reason: "fixture path has no parent".into(),
    })?;
    let arch_path =
        parent
            .join(arch_str)
            .canonicalize()
            .map_err(|e| FixtureError::ArchResolve {
                path: path.to_path_buf(),
                arch: arch_str.into(),
                reason: e.to_string(),
            })?;
    Ok((problem, arch_path))
}

/// `schemars`-generated JSON Schema for the Problem enum, used by the
/// primer generator's AUTOGEN: schema section (Task 23).
pub fn json_schema_pretty() -> String {
    let schema = schemars::schema_for!(Problem);
    serde_json::to_string_pretty(&schema).expect("schema serialize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_arch_stub(dir: &Path) -> PathBuf {
        let p = dir.join("arch.json");
        std::fs::write(&p, br#"{"version": 1, "kind": "stub"}"#).unwrap();
        p
    }

    fn write_fixture(dir: &Path, body: &str) -> PathBuf {
        let p = dir.join("problem.json");
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p
    }

    #[test]
    fn loads_move_fixture_and_resolves_arch() {
        let tmp = TempDir::new().unwrap();
        write_arch_stub(tmp.path());
        let p = write_fixture(
            tmp.path(),
            r#"{"v":1,"kind":"move","arch":"arch.json",
                "initial":[[0,[1,0,0]]],"target":[[0,[1,0,1]]],
                "blocked":[],"policy_params":{}}"#,
        );
        let (prob, arch_path) = load(&p).unwrap();
        match prob {
            Problem::Move(m) => assert_eq!(m.initial.len(), 1),
            _ => panic!("expected Move"),
        }
        assert!(arch_path.ends_with("arch.json"));
    }

    #[test]
    fn loads_target_fixture() {
        let tmp = TempDir::new().unwrap();
        write_arch_stub(tmp.path());
        let p = write_fixture(
            tmp.path(),
            r#"{"v":1,"kind":"target","arch":"arch.json",
                "current_placement":[[0,[1,0,0]]],"controls":[0],"targets":[1]}"#,
        );
        let (prob, _) = load(&p).unwrap();
        assert!(matches!(prob, Problem::Target(_)));
    }

    #[test]
    fn rejects_unknown_kind() {
        let tmp = TempDir::new().unwrap();
        let p = write_fixture(tmp.path(), r#"{"v":1,"kind":"flavor","arch":"x.json"}"#);
        let err = load(&p).unwrap_err();
        assert!(matches!(err, FixtureError::Parse { .. }));
    }

    #[test]
    fn rejects_schema_version_mismatch() {
        let tmp = TempDir::new().unwrap();
        write_arch_stub(tmp.path());
        let p = write_fixture(
            tmp.path(),
            r#"{"v":99,"kind":"move","arch":"arch.json",
                "initial":[],"target":[]}"#,
        );
        let err = load(&p).unwrap_err();
        assert!(matches!(err, FixtureError::SchemaVersion { got: 99, .. }));
    }

    #[test]
    fn json_schema_renders() {
        let s = json_schema_pretty();
        // schemars 0.8 inlines tagged-union variants; check that both
        // discriminator values appear and the schema is well-formed.
        assert!(s.contains(r#""move""#), "schema missing move discriminator");
        assert!(
            s.contains(r#""target""#),
            "schema missing target discriminator"
        );
        assert!(s.contains("Budget"), "schema missing Budget definition");
    }
}
