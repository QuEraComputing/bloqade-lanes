//! Plan C snapshot-fixture regression test.
//!
//! Walks `policies/fixtures/{move,target}/<size>/` and for every
//! `expected.<policy>.json` runs the matching policy against
//! `problem.json` and structurally compares the result.
//!
//! Comparison fields (Plan C spec §9.3):
//!   - Move:   {status, halt_reason, expansions, max_depth}
//!   - Target: {ok, num_candidates, first_candidate_size}
//!
//! Failure messages include a hint to run `just regenerate-fixtures`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::fixture::{self, Problem};
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{NoOpMoveObserver, PolicyOptions, solve_with_policy};
use bloqade_lanes_search::target_generator_dsl::{NoOpTargetObserver, run_target_policy};

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/fixtures")
}

fn policies_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../policies/reference")
}

fn loc_from_triple(t: &[i32; 3]) -> LocationAddr {
    LocationAddr {
        zone_id: t[0] as u32,
        word_id: t[1] as u32,
        site_id: t[2] as u32,
    }
}

#[derive(serde::Deserialize, PartialEq, Debug)]
struct ExpectedMove {
    status: String,
    halt_reason: Option<String>,
    expansions: u64,
    max_depth: u32,
}

#[derive(serde::Deserialize, PartialEq, Debug)]
struct ExpectedTarget {
    ok: bool,
    num_candidates: usize,
    first_candidate_size: usize,
}

#[test]
fn snapshot_corpus_passes_structural_match() {
    let root = fixture_root();
    let mut failures: Vec<String> = Vec::new();
    let mut total = 0;

    for kind_dir in ["move", "target"].iter().map(|k| root.join(k)) {
        if !kind_dir.exists() {
            continue;
        }
        for size in std::fs::read_dir(&kind_dir).unwrap().flatten() {
            let size_path = size.path();
            if !size_path.is_dir() {
                continue;
            }
            let problem = size_path.join("problem.json");
            if !problem.exists() {
                continue;
            }
            for entry in std::fs::read_dir(&size_path).unwrap().flatten() {
                let p = entry.path();
                let name = match p.file_name().and_then(|s| s.to_str()) {
                    Some(n) => n,
                    None => continue,
                };
                let policy_name = match name
                    .strip_prefix("expected.")
                    .and_then(|n| n.strip_suffix(".json"))
                {
                    Some(s) => s,
                    None => continue,
                };
                total += 1;
                let policy_path = policies_dir().join(format!("{policy_name}.star"));
                match run_one(&problem, &p, &policy_path) {
                    Ok(()) => {}
                    Err(msg) => failures.push(format!("{}: {msg}", p.display())),
                }
            }
        }
    }

    assert!(
        total > 0,
        "no snapshot fixtures found at {}",
        root.display()
    );
    if !failures.is_empty() {
        let joined = failures.join("\n  ");
        panic!(
            "snapshot mismatches ({} of {}):\n  {joined}\n\n\
             Hint: if these are intentional baseline shifts, run `just regenerate-fixtures`.",
            failures.len(),
            total
        );
    }
}

fn run_one(problem_path: &Path, expected_path: &Path, policy_path: &Path) -> Result<(), String> {
    let (parsed, arch_path) = fixture::load(problem_path).map_err(|e| e.to_string())?;
    let arch_json = std::fs::read_to_string(&arch_path).map_err(|e| e.to_string())?;
    let arch = ArchSpec::from_json(&arch_json).map_err(|e| e.to_string())?;
    match parsed {
        Problem::Move(mp) => {
            let index = Arc::new(LaneIndex::new(arch));
            let initial: Vec<_> = mp
                .initial
                .iter()
                .map(|(q, t)| (*q, loc_from_triple(t)))
                .collect();
            let target: Vec<_> = mp
                .target
                .iter()
                .map(|(q, t)| (*q, loc_from_triple(t)))
                .collect();
            let blocked: Vec<_> = mp.blocked.iter().map(loc_from_triple).collect();
            let opts = PolicyOptions {
                policy_path: policy_path.display().to_string(),
                sandbox: SandboxConfig::default(),
                policy_params: mp.policy_params.clone(),
                max_expansions: mp.budget.as_ref().map(|b| b.max_expansions).unwrap_or(5000),
                timeout_s: Some(mp.budget.as_ref().map(|b| b.timeout_s).unwrap_or(10.0)),
            };
            let mut obs = NoOpMoveObserver;
            let res = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
                .map_err(|e| e.to_string())?;
            let expected: ExpectedMove =
                serde_json::from_slice(&std::fs::read(expected_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let actual = ExpectedMove {
                status: status_label(&res.status).into(),
                halt_reason: halt_reason(&res.status),
                expansions: res.nodes_expanded as u64,
                max_depth: res.move_layers.len() as u32,
            };
            if actual != expected {
                return Err(format!("expected {expected:?}, got {actual:?}"));
            }
            Ok(())
        }
        Problem::Target(tp) => {
            let index = Arc::new(LaneIndex::new(arch));
            let placement: Vec<_> = tp
                .current_placement
                .iter()
                .map(|(q, t)| (*q, loc_from_triple(t)))
                .collect();
            let cfg = SandboxConfig::default();
            let mut obs = NoOpTargetObserver;
            let result = run_target_policy(
                policy_path,
                index,
                placement,
                tp.controls.clone(),
                tp.targets.clone(),
                tp.lookahead_cz_layers.clone(),
                tp.cz_stage_index,
                tp.policy_params.clone(),
                &cfg,
                &mut obs,
            );
            let expected: ExpectedTarget =
                serde_json::from_slice(&std::fs::read(expected_path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            let (num_candidates, first_candidate_size) = match &result {
                Ok(c) => (c.len(), c.first().map_or(0, |v| v.len())),
                Err(_) => (0, 0),
            };
            let actual = ExpectedTarget {
                ok: result.is_ok(),
                num_candidates,
                first_candidate_size,
            };
            if actual != expected {
                return Err(format!("expected {expected:?}, got {actual:?}"));
            }
            Ok(())
        }
    }
}

fn status_label(s: &bloqade_lanes_search::move_policy_dsl::PolicyStatus) -> &'static str {
    use bloqade_lanes_search::move_policy_dsl::PolicyStatus::*;
    match s {
        Solved => "Solved",
        Unsolvable => "Unsolvable",
        BudgetExhausted => "BudgetExhausted",
        Timeout => "Timeout",
        Fallback(_) => "Fallback",
        SyntaxError(_) => "SyntaxError",
        RuntimeError(_) => "RuntimeError",
        SchemaError(_) => "SchemaError",
        BadPolicy(_) => "BadPolicy",
        StarlarkBudget => "StarlarkBudget",
        StarlarkOOM => "StarlarkOOM",
    }
}

fn halt_reason(s: &bloqade_lanes_search::move_policy_dsl::PolicyStatus) -> Option<String> {
    use bloqade_lanes_search::move_policy_dsl::PolicyStatus::*;
    match s {
        Solved => Some("policy_halt".into()),
        Fallback(r) => Some(r.clone()),
        _ => None,
    }
}
