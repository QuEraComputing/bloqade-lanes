//! `eval-policy` subcommand: run-once with summary output.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::fixture::{self, Problem};
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    NoOpMoveObserver, PolicyOptions, PolicyStatus, solve_with_policy,
};
use bloqade_lanes_search::target_generator_dsl::{NoOpTargetObserver, run_target_policy};

use super::output::{EvalEnvelope, TargetEvalEnvelope, print_human_move, print_human_target};

const SCHEMA_VERSION: u32 = 1;

pub fn run_eval_policy(
    policy: &Path,
    problem: &Path,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
    _seed: Option<u64>,
) -> Result<(), String> {
    let (parsed, arch_path) = fixture::load(problem).map_err(|e| format!("error: {e}"))?;
    let arch_json = std::fs::read_to_string(&arch_path)
        .map_err(|e| format!("error: reading arch {}: {e}", arch_path.display()))?;
    let arch = bloqade_lanes_bytecode_core::arch::ArchSpec::from_json(&arch_json)
        .map_err(|e| format!("error: parsing arch {}: {e}", arch_path.display()))?;

    match parsed {
        Problem::Move(mp) => {
            let exit = run_move(
                policy,
                problem,
                mp,
                arch,
                params,
                max_expansions,
                timeout_s,
                json,
            )?;
            std::process::exit(exit);
        }
        Problem::Target(tp) => {
            let exit = run_target(policy, problem, tp, arch, params, json)?;
            std::process::exit(exit);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_move(
    policy: &Path,
    problem: &Path,
    mp: bloqade_lanes_search::fixture::MoveProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
) -> Result<i32, String> {
    let index = Arc::new(LaneIndex::new(arch));
    let initial = mp.initial.iter().map(|(q, [l, r, c])| {
        (
            *q,
            LocationAddr {
                zone_id: *l as u32,
                word_id: *r as u32,
                site_id: *c as u32,
            },
        )
    });
    let target = mp.target.iter().map(|(q, [l, r, c])| {
        (
            *q,
            LocationAddr {
                zone_id: *l as u32,
                word_id: *r as u32,
                site_id: *c as u32,
            },
        )
    });
    let blocked = mp.blocked.iter().map(|[l, r, c]| LocationAddr {
        zone_id: *l as u32,
        word_id: *r as u32,
        site_id: *c as u32,
    });

    let opts = PolicyOptions {
        policy_path: policy.display().to_string(),
        sandbox: SandboxConfig::default(),
        policy_params: load_params(params, &mp.policy_params)?,
        max_expansions: max_expansions
            .or(mp.budget.as_ref().map(|b| b.max_expansions))
            .unwrap_or(5_000),
        timeout_s: Some(
            timeout_s
                .or(mp.budget.as_ref().map(|b| b.timeout_s))
                .unwrap_or(10.0),
        ),
    };
    let mut obs = NoOpMoveObserver;
    let t0 = Instant::now();
    let res = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
        .map_err(|e| format!("error: {e}"))?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let status_str = status_label(&res.status);
    let halt_reason = halt_reason(&res.status);
    let env = EvalEnvelope {
        v: SCHEMA_VERSION,
        kind: "move",
        policy: policy.to_str().unwrap_or(""),
        problem: problem.to_str().unwrap_or(""),
        status: status_str,
        halt_reason: halt_reason.as_deref(),
        expansions: res.nodes_expanded as u64,
        max_depth: res.move_layers.len() as u32,
        wall_time_ms: wall_ms,
    };
    if json {
        println!("{}", serde_json::to_string(&env).unwrap());
    } else {
        print_human_move(&env);
    }
    Ok(exit_code(&res.status))
}

fn run_target(
    policy: &Path,
    problem: &Path,
    tp: bloqade_lanes_search::fixture::TargetProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    json: bool,
) -> Result<i32, String> {
    let index = Arc::new(LaneIndex::new(arch));
    let placement: Vec<(u32, LocationAddr)> = tp
        .current_placement
        .iter()
        .map(|(q, [l, r, c])| {
            (
                *q,
                LocationAddr {
                    zone_id: *l as u32,
                    word_id: *r as u32,
                    site_id: *c as u32,
                },
            )
        })
        .collect();
    let cfg = SandboxConfig::default();
    let mut obs = NoOpTargetObserver;
    let t0 = Instant::now();
    let result = run_target_policy(
        policy,
        index,
        placement,
        tp.controls.clone(),
        tp.targets.clone(),
        tp.lookahead_cz_layers.clone(),
        tp.cz_stage_index,
        load_params(params, &tp.policy_params)?,
        &cfg,
        &mut obs,
    );
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let (num_candidates, first_candidate_size) = match &result {
        Ok(cands) => (cands.len(), cands.first().map_or(0, |c| c.len())),
        Err(_) => (0, 0),
    };
    let env = TargetEvalEnvelope {
        v: SCHEMA_VERSION,
        kind: "target",
        policy: policy.to_str().unwrap_or(""),
        problem: problem.to_str().unwrap_or(""),
        ok: result.is_ok(),
        num_candidates,
        first_candidate_size,
        wall_time_ms: wall_ms,
    };
    if json {
        println!("{}", serde_json::to_string(&env).unwrap());
    } else {
        print_human_target(&env);
    }
    Ok(if result.is_ok() { 0 } else { 2 })
}

pub(crate) fn load_params(
    file: Option<&Path>,
    fallback: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    match file {
        Some(p) => {
            let bytes =
                std::fs::read(p).map_err(|e| format!("error: reading {}: {e}", p.display()))?;
            serde_json::from_slice(&bytes)
                .map_err(|e| format!("error: parsing {}: {e}", p.display()))
        }
        None => Ok(fallback.clone()),
    }
}

fn status_label(s: &PolicyStatus) -> &'static str {
    match s {
        PolicyStatus::Solved => "Solved",
        PolicyStatus::Unsolvable => "Unsolvable",
        PolicyStatus::BudgetExhausted => "BudgetExhausted",
        PolicyStatus::Timeout => "Timeout",
        PolicyStatus::Fallback(_) => "Fallback",
        PolicyStatus::SyntaxError(_) => "SyntaxError",
        PolicyStatus::RuntimeError(_) => "RuntimeError",
        PolicyStatus::SchemaError(_) => "SchemaError",
        PolicyStatus::BadPolicy(_) => "BadPolicy",
        PolicyStatus::StarlarkBudget => "StarlarkBudget",
        PolicyStatus::StarlarkOOM => "StarlarkOOM",
    }
}

fn halt_reason(s: &PolicyStatus) -> Option<String> {
    match s {
        PolicyStatus::Solved => Some("policy_halt".into()),
        PolicyStatus::Fallback(r) => Some(r.clone()),
        _ => None,
    }
}

fn exit_code(s: &PolicyStatus) -> i32 {
    match s {
        PolicyStatus::Solved => 0,
        PolicyStatus::BudgetExhausted
        | PolicyStatus::Timeout
        | PolicyStatus::Fallback(_)
        | PolicyStatus::Unsolvable => 2,
        _ => 1,
    }
}
