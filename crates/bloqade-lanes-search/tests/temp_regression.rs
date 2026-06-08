//! Temporary regression suite for the search-crate refactor.
//!
//! Loads `tests/temp_regression/arch.json` once, then replays every
//! `tests/temp_regression/fixtures/case_*.json` through `MoveSolver::solve`
//! and asserts bitwise equality of the captured outputs. See the sibling
//! `tests/temp_regression/README.md` for regeneration instructions and the
//! lifecycle of this suite (it is meant to be deleted after the refactor).
//!
//! A single failing fixture aborts the test with a per-field diff. All
//! fixtures run in one `#[test]` so a regen-induced shape mismatch fails
//! fast rather than spamming the test runner.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_search::{
    DeadlockPolicy,
    search::solve::{
        EntropyOptions, InnerStrategy, MoveSolver, SolveOptions, SolveStatus, Strategy,
    },
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    initial: BTreeMap<String, u64>,
    target: BTreeMap<String, u64>,
    blocked: Vec<u64>,
    options: FixtureOptions,
    entropy_options: FixtureEntropyOptions,
    max_expansions: Option<u32>,
    expected: ExpectedResult,
}

#[derive(Deserialize)]
struct FixtureOptions {
    strategy: String,
    weight: f64,
    restarts: u32,
    deadlock_policy: String,
    lookahead: bool,
    top_c: Option<usize>,
}

#[derive(Deserialize)]
struct FixtureEntropyOptions {
    max_movesets_per_group: usize,
    max_goal_candidates: usize,
    w_t: f64,
    collect_entropy_trace: bool,
}

#[derive(Deserialize)]
struct ExpectedResult {
    status: String,
    cost: f64,
    nodes_expanded: u32,
    deadlocks: u32,
    goal_config: BTreeMap<String, u64>,
    move_layers: Vec<Vec<u64>>,
}

fn temp_regression_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/temp_regression")
}

fn parse_strategy(s: &str) -> Strategy {
    match s {
        "astar" => Strategy::AStar,
        "dfs" => Strategy::HeuristicDfs,
        "bfs" => Strategy::Bfs,
        "greedy" => Strategy::GreedyBestFirst,
        "ids" => Strategy::Ids,
        "entropy" => Strategy::Entropy,
        "cascade-ids" => Strategy::Cascade {
            inner: InnerStrategy::Ids,
        },
        "cascade-dfs" => Strategy::Cascade {
            inner: InnerStrategy::Dfs,
        },
        "cascade-entropy" => Strategy::Cascade {
            inner: InnerStrategy::Entropy,
        },
        other => panic!("unknown strategy `{other}` in fixture"),
    }
}

fn parse_deadlock(s: &str) -> DeadlockPolicy {
    match s {
        "skip" => DeadlockPolicy::Skip,
        "move_blockers" => DeadlockPolicy::MoveBlockers,
        "all_moves" => DeadlockPolicy::AllMoves,
        other => panic!("unknown deadlock policy `{other}` in fixture"),
    }
}

fn status_str(status: SolveStatus) -> &'static str {
    match status {
        SolveStatus::Solved => "solved",
        SolveStatus::Unsolvable => "unsolvable",
        SolveStatus::BudgetExceeded => "budget_exceeded",
    }
}

fn decode_qubits(m: &BTreeMap<String, u64>) -> Vec<(u32, LocationAddr)> {
    m.iter()
        .map(|(k, &bits)| {
            let qid: u32 = k.parse().unwrap_or_else(|e| {
                panic!("fixture qubit id `{k}` is not a u32: {e}");
            });
            (qid, LocationAddr::decode(bits))
        })
        .collect()
}

fn run_fixture(solver: &MoveSolver, fx: &Fixture) -> Result<(), String> {
    let initial = decode_qubits(&fx.initial);
    let target = decode_qubits(&fx.target);
    let blocked: Vec<LocationAddr> = fx
        .blocked
        .iter()
        .map(|&b| LocationAddr::decode(b))
        .collect();

    let opts = SolveOptions {
        strategy: parse_strategy(&fx.options.strategy),
        weight: fx.options.weight,
        restarts: fx.options.restarts,
        deadlock_policy: parse_deadlock(&fx.options.deadlock_policy),
        lookahead: fx.options.lookahead,
        top_c: fx.options.top_c,
    };
    let ent = EntropyOptions {
        max_movesets_per_group: fx.entropy_options.max_movesets_per_group,
        max_goal_candidates: fx.entropy_options.max_goal_candidates,
        w_t: fx.entropy_options.w_t,
        collect_entropy_trace: fx.entropy_options.collect_entropy_trace,
    };

    let result = solver
        .solve(
            initial,
            target,
            blocked,
            fx.max_expansions,
            &opts,
            Some(&ent),
        )
        .map_err(|e| format!("solver returned error: {e:?}"))?;

    let actual_status = status_str(result.status);
    if actual_status != fx.expected.status {
        return Err(format!(
            "status: expected {expected} got {actual}",
            expected = fx.expected.status,
            actual = actual_status,
        ));
    }
    // Bitwise float compare — refactors that preserve numerics will pass,
    // refactors that perturb cost arithmetic (different traversal order,
    // float reordering) will fail loudly. That's the point.
    if result.cost.to_bits() != fx.expected.cost.to_bits() {
        return Err(format!(
            "cost: expected {} got {}",
            fx.expected.cost, result.cost
        ));
    }
    if result.nodes_expanded != fx.expected.nodes_expanded {
        return Err(format!(
            "nodes_expanded: expected {} got {}",
            fx.expected.nodes_expanded, result.nodes_expanded
        ));
    }
    if result.deadlocks != fx.expected.deadlocks {
        return Err(format!(
            "deadlocks: expected {} got {}",
            fx.expected.deadlocks, result.deadlocks
        ));
    }

    let actual_goal: BTreeMap<String, u64> = result
        .goal_config
        .iter()
        .map(|(qid, loc)| (qid.to_string(), loc.encode()))
        .collect();
    if actual_goal != fx.expected.goal_config {
        return Err(format!(
            "goal_config mismatch:\n  expected {:?}\n  actual   {:?}",
            fx.expected.goal_config, actual_goal,
        ));
    }

    let actual_layers: Vec<Vec<u64>> = result
        .move_layers
        .iter()
        .map(|ms| ms.encoded_lanes().to_vec())
        .collect();
    if actual_layers != fx.expected.move_layers {
        return Err(format!(
            "move_layers mismatch:\n  expected {:?}\n  actual   {:?}",
            fx.expected.move_layers, actual_layers,
        ));
    }

    let _ = &fx.name; // name is used in the outer error path
    Ok(())
}

#[test]
fn fixtures_replay() {
    let dir = temp_regression_dir();
    let arch_path = dir.join("arch.json");
    let arch_json = fs::read_to_string(&arch_path)
        .unwrap_or_else(|e| panic!("read {arch_path:?}: {e} — run generate_fixtures.py first"));
    let solver =
        MoveSolver::from_json(&arch_json).unwrap_or_else(|e| panic!("parse {arch_path:?}: {e}"));

    let fixtures_dir = dir.join("fixtures");
    let mut paths: Vec<PathBuf> = fs::read_dir(&fixtures_dir)
        .unwrap_or_else(|e| panic!("read fixtures dir {fixtures_dir:?}: {e}"))
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    assert!(
        !paths.is_empty(),
        "no fixtures in {fixtures_dir:?} — run generate_fixtures.py"
    );

    let mut failures: Vec<String> = Vec::new();
    for path in &paths {
        let body =
            fs::read_to_string(path).unwrap_or_else(|e| panic!("read fixture {path:?}: {e}"));
        let fx: Fixture =
            serde_json::from_str(&body).unwrap_or_else(|e| panic!("parse fixture {path:?}: {e}"));
        if let Err(msg) = run_fixture(&solver, &fx) {
            failures.push(format!("[{}] {msg}", fx.name));
        }
    }

    assert!(
        failures.is_empty(),
        "{}/{} fixture(s) failed:\n{}",
        failures.len(),
        paths.len(),
        failures.join("\n\n"),
    );
}
