//! Head-to-head comparison of every fixed-target routing strategy.
//!
//! ```text
//! cargo run --release -p bloqade-lanes-search --example router_comparison
//! cargo run --release -p bloqade-lanes-search --example router_comparison -- --csv
//! ```
//!
//! Complements the Python benchmark harness, which measures whole-kernel
//! compilation. This isolates the *router*: same instance, same budget, one
//! strategy at a time, so a difference is attributable to routing rather than
//! to target selection or circuit structure.
//!
//! ## What is measured
//!
//! Every instance is **reachable by construction** — atoms are scattered, then
//! a random walk of legal single-atom slides produces the target. So a
//! solution provably exists and a failure is a real deficiency of the router,
//! not a hard problem. That is what makes the success rate meaningful.
//!
//! | column | meaning |
//! |---|---|
//! | `solved` | instances routed, of those attempted |
//! | `ops` | AOD operations — the primary cost metric |
//! | `xfer_us` | transport time: per operation the slowest lane, summed |
//! | `lanes` | total single-atom moves — **informational only** |
//! | `ms` | wall time of the solve |
//!
//! `lanes` is not a quality measure. One operation moves a whole rectangle, so
//! packing more atoms per operation raises the lane count while lowering the
//! real cost — optimising for fewer lanes optimises for serialisation.
//!
//! Cost columns are summed over *solved* instances only, so they are
//! comparable only between strategies with the same solved count, which is why
//! that count is printed alongside. Everything except `ms` is deterministic.

use std::sync::Arc;
use std::time::Instant;

use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::push_rotate::instances::{Instance, generate};
use bloqade_lanes_search::search::engine::SearchEngine;
use bloqade_lanes_search::search::move_search::MoveSearch;
use bloqade_lanes_search::search::options::{SolveOptions, Strategy};
use bloqade_lanes_search::search::result::SolveStatus;
use bloqade_lanes_search::search::target_solver::TargetSolver;

const PHYSICAL: &str =
    include_str!("../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json");
const LOGICAL: &str =
    include_str!("../../../python/bloqade/lanes/arch/gemini/logical/_logical_spec.json");

/// Expansion budget every search strategy gets, held equal so the comparison
/// is not a budget comparison.
const MAX_EXPANSIONS: u32 = 5_000;

struct Outcome {
    solved: bool,
    ops: usize,
    lanes: usize,
    xfer_us: f64,
    micros: u128,
}

/// Duration of one AOD operation: its slowest lane, since the atoms move
/// concurrently and the operation ends when the last one arrives.
fn layer_duration(index: &LaneIndex, lanes: &[LaneAddr]) -> f64 {
    lanes
        .iter()
        .filter_map(|l| index.lane_duration_us(l))
        .fold(0.0f64, f64::max)
}

fn run(engine: &Arc<SearchEngine>, instance: &Instance, strategy: Strategy) -> Outcome {
    let search = MoveSearch::default().with_options(SolveOptions {
        strategy,
        ..Default::default()
    });
    let solver = TargetSolver::new(Arc::clone(engine), search);
    let decode = |v: &[(u32, u64)]| -> Vec<(u32, LocationAddr)> {
        v.iter()
            .map(|&(q, l)| (q, LocationAddr::decode(l)))
            .collect()
    };

    let start = Instant::now();
    let result = solver.solve(
        decode(&instance.initial),
        decode(&instance.target),
        Vec::new(),
        Some(MAX_EXPANSIONS),
    );
    let micros = start.elapsed().as_micros();

    match result {
        Ok(r) => Outcome {
            solved: r.status == SolveStatus::Solved,
            ops: r.move_layers.len(),
            lanes: r.move_layers.iter().map(|m| m.decode().len()).sum(),
            xfer_us: r
                .move_layers
                .iter()
                .map(|m| layer_duration(engine.index(), &m.decode()))
                .sum(),
            micros,
        },
        Err(_) => Outcome {
            solved: false,
            ops: 0,
            lanes: 0,
            xfer_us: 0.0,
            micros,
        },
    }
}

fn main() {
    let csv = std::env::args().any(|a| a == "--csv");
    let strategies = [
        ("astar", Strategy::AStar),
        ("ids", Strategy::Ids),
        ("dfs", Strategy::HeuristicDfs),
        ("entropy", Strategy::Entropy),
        ("push-rotate", Strategy::PushRotate),
    ];

    if csv {
        println!("instance,arch,atoms,strategy,solved,ops,lanes,xfer_us,micros");
    } else {
        println!(
            "\n{:<16} {:<13} {:>9} {:>7} {:>10} {:>8} {:>9}",
            "group", "strategy", "solved", "ops", "xfer_us", "lanes", "ms"
        );
        println!("{}", "-".repeat(78));
    }

    for (arch_name, arch) in [("physical", PHYSICAL), ("logical", LOGICAL)] {
        let engine = Arc::new(SearchEngine::from_json(arch).expect("spec parses"));
        for &k in &[1usize, 2, 4, 8, 16] {
            let mut totals = vec![(0usize, 0usize, 0usize, 0.0f64, 0u128); strategies.len()];
            for seed in 0..5u64 {
                let id = format!("{arch_name}/k{k}/seed{seed}");
                let Some(instance) = generate(id.clone(), arch, k, 4 * k, seed) else {
                    continue;
                };
                for (si, (name, strategy)) in strategies.iter().enumerate() {
                    let o = run(&engine, &instance, *strategy);
                    if csv {
                        println!(
                            "{id},{arch_name},{k},{name},{},{},{},{:.1},{}",
                            o.solved, o.ops, o.lanes, o.xfer_us, o.micros
                        );
                    }
                    let t = &mut totals[si];
                    if o.solved {
                        t.0 += 1;
                        t.1 += o.ops;
                        t.2 += o.lanes;
                        t.3 += o.xfer_us;
                    }
                    t.4 += o.micros;
                }
            }
            if !csv {
                for (si, (name, _)) in strategies.iter().enumerate() {
                    let (solved, ops, lanes, xfer, micros) = totals[si];
                    println!(
                        "{:<16} {:<13} {:>4}/{:<4} {:>7} {:>10.0} {:>8} {:>9.1}",
                        format!("{arch_name}/k{k}"),
                        name,
                        solved,
                        5,
                        ops,
                        xfer,
                        lanes,
                        micros as f64 / 1000.0
                    );
                }
                println!();
            }
        }
    }

    if !csv {
        println!(
            "ops = AOD operations (primary cost). lanes is informational: more atoms\n\
             per operation raises it while LOWERING cost. Summed over SOLVED instances\n\
             only, so compare only between strategies with the same solved count."
        );
    }
}
