//! Head-to-head comparison of every fixed-target routing strategy.
//!
//! ```text
//! cargo run --release -p bloqade-lanes-search --example router_comparison
//! cargo run --release -p bloqade-lanes-search --example router_comparison -- --csv
//! cargo run --release -p bloqade-lanes-search --example router_comparison -- --bnb
//! ```
//!
//! `--bnb` adds the branch-and-bound evaluation rows from the design spec's
//! *Evaluation* section — frontier `{lifo, dfs, ids}` × ordering `{h_sum, h0}`
//! × objective `{uniform, weighted_duration}` × schedule `{entropy_only,
//! entropy_then_exhaustive}` — plus `entropy_bounded`, the bounded entropy
//! driver they are measured against, `bnb_lifo_{u,w}_p`, the complete
//! schedule with widening unlimited, whose exhaustion is a proof, and
//! `bnb_lifo_{u,w}_hx`, the heuristic generator as stage 0. Those rows
//! also report how many solves were proofs, the per-stage expansion sums and
//! the widest stage a plan needed.
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
use bloqade_lanes_search::drivers::branch_and_bound::Widening;
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::push_rotate::instances::{Instance, generate};
use bloqade_lanes_search::search::engine::SearchEngine;
use bloqade_lanes_search::search::move_search::MoveSearch;
use bloqade_lanes_search::search::options::{
    BnbFrontier, BnbOptions, BnbOrdering, BoundKind, EntropyOptions, ObjectiveKind, ScheduleKind,
    SolveOptions, Strategy,
};
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
    proven: bool,
    stage_expansions: Vec<u32>,
    plan_stage: Option<u8>,
}

/// One row of the comparison: a name and the search it runs.
struct Row {
    name: String,
    search: MoveSearch,
}

fn plain(name: &str, strategy: Strategy) -> Row {
    Row {
        name: name.to_string(),
        search: MoveSearch::default().with_options(SolveOptions {
            strategy,
            ..Default::default()
        }),
    }
}

/// The *Evaluation* product plus the bounded entropy reference.
fn branch_and_bound_rows() -> Vec<Row> {
    let mut rows = vec![Row {
        name: "entropy_bounded".to_string(),
        search: MoveSearch::entropy().with_entropy_options(EntropyOptions {
            completion_bound: Some(BoundKind::WeightedDistance),
            ..EntropyOptions::default()
        }),
    }];
    let frontiers = [
        ("lifo", BnbFrontier::Lifo, BnbOrdering::HopSum),
        ("dfs_hsum", BnbFrontier::Dfs, BnbOrdering::HopSum),
        ("ids_hsum", BnbFrontier::Ids, BnbOrdering::HopSum),
        ("dfs_h0", BnbFrontier::Dfs, BnbOrdering::Bound),
        ("ids_h0", BnbFrontier::Ids, BnbOrdering::Bound),
    ];
    let objectives = [
        ("u", ObjectiveKind::Uniform),
        ("w", ObjectiveKind::WeightedDuration { tau: None }),
    ];
    let schedules = [
        ("e", ScheduleKind::EntropyOnly),
        ("ex", ScheduleKind::EntropyThenExhaustive),
    ];
    for (fname, frontier, ordering) in frontiers {
        for (oname, objective) in objectives {
            for (sname, schedule) in schedules {
                let search = MoveSearch::branch_and_bound()
                    .with_entropy_options(EntropyOptions {
                        completion_bound: Some(BoundKind::WeightedDistance),
                        objective,
                        ..EntropyOptions::default()
                    })
                    .with_bnb_options(BnbOptions {
                        frontier,
                        ordering,
                        schedule,
                        ..BnbOptions::default()
                    });
                rows.push(Row {
                    name: format!("bnb_{fname}_{oname}_{sname}"),
                    search,
                });
            }
        }
    }
    // What proving costs: the complete schedule with widening never withheld,
    // so exhaustion within the budget is a proof of optimality.
    for (oname, objective) in objectives {
        rows.push(Row {
            name: format!("bnb_lifo_{oname}_p"),
            search: MoveSearch::branch_and_bound()
                .with_entropy_options(EntropyOptions {
                    completion_bound: Some(BoundKind::WeightedDistance),
                    objective,
                    ..EntropyOptions::default()
                })
                .with_bnb_options(BnbOptions {
                    widening: Widening::UNLIMITED,
                    ..BnbOptions::default()
                }),
        });
    }
    // The other stage-0 branching rule: the frontier path's heuristic
    // generator (with the solve's deadlock policy) ahead of the exhaustive
    // ladder — spec open question 2.
    for (oname, objective) in objectives {
        rows.push(Row {
            name: format!("bnb_lifo_{oname}_hx"),
            search: MoveSearch::branch_and_bound()
                .with_entropy_options(EntropyOptions {
                    completion_bound: Some(BoundKind::WeightedDistance),
                    objective,
                    ..EntropyOptions::default()
                })
                .with_bnb_options(BnbOptions {
                    schedule: ScheduleKind::HeuristicThenExhaustive,
                    ..BnbOptions::default()
                }),
        });
    }
    rows
}

/// Duration of one AOD operation: its slowest lane, since the atoms move
/// concurrently and the operation ends when the last one arrives.
fn layer_duration(index: &LaneIndex, lanes: &[LaneAddr]) -> f64 {
    lanes
        .iter()
        .filter_map(|l| index.lane_duration_us(l))
        .fold(0.0f64, f64::max)
}

fn run(engine: &Arc<SearchEngine>, instance: &Instance, search: &MoveSearch) -> Outcome {
    let solver = TargetSolver::new(Arc::clone(engine), search.clone());
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
            proven: r.proven,
            stage_expansions: r.stage_expansions,
            plan_stage: r.plan_stage,
        },
        Err(_) => Outcome {
            solved: false,
            ops: 0,
            lanes: 0,
            xfer_us: 0.0,
            micros,
            proven: false,
            stage_expansions: Vec::new(),
            plan_stage: None,
        },
    }
}

/// Per-strategy totals over one `(arch, k)` group.
#[derive(Default)]
struct Totals {
    solved: usize,
    ops: usize,
    lanes: usize,
    xfer_us: f64,
    micros: u128,
    proven: usize,
    stage_expansions: Vec<u32>,
    plan_stage_max: Option<u8>,
}

impl Totals {
    fn add(&mut self, o: &Outcome) {
        if o.solved {
            self.solved += 1;
            self.ops += o.ops;
            self.lanes += o.lanes;
            self.xfer_us += o.xfer_us;
        }
        self.micros += o.micros;
        if o.proven {
            self.proven += 1;
        }
        if self.stage_expansions.len() < o.stage_expansions.len() {
            self.stage_expansions.resize(o.stage_expansions.len(), 0);
        }
        for (t, n) in self.stage_expansions.iter_mut().zip(&o.stage_expansions) {
            *t += n;
        }
        if let Some(s) = o.plan_stage {
            self.plan_stage_max = Some(self.plan_stage_max.map_or(s, |m| m.max(s)));
        }
    }

    fn stages(&self) -> String {
        if self.stage_expansions.is_empty() {
            "-".to_string()
        } else {
            self.stage_expansions
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/")
        }
    }
}

fn main() {
    let csv = std::env::args().any(|a| a == "--csv");
    let bnb = std::env::args().any(|a| a == "--bnb");
    let mut rows = vec![
        plain("astar", Strategy::AStar),
        plain("ids", Strategy::Ids),
        plain("dfs", Strategy::HeuristicDfs),
        plain("entropy", Strategy::Entropy),
        plain("push-rotate", Strategy::PushRotate),
    ];
    if bnb {
        rows.extend(branch_and_bound_rows());
    }

    if csv {
        println!(
            "instance,arch,atoms,strategy,solved,ops,lanes,xfer_us,micros,proven,stages,plan_stage"
        );
    } else {
        println!(
            "\n{:<16} {:<18} {:>9} {:>7} {:>10} {:>8} {:>9} {:>6} {:>14} {:>5}",
            "group",
            "strategy",
            "solved",
            "ops",
            "xfer_us",
            "lanes",
            "ms",
            "proven",
            "stages",
            "pstg"
        );
        println!("{}", "-".repeat(112));
    }

    for (arch_name, arch) in [("physical", PHYSICAL), ("logical", LOGICAL)] {
        let engine = Arc::new(SearchEngine::from_json(arch).expect("spec parses"));
        for &k in &[1usize, 2, 4, 8, 16] {
            let mut totals: Vec<Totals> = rows.iter().map(|_| Totals::default()).collect();
            for seed in 0..5u64 {
                let id = format!("{arch_name}/k{k}/seed{seed}");
                let Some(instance) = generate(id.clone(), arch, k, 4 * k, seed) else {
                    continue;
                };
                for (si, row) in rows.iter().enumerate() {
                    let o = run(&engine, &instance, &row.search);
                    if csv {
                        let stages = o
                            .stage_expansions
                            .iter()
                            .map(|n| n.to_string())
                            .collect::<Vec<_>>()
                            .join("/");
                        println!(
                            "{id},{arch_name},{k},{},{},{},{},{:.1},{},{},{},{}",
                            row.name,
                            o.solved,
                            o.ops,
                            o.lanes,
                            o.xfer_us,
                            o.micros,
                            o.proven,
                            stages,
                            o.plan_stage.map_or(String::new(), |s| s.to_string())
                        );
                    }
                    totals[si].add(&o);
                }
            }
            if !csv {
                for (si, row) in rows.iter().enumerate() {
                    let t = &totals[si];
                    println!(
                        "{:<16} {:<18} {:>4}/{:<4} {:>7} {:>10.0} {:>8} {:>9.1} {:>6} {:>14} {:>5}",
                        format!("{arch_name}/k{k}"),
                        row.name,
                        t.solved,
                        5,
                        t.ops,
                        t.xfer_us,
                        t.lanes,
                        t.micros as f64 / 1000.0,
                        t.proven,
                        t.stages(),
                        t.plan_stage_max.map_or("-".to_string(), |s| s.to_string())
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
             only, so compare only between strategies with the same solved count.\n\
             proven = solves whose verdict was a proof (branch and bound with a complete\n\
             schedule); stages = expansions per schedule stage, summed; pstg = widest\n\
             stage any plan needed. Rows named bnb_<frontier>_<u|w>_<e|ex> are the\n\
             evaluation product: objective u = uniform, w = weighted duration; schedule\n\
             e = entropy only, ex = entropy then exhaustive (widening off after the\n\
             first plan), p = entropy then exhaustive with widening unlimited (lifo\n\
             only): exhaustion within the budget proves optimality, hx = the\n\
             heuristic generator then exhaustive (lifo only)."
        );
    }
}
