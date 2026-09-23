//! The behaviour net's single binding to the `bloqade-lanes-search` API.
//!
//! **Contract.** This module is the *only* place in the behaviour net that
//! imports the crate. Cases (`cases.rs`) and the result types (`spec.rs`) are
//! plain data; this module turns a [`ProblemSpec`] into crate calls and the
//! crate's result back into an [`Outcome`]. A guard test in `main.rs` fails if
//! any other module names the crate.
//!
//! **If a crate-level interface changes, fix THIS module — do not edit case
//! data.**
//! - A compile error or a failed mapping here means the API moved: re-map it
//!   here, and leave the cases and the golden file alone.
//! - A golden or semantic assertion failure means *behaviour* changed:
//!   investigate the crate change. Regenerate the golden only for a change
//!   that is intended, and review the diff.
//!
//! The binding goes through the outer solver surface — `TargetSolver` and the
//! `CzPlacement` implementations — rather than the inner driver entry points,
//! so demoting or collapsing those cannot break the net.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_search::placement::nohome::NoHomeOptions;
use bloqade_lanes_search::search::options::{BoundKind, EntanglingOptions, EntropyOptions};
use bloqade_lanes_search::search::result::{SolveResult, SolveStatus};
use bloqade_lanes_search::{
    AodCapacity, CzPlacement, DeadlockPolicy, DefaultTargetGenerator, InnerStrategy,
    LooseGoalCzPlacement, MoveSearch, MultiSolveResult, NoHomeCzPlacement,
    RecedingHorizonCzPlacement, RecedingHorizonOptions, SearchEngine, SingleHeuristicCzPlacement,
    SolveOptions, Strategy as CrateStrategy, TargetContext, TargetGenerator, TargetSolver,
    Termination as CrateTermination,
};

use crate::spec::{
    Arch, Attempt, AttemptLog, BoundSummary, Deadlock, Knobs, Loc, Outcome, Placement, Problem,
    ProblemSpec, Run, Status, Strategy, Termination,
};

const GEMINI_LOGICAL: &str =
    include_str!("../../../../python/bloqade/lanes/arch/gemini/logical/_logical_spec.json");
const GEMINI_PHYSICAL: &str =
    include_str!("../../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json");
// Snapshots of the crate's synthetic unit-test specs (`src/test_utils.rs`).
const EXAMPLE: &str = include_str!("../fixtures/behaviour/arch/example.json");
const CHAIN: &str = include_str!("../fixtures/behaviour/arch/chain.json");
const CHAIN_WITH_SIDING: &str = include_str!("../fixtures/behaviour/arch/chain_with_siding.json");
const TWO_ZONE_BUS: &str = include_str!("../fixtures/behaviour/arch/two_zone_bus.json");
const TWO_ZONE_ALIGNED_SITE_BUS: &str =
    include_str!("../fixtures/behaviour/arch/two_zone_aligned_site_bus.json");
const ASYMMETRIC_DURATION: &str =
    include_str!("../fixtures/behaviour/arch/asymmetric_duration.json");

/// Run one case. Never panics: a panic inside the crate becomes
/// [`Outcome::Panicked`].
pub fn run(spec: &ProblemSpec) -> Outcome {
    match catch_unwind(AssertUnwindSafe(|| run_inner(spec))) {
        Ok(Ok(run)) => Outcome::Ran(run),
        Ok(Err(message)) => Outcome::Error(message),
        Err(payload) => Outcome::Panicked(panic_message(payload)),
    }
}

fn run_inner(spec: &ProblemSpec) -> Result<Run, String> {
    // A fresh engine per case, so no case sees another's lazily built caches.
    let json = match spec.arch {
        Arch::GeminiLogical => GEMINI_LOGICAL,
        Arch::GeminiPhysical => GEMINI_PHYSICAL,
        Arch::Example => EXAMPLE,
        Arch::Chain => CHAIN,
        Arch::ChainWithSiding => CHAIN_WITH_SIDING,
        Arch::TwoZoneBus => TWO_ZONE_BUS,
        Arch::TwoZoneAlignedSiteBus => TWO_ZONE_ALIGNED_SITE_BUS,
        Arch::AsymmetricDuration => ASYMMETRIC_DURATION,
    };
    let engine = Arc::new(SearchEngine::from_json_validated(json).map_err(|e| e.to_string())?);
    let search = move_search(spec.strategy, &spec.knobs);

    match &spec.problem {
        Problem::Route {
            initial,
            target,
            blocked,
        } => {
            let solver = TargetSolver::new(engine, search);
            let result = solver
                .solve(addrs(initial), addrs(target), locs(blocked), spec.budget)
                .map_err(|e| e.to_string())?;
            Ok(from_result(&result))
        }
        Problem::CzStage {
            placement,
            initial,
            controls,
            targets,
            blocked,
            future,
        } => {
            let blocked_locs: Vec<LocationAddr> = locs(blocked).collect();
            let initial_addrs: Vec<(u32, LocationAddr)> = addrs(initial).collect();
            let pairs: Vec<(u32, u32)> = controls
                .iter()
                .copied()
                .zip(targets.iter().copied())
                .collect();
            let mismatched = controls.len() != targets.len();

            // Mismatched lengths are only expressible through the trait method.
            if mismatched {
                let placement = cz_placement(placement, engine, search);
                let result = placement
                    .solve(
                        &initial_addrs,
                        controls,
                        targets,
                        &blocked_locs,
                        spec.budget,
                    )
                    .map_err(|e| e.to_string())?;
                return Ok(from_result(&result));
            }

            match placement {
                Placement::SingleHeuristic { candidates } => {
                    let generator: Box<dyn TargetGenerator> = match candidates {
                        None => Box::new(DefaultTargetGenerator),
                        Some(list) => Box::new(FixedCandidates(
                            list.iter().map(|c| addrs(c).collect()).collect(),
                        )),
                    };
                    let placement = SingleHeuristicCzPlacement::new(
                        TargetSolver::new(engine, search),
                        generator,
                    );
                    let result = placement
                        .solve_with_attempts(
                            initial_addrs,
                            controls,
                            targets,
                            blocked_locs,
                            spec.budget,
                        )
                        .map_err(|e| e.to_string())?;
                    Ok(from_multi(&result))
                }
                Placement::LooseGoal => {
                    let placement =
                        LooseGoalCzPlacement::new(engine, search, EntanglingOptions::default());
                    let result = placement
                        .solve_pairs(initial_addrs, &pairs, blocked_locs, spec.budget, future)
                        .map_err(|e| e.to_string())?;
                    Ok(from_result(&result))
                }
                Placement::NoHome => {
                    let placement =
                        NoHomeCzPlacement::new(engine, search, NoHomeOptions::default());
                    let result = placement
                        .solve_pairs(initial_addrs, &pairs, blocked_locs, spec.budget, future)
                        .map_err(|e| e.to_string())?;
                    Ok(from_result(&result))
                }
                Placement::RecedingHorizon => {
                    let placement = RecedingHorizonCzPlacement::new(
                        engine,
                        search,
                        EntanglingOptions::default(),
                        RecedingHorizonOptions::default(),
                    );
                    let result = placement
                        .solve_pairs(initial_addrs, &pairs, blocked_locs, spec.budget, future)
                        .map_err(|e| e.to_string())?;
                    Ok(from_result(&result))
                }
            }
        }
    }
}

/// The placement behind the `CzPlacement` trait, for the trait-method path.
fn cz_placement(
    placement: &Placement,
    engine: Arc<SearchEngine>,
    search: MoveSearch,
) -> Box<dyn CzPlacement> {
    match placement {
        Placement::SingleHeuristic { candidates } => {
            let generator: Box<dyn TargetGenerator> = match candidates {
                None => Box::new(DefaultTargetGenerator),
                Some(list) => Box::new(FixedCandidates(
                    list.iter().map(|c| addrs(c).collect()).collect(),
                )),
            };
            Box::new(SingleHeuristicCzPlacement::new(
                TargetSolver::new(engine, search),
                generator,
            ))
        }
        Placement::LooseGoal => Box::new(LooseGoalCzPlacement::new(
            engine,
            search,
            EntanglingOptions::default(),
        )),
        Placement::NoHome => Box::new(NoHomeCzPlacement::new(
            engine,
            search,
            NoHomeOptions::default(),
        )),
        Placement::RecedingHorizon => Box::new(RecedingHorizonCzPlacement::new(
            engine,
            search,
            EntanglingOptions::default(),
            RecedingHorizonOptions::default(),
        )),
    }
}

/// Offers exactly the listed candidates, in order, whatever the context.
struct FixedCandidates(Vec<Vec<(u32, LocationAddr)>>);

impl TargetGenerator for FixedCandidates {
    fn generate(&self, _ctx: &TargetContext) -> Vec<Vec<(u32, LocationAddr)>> {
        self.0.clone()
    }
}

fn move_search(strategy: Strategy, knobs: &Knobs) -> MoveSearch {
    let strategy = match strategy {
        Strategy::AStar => CrateStrategy::AStar,
        Strategy::Dfs => CrateStrategy::HeuristicDfs,
        Strategy::Bfs => CrateStrategy::Bfs,
        Strategy::Greedy => CrateStrategy::GreedyBestFirst,
        Strategy::Ids => CrateStrategy::Ids,
        Strategy::CascadeIds => CrateStrategy::Cascade {
            inner: InnerStrategy::Ids,
        },
        Strategy::CascadeDfs => CrateStrategy::Cascade {
            inner: InnerStrategy::Dfs,
        },
        Strategy::CascadeEntropy => CrateStrategy::Cascade {
            inner: InnerStrategy::Entropy,
        },
        Strategy::Entropy => CrateStrategy::Entropy,
        Strategy::PushRotate => CrateStrategy::PushRotate,
    };
    let solve_defaults = SolveOptions::default();
    let options = SolveOptions {
        strategy,
        weight: knobs.weight.unwrap_or(solve_defaults.weight),
        restarts: knobs.restarts.unwrap_or(solve_defaults.restarts),
        fallback_push_rotate: knobs.fallback_push_rotate,
        backwards_search: knobs.backwards_search,
        deadlock_policy: match knobs.deadlock_policy {
            None => solve_defaults.deadlock_policy,
            Some(Deadlock::Skip) => DeadlockPolicy::Skip,
            Some(Deadlock::MoveBlockers) => DeadlockPolicy::MoveBlockers,
            Some(Deadlock::AllMoves) => DeadlockPolicy::AllMoves,
        },
        lookahead: knobs.lookahead,
        top_c: knobs.top_c.or(solve_defaults.top_c),
        aod_capacity: match knobs.aod_capacity {
            None => solve_defaults.aod_capacity,
            Some((x, y)) => Some(AodCapacity::new(x, y).expect("a capacity has no zero axis")),
        },
        // Every field is named on purpose, with no `..` fill: a new
        // `SolveOptions` field then fails to compile here, which is where it
        // has to be mapped.
    };
    let entropy_defaults = EntropyOptions::default();
    let entropy = EntropyOptions {
        max_goal_candidates: knobs
            .max_goal_candidates
            .unwrap_or(entropy_defaults.max_goal_candidates),
        seed: knobs.seed.unwrap_or(entropy_defaults.seed),
        completion_bound: knobs
            .completion_bound
            .then_some(BoundKind::WeightedDistance),
        bound_terminates: knobs
            .bound_terminates
            .unwrap_or(entropy_defaults.bound_terminates),
        w_t: knobs.w_t.unwrap_or(entropy_defaults.w_t),
        ..entropy_defaults
    };
    MoveSearch::new(options, entropy)
}

// ── Mapping results back into the test domain ──────────────────────────────

fn from_result(result: &SolveResult) -> Run {
    let status = match result.status {
        SolveStatus::Solved => Status::Solved,
        SolveStatus::Unsolvable => Status::Unsolvable,
        SolveStatus::BudgetExceeded => Status::BudgetExceeded,
    };
    let termination = match result.termination {
        CrateTermination::Budget => Termination::Budget,
        CrateTermination::Exhausted { proof } => Termination::Exhausted { proof },
        CrateTermination::Stopped => Termination::Stopped,
    };
    let final_placement = result
        .goal_config
        .iter()
        .map(|(qubit, addr)| (qubit, to_loc(addr)))
        .collect();
    let stats = &result.bound_stats;
    let bound = stats.bound_enabled.then_some(BoundSummary {
        cuts_by_g: stats.cuts_by_g,
        cuts_by_h: stats.cuts_by_h,
        cuts_infeasible: stats.cuts_infeasible,
        root_lower_bound: stats.root_lower_bound,
        incumbent_cost: stats.incumbent_cost,
    });
    Run {
        status,
        layers: result.move_layers.len(),
        lanes: result.move_layers.iter().map(|m| m.len()).sum(),
        cost: result.cost,
        nodes_expanded: result.nodes_expanded,
        deadlocks: result.deadlocks,
        proven: result.proven,
        termination,
        final_placement,
        plan_digest: plan_digest(result),
        bound,
        attempts: None,
    }
}

fn from_multi(result: &MultiSolveResult) -> Run {
    let mut run = from_result(&result.result);
    let status = |s: SolveStatus| match s {
        SolveStatus::Solved => Status::Solved,
        SolveStatus::Unsolvable => Status::Unsolvable,
        SolveStatus::BudgetExceeded => Status::BudgetExceeded,
    };
    run.attempts = Some(AttemptLog {
        chosen: result.candidate_index,
        total_expansions: result.total_expansions,
        attempts: result
            .attempts
            .iter()
            .map(|a| Attempt {
                index: a.candidate_index,
                status: status(a.status),
                nodes_expanded: a.nodes_expanded,
            })
            .collect(),
    });
    run
}

/// FNV-1a over each layer's lane count and encoded lanes. Stable across
/// platforms and Rust versions, unlike `DefaultHasher`.
fn plan_digest(result: &SolveResult) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    let mut feed = |value: u64| {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(PRIME);
        }
    };
    for layer in &result.move_layers {
        feed(layer.len() as u64);
        for &lane in layer.encoded_lanes() {
            feed(lane);
        }
    }
    hash
}

fn addrs(pairs: &[(u32, Loc)]) -> impl Iterator<Item = (u32, LocationAddr)> + '_ {
    pairs.iter().map(|&(qubit, l)| (qubit, to_addr(l)))
}

fn locs(list: &[Loc]) -> impl Iterator<Item = LocationAddr> + '_ {
    list.iter().map(|&l| to_addr(l))
}

fn to_addr(l: Loc) -> LocationAddr {
    LocationAddr {
        zone_id: l.zone,
        word_id: l.word,
        site_id: l.site,
    }
}

fn to_loc(addr: LocationAddr) -> Loc {
    Loc {
        zone: addr.zone_id,
        word: addr.word_id,
        site: addr.site_id,
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    let text = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<non-string panic payload>".to_string());
    text.lines().next().unwrap_or_default().to_string()
}
