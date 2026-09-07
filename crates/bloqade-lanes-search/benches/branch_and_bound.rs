//! Self-contained micro-benchmark for the branch-and-bound driver, the
//! sibling of `benches/entropy.rs`: same shape, so the per-expansion cost of
//! the two drivers can be read side by side.
//!
//! No external bench framework: `harness = false`, plain `main()` using
//! `std::time::Instant`. Deterministic (seed 0, `NoOpObserver`), so the
//! per-scenario fingerprint — `nodes_expanded`, `stage_expansions`, goal depth
//! and `termination` — MUST stay identical across optimizations; it is the
//! behaviour guard, printed alongside timing.
//!
//! The example arch is used rather than `full.json`: the latter violates the
//! exhaustive generator's P1 precondition (coincident words), so a complete
//! schedule cannot be built on it. Its sites form five independent four-node
//! paths `(0,i) – (0,i+5) – (1,i+5) – (1,i)`; every scenario keeps each atom in
//! its column and respects the order along the path, so all are solvable.
//!
//! Run: `cargo bench -p bloqade-lanes-search --bench branch_and_bound`

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_search::bounds::WeightedDistanceBound;
use bloqade_lanes_search::cost::UniformCost;
use bloqade_lanes_search::drivers::branch_and_bound::{BranchAndBound, Schedule, Widening};
use bloqade_lanes_search::drivers::entropy::{EntropyParams, HeuristicTables};
use bloqade_lanes_search::drivers::frontier::{IdsFrontier, LifoFrontier};
use bloqade_lanes_search::drivers::result::Termination;
use bloqade_lanes_search::generators::EntropyGenerator;
use bloqade_lanes_search::generators::exhaustive::{ExhaustiveGenerator, SeedPolicy};
use bloqade_lanes_search::goals::AllAtTarget;
use bloqade_lanes_search::observer::NoOpObserver;
use bloqade_lanes_search::primitives::context::{AodCapacity, SearchContext};
use bloqade_lanes_search::primitives::distance::{DistanceTable, HopDistanceHeuristic};
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::traits::MoveGenerator;
use bloqade_lanes_search::{Config, SearchResult};

/// Two-word, one-zone architecture: one site bus (`0..4 → 5..9`) per word and
/// one word bus joining the words at sites `5..9` (the crate's example arch).
const EXAMPLE_ARCH_JSON: &str = r#"{
    "version": "2.0",
    "words": [
        { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]] },
        { "sites": [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2], [0, 3], [1, 3], [2, 3], [3, 3], [4, 3]] }
    ],
    "zones": [
        {
            "grid": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5, 7.5, 2.5] },
            "site_buses": [ { "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] } ],
            "word_buses": [ { "src": [0], "dst": [1] } ],
            "words_with_site_buses": [0, 1],
            "sites_with_word_buses": [5, 6, 7, 8, 9],
            "entangling_pairs": [[0, 1]]
        }
    ],
    "zone_buses": [],
    "modes": [ { "name": "default", "zones": [0], "bitstring_order": [] } ]
}"#;

fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
}

struct Scenario {
    name: &'static str,
    initial: Vec<(u32, LocationAddr)>,
    target: Vec<(u32, LocationAddr)>,
    max_expansions: Option<u32>,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "single_multistep",
            initial: vec![(0, loc(0, 0))],
            target: vec![(0, loc(1, 0))],
            max_expansions: Some(2000),
        },
        Scenario {
            name: "route4_parallel",
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
            ],
            target: vec![
                (0, loc(1, 5)),
                (1, loc(1, 6)),
                (2, loc(1, 7)),
                (3, loc(1, 8)),
            ],
            max_expansions: Some(3000),
        },
        Scenario {
            name: "route4_intraword",
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
            ],
            target: vec![
                (0, loc(0, 5)),
                (1, loc(0, 6)),
                (2, loc(0, 7)),
                (3, loc(0, 8)),
            ],
            max_expansions: Some(3000),
        },
        Scenario {
            name: "route6_cross",
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(1, 0)),
                (4, loc(1, 1)),
                (5, loc(1, 2)),
            ],
            target: vec![
                (0, loc(0, 5)),
                (1, loc(0, 6)),
                (2, loc(0, 7)),
                (3, loc(1, 5)),
                (4, loc(1, 6)),
                (5, loc(1, 7)),
            ],
            max_expansions: Some(4000),
        },
        // Five atoms in word 0 lift in one shot (a 5x1 rectangle), two of them
        // continue across the word bus, and three word-1 atoms walk back down
        // their columns together with the arriving two: three shots, with a
        // complete schedule needed once the improving-only stage 0 dead-ends.
        Scenario {
            name: "route8_dense",
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
                (4, loc(0, 4)),
                (5, loc(1, 5)),
                (6, loc(1, 6)),
                (7, loc(1, 7)),
            ],
            target: vec![
                (0, loc(0, 5)),
                (1, loc(0, 6)),
                (2, loc(0, 7)),
                (3, loc(1, 3)),
                (4, loc(1, 4)),
                (5, loc(1, 0)),
                (6, loc(1, 1)),
                (7, loc(1, 2)),
            ],
            max_expansions: Some(6000),
        },
    ]
}

#[derive(Clone, Copy)]
enum FrontierKind {
    Lifo,
    Ids,
}

#[derive(Clone, Copy)]
enum ScheduleKind {
    EntropyOnly,
    EntropyThenExhaustive,
}

struct Configuration {
    name: &'static str,
    frontier: FrontierKind,
    schedule: ScheduleKind,
}

const CONFIGURATIONS: [Configuration; 3] = [
    Configuration {
        name: "lifo/entropy_only",
        frontier: FrontierKind::Lifo,
        schedule: ScheduleKind::EntropyOnly,
    },
    Configuration {
        name: "lifo/complete",
        frontier: FrontierKind::Lifo,
        schedule: ScheduleKind::EntropyThenExhaustive,
    },
    Configuration {
        name: "ids/complete",
        frontier: FrontierKind::Ids,
        schedule: ScheduleKind::EntropyThenExhaustive,
    },
];

/// Everything solve-scoped, built once per scenario (NOT timed).
struct Prepared<'a> {
    root: Config,
    ctx: SearchContext<'a>,
    goal: AllAtTarget,
    bound: WeightedDistanceBound<UniformCost>,
    tables: HeuristicTables,
    levels: Vec<ExhaustiveGenerator>,
    h_sum: HopDistanceHeuristic<'a>,
    max_expansions: Option<u32>,
}

struct Borrowed<'a> {
    dist_table: &'a DistanceTable,
    blocked: &'a HashSet<u64>,
    target_encoded: &'a [(u32, u64)],
}

fn prepare<'a>(index: &'a LaneIndex, s: &Scenario, b: Borrowed<'a>) -> Prepared<'a> {
    let ctx = SearchContext {
        index,
        dist_table: b.dist_table,
        blocked: b.blocked,
        targets: b.target_encoded,
        cz_pairs: None,
        capacity: None,
    };
    let params = EntropyParams::default();
    let tables = HeuristicTables::build(&ctx, params.w_t, params.lookahead);
    let levels: Vec<ExhaustiveGenerator> = [
        (
            SeedPolicy::Unresolved,
            Some(AodCapacity::new(1, 1).unwrap()),
        ),
        (
            SeedPolicy::Unresolved,
            Some(AodCapacity::new(2, 2).unwrap()),
        ),
        (SeedPolicy::Unresolved, None),
        (SeedPolicy::Any, None),
    ]
    .into_iter()
    .map(|(seed, cap)| ExhaustiveGenerator::for_solve(&ctx, seed, cap).expect("P1/P2 hold"))
    .collect();
    Prepared {
        root: Config::new(s.initial.iter().copied()).unwrap(),
        goal: AllAtTarget::new(b.target_encoded),
        bound: WeightedDistanceBound::new(&UniformCost, b.target_encoded, index, b.blocked),
        tables,
        levels,
        h_sum: HopDistanceHeuristic::new(s.target.iter().copied(), b.dist_table),
        max_expansions: s.max_expansions,
        ctx,
    }
}

/// Time ONLY the driver call; everything solve-scoped is prebuilt.
fn run_driver(p: &Prepared<'_>, c: &Configuration) -> SearchResult {
    let stage0 = EntropyGenerator::with_tables(EntropyParams::default(), 0, &p.tables);
    let schedule = match c.schedule {
        ScheduleKind::EntropyOnly => Schedule::partial(vec![&stage0 as &dyn MoveGenerator]),
        ScheduleKind::EntropyThenExhaustive => {
            let (terminal, prefix) = p.levels.split_last().unwrap();
            let mut stages: Vec<&dyn MoveGenerator> = vec![&stage0];
            stages.extend(prefix.iter().map(|g| g as &dyn MoveGenerator));
            Schedule::complete(stages, terminal, None)
        }
    };
    let driver = BranchAndBound::new(
        &UniformCost,
        &p.bound,
        &p.goal,
        &p.ctx,
        p.max_expansions,
        Widening::default(),
    );
    match c.frontier {
        FrontierKind::Lifo => driver.run(
            p.root.clone(),
            &schedule,
            LifoFrontier::new(),
            &mut NoOpObserver,
            None,
        ),
        FrontierKind::Ids => driver.run(
            p.root.clone(),
            &schedule,
            IdsFrontier::new(|cfg: &Config| p.h_sum.estimate_sum(cfg)),
            &mut NoOpObserver,
            None,
        ),
    }
}

fn termination_label(t: Termination) -> &'static str {
    match t {
        Termination::Budget => "budget",
        Termination::Exhausted { proof: false } => "exhausted",
        Termination::Exhausted { proof: true } => "proof",
        Termination::Stopped => "stopped",
    }
}

fn main() {
    let spec: ArchSpec = serde_json::from_str(EXAMPLE_ARCH_JSON).expect("parse example arch");
    let index = LaneIndex::new(spec);

    const WARMUP: usize = 10;
    const SAMPLES: usize = 100;

    println!(
        "{:<18} {:<18} {:>8} {:>14} {:>5} {:>9} {:>10} {:>10} {:>10}",
        "scenario",
        "configuration",
        "expanded",
        "per_stage",
        "depth",
        "term",
        "min_us",
        "median_us",
        "mean_us"
    );

    for s in scenarios() {
        let target_encoded: Vec<(u32, u64)> =
            s.target.iter().map(|&(q, l)| (q, l.encode())).collect();
        let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &index).with_time_distances(&index);
        let blocked = HashSet::new();
        let p = prepare(
            &index,
            &s,
            Borrowed {
                dist_table: &dist_table,
                blocked: &blocked,
                target_encoded: &target_encoded,
            },
        );

        for c in &CONFIGURATIONS {
            // Fingerprint: deterministic search outcome (must not change).
            let r0 = run_driver(&p, c);
            let expanded = r0.nodes_expanded;
            let per_stage = r0
                .stage_expansions
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("/");
            let depth = r0.goal.map(|g| r0.graph.depth(g) as i64).unwrap_or(-1);
            let term = termination_label(r0.termination);

            for _ in 0..WARMUP {
                black_box(run_driver(&p, c));
            }
            let mut times_us: Vec<f64> = Vec::with_capacity(SAMPLES);
            for _ in 0..SAMPLES {
                let t = Instant::now();
                let r = run_driver(&p, c);
                black_box(&r);
                times_us.push(t.elapsed().as_nanos() as f64 / 1000.0);
            }
            times_us.sort_by(|a, b| a.total_cmp(b));
            let min = times_us[0];
            let median = times_us[times_us.len() / 2];
            let mean = times_us.iter().sum::<f64>() / times_us.len() as f64;

            println!(
                "{:<18} {:<18} {:>8} {:>14} {:>5} {:>9} {:>10.2} {:>10.2} {:>10.2}",
                s.name, c.name, expanded, per_stage, depth, term, min, median, mean
            );
        }
    }
}
