//! Self-contained micro-benchmark for the entropy-guided search driver.
//!
//! No external bench framework: `harness = false`, plain `main()` using
//! `std::time::Instant`. Deterministic (fixed seed=0, NoOpObserver), so the
//! per-scenario `nodes_expanded`/goal-depth fingerprint MUST stay identical
//! across optimizations — it is the behavior guard, printed alongside timing.
//!
//! Run: `cargo bench -p bloqade-lanes-search --bench entropy`

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_search::drivers::entropy::{EntropyParams, entropy_search};
use bloqade_lanes_search::goals::AllAtTarget;
use bloqade_lanes_search::observer::NoOpObserver;
use bloqade_lanes_search::primitives::context::SearchContext;
use bloqade_lanes_search::primitives::distance::DistanceTable;
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::{Config, SearchResult};

/// Three-word, two-zone architecture (9 site buses + 1 word bus in zone 0).
const FULL_ARCH_JSON: &str = include_str!("../../../examples/arch/full.json");

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
        // 1 qubit, multi-step cross-word (matches passing solve_multi_step).
        Scenario {
            name: "single_multistep",
            initial: vec![(0, loc(0, 0))],
            target: vec![(0, loc(1, 5))],
            max_expansions: Some(2000),
        },
        // 4 qubits routed in parallel: site-bus 0..3 -> 5..8, then word bus.
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
        // 4 qubits within word 0: site-bus shuffle only.
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
        // 6 qubits split across words, cross-routing (contention).
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
                (0, loc(1, 5)),
                (1, loc(1, 6)),
                (2, loc(1, 7)),
                (3, loc(0, 5)),
                (4, loc(0, 6)),
                (5, loc(0, 7)),
            ],
            max_expansions: Some(4000),
        },
        // 8 qubits, dense parallel routing.
        Scenario {
            name: "route8_dense",
            initial: vec![
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
                (4, loc(0, 4)),
                (5, loc(1, 0)),
                (6, loc(1, 1)),
                (7, loc(1, 2)),
            ],
            target: vec![
                (0, loc(0, 5)),
                (1, loc(0, 6)),
                (2, loc(0, 7)),
                (3, loc(0, 8)),
                (4, loc(0, 9)),
                (5, loc(1, 7)),
                (6, loc(1, 8)),
                (7, loc(1, 9)),
            ],
            max_expansions: Some(6000),
        },
    ]
}

/// Everything the driver needs, built once per scenario (NOT timed).
struct Prepared {
    root: Config,
    dist_table: DistanceTable,
    blocked: HashSet<u64>,
    target_encoded: Vec<(u32, u64)>,
    goal: AllAtTarget,
    params: EntropyParams,
    max_expansions: Option<u32>,
}

fn prepare(index: &LaneIndex, s: &Scenario) -> Prepared {
    let root = Config::new(s.initial.iter().copied()).unwrap();
    let target_encoded: Vec<(u32, u64)> = s.target.iter().map(|&(q, l)| (q, l.encode())).collect();
    let target_locs: Vec<u64> = target_encoded.iter().map(|&(_, l)| l).collect();
    let dist_table = DistanceTable::new(&target_locs, index).with_time_distances(index);
    let goal = AllAtTarget::new(&target_encoded);
    Prepared {
        root,
        dist_table,
        blocked: HashSet::new(),
        target_encoded,
        goal,
        params: EntropyParams::default(),
        max_expansions: s.max_expansions,
    }
}

/// Time ONLY the driver call; context is prebuilt and reused.
fn run_driver(index: &LaneIndex, p: &Prepared) -> SearchResult {
    let ctx = SearchContext {
        index,
        dist_table: &p.dist_table,
        blocked: &p.blocked,
        targets: &p.target_encoded,
        cz_pairs: None,
    };
    entropy_search(
        p.root.clone(),
        &p.goal,
        &p.params,
        &ctx,
        p.max_expansions,
        None,
        0,
        &mut NoOpObserver,
    )
}

fn main() {
    let spec: ArchSpec = serde_json::from_str(FULL_ARCH_JSON).expect("parse full arch");
    let index = LaneIndex::new(spec);

    const WARMUP: usize = 10;
    const SAMPLES: usize = 100;

    println!(
        "{:<20} {:>10} {:>8} {:>12} {:>12} {:>12}",
        "scenario", "expanded", "depth", "min_us", "median_us", "mean_us"
    );

    for s in scenarios() {
        let p = prepare(&index, &s);

        // Fingerprint: deterministic search outcome (must not change).
        let r0 = run_driver(&index, &p);
        let expanded = r0.nodes_expanded;
        let depth = r0.goal.map(|g| r0.graph.depth(g) as i64).unwrap_or(-1);

        for _ in 0..WARMUP {
            black_box(run_driver(&index, &p));
        }

        let mut times_us: Vec<f64> = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let t = Instant::now();
            let r = run_driver(&index, &p);
            black_box(&r);
            times_us.push(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        times_us.sort_by(|a, b| a.total_cmp(b));
        let min = times_us[0];
        let median = times_us[times_us.len() / 2];
        let mean = times_us.iter().sum::<f64>() / times_us.len() as f64;

        println!(
            "{:<20} {:>10} {:>8} {:>12.2} {:>12.2} {:>12.2}",
            s.name, expanded, depth, min, median, mean
        );
    }
}
