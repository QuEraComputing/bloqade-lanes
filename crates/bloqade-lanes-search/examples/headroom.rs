//! How much parallelism is the rectangle constraint costing us?
//!
//! Compares three numbers for Push and Rotate's output:
//!
//! * `moves` — the sequential plan length; what pnr costs with no batching.
//! * `DAG_depth` — the longest path through the dependency graph. This is the
//!   operation count the scheduler would reach if *any* set of independent
//!   moves could share an operation, i.e. the AOD rectangle constraint were
//!   free. A hard lower bound.
//! * `scheduler_ops` — what the real scheduler achieves, rectangles and all.
//!
//! The gap between `DAG_depth` and `scheduler_ops` is the cost of geometry
//! alone: moves that are ready simultaneously but whose source positions do
//! not form a complete X×Y grid on one bus. `max_level_width` shows how many
//! moves are typically ready at once, so a wide level with few merges means
//! the candidates are there and only alignment is missing.
//!
//! Run: `cargo run --release -p bloqade-lanes-search --example headroom`

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_search::feasibility::graph::{LaneGraph, VertexId};
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::push_rotate::instances::generate;
use bloqade_lanes_search::push_rotate::{plan, schedule::schedule};
use std::collections::HashMap;
const PHYSICAL: &str =
    include_str!("../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json");
fn main() {
    let spec: ArchSpec = serde_json::from_str(PHYSICAL).unwrap();
    let index = LaneIndex::new(spec);
    let g = LaneGraph::build(&index, &Default::default());
    for k in [4usize, 8, 16] {
        let (mut mv, mut depth, mut ops, mut maxready) = (0usize, 0usize, 0usize, 0usize);
        for seed in 0..5u64 {
            let i = generate("x".into(), PHYSICAL, k, 4 * k, seed).unwrap();
            let tov = |v: &[(u32, u64)]| {
                v.iter()
                    .map(|&(q, l)| (q, g.vertex_of(l).unwrap()))
                    .collect::<Vec<(u32, VertexId)>>()
            };
            let p = plan(&index, &g, &tov(&i.initial), &tov(&i.target), 500_000).unwrap();
            let b = schedule(&index, &g, &p.moves).unwrap();
            mv += p.moves.len();
            ops += b.len();
            // DAG longest path: lower bound on ops if rectangles were free
            let n = p.moves.len();
            let mut lvl = vec![0usize; n];
            let mut last_v: HashMap<VertexId, usize> = HashMap::new();
            let mut last_a: HashMap<u32, usize> = HashMap::new();
            let mut level_count: HashMap<usize, usize> = HashMap::new();
            for (idx, m) in p.moves.iter().enumerate() {
                let mut l = 0;
                for v in [m.from, m.to] {
                    if let Some(&pr) = last_v.get(&v) {
                        l = l.max(lvl[pr] + 1);
                    }
                }
                if let Some(&pr) = last_a.get(&m.agent) {
                    l = l.max(lvl[pr] + 1);
                }
                lvl[idx] = l;
                last_v.insert(m.from, idx);
                last_v.insert(m.to, idx);
                last_a.insert(m.agent, idx);
                *level_count.entry(l).or_default() += 1;
            }
            depth += lvl.iter().map(|&x| x + 1).max().unwrap_or(0);
            maxready += level_count.values().copied().max().unwrap_or(0);
        }
        println!(
            "k={k:>2}: moves={mv:>4}  DAG_depth={depth:>4}  scheduler_ops={ops:>4}  max_level_width={maxready:>3}"
        );
    }
}
