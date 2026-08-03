//! Turn a sequential move list into a parallel AOD schedule.
//!
//! This is the "condenser" of `docs/design.md` phase 2b, and it is a **list
//! scheduler over a dependency DAG**, not a peephole merger. Push and Rotate
//! moves one agent to its destination before starting the next, so two moves
//! that could share an AOD operation are typically hundreds of positions apart
//! in the sequence. Nothing can be merged without reordering first.
//!
//! ## Dependencies
//!
//! For each vertex, the moves touching it (as source or destination) are
//! totally ordered by the input sequence, and consecutive pairs get a
//! precedence edge. Together with each agent's own move order that is
//! sufficient for validity: a move's destination is vacated before it enters.
//!
//! The edges are *strict* — a move cannot enter a vertex in the same operation
//! that another leaves it. That would need the vertex to be both a source and
//! a destination of one bus group, and every bus has disjoint source and
//! destination sets (see the `feasibility` module in the open crate for why
//! that property holds and why it matters).
//!
//! ## Batching
//!
//! A legal AOD operation is a set of lanes sharing one
//! `(move_type, bus_id, zone_id, direction)` group whose source positions form
//! a complete X×Y rectangle — the Cartesian product of their distinct x and y
//! values. So each scheduling step takes the ready set, partitions it by bus
//! group, and picks the largest rectangle available. Every emitted batch is
//! re-validated with `ArchSpec::check_lanes`, which is the authoritative rule.

use std::collections::HashMap;

use crate::feasibility::graph::{LaneGraph, VertexId};
use crate::primitives::lane_index::LaneIndex;
use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

use crate::push_rotate::state::Move;

/// One AOD operation: moves executed simultaneously, with the lanes realising
/// them.
#[derive(Debug, Clone)]
pub struct Batch {
    pub moves: Vec<Move>,
    pub lanes: Vec<LaneAddr>,
}

/// Bus-group key. Matches `LaneIndex`'s own grouping.
type GroupKey = (u8, u32, u32, u8);

/// Strategy for choosing which AOD operation to emit next.
///
/// At each step the scheduler finds the largest legal rectangle available in
/// each bus group; the policy then decides which of those to take. That is the
/// scheduler's only free choice, and the counterpart to
/// [`PlanHeuristics`](crate::push_rotate::heuristics::PlanHeuristics) on the
/// planner side.
///
/// Scores are compared strictly, so an implementation that ties with the
/// default leaves the default's group ordering intact.
pub trait BatchPolicy {
    /// Rank a candidate batch. **Higher is better.**
    ///
    /// `lanes` are the batch's lanes and `index` is available for geometry —
    /// transport duration, positions — should a policy want to weigh a
    /// slower-but-larger operation against a faster-but-smaller one.
    fn score_batch(&self, moves: &[Move], lanes: &[LaneAddr], index: &LaneIndex) -> f64 {
        let _ = (lanes, index);
        moves.len() as f64
    }
}

/// Take the biggest rectangle available. The behaviour all current benchmark
/// numbers were measured with.
#[derive(Debug, Default, Clone, Copy)]
pub struct LargestBatch;

impl BatchPolicy for LargestBatch {}

fn group_key(l: &LaneAddr) -> GroupKey {
    (l.move_type as u8, l.bus_id, l.zone_id, l.direction as u8)
}

/// Schedule `moves` into AOD operations, preserving the placement they
/// produce.
///
/// Returns `None` if any move has no lane realising it, which would mean the
/// plan and the architecture disagree.
pub fn schedule(index: &LaneIndex, graph: &LaneGraph, moves: &[Move]) -> Option<Vec<Batch>> {
    schedule_with(index, graph, moves, &LargestBatch)
}

/// Schedule with an explicit batch policy.
pub fn schedule_with(
    index: &LaneIndex,
    graph: &LaneGraph,
    moves: &[Move],
    policy: &dyn BatchPolicy,
) -> Option<Vec<Batch>> {
    if moves.is_empty() {
        return Some(Vec::new());
    }
    let n = moves.len();

    // Resolve each move to the lane that realises it, plus its source position.
    let mut lanes: Vec<LaneAddr> = Vec::with_capacity(n);
    let mut src_pos: Vec<(u64, u64)> = Vec::with_capacity(n);
    for m in moves {
        let lane = lane_between(index, graph, m.from, m.to)?;
        let src = LocationAddr::decode(graph.location_of(m.from));
        let (x, y) = index.position(src)?;
        lanes.push(lane);
        src_pos.push((x.to_bits(), y.to_bits()));
    }

    // ── Dependency edges ───────────────────────────────────────────
    // Per vertex, chain the moves touching it in input order; per agent,
    // chain its own moves. Both are captured by "last event at X".
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg = vec![0usize; n];
    let mut last_at_vertex: HashMap<VertexId, usize> = HashMap::new();
    let mut last_of_agent: HashMap<u32, usize> = HashMap::new();

    let add_edge = |a: usize, b: usize, succ: &mut Vec<Vec<usize>>, indeg: &mut Vec<usize>| {
        if a != b && !succ[a].contains(&b) {
            succ[a].push(b);
            indeg[b] += 1;
        }
    };

    for (i, m) in moves.iter().enumerate() {
        for v in [m.from, m.to] {
            if let Some(&prev) = last_at_vertex.get(&v) {
                add_edge(prev, i, &mut succ, &mut indeg);
            }
        }
        // Register after adding edges so a move touching the same vertex
        // twice does not depend on itself.
        last_at_vertex.insert(m.from, i);
        last_at_vertex.insert(m.to, i);

        if let Some(&prev) = last_of_agent.get(&m.agent) {
            add_edge(prev, i, &mut succ, &mut indeg);
        }
        last_of_agent.insert(m.agent, i);
    }

    // ── List scheduling ────────────────────────────────────────────
    let arch = index.arch_spec();
    let mut ready: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut scheduled = vec![false; n];
    let mut out: Vec<Batch> = Vec::new();

    while !ready.is_empty() {
        // Partition the ready set by bus group and take the biggest legal
        // rectangle across all groups.
        let mut by_group: HashMap<GroupKey, Vec<usize>> = HashMap::new();
        for &i in &ready {
            by_group.entry(group_key(&lanes[i])).or_default().push(i);
        }
        let mut best: Vec<usize> = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut keys: Vec<&GroupKey> = by_group.keys().collect();
        keys.sort_unstable();
        for k in keys {
            let pick = largest_rectangle(&by_group[k], &src_pos);
            if pick.is_empty() {
                continue;
            }
            let pick_moves: Vec<Move> = pick.iter().map(|&i| moves[i]).collect();
            let pick_lanes: Vec<LaneAddr> = pick.iter().map(|&i| lanes[i]).collect();
            let score = policy.score_batch(&pick_moves, &pick_lanes, index);
            // Strictly greater, so a tie keeps the earlier group — which is
            // what makes the default identical to a plain size comparison.
            if score > best_score {
                best_score = score;
                best = pick;
            }
        }
        if best.is_empty() {
            // Should not happen — a single move is always a 1×1 rectangle.
            best = vec![*ready.iter().min().expect("ready is non-empty")];
        }

        // Authoritative check. If the arch rejects the group, fall back to
        // emitting one move, which is always legal.
        let batch_lanes: Vec<LaneAddr> = best.iter().map(|&i| lanes[i]).collect();
        let best = if batch_lanes.len() > 1 && !arch.check_lanes(&batch_lanes).is_empty() {
            vec![best[0]]
        } else {
            best
        };

        let batch_lanes: Vec<LaneAddr> = best.iter().map(|&i| lanes[i]).collect();
        out.push(Batch {
            moves: best.iter().map(|&i| moves[i]).collect(),
            lanes: batch_lanes,
        });

        for &i in &best {
            scheduled[i] = true;
            for &s in &succ[i] {
                indeg[s] -= 1;
            }
        }
        ready = (0..n).filter(|&i| !scheduled[i] && indeg[i] == 0).collect();
    }

    debug_assert!(scheduled.iter().all(|&s| s), "dependency graph had a cycle");
    Some(out)
}

/// Largest subset of `cand` whose source positions form a complete X×Y grid.
///
/// Enumerates subsets of the distinct y values and, for each, intersects the
/// x values present on those rows — the largest `|rows| × |common x|` wins.
/// Exponential in the number of distinct rows, which is why it is capped;
/// observed rectangles on Gemini top out at 3×3.
fn largest_rectangle(cand: &[usize], src_pos: &[(u64, u64)]) -> Vec<usize> {
    const MAX_ROWS_EXHAUSTIVE: usize = 12;

    if cand.len() <= 1 {
        return cand.to_vec();
    }
    // rows: y -> { x -> move index }
    let mut rows: Vec<(u64, HashMap<u64, usize>)> = Vec::new();
    for &i in cand {
        let (x, y) = src_pos[i];
        match rows.iter_mut().find(|(ry, _)| *ry == y) {
            Some((_, m)) => {
                m.entry(x).or_insert(i);
            }
            None => {
                let mut m = HashMap::new();
                m.insert(x, i);
                rows.push((y, m));
            }
        }
    }
    rows.sort_by_key(|(y, _)| *y);

    if rows.len() > MAX_ROWS_EXHAUSTIVE {
        // Degrade to the single best row rather than enumerate 2^n.
        let best = rows
            .iter()
            .max_by_key(|(_, m)| m.len())
            .expect("rows is non-empty");
        let mut xs: Vec<(&u64, &usize)> = best.1.iter().collect();
        xs.sort_unstable();
        return xs.into_iter().map(|(_, &i)| i).collect();
    }

    let mut best: Vec<usize> = Vec::new();
    for mask in 1u32..(1 << rows.len()) {
        let chosen: Vec<usize> = (0..rows.len()).filter(|b| mask >> b & 1 == 1).collect();
        // x values present on every chosen row.
        let mut common: Vec<u64> = rows[chosen[0]].1.keys().copied().collect();
        for &r in &chosen[1..] {
            common.retain(|x| rows[r].1.contains_key(x));
        }
        if common.len() * chosen.len() <= best.len() {
            continue;
        }
        common.sort_unstable();
        let mut pick: Vec<usize> = Vec::with_capacity(common.len() * chosen.len());
        for &r in &chosen {
            for x in &common {
                pick.push(rows[r].1[x]);
            }
        }
        if pick.len() > best.len() {
            best = pick;
        }
    }
    best
}

/// The lane realising the edge `from -> to`.
fn lane_between(
    index: &LaneIndex,
    graph: &LaneGraph,
    from: VertexId,
    to: VertexId,
) -> Option<LaneAddr> {
    let src = LocationAddr::decode(graph.location_of(from));
    let dst_enc = graph.location_of(to);
    index
        .outgoing_lanes(src)
        .iter()
        .find(|lane| {
            index
                .endpoints(lane)
                .is_some_and(|(_, d)| d.encode() == dst_enc)
        })
        .copied()
}
