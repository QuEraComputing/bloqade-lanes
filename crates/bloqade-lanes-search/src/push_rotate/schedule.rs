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
//! Most of those edges are **strict**: the successor's operation must come
//! after the predecessor's. One pattern is not. When the predecessor *vacates*
//! the vertex the successor *enters*, and both moves ride the same bus group,
//! the execution model's uniform destination rule (issue #866) makes the pair
//! legal in one operation — that is precisely a conveyor chain hop. Those are
//! recorded as **chain** dependencies, meaning "the same operation or a later
//! one, never an earlier one": a batch may include such a move only if every
//! outstanding chain dependency of it is in the same batch.
//!
//! Chain dependencies only arise where one bus's source and destination sets
//! overlap. That is legal — per-bus acyclicity, not endpoint disjointness, is
//! the invariant `ArchSpec::validate` enforces (issue #874) — and at least one
//! architecture uses it deliberately to widen its move options. The shipped
//! Gemini specs do keep their endpoints disjoint, and there no vertex can be
//! both a source and a destination of one bus group, so every edge is strict
//! and this scheduler behaves exactly as it did before the relaxation.
//!
//! ## Batching
//!
//! A legal AOD operation is a set of lanes sharing one
//! `(move_type, bus_id, zone_id, direction)` group whose source positions form
//! a complete X×Y rectangle — the Cartesian product of their distinct x and y
//! values. So each scheduling step takes the ready set, partitions it by bus
//! group, and picks the largest rectangle that carries its chain dependencies
//! with it. Every emitted batch is then re-validated with
//! `AtomStateData::validate_moves` against the replayed placement — the
//! canonical executability check, `ArchSpec::check_lanes` geometry *plus* the
//! occupancy rules.
//!
//! That check is an assertion, not a fallback: the batches built here are legal
//! by construction, so a rejection is a defect in this module and **panics**
//! naming the batch, exactly as [`crate::search::verify`] does for a packaged
//! plan. Emitting a smaller operation instead would keep the plan valid and
//! leave the defect invisible.

use std::collections::{HashMap, HashSet};

use crate::feasibility::graph::{LaneGraph, VertexId};
use crate::primitives::lane_index::LaneIndex;
use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
use bloqade_lanes_bytecode_core::atom_state::AtomStateData;

use crate::push_rotate::context::GroupKey;
use crate::push_rotate::state::Move;

/// One AOD operation: moves executed simultaneously, with the lanes realising
/// them.
#[derive(Debug, Clone)]
pub struct Batch {
    pub moves: Vec<Move>,
    pub lanes: Vec<LaneAddr>,
}

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

    let Deps {
        succ,
        mut indeg,
        chain_preds,
    } = build_deps(moves, &lanes);

    // ── List scheduling ────────────────────────────────────────────
    let arch = index.arch_spec();
    let mut scheduled = vec![false; n];
    let mut done = 0usize;
    let mut out: Vec<Batch> = Vec::new();
    // Occupancy of the plan's movers, advanced one operation at a time, so a
    // batch can be judged by the execution model rather than by geometry
    // alone.
    let mut state = mover_state(graph, moves);

    while done < n {
        let ready = ready_set(&scheduled, &indeg, &chain_preds);
        // Every edge runs forward through the plan, so the lowest unscheduled
        // move has no unscheduled predecessor of either kind: the ready set is
        // never empty, and always holds at least one move with nothing
        // outstanding. Such a move is legal on its own — the plan put it after
        // whatever vacated its destination — which is what makes `solo` a safe
        // pick when every group's rectangle is stranded.
        debug_assert!(!ready.is_empty(), "dependency graph had a cycle");
        let needs = outstanding_chain_preds(&ready, &scheduled, &chain_preds);
        let solo = ready.iter().copied().find(|i| !needs.contains_key(i))?;

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
            let pick = largest_rectangle(&by_group[k], &src_pos, &needs);
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
            // Every group's rectangle was stranded by a chain dependency it
            // could not carry; one move at a time still makes progress.
            best = vec![solo];
        }

        // Authoritative check: lane-group geometry *and* the occupancy rules,
        // against the replayed placement.
        //
        // Every batch this loop builds is legal by construction — the pick is a
        // complete rectangle on one bus group, and chain closure leaves each
        // destination either free or vacated by a move in the same operation —
        // so a rejection here is a defect in the condenser, not a situation to
        // schedule around. It fails loudly for the same reason
        // [`crate::search::verify`] does: degrading to a smaller operation
        // would hide the defect behind a slower-but-valid plan, and the batch
        // that provoked it is only in scope right here.
        let batch = best;
        let batch_lanes: Vec<LaneAddr> = batch.iter().map(|&i| lanes[i]).collect();
        let validated = state
            .validate_moves(&batch_lanes, arch)
            .unwrap_or_else(|errors| {
                panic!(
                    "the condenser built an operation the execution model \
                     rejects (this is a bug in the scheduler, not in the \
                     request): {} of {n} moves, {} lanes{}",
                    batch.len(),
                    batch_lanes.len(),
                    errors
                        .iter()
                        .map(|e| format!("\n  - {e}"))
                        .collect::<String>(),
                )
            });
        state = state
            .apply_validated(&validated)
            .expect("the token was just validated against this state");

        for &i in &batch {
            scheduled[i] = true;
            done += 1;
            for &s in &succ[i] {
                indeg[s] -= 1;
            }
        }
        out.push(Batch {
            moves: batch.iter().map(|&i| moves[i]).collect(),
            lanes: batch_lanes,
        });
    }

    Some(out)
}

/// The precedence structure of a move list.
struct Deps {
    /// Strict successors: `succ[i]` cannot start until `i`'s operation is done.
    succ: Vec<Vec<usize>>,
    /// Count of unscheduled strict predecessors.
    indeg: Vec<usize>,
    /// Predecessors that may instead ride along in the same operation — see
    /// [`is_conveyor_pair`] and the module docs.
    chain_preds: Vec<Vec<usize>>,
}

/// Order the move list: per vertex, chain the moves touching it in input order;
/// per agent, chain its own moves. Both are captured by "last event at X".
fn build_deps(moves: &[Move], lanes: &[LaneAddr]) -> Deps {
    let n = moves.len();
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indeg = vec![0usize; n];
    let mut chain_preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut last_at_vertex: HashMap<VertexId, usize> = HashMap::new();
    let mut last_of_agent: HashMap<u32, usize> = HashMap::new();

    let add_edge = |a: usize, b: usize, succ: &mut Vec<Vec<usize>>, indeg: &mut Vec<usize>| {
        if a != b && !succ[a].contains(&b) {
            succ[a].push(b);
            indeg[b] += 1;
        }
    };

    for (i, m) in moves.iter().enumerate() {
        // Conveyor pairs are collected rather than recorded straight away: the
        // same two moves can meet at both of `i`'s endpoints, or be one agent's
        // consecutive steps, and a strict edge between them always wins over a
        // chain dependency.
        let mut conveyor: Vec<usize> = Vec::new();
        for v in [m.from, m.to] {
            if let Some(&prev) = last_at_vertex.get(&v) {
                if is_conveyor_pair(&moves[prev], m, v, &lanes[prev], &lanes[i]) {
                    conveyor.push(prev);
                } else {
                    add_edge(prev, i, &mut succ, &mut indeg);
                }
            }
        }
        // Register after adding edges so a move touching the same vertex twice
        // does not depend on itself.
        last_at_vertex.insert(m.from, i);
        last_at_vertex.insert(m.to, i);

        if let Some(&prev) = last_of_agent.get(&m.agent) {
            add_edge(prev, i, &mut succ, &mut indeg);
        }
        last_of_agent.insert(m.agent, i);

        for prev in conveyor {
            if !succ[prev].contains(&i) && !chain_preds[i].contains(&prev) {
                chain_preds[i].push(prev);
            }
        }
    }

    Deps {
        succ,
        indeg,
        chain_preds,
    }
}

/// Does `later` enter the vertex `at` in the same breath that `earlier` leaves
/// it — the conveyor case the uniform destination rule permits?
///
/// Sharing a bus group is required because one AOD operation drives exactly
/// one; distinct agents because a single atom cannot ride two lanes at once.
fn is_conveyor_pair(
    earlier: &Move,
    later: &Move,
    at: VertexId,
    earlier_lane: &LaneAddr,
    later_lane: &LaneAddr,
) -> bool {
    earlier.from == at
        && later.to == at
        && earlier.agent != later.agent
        && group_key(earlier_lane) == group_key(later_lane)
}

/// Where the plan's movers start out, as an [`AtomStateData`] to replay.
///
/// An atom that never moves is invisible here, and that is sufficient: the
/// planner only ever steps onto a vertex it has established is empty, so a
/// stationary atom cannot be sitting on any move's destination. The only
/// occupant a batch has to reason about is another mover.
///
/// Agent ids stand in for qubit ids — the mapping is a bijection and this
/// state never leaves the scheduler.
fn mover_state(graph: &LaneGraph, moves: &[Move]) -> AtomStateData {
    let mut first: HashMap<u32, LocationAddr> = HashMap::new();
    for m in moves {
        first
            .entry(m.agent)
            .or_insert_with(|| LocationAddr::decode(graph.location_of(m.from)));
    }
    let atoms: Vec<(u32, LocationAddr)> = first.into_iter().collect();
    AtomStateData::from_locations(&atoms)
}

/// Moves that may go in the next operation, in plan order.
///
/// A move qualifies once every strict predecessor is scheduled and every chain
/// predecessor is either scheduled or itself in the set — a chain dependency
/// can be met inside the same operation, but only by a move that can actually
/// be in it.
fn ready_set(scheduled: &[bool], indeg: &[usize], chain_preds: &[Vec<usize>]) -> Vec<usize> {
    let n = scheduled.len();
    let mut ready: Vec<bool> = (0..n).map(|i| !scheduled[i] && indeg[i] == 0).collect();
    // One forward pass reaches the fixpoint: a chain predecessor always sits
    // earlier in the plan than its successor, so a withdrawal is already
    // visible by the time the pass reaches whatever depended on it.
    for i in 0..n {
        if ready[i] && chain_preds[i].iter().any(|&j| !scheduled[j] && !ready[j]) {
            ready[i] = false;
        }
    }
    (0..n).filter(|&i| ready[i]).collect()
}

/// Per ready move, the chain predecessors it still has to share an operation
/// with. Moves with nothing outstanding are left out, so the map is empty on
/// an endpoint-disjoint spec.
fn outstanding_chain_preds(
    ready: &[usize],
    scheduled: &[bool],
    chain_preds: &[Vec<usize>],
) -> HashMap<usize, Vec<usize>> {
    let mut needs: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in ready {
        let pending: Vec<usize> = chain_preds[i]
            .iter()
            .copied()
            .filter(|&j| !scheduled[j])
            .collect();
        if !pending.is_empty() {
            needs.insert(i, pending);
        }
    }
    needs
}

/// Largest subset of `cand` whose source positions form a complete X×Y grid
/// and which carries every outstanding chain dependency in `needs`.
///
/// Enumerates subsets of the distinct y values and, for each, intersects the
/// x values present on those rows — the largest `|rows| × |common x|` wins.
/// Exponential in the number of distinct rows, which is why it is capped;
/// observed rectangles on Gemini top out at 3×3.
///
/// Returns empty when nothing in `cand` can be batched without leaving a
/// chain dependency behind.
fn largest_rectangle(
    cand: &[usize],
    src_pos: &[(u64, u64)],
    needs: &HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    const MAX_ROWS_EXHAUSTIVE: usize = 12;

    if cand.is_empty() {
        return Vec::new();
    }
    if cand.len() == 1 {
        // A lone move has nothing to satisfy a chain dependency with.
        return if needs.contains_key(&cand[0]) {
            Vec::new()
        } else {
            cand.to_vec()
        };
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
            .enumerate()
            .max_by_key(|(_, (_, m))| m.len())
            .expect("rows is non-empty")
            .0;
        let mut xs: Vec<u64> = rows[best].1.keys().copied().collect();
        xs.sort_unstable();
        close_rectangle(&rows, &[best], &mut xs, needs);
        return cells(&rows, &[best], &xs);
    }

    let mut best: Vec<usize> = Vec::new();
    for mask in 1u32..(1 << rows.len()) {
        let chosen: Vec<usize> = (0..rows.len()).filter(|b| mask >> b & 1 == 1).collect();
        // x values present on every chosen row.
        let mut common: Vec<u64> = rows[chosen[0]].1.keys().copied().collect();
        for &r in &chosen[1..] {
            common.retain(|x| rows[r].1.contains_key(x));
        }
        common.sort_unstable();
        close_rectangle(&rows, &chosen, &mut common, needs);
        // The trimmed product is the exact size of this candidate, so a tie
        // keeps the earlier mask — as it did before chains were relaxed.
        if common.len() * chosen.len() <= best.len() {
            continue;
        }
        best = cells(&rows, &chosen, &common);
    }
    best
}

/// The moves at `chosen` rows × `xs` columns, row-major.
fn cells(rows: &[(u64, HashMap<u64, usize>)], chosen: &[usize], xs: &[u64]) -> Vec<usize> {
    let mut pick: Vec<usize> = Vec::with_capacity(chosen.len() * xs.len());
    for &r in chosen {
        for x in xs {
            pick.push(rows[r].1[x]);
        }
    }
    pick
}

/// Drop columns from `xs` until every move in `chosen × xs` has its
/// outstanding chain dependencies inside the rectangle.
///
/// A chain dependency can only be met by another cell of the same operation,
/// so a move whose vacating partner falls outside the rectangle cannot ride in
/// it — and dropping that move means dropping its whole column, since the AOD
/// drives a complete Cartesian product and rectangularity is the one thing it
/// cannot compromise on. Dropping a column can strand further moves, hence the
/// loop; each pass removes at least one column, so it terminates.
///
/// The rows are left alone: every choice of rows is enumerated separately by
/// [`largest_rectangle`], so a stranding that another row set would fix is
/// already covered there.
fn close_rectangle(
    rows: &[(u64, HashMap<u64, usize>)],
    chosen: &[usize],
    xs: &mut Vec<u64>,
    needs: &HashMap<usize, Vec<usize>>,
) {
    if needs.is_empty() {
        return;
    }
    while !xs.is_empty() {
        let picked: HashSet<usize> = cells(rows, chosen, xs).into_iter().collect();
        let mut stranded: HashSet<u64> = HashSet::new();
        for &r in chosen {
            for x in xs.iter() {
                let i = rows[r].1[x];
                if needs
                    .get(&i)
                    .is_some_and(|req| req.iter().any(|j| !picked.contains(j)))
                {
                    stranded.insert(*x);
                }
            }
        }
        if stranded.is_empty() {
            return;
        }
        xs.retain(|x| !stranded.contains(x));
    }
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

#[cfg(test)]
mod tests {
    //! Batching on an **overlapping-bus** spec, where a bus's source and
    //! destination sets intersect and conveyor chains are therefore reachable.
    //!
    //! [`chain_arch_json`] rewires site bus 0 of the example arch into the
    //! chain `0→1→2→3→4`, in both of its words. Words 0 and 1 sit on different
    //! grid rows, so one bus group offers a two-row rectangle — enough to
    //! exercise the interaction between chain dependencies and the AOD's
    //! Cartesian-product constraint, which is where batching a chain is harder
    //! than merely permitting it.
    //!
    //! Move lists are written by hand rather than planned: the point is what
    //! the condenser does with a given dependency structure, and a fabricated
    //! list pins that structure exactly.

    use super::*;
    use crate::search::result::SolveStatus;
    use crate::test_utils::{chain_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    struct Fixture {
        index: LaneIndex,
        graph: LaneGraph,
    }

    impl Fixture {
        fn new() -> Self {
            let spec: ArchSpec = serde_json::from_str(&chain_arch_json()).expect("fixture parses");
            let index = LaneIndex::new(spec);
            let graph = LaneGraph::build(&index, &Default::default());
            Self { index, graph }
        }

        /// The vertex holding `(word, site)`.
        fn at(&self, word: u32, site: u32) -> VertexId {
            self.graph
                .vertex_of(loc(word, site).encode())
                .expect("site is on the graph")
        }

        /// One hop `site → site + 1` in `word`, carried by `agent`.
        fn hop(&self, agent: u32, word: u32, site: u32) -> Move {
            Move {
                agent,
                from: self.at(word, site),
                to: self.at(word, site + 1),
            }
        }

        fn schedule(&self, moves: &[Move]) -> Vec<Batch> {
            schedule(&self.index, &self.graph, moves).expect("the fixture schedules")
        }
    }

    /// Batch sizes in emission order — the shape assertions below are about
    /// how much rode together, not which lane landed where.
    fn sizes(batches: &[Batch]) -> Vec<usize> {
        batches.iter().map(|b| b.lanes.len()).collect()
    }

    /// The regression this module exists for (issue #892): three hops of one
    /// conveyor chain are one AOD operation, not three.
    ///
    /// Each hop enters the site the next one leaves, so under strict
    /// precedence every pair was forced apart and the chain serialised.
    #[test]
    fn chain_hops_ride_one_operation() {
        let fx = Fixture::new();
        // Leader first, as the planner emits it: pushing atom 0 to site 3
        // clears site 2 for atom 1, and so on.
        let moves = [fx.hop(0, 0, 2), fx.hop(1, 0, 1), fx.hop(2, 0, 0)];
        let batches = fx.schedule(&moves);

        assert_eq!(sizes(&batches), vec![3], "the chain must ride in one shot");
        assert_eq!(batches[0].moves.len(), 3);
    }

    /// Chains in two words share the operation when their columns line up: a
    /// 2×2 rectangle whose every row is a chain.
    #[test]
    fn aligned_chains_in_two_words_share_one_operation() {
        let fx = Fixture::new();
        let moves = [
            fx.hop(0, 0, 1),
            fx.hop(1, 1, 1),
            fx.hop(2, 0, 0),
            fx.hop(3, 1, 0),
        ];
        let batches = fx.schedule(&moves);

        assert_eq!(sizes(&batches), vec![4], "both rows batch together");
    }

    /// A chain dependency is "same operation or later" — never earlier.
    ///
    /// Here the vacating hop is itself held back by a strict edge (its agent
    /// has to arrive first), so the follower must wait for it rather than
    /// slide into a site that is still occupied.
    #[test]
    fn a_chain_hop_never_precedes_the_move_it_waits_on() {
        let fx = Fixture::new();
        // Agent 1 walks site 2 → 1 → 2; agent 0 follows it into site 1.
        let arrive = Move {
            agent: 1,
            from: fx.at(0, 2),
            to: fx.at(0, 1),
        };
        let moves = [arrive, fx.hop(1, 0, 1), fx.hop(0, 0, 0)];
        let batches = fx.schedule(&moves);

        assert_eq!(sizes(&batches), vec![1, 2]);
        assert_eq!(
            batches[0].moves,
            vec![arrive],
            "the follower must not overtake the hop that clears its way"
        );
    }

    /// A rectangle that would strand a chain leader is shrunk, not broken.
    ///
    /// Word 0 carries a three-hop chain and word 1 a two-hop chain, so the
    /// widest *geometric* rectangle is the 2×2 over their shared columns — and
    /// it is illegal, because word 0's middle hop needs the leader that sits in
    /// the column the intersection drops. Trimming has to give up a column (the
    /// AOD cannot drive a partial row), which leaves word 0's full chain as the
    /// best legal operation.
    #[test]
    fn a_rectangle_that_would_strand_a_chain_leader_is_trimmed() {
        let fx = Fixture::new();
        let moves = [
            fx.hop(0, 0, 2),
            fx.hop(1, 1, 1),
            fx.hop(2, 0, 1),
            fx.hop(3, 1, 0),
            fx.hop(4, 0, 0),
        ];
        let batches = fx.schedule(&moves);

        assert_eq!(
            sizes(&batches),
            vec![3, 2],
            "word 0's chain goes whole, then word 1's — never a stranded 2x2"
        );
        assert!(
            batches[0]
                .moves
                .iter()
                .all(|m| m.agent != 1 && m.agent != 3),
            "the first operation is word 0's chain: {:?}",
            batches[0].moves
        );
    }

    /// Every emitted operation executes against the placement the previous ones
    /// produced, checked with the canonical model rather than with this
    /// module's own reasoning about it.
    #[test]
    fn every_emitted_operation_is_executable() {
        let fx = Fixture::new();
        let moves = [
            fx.hop(0, 0, 2),
            fx.hop(1, 1, 1),
            fx.hop(2, 0, 1),
            fx.hop(3, 1, 0),
            fx.hop(4, 0, 0),
        ];
        let batches = fx.schedule(&moves);

        let mut state = mover_state(&fx.graph, &moves);
        for (i, b) in batches.iter().enumerate() {
            let validated = state
                .validate_moves(&b.lanes, fx.index.arch_spec())
                .unwrap_or_else(|e| panic!("operation {i} does not execute: {e:?}"));
            state = state.apply_validated(&validated).expect("token is fresh");
        }
        assert!(state.collision.is_empty());
        for m in &moves {
            assert_eq!(
                state.qubit_to_locations.get(&m.agent),
                Some(&LocationAddr::decode(fx.graph.location_of(m.to))),
                "agent {} did not land on its destination",
                m.agent
            );
        }
    }

    /// The whole router, not just the condenser: asking three atoms to shift
    /// one site along the chain comes back as a single AOD operation.
    ///
    /// `solve_push_rotate` replays what it packages through the canonical
    /// execution model before returning (`search::verify`), from the *full*
    /// root placement rather than this module's mover-only state — so a chain
    /// batch that only looked legal to the scheduler would panic there.
    #[test]
    fn the_router_batches_a_chain_into_one_operation() {
        let fx = Fixture::new();
        let initial: Vec<(u32, LocationAddr)> = (0..3u32).map(|q| (q, loc(0, q))).collect();
        let target: Vec<(u32, LocationAddr)> = (0..3u32).map(|q| (q, loc(0, q + 1))).collect();

        let result =
            crate::push_rotate::solve_push_rotate(&fx.index, &initial, &target, &[], 10_000)
                .expect("valid config");

        assert_eq!(result.status, SolveStatus::Solved);
        let lanes_per_layer: Vec<usize> = result
            .move_layers
            .iter()
            .map(|l| l.decode().len())
            .collect();
        assert_eq!(
            lanes_per_layer,
            vec![3],
            "the shift is one conveyor operation, not one per hop"
        );
    }
}
