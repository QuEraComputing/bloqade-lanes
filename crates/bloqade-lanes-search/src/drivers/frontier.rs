//! Frontier trait and generic search loop.
//!
//! Implements the traversal strategy abstraction from issue #427:
//! a [`Frontier`] trait controls node ordering and goal-check timing,
//! while [`run_search`] provides the shared search loop.
//!
//! Concrete frontiers: [`PriorityFrontier`] (A* / greedy best-first),
//! [`BfsFrontier`], and [`DfsFrontier`] (heuristic depth-first).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::drivers::result::SearchResult;
use crate::observer::{SearchEvent, SearchObserver};
use crate::primitives::config::Config;
use crate::primitives::context::{MoveCandidate, SearchContext, SearchState};
use crate::primitives::graph::{NodeId, SearchGraph};
use crate::traits::{CandidateScorer, CostFn, Goal, MoveGenerator};

/// Number of parent-chain steps to inspect when computing the
/// per-atom recent-source map for the IDS frontier's reversal
/// tiebreaker. Cycles in the diagnosed failure mode oscillate every
/// 1–2 steps, so a small window suffices; larger windows catch longer
/// cycles at modest extra cost.
const IDS_REVERSAL_LOOKBACK: usize = 10;

/// Penalty added to a child's `h_score` for each atom whose move
/// (parent → child) returns it to a position vacated within the last
/// [`IDS_REVERSAL_LOOKBACK`] steps. Breaks the symmetry between
/// forward and backward moves of the same atom inside h-plateaus,
/// where the heuristic alone cannot tell the two apart.
///
/// Calibrated to be small (≪ 1 atom-distance) so that genuinely
/// improving moves still dominate the priority order; the penalty
/// only flips ordering when h would otherwise tie.
const IDS_REVERSAL_PENALTY: f64 = 0.5;

// ── Frontier trait ──────────────────────────────────────────────────

/// Trait for frontier data structures that drive the search loop.
///
/// Each implementation controls how nodes are ordered (FIFO, priority,
/// LIFO) and when the goal is checked (on pop vs. on generate).
pub trait Frontier {
    /// Remove and return the next node to process, or `None` if empty.
    fn select_next(&mut self) -> Option<NodeId>;

    /// Receive newly created child nodes after expansion.
    /// Use `graph` to look up configs and g-scores for ordering.
    fn receive_children(&mut self, children: &[NodeId], graph: &SearchGraph);

    /// Check goal when a node is popped (before expansion)?
    /// `true` for A* (guarantees optimality). Default: `false`.
    fn check_goal_on_pop(&self) -> bool {
        false
    }

    /// Check goal when a child is generated (before pushing)?
    /// `true` for BFS/DFS (find earliest). Default: `true`.
    fn check_goal_on_generate(&self) -> bool {
        true
    }
}

// ── PriorityFrontier (A* / Greedy) ─────────────────────────────────

/// Priority queue entry, ordered by f-score (lower = higher priority).
struct PriorityEntry {
    f_score: f64,
    g_score: f64,
    node_id: NodeId,
}

impl Eq for PriorityEntry {}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_score.total_cmp(&other.f_score) == Ordering::Equal
            && self.g_score.total_cmp(&other.g_score) == Ordering::Equal
            && self.node_id == other.node_id
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap. Tie-break: prefer higher g (deeper),
        // then lower node_id for deterministic ordering.
        other
            .f_score
            .total_cmp(&self.f_score)
            .then(self.g_score.total_cmp(&other.g_score))
            .then(other.node_id.0.cmp(&self.node_id.0))
    }
}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority-queue frontier for A* and greedy best-first search.
///
/// - A*: `f = g + h`, goal checked on pop (optimal).
/// - Greedy: `f = h`, goal checked on generate (fast, not optimal).
pub struct PriorityFrontier<H> {
    heap: BinaryHeap<PriorityEntry>,
    heuristic: H,
    weight: f64,
    use_cost: bool,
    /// When `true`, goal is checked on pop (A* semantics). This guarantees
    /// optimality only when `weight == 1.0` with an admissible heuristic.
    /// With `weight > 1.0`, the guarantee weakens to bounded suboptimal
    /// (cost ≤ weight × optimal).
    goal_on_pop: bool,
}

impl<H> PriorityFrontier<H> {
    /// Create an A* frontier: `f = g + weight * h`, goal on pop.
    ///
    /// - `weight = 1.0`: standard A* (optimal with admissible heuristic).
    /// - `weight > 1.0`: weighted A* (bounded suboptimal, cost ≤ weight × optimal).
    pub fn astar(heuristic: H, weight: f64) -> Self {
        Self {
            heap: BinaryHeap::new(),
            heuristic,
            weight,
            use_cost: true,
            goal_on_pop: true,
        }
    }

    /// Create a greedy best-first frontier: `f = h`, goal on generate.
    pub fn greedy(heuristic: H) -> Self {
        Self {
            heap: BinaryHeap::new(),
            heuristic,
            weight: 0.0, // unused — greedy ignores cost
            use_cost: false,
            goal_on_pop: false,
        }
    }
}

impl<H: crate::traits::Heuristic> Frontier for PriorityFrontier<H> {
    fn select_next(&mut self) -> Option<NodeId> {
        self.heap.pop().map(|e| e.node_id)
    }

    fn receive_children(&mut self, children: &[NodeId], graph: &SearchGraph) {
        for &child_id in children {
            let g = graph.g_score(child_id);
            let h = self.heuristic.estimate(graph.config(child_id));
            let f = if self.use_cost {
                g + self.weight * h
            } else {
                h
            };
            self.heap.push(PriorityEntry {
                f_score: f,
                g_score: g,
                node_id: child_id,
            });
        }
    }

    fn check_goal_on_pop(&self) -> bool {
        self.goal_on_pop
    }

    fn check_goal_on_generate(&self) -> bool {
        !self.goal_on_pop
    }
}

// ── BfsFrontier ─────────────────────────────────────────────────────

/// FIFO frontier for breadth-first search.
pub struct BfsFrontier {
    queue: VecDeque<NodeId>,
}

impl BfsFrontier {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::new(),
        }
    }
}

impl Default for BfsFrontier {
    fn default() -> Self {
        Self::new()
    }
}

impl Frontier for BfsFrontier {
    fn select_next(&mut self) -> Option<NodeId> {
        self.queue.pop_front()
    }

    fn receive_children(&mut self, children: &[NodeId], _graph: &SearchGraph) {
        self.queue.extend(children);
    }
}

// ── DfsFrontier ─────────────────────────────────────────────────────

/// LIFO frontier for heuristic depth-first search.
///
/// Sorts children by heuristic (best last on stack = popped first).
/// Commits to the best candidate and backtracks when stuck.
/// Memory: O(depth × branching factor at backtrack points).
pub struct DfsFrontier<H> {
    stack: Vec<NodeId>,
    heuristic: H,
}

impl<H> DfsFrontier<H> {
    pub fn new(heuristic: H) -> Self {
        Self {
            stack: Vec::new(),
            heuristic,
        }
    }
}

impl<H: crate::traits::Heuristic> Frontier for DfsFrontier<H> {
    fn select_next(&mut self) -> Option<NodeId> {
        self.stack.pop()
    }

    fn receive_children(&mut self, children: &[NodeId], graph: &SearchGraph) {
        if children.is_empty() {
            return;
        }
        // Sort by heuristic descending (worst first on stack).
        // Best child is pushed last → popped first (LIFO).
        let mut scored: Vec<(f64, NodeId)> = children
            .iter()
            .map(|&id| (self.heuristic.estimate(graph.config(id)), id))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (_, id) in scored {
            self.stack.push(id);
        }
    }

    fn check_goal_on_pop(&self) -> bool {
        false
    }

    fn check_goal_on_generate(&self) -> bool {
        true
    }
}

// ── IdsFrontier (Iterative Diving Search) ───────────────────────────

/// Priority entry for IDS: best-first dive (paper-style "Iterative Diving
/// Search").
///
/// Ordering (max-heap): lower `h_score` first, then deeper (dive when scores
/// tie), then earlier insertion (preserves expander ranking).
///
/// This matches the algorithm in arxiv:2512.13790 — the heuristic drives the
/// pop order, with depth as a tiebreaker so the search prefers diving into
/// the cheapest path while still backing up to a shallower node when the
/// shallower path looks better.
///
/// Note: `insertion_order` is unique across all entries (one increment per
/// `receive_children` call), so it always breaks ties decisively. This
/// means earlier orderings where `insertion_order` came before `h_score`
/// effectively never consulted the heuristic — the change to put
/// `h_score` first activates a previously-dormant signal.
struct IdsEntry {
    depth: u32,
    insertion_order: u64,
    h_score: f64,
    node_id: NodeId,
}

impl Eq for IdsEntry {}

impl PartialEq for IdsEntry {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.insertion_order == other.insertion_order
            && self.h_score.total_cmp(&other.h_score) == Ordering::Equal
    }
}

impl Ord for IdsEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Primary: lower h_score = higher priority (best-first).
        other
            .h_score
            .total_cmp(&self.h_score)
            // Secondary: deeper = higher priority (dive within score tier).
            .then(self.depth.cmp(&other.depth))
            // Tertiary: earlier insertion = higher priority (preserves
            // expander ranking among same-depth same-score entries).
            .then(other.insertion_order.cmp(&self.insertion_order))
    }
}

impl PartialOrd for IdsEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority-queue frontier for Iterative Diving Search.
///
/// Pop order is primarily by lowest heuristic score (`h_score`) with
/// depth as a tiebreaker (deeper first within an h-plateau, so the
/// search dives when scores tie). When the current best-h path
/// dead-ends, the heap naturally pops the next-best-h node anywhere in
/// the tree — the "jump-back". A small reversal-aware penalty is mixed
/// into `h_score` in `receive_children` to break symmetric cycles
/// within an h-plateau.
///
/// Inspired by Iterative Diving Search (arxiv:2512.13790); the
/// h-primary ordering matches the algorithm in that paper.
pub struct IdsFrontier<H> {
    heap: BinaryHeap<IdsEntry>,
    heuristic: H,
    insertion_counter: u64,
}

impl<H> IdsFrontier<H> {
    pub fn new(heuristic: H) -> Self {
        Self {
            heap: BinaryHeap::new(),
            heuristic,
            insertion_counter: 0,
        }
    }
}

impl<H: crate::traits::Heuristic> Frontier for IdsFrontier<H> {
    fn select_next(&mut self) -> Option<NodeId> {
        self.heap.pop().map(|e| e.node_id)
    }

    fn receive_children(&mut self, children: &[NodeId], graph: &SearchGraph) {
        if children.is_empty() {
            return;
        }
        // Path-history reversal tiebreaker: within an h-plateau, the
        // pure h_score / depth ordering can dive into a cycle where
        // an atom oscillates between two positions while the rest of
        // the configuration is stuck. Adding a small penalty for
        // moves that return an atom to a recently-vacated position
        // breaks the symmetry between the forward and backward leg of
        // such a cycle without disturbing the priority ordering when
        // h actually moves.
        //
        // All children of a `receive_children` call share the same
        // parent (the just-expanded node), so the parent-chain walk
        // and recent-source map are computed once per call, not per
        // child.
        let parent_id_opt = graph.parent(children[0]);
        let recent_sources: Option<HashMap<u32, HashSet<u64>>> =
            parent_id_opt.map(|pid| compute_recent_sources(graph, pid, IDS_REVERSAL_LOOKBACK));
        for &child_id in children {
            let h = self.heuristic.estimate(graph.config(child_id));
            let depth = graph.depth(child_id);
            let penalty = match (parent_id_opt, &recent_sources) {
                (Some(pid), Some(rs)) => count_reversals(graph, pid, child_id, rs),
                _ => 0,
            };
            let h_with_penalty = h + IDS_REVERSAL_PENALTY * penalty as f64;
            self.heap.push(IdsEntry {
                depth,
                insertion_order: self.insertion_counter,
                h_score: h_with_penalty,
                node_id: child_id,
            });
            self.insertion_counter += 1;
        }
    }
}

/// Walk the parent chain back up to `lookback` steps from `node_id`
/// and accumulate, for each qubit that moved along that prefix, the
/// set of locations it was at *before* each move (i.e. the
/// destinations of any candidate that would *return* the atom to a
/// recently-occupied slot).
fn compute_recent_sources(
    graph: &SearchGraph,
    node_id: NodeId,
    lookback: usize,
) -> HashMap<u32, HashSet<u64>> {
    let mut sources: HashMap<u32, HashSet<u64>> = HashMap::new();
    let mut cur = node_id;
    let mut steps = 0;
    while let Some(parent) = graph.parent(cur) {
        if steps >= lookback {
            break;
        }
        let parent_cfg = graph.config(parent);
        let child_cfg = graph.config(cur);
        for (qid, child_loc) in child_cfg.iter() {
            if let Some(parent_loc) = parent_cfg.location_of(qid) {
                let parent_enc = parent_loc.encode();
                if parent_enc != child_loc.encode() {
                    sources.entry(qid).or_default().insert(parent_enc);
                }
            }
        }
        cur = parent;
        steps += 1;
    }
    sources
}

/// Count, for the move from `parent_id` to `child_id`, how many
/// atoms' new positions are in their recent-source set — i.e. how
/// many atoms are reversing a recent move.
fn count_reversals(
    graph: &SearchGraph,
    parent_id: NodeId,
    child_id: NodeId,
    recent_sources: &HashMap<u32, HashSet<u64>>,
) -> u32 {
    let parent_cfg = graph.config(parent_id);
    let child_cfg = graph.config(child_id);
    let mut count = 0u32;
    for (qid, child_loc) in child_cfg.iter() {
        let Some(parent_loc) = parent_cfg.location_of(qid) else {
            continue;
        };
        let dst = child_loc.encode();
        if parent_loc.encode() == dst {
            continue;
        }
        if let Some(sources) = recent_sources.get(&qid)
            && sources.contains(&dst)
        {
            count += 1;
        }
    }
    count
}

// ── Trait-based search loop (v2) ────────────────────────────────────

/// Debug-only AOD lane-group validation for generator output.
///
/// ## Convention
///
/// Every candidate emitted by a [`MoveGenerator`] must represent a legal
/// AOD move: a single `(move_type, bus_id, direction)` group whose source
/// positions form a complete X×Y rectangle, with no duplicate or unknown
/// lane addresses. The generators enforce this structurally (by how they
/// build rectangles), but this helper acts as a **debug-only safety net**
/// that re-validates each candidate against `ArchSpec::check_lanes` before
/// it is inserted into the search graph.
///
/// Call this once, right after `MoveGenerator::generate`, from any new
/// search loop. Under `#[cfg(debug_assertions)]` it verifies every
/// candidate; in release builds it is a zero-cost no-op.
///
/// Do **not** call this from hot production paths outside the search loop
/// — `ArchSpec::check_lanes` is linear in the group size and allocates.
#[inline]
fn debug_assert_candidates_valid(candidates: &[MoveCandidate], ctx: &SearchContext<'_>) {
    #[cfg(debug_assertions)]
    {
        let arch = ctx.index.arch_spec();
        for candidate in candidates {
            let lanes = candidate.move_set.decode();
            let errors = arch.check_lanes(&lanes);
            debug_assert!(
                errors.is_empty(),
                "generator emitted invalid AOD lane group: {:?} (lanes={:?})",
                errors,
                lanes,
            );
        }
    }
    #[cfg(not(debug_assertions))]
    {
        let _ = (candidates, ctx);
    }
}

/// Run a search using the composable trait-based API.
///
/// Uses separate [`MoveGenerator`], [`CandidateScorer`], [`CostFn`], and
/// [`Goal`] traits. The [`Frontier`] controls node ordering and goal-check timing.
///
/// Two independent limits, deliberately not collapsed into one:
///
/// - `max_depth` bounds the *tree depth* — a horizon in move layers, as the
///   receding-horizon rollout uses.
/// - `max_cost` bounds the accumulated [`CostFn`] cost: a node whose `g` reaches
///   it is neither expanded **nor returned as a goal**. Callers refining an
///   existing solution pass that solution's cost, so only strictly cheaper
///   plans stay reachable — including the plan this call reports.
///
/// They coincide only while `g == depth` (i.e. under
/// [`UniformCost`](crate::cost::UniformCost)). Under a non-uniform objective a
/// cheaper plan can be *deeper* — more shots, each cheaper — so expressing an
/// incumbent bound as a depth cap would exclude exactly the improvements the
/// caller is looking for.
/// Whether `g` has reached the caller's cost cap, i.e. nothing at or beyond
/// this node can be *strictly* cheaper than the incumbent the cap came from.
///
/// Applied to the goal-return paths as well as to expansion: a cap that only
/// stopped expansion still let `run_search` report a plan costlier than the
/// incumbent it was given, which is the opposite of what a refinement pass asks
/// for.
#[inline]
fn reaches_cost_cap(g: f64, max_cost: Option<f64>) -> bool {
    max_cost.is_some_and(|cap| g >= cap)
}

#[allow(clippy::too_many_arguments)]
pub fn run_search<G, S, C, Go, F, O>(
    root: Config,
    generator: &G,
    scorer: &S,
    cost_fn: &C,
    goal: &Go,
    frontier: &mut F,
    ctx: &SearchContext,
    state: &mut SearchState,
    observer: &mut O,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
    max_cost: Option<f64>,
) -> SearchResult
where
    G: MoveGenerator,
    S: CandidateScorer,
    C: CostFn,
    Go: Goal,
    F: Frontier,
    O: SearchObserver,
{
    // Early check: root is already a goal.
    if goal.is_goal(&root) {
        return SearchResult {
            goal: Some(NodeId(0)),
            nodes_expanded: 0,
            max_depth_reached: 0,
            graph: SearchGraph::new(root),
            // The frontier drivers do not prune against an incumbent.
            bound_stats: crate::bounds::BoundStats::default(),
        };
    }

    let mut graph = SearchGraph::new(root);
    let root_id = graph.root();

    // Seed the frontier.
    frontier.receive_children(&[root_id], &graph);

    let mut nodes_expanded: u32 = 0;
    let mut max_depth_seen: u32 = 0;
    let mut closed: Vec<bool> = vec![false; 64];
    let mut candidates: Vec<MoveCandidate> = Vec::new();
    let mut new_children: Vec<NodeId> = Vec::new();

    while let Some(node_id) = frontier.select_next() {
        if let Some(max) = max_expansions
            && nodes_expanded >= max
        {
            break;
        }

        let idx = node_id.0 as usize;

        // Closed set check.
        if idx >= closed.len() {
            closed.resize(idx + 1, false);
        }
        if closed[idx] {
            continue;
        }
        closed[idx] = true;

        // Goal check on pop (A* optimality). A goal at or past the cost cap is
        // not an improvement on the incumbent that cap came from, so it is not
        // reported; falling through leaves the cost gate below to drop it.
        if frontier.check_goal_on_pop()
            && goal.is_goal(graph.config(node_id))
            && !reaches_cost_cap(graph.g_score(node_id), max_cost)
        {
            observer.on_event(SearchEvent::GoalFound {
                depth: graph.depth(node_id),
                node_id,
                config: graph.config(node_id),
            });
            return SearchResult {
                goal: Some(node_id),
                nodes_expanded,
                max_depth_reached: max_depth_seen,
                graph,
                bound_stats: crate::bounds::BoundStats::default(),
            };
        }

        // Depth tracking, then both limits.
        let depth = graph.depth(node_id);
        max_depth_seen = max_depth_seen.max(depth);
        if let Some(max_d) = max_depth
            && depth >= max_d
        {
            continue; // Beyond the caller's layer horizon.
        }
        if let Some(max_c) = max_cost
            && graph.g_score(node_id) >= max_c
        {
            continue; // Cannot beat the caller's incumbent cost.
        }

        // Expand.
        nodes_expanded += 1;
        let current_g = graph.g_score(node_id);

        candidates.clear();
        generator.generate(graph.config(node_id), node_id, ctx, state, &mut candidates);
        debug_assert_candidates_valid(&candidates, ctx);

        observer.on_event(SearchEvent::NodeExpanded {
            depth,
            num_candidates: candidates.len(),
            node_id,
            config: graph.config(node_id),
        });

        // Sort by scorer (higher = better, so sort descending).
        candidates.sort_by(|a, b| {
            scorer
                .score(b, graph.config(node_id), ctx)
                .partial_cmp(&scorer.score(a, graph.config(node_id), ctx))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        new_children.clear();

        for candidate in candidates.drain(..) {
            let edge_cost = cost_fn.edge_cost(
                &candidate.move_set,
                graph.config(node_id),
                &candidate.new_config,
            );
            debug_assert!(edge_cost.is_finite(), "edge_cost must be finite");
            let new_g = current_g + edge_cost;
            let (child_id, is_new) =
                graph.insert(node_id, candidate.move_set, candidate.new_config, new_g);

            let child_idx = child_id.0 as usize;
            let child_closed = child_idx < closed.len() && closed[child_idx];

            if is_new && !child_closed {
                // Goal check on generate (BFS/DFS). Same cap rule as the
                // on-pop path: a child that already reaches the cap cannot be
                // a strictly cheaper plan, so it is queued rather than
                // returned, and the cost gate drops it when popped.
                if frontier.check_goal_on_generate()
                    && goal.is_goal(graph.config(child_id))
                    && !reaches_cost_cap(new_g, max_cost)
                {
                    observer.on_event(SearchEvent::GoalFound {
                        depth: graph.depth(child_id),
                        node_id: child_id,
                        config: graph.config(child_id),
                    });
                    return SearchResult {
                        goal: Some(child_id),
                        nodes_expanded,
                        max_depth_reached: max_depth_seen.max(graph.depth(child_id)),
                        graph,
                        bound_stats: crate::bounds::BoundStats::default(),
                    };
                }
                new_children.push(child_id);
            }
        }

        if !new_children.is_empty() {
            frontier.receive_children(&new_children, &graph);
        }
    }

    SearchResult {
        goal: None,
        nodes_expanded,
        max_depth_reached: max_depth_seen,
        graph,
        bound_stats: crate::bounds::BoundStats::default(),
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::UniformCost;
    use crate::primitives::context::{MoveCandidate, SearchContext, SearchState};
    use crate::primitives::distance::DistanceTable;
    use crate::primitives::graph::MoveSet;
    use crate::primitives::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use std::collections::HashSet;

    // ── v2 fixture ──
    //
    // These doubles replace the retired `Expander` test doubles. The
    // legacy synthetic graphs (line / two-path / diamond) had no real
    // architecture; the v2 `run_search` loop, however, validates every
    // emitted candidate against `ArchSpec::check_lanes` under
    // `debug_assertions`. Each generator therefore emits candidates with
    // an *empty* `MoveSet` (`check_lanes(&[]) == []`) so the synthetic
    // transition graph stays fully under the test's control while the
    // debug validation passes. Edge cost lives in a separate `CostFn`
    // (the legacy `Expander` carried it inline in the successor tuple).

    /// Build a throwaway [`SearchContext`] for the synthetic generators.
    ///
    /// The generators ignore every field; `run_search` only needs a
    /// well-formed context (and a real `index` for debug candidate
    /// validation, which the empty move sets trivially pass).
    struct Fixture {
        index: LaneIndex,
        table: DistanceTable,
        blocked: HashSet<u64>,
        targets: Vec<(u32, u64)>,
    }

    impl Fixture {
        fn new() -> Self {
            let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
            let index = LaneIndex::new(spec);
            let target_locs = [loc(0, 5).encode()];
            let table = DistanceTable::new(&target_locs, &index);
            Self {
                index,
                table,
                blocked: HashSet::new(),
                targets: vec![(0u32, loc(0, 5).encode())],
            }
        }

        fn ctx(&self) -> SearchContext<'_> {
            SearchContext {
                index: &self.index,
                dist_table: &self.table,
                blocked: &self.blocked,
                targets: &self.targets,
                cz_pairs: None,
            }
        }
    }

    /// Empty-move-set candidate moving qubit 0 to `site` (and back into
    /// word 0). Empty move set ⇒ passes `check_lanes` under debug.
    fn step_to(config: &Config, site: u32) -> MoveCandidate {
        MoveCandidate {
            move_set: MoveSet::new([]),
            new_config: config.with_moves(&[(0, loc(0, site))]),
        }
    }

    /// 1D line generator: qubit 0 moves left or right along the site axis.
    struct LineGen {
        max_site: u32,
    }

    impl MoveGenerator for LineGen {
        fn generate(
            &self,
            config: &Config,
            _node_id: NodeId,
            _ctx: &SearchContext,
            _state: &mut SearchState,
            out: &mut Vec<MoveCandidate>,
        ) {
            let site = config.location_of(0).expect("qubit 0").site_id;
            if site > 0 {
                out.push(step_to(config, site - 1));
            }
            if site < self.max_site {
                out.push(step_to(config, site + 1));
            }
        }
    }

    /// Two-path generator (non-uniform cost): from site 0 there is a
    /// direct expensive hop to site 1 and a cheap two-hop detour through
    /// site 2. Optimal cost-1 path is 0 → 2 → 1 (cost 1 + 1 = 2), beating
    /// the direct 0 → 1 (cost 10).
    struct TwoPathGen;

    impl MoveGenerator for TwoPathGen {
        fn generate(
            &self,
            config: &Config,
            _node_id: NodeId,
            _ctx: &SearchContext,
            _state: &mut SearchState,
            out: &mut Vec<MoveCandidate>,
        ) {
            let site = config.location_of(0).unwrap().site_id;
            match site {
                0 => {
                    out.push(step_to(config, 1));
                    out.push(step_to(config, 2));
                }
                2 => out.push(step_to(config, 1)),
                _ => {}
            }
        }
    }

    /// Edge costs for [`TwoPathGen`]: 0→1 = 10, 0→2 = 1, 2→1 = 1.
    struct TwoPathCost;

    impl CostFn for TwoPathCost {
        fn edge_cost(&self, _move_set: &MoveSet, from: &Config, to: &Config) -> f64 {
            let f = from.location_of(0).unwrap().site_id;
            let t = to.location_of(0).unwrap().site_id;
            match (f, t) {
                (0, 1) => 10.0,
                (0, 2) => 1.0,
                (2, 1) => 1.0,
                _ => 1.0,
            }
        }
    }

    /// Diamond generator: 0 → {3 (cost 5), 1 (cost 1)}, 1 → 3 (cost 1),
    /// 3 → 4 (cost 1). Reaching the goal (site 4) optimally goes
    /// 0 → 1 → 3 → 4 (cost 3), even though 0 → 3 is one hop (cost 5).
    /// Site 3 is reached by two paths (transposition); the cheaper one
    /// (via 1) must win, exercising the closed-set / g-score interaction.
    struct DiamondGen;

    impl MoveGenerator for DiamondGen {
        fn generate(
            &self,
            config: &Config,
            _node_id: NodeId,
            _ctx: &SearchContext,
            _state: &mut SearchState,
            out: &mut Vec<MoveCandidate>,
        ) {
            let site = config.location_of(0).unwrap().site_id;
            match site {
                0 => {
                    out.push(step_to(config, 3));
                    out.push(step_to(config, 1));
                }
                1 => out.push(step_to(config, 3)),
                3 => out.push(step_to(config, 4)),
                _ => {}
            }
        }
    }

    /// Edge costs for [`DiamondGen`]: 0→3 = 5, all other edges = 1.
    struct DiamondCost;

    impl CostFn for DiamondCost {
        fn edge_cost(&self, _move_set: &MoveSet, from: &Config, to: &Config) -> f64 {
            let f = from.location_of(0).unwrap().site_id;
            let t = to.location_of(0).unwrap().site_id;
            match (f, t) {
                (0, 3) => 5.0,
                _ => 1.0,
            }
        }
    }

    /// Goal: qubit 0 has reached `target` on the site axis.
    struct SiteGoal {
        target: u32,
    }

    impl Goal for SiteGoal {
        fn is_goal(&self, config: &Config) -> bool {
            config
                .location_of(0)
                .is_some_and(|l| l.site_id == self.target)
        }
    }

    /// Scorer with no ranking preference (synthetic graphs do not depend
    /// on generator-emission order for the invariants under test;
    /// frontier ordering alone determines correctness).
    struct ZeroScorer;

    impl CandidateScorer for ZeroScorer {
        fn score(&self, _candidate: &MoveCandidate, _config: &Config, _ctx: &SearchContext) -> f64 {
            0.0
        }
    }

    fn manhattan(target: u32) -> impl Fn(&Config) -> f64 {
        move |c: &Config| {
            let s = c.location_of(0).expect("qubit 0").site_id;
            (s as f64 - target as f64).abs()
        }
    }

    /// Run the v2 loop over a synthetic generator + cost with `ZeroScorer`,
    /// a [`SiteGoal`], and a `NoOpObserver`, hiding the fixture/context
    /// boilerplate so each test reads like the old `run_search_legacy` call.
    #[allow(clippy::too_many_arguments)]
    fn run<G, C, F>(
        fixture: &Fixture,
        root: Config,
        generator: &G,
        cost: &C,
        goal_site: u32,
        frontier: &mut F,
        max_expansions: Option<u32>,
        max_depth: Option<u32>,
        max_cost: Option<f64>,
    ) -> SearchResult
    where
        G: MoveGenerator,
        C: CostFn,
        F: Frontier,
    {
        let ctx = fixture.ctx();
        let mut state = SearchState::default();
        run_search(
            root,
            generator,
            &ZeroScorer,
            cost,
            &SiteGoal { target: goal_site },
            frontier,
            &ctx,
            &mut state,
            &mut crate::observer::NoOpObserver,
            max_expansions,
            max_depth,
            max_cost,
        )
    }

    // ── BFS ──

    #[test]
    fn bfs_finds_shallowest() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = BfsFrontier::new();
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            3,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.solution_path().unwrap().len(), 3);
    }

    /// `max_depth` is a horizon on *move layers*: a plan of exactly `max_depth`
    /// layers is admissible, one layer more is not.
    ///
    /// Both sides of the boundary are checked. A goal placed well beyond the
    /// cap would pass whether the gate reads `depth >= max` or `depth > max`,
    /// so it would leave the off-by-one untested — and `max_depth` is what the
    /// receding-horizon rollout uses to size its window.
    #[test]
    fn bfs_respects_max_depth() {
        let fx = Fixture::new();
        let at_horizon = run(
            &fx,
            Config::new([(0, loc(0, 0))]).unwrap(),
            &LineGen { max_site: 10 },
            &UniformCost,
            3,
            &mut BfsFrontier::new(),
            None,
            Some(3),
            None,
        );
        assert!(
            at_horizon.goal.is_some(),
            "a plan of exactly max_depth layers is within the horizon"
        );
        assert_eq!(at_horizon.solution_path().unwrap().len(), 3);

        let past_horizon = run(
            &fx,
            Config::new([(0, loc(0, 0))]).unwrap(),
            &LineGen { max_site: 10 },
            &UniformCost,
            4,
            &mut BfsFrontier::new(),
            None,
            Some(3),
            None,
        );
        assert!(
            past_horizon.goal.is_none(),
            "a node at depth == max_depth must not be expanded"
        );
    }

    // ── A* ──

    #[test]
    fn astar_finds_optimal() {
        // Non-uniform cost: cheapest path 0 → 2 → 1 (cost 2) beats the
        // direct 0 → 1 (cost 10). A* on pop with a zero heuristic must
        // settle the goal at g = 2.
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let result = run(
            &fx,
            root,
            &TwoPathGen,
            &TwoPathCost,
            1,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.graph.g_score(result.goal.unwrap()), 2.0);
        assert_eq!(result.solution_path().unwrap().len(), 2);
    }

    #[test]
    fn astar_with_heuristic() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(manhattan(3), 1.0);
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            3,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.solution_path().unwrap().len(), 3);
        // With an exact manhattan heuristic, A* expands exactly 3 nodes
        // on the line (no off-path expansions).
        assert_eq!(result.nodes_expanded, 3);
    }

    #[test]
    fn closed_set_prevents_reexpansion() {
        // Reaching site 3 on a bidirectional line: the closed set must
        // prevent re-expanding already-settled nodes, so exactly 3 nodes
        // (sites 0,1,2) are expanded before the goal pops.
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(manhattan(3), 1.0);
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            3,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.nodes_expanded, 3);
    }

    #[test]
    fn transposition_and_closed_set_interaction() {
        // Site 3 is reachable via 0→3 (cost 5) and 0→1→3 (cost 2). A*
        // must settle site 3 through the cheaper path, yielding an
        // optimal 0→1→3→4 solution at g = 3 (len 3), not the 0→3→4
        // path at g = 6.
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let result = run(
            &fx,
            root,
            &DiamondGen,
            &DiamondCost,
            4,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.graph.g_score(result.goal.unwrap()), 3.0);
        assert_eq!(result.solution_path().unwrap().len(), 3);
    }

    #[test]
    fn astar_max_expansions_respected() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(manhattan(100), 1.0);
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 200 },
            &UniformCost,
            100,
            &mut f,
            Some(5),
            None,
            None,
        );
        assert!(result.goal.is_none());
        assert!(result.nodes_expanded <= 5);
    }

    #[test]
    fn no_path_disconnected() {
        // max_site 0 ⇒ qubit 0 cannot leave site 0, so site 5 is
        // unreachable.
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(manhattan(5), 1.0);
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 0 },
            &UniformCost,
            5,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_none());
    }

    // ── Greedy ──

    #[test]
    fn greedy_finds_goal() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::greedy(manhattan(5));
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            5,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
    }

    // ── DFS ──

    #[test]
    fn dfs_finds_goal() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = DfsFrontier::new(manhattan(5));
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            5,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.solution_path().unwrap().len(), 5);
    }

    #[test]
    fn dfs_respects_max_expansions() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = DfsFrontier::new(manhattan(100));
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 200 },
            &UniformCost,
            100,
            &mut f,
            Some(5),
            None,
            None,
        );
        assert!(result.goal.is_none());
        assert!(result.nodes_expanded <= 5);
    }

    #[test]
    fn dfs_depth_first_ordering() {
        // DFS with a perfect heuristic dives straight to the goal without
        // exploring siblings — fewer (or equal) expansions than BFS.
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();

        let mut dfs = DfsFrontier::new(manhattan(5));
        let dfs_result = run(
            &fx,
            root.clone(),
            &LineGen { max_site: 10 },
            &UniformCost,
            5,
            &mut dfs,
            None,
            None,
            None,
        );

        let mut bfs = BfsFrontier::new();
        let bfs_result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            5,
            &mut bfs,
            None,
            None,
            None,
        );

        assert!(dfs_result.goal.is_some());
        assert!(bfs_result.goal.is_some());
        assert!(dfs_result.nodes_expanded <= bfs_result.nodes_expanded);
    }

    // ── IDS ──

    #[test]
    fn ids_finds_goal() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = IdsFrontier::new(manhattan(5));
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            5,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.solution_path().unwrap().len(), 5);
    }

    #[test]
    fn ids_dives_depth_first() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();

        let mut ids = IdsFrontier::new(manhattan(5));
        let ids_result = run(
            &fx,
            root.clone(),
            &LineGen { max_site: 10 },
            &UniformCost,
            5,
            &mut ids,
            None,
            None,
            None,
        );

        let mut bfs = BfsFrontier::new();
        let bfs_result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            5,
            &mut bfs,
            None,
            None,
            None,
        );

        assert!(ids_result.goal.is_some());
        assert!(bfs_result.goal.is_some());
        assert!(ids_result.nodes_expanded <= bfs_result.nodes_expanded);
    }

    #[test]
    fn ids_respects_max_expansions() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = IdsFrontier::new(manhattan(100));
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 200 },
            &UniformCost,
            100,
            &mut f,
            Some(5),
            None,
            None,
        );
        assert!(result.goal.is_none());
        assert!(result.nodes_expanded <= 5);
    }

    // ── Root is goal ──

    #[test]
    fn root_is_goal_all_strategies() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 3))]).unwrap();
        let generator = LineGen { max_site: 10 };

        for (name, result) in [
            ("bfs", {
                let mut f = BfsFrontier::new();
                run(
                    &fx,
                    root.clone(),
                    &generator,
                    &UniformCost,
                    3,
                    &mut f,
                    None,
                    None,
                    None,
                )
            }),
            ("astar", {
                let mut f = PriorityFrontier::astar(manhattan(3), 1.0);
                run(
                    &fx,
                    root.clone(),
                    &generator,
                    &UniformCost,
                    3,
                    &mut f,
                    None,
                    None,
                    None,
                )
            }),
            ("dfs", {
                let mut f = DfsFrontier::new(manhattan(3));
                run(
                    &fx,
                    root.clone(),
                    &generator,
                    &UniformCost,
                    3,
                    &mut f,
                    None,
                    None,
                    None,
                )
            }),
            ("ids", {
                let mut f = IdsFrontier::new(manhattan(3));
                run(
                    &fx,
                    root.clone(),
                    &generator,
                    &UniformCost,
                    3,
                    &mut f,
                    None,
                    None,
                    None,
                )
            }),
        ] {
            assert!(result.goal.is_some(), "{name} should find root-is-goal");
            assert_eq!(result.nodes_expanded, 0, "{name} should expand 0 nodes");
        }
    }

    // ── max_cost ──

    /// A node whose accumulated `g` *reaches* the cap is neither expanded nor
    /// returned, so no plan costing the cap or more can come back.
    ///
    /// The diamond puts the boundary where a sloppy comparison cannot survive:
    /// the optimal plan `0 → 1 → 3 → 4` costs 3 and runs through a node at
    /// `g = 2`. At `max_cost = 4` the plan is admissible; at `max_cost = 3` the
    /// goal is exactly at the cap and must be refused (gating on `>` would
    /// return it); at `max_cost = 2` the waypoint itself is at the cap, so
    /// nothing expands past it.
    #[test]
    fn max_cost_gates_expansion_and_the_returned_plan() {
        let fx = Fixture::new();

        // Cap above the optimum: the goal is reached, at its true cost.
        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let reachable = run(
            &fx,
            Config::new([(0, loc(0, 0))]).unwrap(),
            &DiamondGen,
            &DiamondCost,
            4,
            &mut f,
            None,
            None,
            Some(4.0),
        );
        let goal = reachable.goal.expect("cost-3 plan is under a cap of 4");
        assert_eq!(reachable.graph.g_score(goal), 3.0);

        // Cap *at* the goal's own cost: not an improvement, so not returned.
        // A refinement pass asks for something strictly cheaper.
        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let at_cap = run(
            &fx,
            Config::new([(0, loc(0, 0))]).unwrap(),
            &DiamondGen,
            &DiamondCost,
            4,
            &mut f,
            None,
            None,
            Some(3.0),
        );
        assert!(
            at_cap.goal.is_none(),
            "a goal at g == max_cost is not strictly cheaper and must not be returned"
        );

        // Cap at the cost of the only viable waypoint: nothing expands past it.
        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let blocked = run(
            &fx,
            Config::new([(0, loc(0, 0))]).unwrap(),
            &DiamondGen,
            &DiamondCost,
            4,
            &mut f,
            None,
            None,
            Some(2.0),
        );
        assert!(
            blocked.goal.is_none(),
            "a node at g == max_cost must not be expanded"
        );
    }

    /// The cap bounds the *returned* plan on both goal-check timings, not just
    /// expansion.
    ///
    /// This is the regression the cap originally missed: the root passes the
    /// gate at `g = 0`, expands, and the child arrives already over the cap —
    /// so a gate on expansion alone returned a plan costing 8 under a cap of 3.
    /// A* checks the goal on pop and BFS on generate, and both paths leaked.
    #[test]
    fn max_cost_bounds_the_returned_plan_on_both_goal_paths() {
        let fx = Fixture::new();

        for (name, goal) in [
            ("bfs (goal on generate)", {
                run(
                    &fx,
                    Config::new([(0, loc(0, 0))]).unwrap(),
                    &LineGen { max_site: 10 },
                    &FrontLoadedCost,
                    1,
                    &mut BfsFrontier::new(),
                    None,
                    None,
                    Some(3.0),
                )
                .goal
            }),
            ("astar (goal on pop)", {
                run(
                    &fx,
                    Config::new([(0, loc(0, 0))]).unwrap(),
                    &LineGen { max_site: 10 },
                    &FrontLoadedCost,
                    1,
                    &mut PriorityFrontier::astar(|_: &Config| 0.0, 1.0),
                    None,
                    None,
                    Some(3.0),
                )
                .goal
            }),
        ] {
            assert!(
                goal.is_none(),
                "{name}: the only plan costs 8, which exceeds the cap of 3"
            );
        }
    }

    /// Edge costs for [`LineGen`] with the expense front-loaded: the first hop
    /// off site 0 costs 8, every later hop costs 1. `g` and `depth` therefore
    /// diverge immediately, which is what separates the two limits below.
    struct FrontLoadedCost;

    impl CostFn for FrontLoadedCost {
        fn edge_cost(&self, _move_set: &MoveSet, from: &Config, _to: &Config) -> f64 {
            if from.location_of(0).unwrap().site_id == 0 {
                8.0
            } else {
                1.0
            }
        }
    }

    /// `max_cost` gates on accumulated cost, not on tree depth — the two are
    /// not interchangeable, which is why `run_search` carries both.
    ///
    /// The same numeric cap of 8 is applied twice to the same instance. As a
    /// *cost* cap it binds at the very first node (`g = 8`) and the goal
    /// becomes unreachable; as a *depth* cap it never binds at all (depth 2)
    /// and the search returns a cost-9 plan. A cap that behaved like a depth
    /// limit would let that cost-9 plan through in both cases.
    ///
    /// Under `UniformCost` the two coincide (`g == depth`) and this distinction
    /// is invisible, which is exactly why it needs pinning: the cascade passes
    /// its incumbent's *cost*, and a non-uniform objective is what makes the
    /// difference observable.
    #[test]
    fn max_cost_gates_on_cost_not_depth() {
        let fx = Fixture::new();

        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let by_cost = run(
            &fx,
            Config::new([(0, loc(0, 0))]).unwrap(),
            &LineGen { max_site: 10 },
            &FrontLoadedCost,
            2,
            &mut f,
            None,
            None,
            Some(8.0),
        );
        assert!(
            by_cost.goal.is_none(),
            "the only route runs through a node at g == 8, which the cap must refuse"
        );

        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let by_depth = run(
            &fx,
            Config::new([(0, loc(0, 0))]).unwrap(),
            &LineGen { max_site: 10 },
            &FrontLoadedCost,
            2,
            &mut f,
            None,
            Some(8),
            None,
        );
        let goal = by_depth
            .goal
            .expect("the same cap read as depth never binds on a two-layer plan");
        assert_eq!(by_depth.graph.g_score(goal), 9.0);
    }

    // ── max_depth_reached tracking ──

    #[test]
    fn max_depth_tracked() {
        let fx = Fixture::new();
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = BfsFrontier::new();
        let result = run(
            &fx,
            root,
            &LineGen { max_site: 10 },
            &UniformCost,
            3,
            &mut f,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert!(result.max_depth_reached >= 3);
    }

    // ── trait-based run_search ──

    #[test]
    fn v2_astar_finds_solution() {
        use crate::cost::UniformCost;
        use crate::generators::HeuristicGenerator;
        use crate::goals::AllAtTarget;
        use crate::observer::NoOpObserver;
        use crate::primitives::context::{SearchContext, SearchState};
        use crate::primitives::distance::{DistanceTable, HopDistanceHeuristic};
        use crate::primitives::lane_index::LaneIndex;
        use crate::scorers::DistanceScorer;
        use crate::test_utils::example_arch_json;
        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
        use std::collections::HashSet;

        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);

        let targets = [(0u32, loc(0, 5))];
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        let table = DistanceTable::new(&target_locs, &index);

        let target_enc: Vec<(u32, u64)> = targets.iter().map(|&(q, l)| (q, l.encode())).collect();
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &table,
            blocked: &blocked,
            targets: &target_enc,
            cz_pairs: None,
        };
        let mut state = SearchState::default();

        let generator = HeuristicGenerator::new();
        let scorer = DistanceScorer;
        let cost = UniformCost;
        let goal = AllAtTarget::new(&target_enc);

        let hop = HopDistanceHeuristic::new(targets, &table);
        let h = move |c: &Config| -> f64 { hop.estimate_max(c) };
        let mut frontier = PriorityFrontier::astar(h, 1.0);

        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let result = run_search(
            config,
            &generator,
            &scorer,
            &cost,
            &goal,
            &mut frontier,
            &ctx,
            &mut state,
            &mut NoOpObserver,
            None,
            None,
            None,
        );
        assert!(result.goal.is_some(), "v2 should find a solution");
    }
}
