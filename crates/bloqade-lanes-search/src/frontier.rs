//! Frontier trait and generic search loop.
//!
//! Implements the traversal strategy abstraction from issue #427:
//! a [`Frontier`] trait controls node ordering and goal-check timing,
//! while [`run_search`] provides the shared search loop.
//!
//! Concrete frontiers: [`PriorityFrontier`] (A* / greedy best-first),
//! [`BfsFrontier`], and [`DfsFrontier`] (heuristic depth-first).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use crate::astar::{Expander, SearchResult};
use crate::config::Config;
use crate::graph::{MoveSet, NodeId, SearchGraph};

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

// ── Generic search loop ─────────────────────────────────────────────

/// Run a search with the given frontier strategy.
///
/// This is the shared loop used by all traversal strategies. The
/// frontier controls node ordering and goal-check timing.
pub fn run_search(
    root: Config,
    goal: impl Fn(&Config) -> bool,
    expander: &impl Expander,
    frontier: &mut impl Frontier,
    max_expansions: Option<u32>,
    max_depth: Option<u32>,
) -> SearchResult {
    // Early check: root is already a goal.
    if goal(&root) {
        return SearchResult {
            goal: Some(NodeId(0)),
            nodes_expanded: 0,
            max_depth_reached: 0,
            graph: SearchGraph::new(root),
        };
    }

    let mut graph = SearchGraph::new(root);
    let root_id = graph.root();

    // Seed the frontier.
    frontier.receive_children(&[root_id], &graph);

    let mut nodes_expanded: u32 = 0;
    let mut max_depth_seen: u32 = 0;
    let mut closed: Vec<bool> = vec![false; 64];
    let mut successors: Vec<(MoveSet, Config, f64)> = Vec::new();
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

        // Goal check on pop (A* optimality).
        if frontier.check_goal_on_pop() && goal(graph.config(node_id)) {
            return SearchResult {
                goal: Some(node_id),
                nodes_expanded,
                max_depth_reached: max_depth_seen,
                graph,
            };
        }

        // Depth tracking + limit.
        let depth = graph.depth(node_id);
        max_depth_seen = max_depth_seen.max(depth);
        if let Some(max_d) = max_depth
            && depth >= max_d
        {
            continue; // Don't expand beyond max depth.
        }

        // Expand.
        nodes_expanded += 1;
        let current_g = graph.g_score(node_id);

        successors.clear();
        expander.expand(graph.config(node_id), &mut successors);

        new_children.clear();

        for (move_set, new_config, edge_cost) in successors.drain(..) {
            debug_assert!(edge_cost.is_finite(), "edge_cost must be finite");
            let new_g = current_g + edge_cost;
            let (child_id, is_new) = graph.insert(node_id, move_set, new_config, new_g);

            let child_idx = child_id.0 as usize;
            let child_closed = child_idx < closed.len() && closed[child_idx];

            if is_new && !child_closed {
                // Goal check on generate (BFS/DFS).
                if frontier.check_goal_on_generate() && goal(graph.config(child_id)) {
                    return SearchResult {
                        goal: Some(child_id),
                        nodes_expanded,
                        max_depth_reached: max_depth_seen.max(graph.depth(child_id)),
                        graph,
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

impl<H: for<'a> Fn(&'a Config) -> f64> Frontier for PriorityFrontier<H> {
    fn select_next(&mut self) -> Option<NodeId> {
        self.heap.pop().map(|e| e.node_id)
    }

    fn receive_children(&mut self, children: &[NodeId], graph: &SearchGraph) {
        for &child_id in children {
            let g = graph.g_score(child_id);
            let h = (self.heuristic)(graph.config(child_id));
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

impl<H: for<'a> Fn(&'a Config) -> f64> Frontier for DfsFrontier<H> {
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
            .map(|&id| ((self.heuristic)(graph.config(id)), id))
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

/// Priority entry for IDS: depth-first with heuristic jump-back.
///
/// Ordering (max-heap): higher depth first, then lower insertion order
/// (preserves expander ranking), then lower h (best heuristic on jump-back).
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
        // Max-heap: higher depth = higher priority (depth-first dive).
        self.depth
            .cmp(&other.depth)
            // Tie-break: lower insertion_order = higher priority (expander's ranking).
            .then(other.insertion_order.cmp(&self.insertion_order))
            // Tie-break: lower h = higher priority (best heuristic on jump-back).
            .then(other.h_score.total_cmp(&self.h_score))
    }
}

impl PartialOrd for IdsEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority-queue frontier for Iterative Diving Search.
///
/// Normally dives depth-first (highest depth popped first), picking the
/// expander's best candidate via insertion order. When all nodes at the
/// current depth are exhausted (dead ends, closed), the heap naturally
/// pops a shallower node with the best heuristic — the "jump-back".
///
/// Inspired by Iterative Diving Search (arxiv:2512.13790).
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

impl<H: for<'a> Fn(&'a Config) -> f64> Frontier for IdsFrontier<H> {
    fn select_next(&mut self) -> Option<NodeId> {
        self.heap.pop().map(|e| e.node_id)
    }

    fn receive_children(&mut self, children: &[NodeId], graph: &SearchGraph) {
        for &child_id in children {
            let h = (self.heuristic)(graph.config(child_id));
            let depth = graph.depth(child_id);
            self.heap.push(IdsEntry {
                depth,
                insertion_order: self.insertion_counter,
                h_score: h,
                node_id: child_id,
            });
            self.insertion_counter += 1;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{dummy_lane, loc};

    /// 1D line expander: qubit 0 can move left or right on site axis.
    struct LineExpander {
        max_site: u32,
    }

    impl Expander for LineExpander {
        fn expand(&self, config: &Config, out: &mut Vec<(MoveSet, Config, f64)>) {
            let (_, current_loc) = config.iter().next().expect("config must have qubit 0");
            let site = current_loc.site_id;
            if site > 0 {
                out.push((
                    MoveSet::new([dummy_lane(site)]),
                    config.with_moves(&[(0, loc(0, site - 1))]),
                    1.0,
                ));
            }
            if site < self.max_site {
                out.push((
                    MoveSet::new([dummy_lane(site)]),
                    config.with_moves(&[(0, loc(0, site + 1))]),
                    1.0,
                ));
            }
        }
    }

    fn site_goal(target: u32) -> impl Fn(&Config) -> bool {
        move |c: &Config| c.location_of(0).is_some_and(|l| l.site_id == target)
    }

    fn manhattan(target: u32) -> impl Fn(&Config) -> f64 {
        move |c: &Config| {
            let s = c.location_of(0).expect("qubit 0").site_id;
            (s as f64 - target as f64).abs()
        }
    }

    // ── BFS ──

    #[test]
    fn bfs_finds_shallowest() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = BfsFrontier::new();
        let result = run_search(
            root,
            site_goal(3),
            &LineExpander { max_site: 10 },
            &mut f,
            None,
            None,
        );
        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 3);
    }

    #[test]
    fn bfs_respects_max_depth() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = BfsFrontier::new();
        let result = run_search(
            root,
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut f,
            None,
            Some(3),
        );
        // Goal at depth 5, max_depth 3 → not found.
        assert!(result.goal.is_none());
    }

    // ── A* ──

    #[test]
    fn astar_finds_optimal() {
        // Non-uniform cost expander.
        struct TwoPathExpander;
        impl Expander for TwoPathExpander {
            fn expand(&self, config: &Config, out: &mut Vec<(MoveSet, Config, f64)>) {
                let site = config.location_of(0).unwrap().site_id;
                match site {
                    0 => {
                        out.push((
                            MoveSet::new([dummy_lane(0)]),
                            config.with_moves(&[(0, loc(0, 1))]),
                            10.0,
                        ));
                        out.push((
                            MoveSet::new([dummy_lane(1)]),
                            config.with_moves(&[(0, loc(0, 2))]),
                            1.0,
                        ));
                    }
                    2 => {
                        out.push((
                            MoveSet::new([dummy_lane(2)]),
                            config.with_moves(&[(0, loc(0, 1))]),
                            1.0,
                        ));
                    }
                    _ => {}
                }
            }
        }

        fn zero_heuristic(_: &Config) -> f64 {
            0.0
        }

        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(zero_heuristic, 1.0);
        let result = run_search(root, site_goal(1), &TwoPathExpander, &mut f, None, None);
        assert!(result.goal.is_some());
        assert_eq!(result.graph.g_score(result.goal.unwrap()), 2.0);
    }

    #[test]
    fn astar_with_heuristic() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::astar(manhattan(3), 1.0);
        let result = run_search(
            root,
            site_goal(3),
            &LineExpander { max_site: 10 },
            &mut f,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.solution_path().unwrap().len(), 3);
        // With manhattan heuristic, A* expands exactly 3 nodes on a line.
        assert_eq!(result.nodes_expanded, 3);
    }

    // ── Greedy ──

    #[test]
    fn greedy_finds_goal() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = PriorityFrontier::greedy(manhattan(5));
        let result = run_search(
            root,
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut f,
            None,
            None,
        );
        assert!(result.goal.is_some());
    }

    // ── DFS ──

    #[test]
    fn dfs_finds_goal() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = DfsFrontier::new(manhattan(5));
        let result = run_search(
            root,
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut f,
            None,
            None,
        );
        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn dfs_respects_max_expansions() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = DfsFrontier::new(manhattan(100));
        let result = run_search(
            root,
            site_goal(100),
            &LineExpander { max_site: 200 },
            &mut f,
            Some(5),
            None,
        );
        assert!(result.goal.is_none());
        assert!(result.nodes_expanded <= 5);
    }

    #[test]
    fn dfs_depth_first_ordering() {
        // DFS with perfect heuristic should go straight to the goal
        // without exploring siblings — fewer expansions than BFS.
        let root = Config::new([(0, loc(0, 0))]).unwrap();

        let mut dfs = DfsFrontier::new(manhattan(5));
        let dfs_result = run_search(
            root.clone(),
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut dfs,
            None,
            None,
        );

        let mut bfs = BfsFrontier::new();
        let bfs_result = run_search(
            root,
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut bfs,
            None,
            None,
        );

        assert!(dfs_result.goal.is_some());
        assert!(bfs_result.goal.is_some());
        // DFS with good heuristic should expand fewer or equal nodes.
        assert!(dfs_result.nodes_expanded <= bfs_result.nodes_expanded);
    }

    // ── IDS ──

    #[test]
    fn ids_finds_goal() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = IdsFrontier::new(manhattan(5));
        let result = run_search(
            root,
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut f,
            None,
            None,
        );
        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn ids_dives_depth_first() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();

        let mut ids = IdsFrontier::new(manhattan(5));
        let ids_result = run_search(
            root.clone(),
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut ids,
            None,
            None,
        );

        let mut bfs = BfsFrontier::new();
        let bfs_result = run_search(
            root,
            site_goal(5),
            &LineExpander { max_site: 10 },
            &mut bfs,
            None,
            None,
        );

        assert!(ids_result.goal.is_some());
        assert!(bfs_result.goal.is_some());
        assert!(ids_result.nodes_expanded <= bfs_result.nodes_expanded);
    }

    #[test]
    fn ids_respects_max_expansions() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = IdsFrontier::new(manhattan(100));
        let result = run_search(
            root,
            site_goal(100),
            &LineExpander { max_site: 200 },
            &mut f,
            Some(5),
            None,
        );
        assert!(result.goal.is_none());
        assert!(result.nodes_expanded <= 5);
    }

    // ── Root is goal ──

    #[test]
    fn root_is_goal_all_strategies() {
        let root = Config::new([(0, loc(0, 3))]).unwrap();
        let expander = LineExpander { max_site: 10 };

        for (name, result) in [
            ("bfs", {
                let mut f = BfsFrontier::new();
                run_search(root.clone(), site_goal(3), &expander, &mut f, None, None)
            }),
            ("astar", {
                let mut f = PriorityFrontier::astar(manhattan(3), 1.0);
                run_search(root.clone(), site_goal(3), &expander, &mut f, None, None)
            }),
            ("dfs", {
                let mut f = DfsFrontier::new(manhattan(3));
                run_search(root.clone(), site_goal(3), &expander, &mut f, None, None)
            }),
            ("ids", {
                let mut f = IdsFrontier::new(manhattan(3));
                run_search(root.clone(), site_goal(3), &expander, &mut f, None, None)
            }),
        ] {
            assert!(result.goal.is_some(), "{name} should find root-is-goal");
            assert_eq!(result.nodes_expanded, 0, "{name} should expand 0 nodes");
        }
    }

    // ── max_depth_reached tracking ──

    #[test]
    fn max_depth_tracked() {
        let root = Config::new([(0, loc(0, 0))]).unwrap();
        let mut f = BfsFrontier::new();
        let result = run_search(
            root,
            site_goal(3),
            &LineExpander { max_site: 10 },
            &mut f,
            None,
            None,
        );
        assert!(result.goal.is_some());
        assert!(result.max_depth_reached >= 3);
    }
}
