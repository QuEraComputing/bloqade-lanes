//! A* search algorithm.
//!
//! Provides a generic [`astar`] function that searches over configurations
//! using a pluggable [`Expander`] trait for successor generation.
//!
//! Key correctness properties (vs. the Python implementation):
//! - Uses a proper closed set to prevent re-expansion of nodes.
//! - The transposition table uses g-score (cost), not depth.
//! - Goal is checked when a node is **popped** from the open set,
//!   guaranteeing optimality with an admissible heuristic.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::config::Config;
use crate::graph::{MoveSet, NodeId, SearchGraph};

/// Trait for generating successor configurations.
///
/// This is the integration point for move validation. Implementations
/// use architecture information to enumerate valid move sets and apply
/// them, but the search algorithm itself knows nothing about lanes,
/// buses, or architecture constraints.
pub trait Expander {
    /// Generate all valid successors of the given configuration.
    ///
    /// Appends `(move_set, new_config, edge_cost)` triples to `out`.
    /// The caller clears `out` before each call, allowing buffer reuse
    /// across expansions to avoid per-call allocation.
    ///
    /// `edge_cost` is the cost of this single transition (typically 1.0
    /// for unit-cost search, but could represent lane duration, etc.).
    fn expand(&self, config: &Config, out: &mut Vec<(MoveSet, Config, f64)>);
}

/// Result of an A* search.
#[derive(Debug)]
pub struct SearchResult {
    /// The goal node, if found.
    pub goal: Option<NodeId>,
    /// Number of nodes expanded (popped from open set and not in closed set).
    pub nodes_expanded: u32,
    /// The search graph, for path reconstruction and inspection.
    pub graph: SearchGraph,
}

impl SearchResult {
    /// Reconstruct the solution path (sequence of move sets from root to goal).
    ///
    /// Returns `None` if no goal was found.
    pub fn solution_path(&self) -> Option<Vec<MoveSet>> {
        self.goal.map(|id| self.graph.reconstruct_path(id))
    }
}

/// Priority queue entry for A*.
///
/// Ordered by f-score (lower = higher priority). Ties are broken by
/// preferring higher g-score (deeper nodes first, LIFO-like behavior).
struct OpenEntry {
    f_score: f64,
    g_score: f64,
    node_id: NodeId,
}

impl Eq for OpenEntry {}

impl PartialEq for OpenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.f_score.total_cmp(&other.f_score) == Ordering::Equal
            && self.g_score.total_cmp(&other.g_score) == Ordering::Equal
    }
}

impl Ord for OpenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap, so reverse f_score for min-heap behavior.
        // For ties, prefer higher g_score (deeper node = closer to goal).
        other
            .f_score
            .total_cmp(&self.f_score)
            .then(self.g_score.total_cmp(&other.g_score))
    }
}

impl PartialOrd for OpenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A* search over configurations.
///
/// Finds the lowest-cost path from `root` to a configuration satisfying
/// `goal`, using `heuristic` to estimate cost-to-go and `expander` to
/// generate successors.
///
/// # Optimality
///
/// With an admissible heuristic (never overestimates), A* guarantees
/// finding the optimal (lowest cost) solution.
///
/// # Arguments
///
/// * `root` — Initial configuration.
/// * `goal` — Returns `true` for goal configurations.
/// * `heuristic` — Admissible estimate of cost from a configuration to goal.
/// * `expander` — Generates successor configurations.
/// * `max_expansions` — Optional limit on nodes expanded. `None` for no limit.
pub fn astar(
    root: Config,
    goal: impl Fn(&Config) -> bool,
    heuristic: impl Fn(&Config) -> f64,
    expander: &impl Expander,
    max_expansions: Option<u32>,
) -> SearchResult {
    // Check if root is already a goal.
    if goal(&root) {
        return SearchResult {
            goal: Some(NodeId(0)),
            nodes_expanded: 0,
            graph: SearchGraph::new(root),
        };
    }

    let mut graph = SearchGraph::new(root);
    let mut open: BinaryHeap<OpenEntry> = BinaryHeap::new();
    // Dense closed set: NodeIds are sequential u32 indices, so Vec<bool>
    // is faster than HashSet (no hashing, cache-friendly).
    let mut closed: Vec<bool> = vec![false; 64];

    let root_id = graph.root();
    let h = heuristic(graph.config(root_id));
    open.push(OpenEntry {
        f_score: h,
        g_score: 0.0,
        node_id: root_id,
    });

    let mut nodes_expanded: u32 = 0;
    let mut successors: Vec<(MoveSet, Config, f64)> = Vec::new();

    while let Some(entry) = open.pop() {
        if let Some(max) = max_expansions
            && nodes_expanded >= max
        {
            break;
        }

        let node_id = entry.node_id;
        let idx = node_id.0 as usize;

        // Skip if already expanded (closed set).
        if idx >= closed.len() {
            closed.resize(idx + 1, false);
        }
        if closed[idx] {
            continue;
        }
        closed[idx] = true;

        // Goal check on pop — guarantees optimality.
        if goal(graph.config(node_id)) {
            return SearchResult {
                goal: Some(node_id),
                nodes_expanded,
                graph,
            };
        }

        nodes_expanded += 1;
        let current_g = graph.g_score(node_id);

        successors.clear();
        expander.expand(graph.config(node_id), &mut successors);

        for (move_set, new_config, edge_cost) in successors.drain(..) {
            let new_g = current_g + edge_cost;
            let (child_id, is_new) = graph.insert(node_id, move_set, new_config, new_g);

            let child_idx = child_id.0 as usize;
            let child_closed = child_idx < closed.len() && closed[child_idx];
            if is_new && !child_closed {
                let h = heuristic(graph.config(child_id));
                open.push(OpenEntry {
                    f_score: new_g + h,
                    g_score: new_g,
                    node_id: child_id,
                });
            }
        }
    }

    SearchResult {
        goal: None,
        nodes_expanded,
        graph,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

    // ── Test helpers ──

    fn loc(word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            word_id: word,
            site_id: site,
        }
    }

    fn dummy_lane(id: u32) -> LaneAddr {
        LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            word_id: 0,
            site_id: id,
            bus_id: 0,
        }
    }

    // ── 1D line expander: qubit 0 at site S can move to S-1 or S+1 ──

    struct LineExpander {
        max_site: u32,
    }

    impl Expander for LineExpander {
        fn expand(&self, config: &Config, out: &mut Vec<(MoveSet, Config, f64)>) {
            let (_, current_loc) = config.iter().next().expect("config must have qubit 0");
            let site = current_loc.site_id;

            if site > 0 {
                let new_loc = loc(0, site - 1);
                let ms = MoveSet::new([dummy_lane(site)]);
                out.push((ms, config.with_moves(&[(0, new_loc)]), 1.0));
            }
            if site < self.max_site {
                let new_loc = loc(0, site + 1);
                let ms = MoveSet::new([dummy_lane(site)]);
                out.push((ms, config.with_moves(&[(0, new_loc)]), 1.0));
            }
        }
    }

    fn manhattan_heuristic(target_site: u32) -> impl Fn(&Config) -> f64 {
        move |config: &Config| {
            let current = config.location_of(0).expect("qubit 0 must exist");
            (current.site_id as f64 - target_site as f64).abs()
        }
    }

    fn site_goal(target_site: u32) -> impl Fn(&Config) -> bool {
        move |config: &Config| {
            config
                .location_of(0)
                .is_some_and(|l| l.site_id == target_site)
        }
    }

    // ── Tests ──

    #[test]
    fn trivial_one_step() {
        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(1),
            manhattan_heuristic(1),
            &LineExpander { max_site: 10 },
            None,
        );
        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn multi_step_optimal() {
        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(5),
            manhattan_heuristic(5),
            &LineExpander { max_site: 10 },
            None,
        );
        assert!(result.goal.is_some());
        let path = result.solution_path().unwrap();
        // Optimal path on a line from 0 to 5 is exactly 5 steps.
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn no_path_disconnected() {
        // Qubit at site 0, goal at site 5, but max_site=0 so no moves.
        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(5),
            manhattan_heuristic(5),
            &LineExpander { max_site: 0 },
            None,
        );
        assert!(result.goal.is_none());
    }

    #[test]
    fn max_expansions_respected() {
        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(100),
            manhattan_heuristic(100),
            &LineExpander { max_site: 200 },
            Some(5),
        );
        // With only 5 expansions, shouldn't reach site 100.
        assert!(result.goal.is_none());
        assert!(result.nodes_expanded <= 5);
    }

    #[test]
    fn root_is_goal() {
        let root = Config::new([(0, loc(0, 3))]);
        let result = astar(
            root,
            site_goal(3),
            manhattan_heuristic(3),
            &LineExpander { max_site: 10 },
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.nodes_expanded, 0);
        assert!(result.solution_path().unwrap().is_empty());
    }

    #[test]
    fn nonuniform_cost_finds_cheapest() {
        // Two-path graph: qubit 0 starts at (0,0).
        // Path A: (0,0) → (0,1) cost 10
        // Path B: (0,0) → (0,2) → (0,1) cost 1+1=2
        // A* should find path B.

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

        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(1),
            |_| 0.0, // trivial heuristic
            &TwoPathExpander,
            None,
        );
        assert!(result.goal.is_some());
        let goal_id = result.goal.unwrap();
        // Cheapest path costs 2.0, not 10.0.
        assert_eq!(result.graph.g_score(goal_id), 2.0);
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 2); // two steps via detour
    }

    #[test]
    fn closed_set_prevents_reexpansion() {
        // On a line 0..10, expanding from site 0 toward site 5.
        // Without a closed set, nodes could be re-expanded.
        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(3),
            manhattan_heuristic(3),
            &LineExpander { max_site: 10 },
            None,
        );
        assert!(result.goal.is_some());
        // Optimal path is 3 steps. With good heuristic, A* should
        // expand very few nodes (3 on the optimal path + maybe a
        // few neighbors). Without closed set, it would expand more.
        // On a line with manhattan heuristic, A* expands exactly
        // the nodes on the optimal path (3 expansions).
        assert_eq!(result.nodes_expanded, 3);
    }

    #[test]
    fn solution_path_returns_correct_sequence() {
        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(3),
            manhattan_heuristic(3),
            &LineExpander { max_site: 10 },
            None,
        );
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 3);

        // Verify the path is a valid sequence of single-lane moves.
        for ms in &path {
            assert_eq!(ms.len(), 1);
        }
    }

    #[test]
    fn transposition_and_closed_set_interaction() {
        // Construct a graph where a node is discovered via an expensive
        // path first, then via a cheaper path. Verify the solution uses
        // the cheaper path.
        //
        // Graph:
        //   Start(0,0) --cost 5--> Mid(0,3) --cost 1--> Goal(0,4)
        //   Start(0,0) --cost 1--> Via(0,1) --cost 1--> Mid(0,3) --cost 1--> Goal(0,4)
        //
        // Expensive to Mid: cost 5
        // Cheap to Mid via (0,1): cost 2
        // A* should find the cheap path.

        struct DiamondExpander;

        impl Expander for DiamondExpander {
            fn expand(&self, config: &Config, out: &mut Vec<(MoveSet, Config, f64)>) {
                let site = config.location_of(0).unwrap().site_id;
                match site {
                    0 => {
                        out.push((
                            MoveSet::new([dummy_lane(10)]),
                            config.with_moves(&[(0, loc(0, 3))]),
                            5.0,
                        ));
                        out.push((
                            MoveSet::new([dummy_lane(11)]),
                            config.with_moves(&[(0, loc(0, 1))]),
                            1.0,
                        ));
                    }
                    1 => {
                        out.push((
                            MoveSet::new([dummy_lane(12)]),
                            config.with_moves(&[(0, loc(0, 3))]),
                            1.0,
                        ));
                    }
                    3 => {
                        out.push((
                            MoveSet::new([dummy_lane(13)]),
                            config.with_moves(&[(0, loc(0, 4))]),
                            1.0,
                        ));
                    }
                    _ => {}
                }
            }
        }

        let root = Config::new([(0, loc(0, 0))]);
        let result = astar(
            root,
            site_goal(4),
            |_| 0.0, // no heuristic guidance
            &DiamondExpander,
            None,
        );

        assert!(result.goal.is_some());
        let goal_id = result.goal.unwrap();
        // Cheap path: 0→1→3→4 costs 1+1+1=3, not 0→3→4 costing 5+1=6.
        assert_eq!(result.graph.g_score(goal_id), 3.0);
        let path = result.solution_path().unwrap();
        assert_eq!(path.len(), 3);
    }
}
