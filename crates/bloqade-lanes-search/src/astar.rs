//! A* search algorithm and shared search types.
//!
//! Provides the [`Expander`] trait, [`SearchResult`] type, and an [`astar`]
//! convenience function. The actual search loop is in [`crate::frontier`].

use crate::config::Config;
use crate::frontier::{self, PriorityFrontier};
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

/// Result of a search.
#[derive(Debug)]
pub struct SearchResult {
    /// The goal node, if found.
    pub goal: Option<NodeId>,
    /// Number of nodes expanded (popped from frontier and not in closed set).
    pub nodes_expanded: u32,
    /// Maximum depth reached during search.
    pub max_depth_reached: u32,
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

/// A* search over configurations.
///
/// Convenience wrapper over [`frontier::run_search`] with
/// [`PriorityFrontier::astar`]. Finds the lowest-cost path from `root`
/// to a configuration satisfying `goal`.
///
/// With an admissible heuristic (never overestimates), A* guarantees
/// finding the optimal (lowest cost) solution.
pub fn astar(
    root: Config,
    goal: impl Fn(&Config) -> bool,
    heuristic: impl Fn(&Config) -> f64,
    expander: &impl Expander,
    max_expansions: Option<u32>,
) -> SearchResult {
    let mut f = PriorityFrontier::astar(heuristic, 1.0);
    frontier::run_search(root, goal, expander, &mut f, max_expansions, None, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

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

    #[test]
    fn trivial_one_step() {
        let result = astar(
            Config::new([(0, loc(0, 0))]),
            site_goal(1),
            manhattan_heuristic(1),
            &LineExpander { max_site: 10 },
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.solution_path().unwrap().len(), 1);
    }

    #[test]
    fn multi_step_optimal() {
        let result = astar(
            Config::new([(0, loc(0, 0))]),
            site_goal(5),
            manhattan_heuristic(5),
            &LineExpander { max_site: 10 },
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.solution_path().unwrap().len(), 5);
    }

    #[test]
    fn no_path_disconnected() {
        let result = astar(
            Config::new([(0, loc(0, 0))]),
            site_goal(5),
            manhattan_heuristic(5),
            &LineExpander { max_site: 0 },
            None,
        );
        assert!(result.goal.is_none());
    }

    #[test]
    fn max_expansions_respected() {
        let result = astar(
            Config::new([(0, loc(0, 0))]),
            site_goal(100),
            manhattan_heuristic(100),
            &LineExpander { max_site: 200 },
            Some(5),
        );
        assert!(result.goal.is_none());
        assert!(result.nodes_expanded <= 5);
    }

    #[test]
    fn root_is_goal() {
        let result = astar(
            Config::new([(0, loc(0, 3))]),
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

        fn zero(_: &Config) -> f64 {
            0.0
        }

        let result = astar(
            Config::new([(0, loc(0, 0))]),
            site_goal(1),
            zero,
            &TwoPathExpander,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.graph.g_score(result.goal.unwrap()), 2.0);
        assert_eq!(result.solution_path().unwrap().len(), 2);
    }

    #[test]
    fn closed_set_prevents_reexpansion() {
        let result = astar(
            Config::new([(0, loc(0, 0))]),
            site_goal(3),
            manhattan_heuristic(3),
            &LineExpander { max_site: 10 },
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.nodes_expanded, 3);
    }

    #[test]
    fn transposition_and_closed_set_interaction() {
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

        fn zero(_: &Config) -> f64 {
            0.0
        }

        let result = astar(
            Config::new([(0, loc(0, 0))]),
            site_goal(4),
            zero,
            &DiamondExpander,
            None,
        );
        assert!(result.goal.is_some());
        assert_eq!(result.graph.g_score(result.goal.unwrap()), 3.0);
        assert_eq!(result.solution_path().unwrap().len(), 3);
    }
}
