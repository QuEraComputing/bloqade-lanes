//! High-level solver that ties together the lane index, expander, heuristic,
//! and A* search into a single reusable object.
//!
//! [`MoveSolver`] is constructed once per architecture (parsing JSON and
//! building indexes) and can then solve multiple placement problems.

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::astar;
use crate::config::Config;
use crate::graph::MoveSet;
use crate::heuristic::{DistanceTable, HopDistanceHeuristic};
use crate::heuristic_expander::HeuristicExpander;
use crate::lane_index::LaneIndex;

/// Result of a successful solve.
#[derive(Debug)]
pub struct SolveResult {
    /// Sequence of parallel move steps from initial to goal configuration.
    pub move_layers: Vec<MoveSet>,
    /// Final qubit positions (the goal configuration).
    pub goal_config: Config,
    /// Number of nodes expanded during search.
    pub nodes_expanded: u32,
    /// Total path cost.
    pub cost: f64,
}

/// Reusable move synthesis solver.
///
/// Constructed once per architecture — parses the arch spec JSON and builds
/// the lane index. Then [`solve`](MoveSolver::solve) can be called multiple
/// times with different initial/target placements.
///
/// Works for both physical and logical architectures (same interface,
/// different arch spec JSON).
pub struct MoveSolver {
    index: LaneIndex,
}

impl MoveSolver {
    /// Construct from an [`ArchSpec`] JSON string.
    ///
    /// Parses the JSON, builds the lane index (precomputes all lane lookups,
    /// endpoints, and positions).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let arch_spec = serde_json::from_str(json)?;
        Ok(Self {
            index: LaneIndex::new(arch_spec),
        })
    }

    /// Construct from an existing [`LaneIndex`].
    pub fn from_index(index: LaneIndex) -> Self {
        Self { index }
    }

    /// Access the underlying lane index.
    pub fn index(&self) -> &LaneIndex {
        &self.index
    }

    /// Solve a move synthesis problem.
    ///
    /// Finds the minimum-cost sequence of parallel move steps to move
    /// qubits from `initial` placement to `target` placement, avoiding
    /// `blocked` locations.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions: `(qubit_id, location)` pairs.
    /// * `target` — Desired qubit positions: `(qubit_id, location)` pairs.
    /// * `blocked` — Locations occupied by external atoms (immovable obstacles).
    /// * `max_expansions` — Optional limit on A* node expansions.
    ///
    /// # Returns
    ///
    /// `Some(SolveResult)` if a solution is found, `None` otherwise.
    pub fn solve(
        &self,
        initial: impl IntoIterator<Item = (u32, LocationAddr)>,
        target: impl IntoIterator<Item = (u32, LocationAddr)>,
        blocked: impl IntoIterator<Item = LocationAddr>,
        max_expansions: Option<u32>,
    ) -> Option<SolveResult> {
        let root = Config::new(initial);
        let target_pairs: Vec<(u32, LocationAddr)> = target.into_iter().collect();

        // Build goal predicate: all qubits at their target locations.
        let target_encoded: Vec<(u32, u32)> =
            target_pairs.iter().map(|&(q, l)| (q, l.encode())).collect();
        let goal = |config: &Config| -> bool {
            target_encoded.iter().all(|&(qid, target_enc)| {
                config
                    .location_of(qid)
                    .is_some_and(|l| l.encode() == target_enc)
            })
        };

        // Build distance table (shared between heuristic and expander).
        let target_locs: Vec<u32> = target_encoded.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &self.index);

        // Build heuristic from shared distance table.
        let heuristic = HopDistanceHeuristic::new(target_pairs.iter().copied(), &dist_table);
        let heuristic_fn = |config: &Config| -> f64 { heuristic.estimate(config) };

        // Build expander (heuristic: scores qubit-bus pairs, generates ~5-15 candidates).
        let expander = HeuristicExpander::new(
            &self.index,
            blocked,
            target_pairs.iter().copied(),
            &dist_table,
            3, // top_c
            3, // max_movesets_per_group
        );

        // Run A*.
        let result = astar::astar(root, goal, heuristic_fn, &expander, max_expansions);

        let goal_id = result.goal?;
        let move_layers = result.solution_path().unwrap_or_default();
        let goal_config = result.graph.config(goal_id).clone();
        let cost = result.graph.g_score(goal_id);

        Some(SolveResult {
            move_layers,
            goal_config,
            nodes_expanded: result.nodes_expanded,
            cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_arch_json() -> &'static str {
        r#"{
            "version": "2.0",
            "geometry": {
                "sites_per_word": 10,
                "words": [
                    {
                        "positions": { "x_start": 1.0, "y_start": 2.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                        "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                    },
                    {
                        "positions": { "x_start": 1.0, "y_start": 12.5, "x_spacing": [2.0, 2.0, 2.0, 2.0], "y_spacing": [2.5] },
                        "site_indices": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]]
                    }
                ]
            },
            "buses": {
                "site_buses": [
                    { "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
                ],
                "word_buses": [
                    { "src": [0], "dst": [1] }
                ]
            },
            "words_with_site_buses": [0, 1],
            "sites_with_word_buses": [5, 6, 7, 8, 9],
            "zones": [
                { "words": [0, 1] }
            ],
            "entangling_zones": [[[0, 1]]],
            "blockade_radius": 2.0,
            "measurement_mode_zones": [0]
        }"#
    }

    fn loc(word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            word_id: word,
            site_id: site,
        }
    }

    #[test]
    fn solve_simple_one_step() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
            )
            .unwrap();

        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
    }

    #[test]
    fn solve_already_at_target() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let result = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
            )
            .unwrap();

        assert_eq!(result.cost, 0.0);
        assert!(result.move_layers.is_empty());
        assert_eq!(result.nodes_expanded, 0);
    }

    #[test]
    fn solve_cross_word() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Move qubit from word 0 site 5 to word 1 site 5 (one word bus hop).
        let result = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(100),
            )
            .unwrap();

        assert_eq!(result.cost, 1.0);
        assert_eq!(result.move_layers.len(), 1);
        assert_eq!(result.goal_config.location_of(0), Some(loc(1, 5)));
    }

    #[test]
    fn solve_multi_step() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Word 0 site 0 → word 1 site 5: needs site bus + word bus = 2 steps.
        let result = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(1, 5))],
                std::iter::empty(),
                Some(1000),
            )
            .unwrap();

        assert_eq!(result.cost, 2.0);
        assert_eq!(result.move_layers.len(), 2);
    }

    #[test]
    fn solve_no_solution() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Target a nonexistent location.
        let result = solver.solve(
            [(0, loc(0, 0))],
            [(0, loc(99, 99))],
            std::iter::empty(),
            Some(100),
        );

        assert!(result.is_none());
    }

    #[test]
    fn solver_reusable() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();

        let r1 = solver
            .solve(
                [(0, loc(0, 0))],
                [(0, loc(0, 5))],
                std::iter::empty(),
                Some(100),
            )
            .unwrap();

        let r2 = solver
            .solve(
                [(0, loc(0, 5))],
                [(0, loc(0, 0))],
                std::iter::empty(),
                Some(100),
            )
            .unwrap();

        assert_eq!(r1.cost, 1.0);
        assert_eq!(r2.cost, 1.0);
    }

    #[test]
    fn solve_with_blocked() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Qubit at site 0, target site 5, but site 5 is blocked.
        // Should find no solution (or a longer path if one exists).
        let result = solver.solve([(0, loc(0, 0))], [(0, loc(0, 5))], [loc(0, 5)], Some(100));

        // Can't reach blocked destination.
        assert!(result.is_none());
    }

    #[test]
    fn solve_multiple_qubits() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        // Move two qubits from sites 0,1 to sites 5,6 (parallel site bus move).
        let result = solver
            .solve(
                [(0, loc(0, 0)), (1, loc(0, 1))],
                [(0, loc(0, 5)), (1, loc(0, 6))],
                std::iter::empty(),
                Some(1000),
            )
            .unwrap();

        // Should find the parallel move in 1 step.
        assert_eq!(result.cost, 1.0);
        assert_eq!(result.goal_config.location_of(0), Some(loc(0, 5)));
        assert_eq!(result.goal_config.location_of(1), Some(loc(0, 6)));
    }
}
