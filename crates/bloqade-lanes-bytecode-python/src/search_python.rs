//! PyO3 bindings for the move synthesis solver.
//!
//! Exposes [`PyMoveSolver`] and [`PySolveResult`] to Python. Uses a JSON
//! bridge for `ArchSpec` — the solver is fully decoupled from the bytecode
//! Python bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_search::heuristic_expander::FreeRiderPolicy;
use bloqade_lanes_search::solve::{MoveSolver, SolveResult, SolveStatus, Strategy};

use crate::arch_python::PyArchSpec;

/// Result of a move synthesis solve.
///
/// Contains the sequence of move steps, the final qubit configuration,
/// and search statistics.
#[pyclass(name = "SolveResult", frozen, module = "bloqade.lanes.bytecode")]
pub struct PySolveResult {
    inner: SolveResult,
}

#[pymethods]
impl PySolveResult {
    /// Status of the solve: "solved", "unsolvable", or "budget_exceeded".
    #[getter]
    fn status(&self) -> &'static str {
        match self.inner.status {
            SolveStatus::Solved => "solved",
            SolveStatus::Unsolvable => "unsolvable",
            SolveStatus::BudgetExceeded => "budget_exceeded",
        }
    }

    /// Move layers: list of move steps, each a list of lane address tuples.
    ///
    /// Each lane is represented as `(direction, move_type, word_id, site_id, bus_id)`
    /// where direction is 0=Forward/1=Backward and move_type is 0=SiteBus/1=WordBus.
    #[getter]
    #[allow(clippy::type_complexity)]
    fn move_layers(&self) -> Vec<Vec<(u8, u8, u32, u32, u32)>> {
        self.inner
            .move_layers
            .iter()
            .map(|ms| {
                ms.decode()
                    .into_iter()
                    .map(|lane| {
                        (
                            lane.direction as u8,
                            lane.move_type as u8,
                            lane.word_id,
                            lane.site_id,
                            lane.bus_id,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// Goal configuration: list of (qubit_id, word_id, site_id) tuples.
    #[getter]
    fn goal_config(&self) -> Vec<(u32, u32, u32)> {
        self.inner
            .goal_config
            .iter()
            .map(|(qid, loc)| (qid, loc.word_id, loc.site_id))
            .collect()
    }

    /// Number of nodes expanded during search.
    #[getter]
    fn nodes_expanded(&self) -> u32 {
        self.inner.nodes_expanded
    }

    /// Total path cost.
    #[getter]
    fn cost(&self) -> f64 {
        self.inner.cost
    }

    /// Number of deadlocks encountered during search.
    #[getter]
    fn deadlocks(&self) -> u32 {
        self.inner.deadlocks
    }

    fn __repr__(&self) -> String {
        let status = match self.inner.status {
            SolveStatus::Solved => "solved",
            SolveStatus::Unsolvable => "unsolvable",
            SolveStatus::BudgetExceeded => "budget_exceeded",
        };
        format!(
            "SolveResult(status={}, steps={}, cost={}, expanded={}, deadlocks={})",
            status,
            self.inner.move_layers.len(),
            self.inner.cost,
            self.inner.nodes_expanded,
            self.inner.deadlocks,
        )
    }
}

/// Reusable move synthesis solver.
///
/// Constructed once from an architecture specification JSON string.
/// The constructor parses the JSON and precomputes lane indexes.
/// Then `solve()` can be called multiple times with different placements.
///
/// Works for both physical and logical architectures.
#[pyclass(name = "MoveSolver", frozen, module = "bloqade.lanes.bytecode")]
pub struct PyMoveSolver {
    inner: MoveSolver,
}

#[pymethods]
impl PyMoveSolver {
    /// Create a solver from an ArchSpec JSON string.
    ///
    /// Args:
    ///     arch_spec_json: JSON string of the architecture specification.
    ///
    /// Raises:
    ///     ValueError: If the JSON is invalid.
    #[new]
    fn new(arch_spec_json: &str) -> PyResult<Self> {
        let inner = MoveSolver::from_json(arch_spec_json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec JSON: {e}")))?;
        Ok(Self { inner })
    }

    /// Create a solver from a native ArchSpec object.
    ///
    /// Serializes the ArchSpec to JSON internally, avoiding manual
    /// JSON round-trips.
    #[staticmethod]
    fn from_arch_spec(arch: &PyArchSpec) -> PyResult<Self> {
        let json = serde_json::to_string(&arch.inner)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize arch spec: {e}")))?;
        let inner = MoveSolver::from_json(&json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec: {e}")))?;
        Ok(Self { inner })
    }

    /// Solve a move synthesis problem.
    ///
    /// Finds the minimum-cost sequence of parallel move steps to move
    /// qubits from initial to target placement, avoiding blocked locations.
    ///
    /// Args:
    ///     initial: List of (qubit_id, word_id, site_id) tuples for starting positions.
    ///     target: List of (qubit_id, word_id, site_id) tuples for desired positions.
    ///     blocked: List of (word_id, site_id) tuples for immovable obstacle locations.
    ///     max_expansions: Optional limit on node expansions.
    ///     strategy: Search strategy: "astar", "dfs", "bfs", "greedy", "ids", "cascade".
    ///     top_c: Top bus options per qubit in the heuristic expander (default 3).
    ///     max_movesets_per_group: Max movesets per bus group (default 3).
    ///     weight: Heuristic weight for A* (1.0 = standard, >1.0 = bounded suboptimal).
    ///     mobility_weight: Weight for mobility bonus in expander scoring (0.0 = disabled).
    ///     restarts: Number of parallel restarts with perturbed scoring (1 = no restarts).
    ///     free_riders: Free rider policy: "off", "unblock", "unblock_or_improve".
    ///
    /// Returns:
    ///     SolveResult with status indicating success/failure.
    #[pyo3(signature = (initial, target, blocked, max_expansions=None, strategy="astar", top_c=3, max_movesets_per_group=3, weight=1.0, mobility_weight=0.0, restarts=1, free_riders="off"))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        initial: Vec<(u32, u32, u32)>,
        target: Vec<(u32, u32, u32)>,
        blocked: Vec<(u32, u32)>,
        max_expansions: Option<u32>,
        strategy: &str,
        top_c: usize,
        max_movesets_per_group: usize,
        weight: f64,
        mobility_weight: f64,
        restarts: u32,
        free_riders: &str,
    ) -> PyResult<PySolveResult> {
        // Validate: check for duplicate qubit IDs in target.
        {
            let mut seen = std::collections::HashSet::new();
            for &(qid, _, _) in &target {
                if !seen.insert(qid) {
                    return Err(PyValueError::new_err(format!(
                        "duplicate qubit_id {qid} in target placement"
                    )));
                }
            }
        }

        let initial_pairs: Vec<_> = initial
            .into_iter()
            .map(|(qid, word_id, site_id)| (qid, LocationAddr { word_id, site_id }))
            .collect();

        let target_pairs: Vec<_> = target
            .into_iter()
            .map(|(qid, word_id, site_id)| (qid, LocationAddr { word_id, site_id }))
            .collect();

        let blocked_locs: Vec<_> = blocked
            .into_iter()
            .map(|(word_id, site_id)| LocationAddr { word_id, site_id })
            .collect();

        let strat = match strategy {
            "astar" => Strategy::AStar,
            "dfs" => Strategy::HeuristicDfs,
            "bfs" => Strategy::Bfs,
            "greedy" => Strategy::GreedyBestFirst,
            "ids" => Strategy::Ids,
            "cascade" => Strategy::Cascade,
            "entropy" => Strategy::Entropy,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown strategy '{strategy}', expected: astar, dfs, bfs, greedy, ids, cascade, entropy"
                )));
            }
        };

        let fr_policy = match free_riders {
            "off" => FreeRiderPolicy::Off,
            "unblock" => FreeRiderPolicy::Unblock,
            "unblock_or_improve" => FreeRiderPolicy::UnblockOrImprove,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown free_riders '{free_riders}', expected: off, unblock, unblock_or_improve"
                )));
            }
        };

        // Release the GIL during search (pure Rust, no Python objects needed).
        let result = py
            .allow_threads(|| {
                self.inner.solve(
                    initial_pairs,
                    target_pairs,
                    blocked_locs,
                    max_expansions,
                    strat,
                    top_c,
                    max_movesets_per_group,
                    weight,
                    mobility_weight,
                    restarts,
                    fr_policy,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "MoveSolver(...)".to_string()
    }
}
