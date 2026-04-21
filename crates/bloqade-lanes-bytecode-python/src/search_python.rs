//! PyO3 bindings for the move synthesis solver.
//!
//! Exposes [`PyMoveSolver`] and [`PySolveResult`] to Python. Uses a JSON
//! bridge for `ArchSpec` — the solver is fully decoupled from the bytecode
//! Python bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_search::DeadlockPolicy;
use bloqade_lanes_search::solve::{
    InnerStrategy, MoveSolver, MultiSolveResult, SolveOptions, SolveResult, SolveStatus, Strategy,
};
use bloqade_lanes_search::target_generator::{DefaultTargetGenerator, TargetGenerator};

use crate::arch_python::PyArchSpec;

// ── Enum wrappers ──

/// Search strategy for the move solver.
#[pyclass(
    name = "SearchStrategy",
    eq,
    eq_int,
    hash,
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PySearchStrategy {
    #[pyo3(name = "ASTAR")]
    AStar = 0,
    #[pyo3(name = "DFS")]
    HeuristicDfs = 1,
    #[pyo3(name = "BFS")]
    Bfs = 2,
    #[pyo3(name = "GREEDY")]
    GreedyBestFirst = 3,
    #[pyo3(name = "IDS")]
    Ids = 4,
    #[pyo3(name = "CASCADE_IDS")]
    CascadeIds = 5,
    #[pyo3(name = "CASCADE_DFS")]
    CascadeDfs = 6,
    #[pyo3(name = "CASCADE_ENTROPY")]
    CascadeEntropy = 7,
    #[pyo3(name = "ENTROPY")]
    Entropy = 8,
}

#[pymethods]
impl PySearchStrategy {
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::AStar => "ASTAR",
            Self::HeuristicDfs => "DFS",
            Self::Bfs => "BFS",
            Self::GreedyBestFirst => "GREEDY",
            Self::Ids => "IDS",
            Self::CascadeIds => "CASCADE_IDS",
            Self::CascadeDfs => "CASCADE_DFS",
            Self::CascadeEntropy => "CASCADE_ENTROPY",
            Self::Entropy => "ENTROPY",
        }
    }
}

impl PySearchStrategy {
    fn to_rs(self) -> Strategy {
        match self {
            Self::AStar => Strategy::AStar,
            Self::HeuristicDfs => Strategy::HeuristicDfs,
            Self::Bfs => Strategy::Bfs,
            Self::GreedyBestFirst => Strategy::GreedyBestFirst,
            Self::Ids => Strategy::Ids,
            Self::CascadeIds => Strategy::Cascade {
                inner: InnerStrategy::Ids,
            },
            Self::CascadeDfs => Strategy::Cascade {
                inner: InnerStrategy::Dfs,
            },
            Self::CascadeEntropy => Strategy::Cascade {
                inner: InnerStrategy::Entropy,
            },
            Self::Entropy => Strategy::Entropy,
        }
    }
}

/// Deadlock handling policy for the move solver.
#[pyclass(
    name = "DeadlockPolicy",
    eq,
    eq_int,
    hash,
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyDeadlockPolicy {
    #[pyo3(name = "SKIP")]
    Skip = 0,
    #[pyo3(name = "MOVE_BLOCKERS")]
    MoveBlockers = 1,
}

#[pymethods]
impl PyDeadlockPolicy {
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::Skip => "SKIP",
            Self::MoveBlockers => "MOVE_BLOCKERS",
        }
    }
}

impl PyDeadlockPolicy {
    fn to_rs(self) -> DeadlockPolicy {
        match self {
            Self::Skip => DeadlockPolicy::Skip,
            Self::MoveBlockers => DeadlockPolicy::MoveBlockers,
        }
    }
}

// ── Solve results ──

/// Result of a move synthesis solve.
///
/// Contains the sequence of move steps, the final qubit configuration,
/// and search statistics.
#[pyclass(
    name = "SolveResult",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
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
    /// Each lane is represented as `(direction, move_type, zone_id, word_id, site_id, bus_id)`
    /// where direction is 0=Forward/1=Backward and move_type is 0=SiteBus/1=WordBus/2=ZoneBus.
    #[getter]
    #[allow(clippy::type_complexity)]
    fn move_layers(&self) -> Vec<Vec<(u8, u8, u32, u32, u32, u32)>> {
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
                            lane.zone_id,
                            lane.word_id,
                            lane.site_id,
                            lane.bus_id,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// Goal configuration: list of (qubit_id, zone_id, word_id, site_id) tuples.
    #[getter]
    fn goal_config(&self) -> Vec<(u32, u32, u32, u32)> {
        self.inner
            .goal_config
            .iter()
            .map(|(qid, loc)| (qid, loc.zone_id, loc.word_id, loc.site_id))
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
            "SolveResult(status='{}', steps={}, cost={}, expanded={}, deadlocks={})",
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
#[pyclass(name = "MoveSolver", frozen, module = "bloqade.lanes.bytecode._native")]
pub struct PyMoveSolver {
    inner: MoveSolver,
}

#[pymethods]
impl PyMoveSolver {
    /// Create a solver from an ArchSpec JSON string.
    #[new]
    fn new(arch_spec_json: &str) -> PyResult<Self> {
        let inner = MoveSolver::from_json(arch_spec_json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec JSON: {e}")))?;
        Ok(Self { inner })
    }

    /// Create a solver from a native ArchSpec object.
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
    /// Args:
    ///     initial: List of (qubit_id, zone_id, word_id, site_id) tuples for starting positions.
    ///     target: List of (qubit_id, zone_id, word_id, site_id) tuples for desired positions.
    ///     blocked: List of (zone_id, word_id, site_id) tuples for immovable obstacle locations.
    ///     max_expansions: Optional limit on node expansions.
    ///     strategy: Search strategy string.
    ///     top_c: Top bus options per qubit in the heuristic expander (default 3).
    ///     max_movesets_per_group: Max movesets per bus group (default 3).
    ///     weight: Heuristic weight for A* (1.0 = standard, >1.0 = bounded suboptimal).
    ///     restarts: Number of parallel restarts with perturbed scoring (1 = no restarts).
    ///     lookahead: Enable 2-step lookahead scoring.
    ///     deadlock_policy: Deadlock handling: "skip" or "move_blockers".
    ///     w_t: Time-distance blend weight (0.0 = hop-count only, 1.0 = time only). Affects entropy strategy.
    ///
    /// Returns:
    ///     SolveResult with status indicating success/failure.
    #[pyo3(signature = (initial, target, blocked, max_expansions=None, strategy="astar", top_c=3, max_movesets_per_group=3, weight=1.0, restarts=1, lookahead=false, deadlock_policy="skip", w_t=0.05))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        initial: Vec<(u32, u32, u32, u32)>,
        target: Vec<(u32, u32, u32, u32)>,
        blocked: Vec<(u32, u32, u32)>,
        max_expansions: Option<u32>,
        strategy: &str,
        top_c: usize,
        max_movesets_per_group: usize,
        weight: f64,
        restarts: u32,
        lookahead: bool,
        deadlock_policy: &str,
        w_t: f64,
    ) -> PyResult<PySolveResult> {
        // Validate: check for duplicate qubit IDs in target.
        {
            let mut seen = std::collections::HashSet::new();
            for &(qid, _, _, _) in &target {
                if !seen.insert(qid) {
                    return Err(PyValueError::new_err(format!(
                        "duplicate qubit_id {qid} in target placement"
                    )));
                }
            }
        }

        let initial_pairs = to_placement(&initial);
        let target_pairs = to_placement(&target);
        let blocked_locs = to_blocked_locs(&blocked);
        let opts = build_solve_options(
            strategy,
            top_c,
            max_movesets_per_group,
            weight,
            restarts,
            lookahead,
            deadlock_policy,
            w_t,
        )?;

        // Release the GIL during search (pure Rust, no Python objects needed).
        let result = py
            .allow_threads(|| {
                self.inner.solve(
                    initial_pairs,
                    target_pairs,
                    blocked_locs,
                    max_expansions,
                    &opts,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult { inner: result })
    }

    /// Solve using a target generator: generates candidates, validates each,
    /// and tries them in order with a shared expansion budget.
    ///
    /// Args:
    ///     initial: List of (qubit_id, zone_id, word_id, site_id) tuples for starting positions.
    ///     blocked: List of (zone_id, word_id, site_id) tuples for immovable obstacle locations.
    ///     controls: List of control qubit IDs for the CZ gate layer.
    ///     targets: List of target qubit IDs for the CZ gate layer.
    ///     generator: Optional Rust-side target generator. Defaults to DefaultTargetGenerator.
    ///     max_expansions: Optional limit on total node expansions across all candidates.
    ///     options: Search-tuning parameters (SolveOptions). Defaults to SolveOptions().
    ///
    /// Returns:
    ///     MultiSolveResult with per-candidate debug info.
    #[pyo3(signature = (initial, blocked, controls, targets, generator=None, max_expansions=None, options=None))]
    fn solve_with_generator(
        &self,
        py: Python<'_>,
        initial: Vec<(u32, u32, u32, u32)>,
        blocked: Vec<(u32, u32, u32)>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        generator: Option<&PyDefaultTargetGenerator>,
        max_expansions: Option<u32>,
        options: Option<&PySolveOptions>,
    ) -> PyResult<PyMultiSolveResult> {
        let initial_pairs = to_placement(&initial);
        let blocked_locs = to_blocked_locs(&blocked);
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

        // Currently only DefaultTargetGenerator is supported. Reject explicit
        // generator arguments until multiple generator types are implemented.
        if generator.is_some() {
            return Err(PyValueError::new_err(
                "custom generator parameter is not yet supported; pass None or omit",
            ));
        }
        let rust_gen: Box<dyn TargetGenerator> = Box::new(DefaultTargetGenerator);

        let result = py
            .allow_threads(|| {
                self.inner.solve_with_generator(
                    initial_pairs,
                    blocked_locs,
                    &controls,
                    &targets,
                    rust_gen.as_ref(),
                    max_expansions,
                    &opts,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyMultiSolveResult { inner: result })
    }

    /// Generate and validate candidate target configurations without solving.
    ///
    /// Returns a list of candidates, each a list of (qubit_id, zone_id, word_id, site_id) tuples.
    /// Only validated candidates are included.
    #[pyo3(signature = (initial, controls, targets, generator=None))]
    fn generate_candidates(
        &self,
        initial: Vec<(u32, u32, u32, u32)>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        generator: Option<&PyDefaultTargetGenerator>,
    ) -> PyResult<Vec<Vec<(u32, u32, u32, u32)>>> {
        if generator.is_some() {
            return Err(PyValueError::new_err(
                "custom generator parameter is not yet supported; pass None or omit",
            ));
        }
        let initial_pairs = to_placement(&initial);
        let rust_gen = DefaultTargetGenerator;

        Ok(self
            .inner
            .generate_candidates(&initial_pairs, &controls, &targets, &rust_gen)
            .into_iter()
            .map(|candidate| {
                candidate
                    .into_iter()
                    .map(|(qid, loc)| (qid, loc.zone_id, loc.word_id, loc.site_id))
                    .collect()
            })
            .collect())
    }

    fn __repr__(&self) -> String {
        "MoveSolver(...)".to_string()
    }
}

// ── Solve options ──

/// Search-tuning parameters shared across solve methods.
///
/// Bundles strategy, heuristic weight, restarts, and other tuning knobs
/// into a single reusable object.
#[pyclass(
    name = "SolveOptions",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PySolveOptions {
    inner: SolveOptions,
}

#[pymethods]
impl PySolveOptions {
    #[new]
    #[pyo3(signature = (strategy=PySearchStrategy::AStar, top_c=3, max_movesets_per_group=3, weight=1.0, restarts=1, lookahead=false, deadlock_policy=PyDeadlockPolicy::Skip, w_t=0.05))]
    fn new(
        strategy: PySearchStrategy,
        top_c: usize,
        max_movesets_per_group: usize,
        weight: f64,
        restarts: u32,
        lookahead: bool,
        deadlock_policy: PyDeadlockPolicy,
        w_t: f64,
    ) -> PyResult<Self> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(
                "weight must be a finite float greater than 0.0",
            ));
        }
        if !w_t.is_finite() || !(0.0..=1.0).contains(&w_t) {
            return Err(PyValueError::new_err(
                "w_t must be a finite float in the range [0.0, 1.0]",
            ));
        }
        Ok(Self {
            inner: SolveOptions {
                strategy: strategy.to_rs(),
                top_c,
                max_movesets_per_group,
                weight,
                restarts,
                lookahead,
                deadlock_policy: deadlock_policy.to_rs(),
                w_t,
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveOptions(top_c={}, weight={}, restarts={})",
            self.inner.top_c, self.inner.weight, self.inner.restarts,
        )
    }
}

// ── Helper functions ──

fn to_placement(tuples: &[(u32, u32, u32, u32)]) -> Vec<(u32, LocationAddr)> {
    tuples
        .iter()
        .map(|&(qid, zone_id, word_id, site_id)| {
            (
                qid,
                LocationAddr {
                    zone_id,
                    word_id,
                    site_id,
                },
            )
        })
        .collect()
}

fn to_blocked_locs(blocked: &[(u32, u32, u32)]) -> Vec<LocationAddr> {
    blocked
        .iter()
        .map(|&(zone_id, word_id, site_id)| LocationAddr {
            zone_id,
            word_id,
            site_id,
        })
        .collect()
}

/// Parse a strategy string for backwards-compatible methods.
fn parse_strategy(strategy: &str) -> PyResult<Strategy> {
    match strategy {
        "astar" => Ok(Strategy::AStar),
        "dfs" => Ok(Strategy::HeuristicDfs),
        "bfs" => Ok(Strategy::Bfs),
        "greedy" => Ok(Strategy::GreedyBestFirst),
        "ids" => Ok(Strategy::Ids),
        "cascade" | "cascade-ids" => Ok(Strategy::Cascade {
            inner: InnerStrategy::Ids,
        }),
        "cascade-dfs" => Ok(Strategy::Cascade {
            inner: InnerStrategy::Dfs,
        }),
        "cascade-entropy" => Ok(Strategy::Cascade {
            inner: InnerStrategy::Entropy,
        }),
        "entropy" => Ok(Strategy::Entropy),
        _ => Err(PyValueError::new_err(format!(
            "unknown strategy '{strategy}', expected: astar, dfs, bfs, greedy, ids, cascade, cascade-ids, cascade-dfs, cascade-entropy, entropy"
        ))),
    }
}

/// Parse a deadlock policy string for backwards-compatible methods.
fn parse_deadlock_policy(policy: &str) -> PyResult<DeadlockPolicy> {
    match policy {
        "skip" => Ok(DeadlockPolicy::Skip),
        "move_blockers" => Ok(DeadlockPolicy::MoveBlockers),
        _ => Err(PyValueError::new_err(format!(
            "unknown deadlock_policy '{policy}', expected: skip, move_blockers"
        ))),
    }
}

/// Build SolveOptions from individual string-based parameters (backwards compat).
#[allow(clippy::too_many_arguments)]
fn build_solve_options(
    strategy: &str,
    top_c: usize,
    max_movesets_per_group: usize,
    weight: f64,
    restarts: u32,
    lookahead: bool,
    deadlock_policy: &str,
    w_t: f64,
) -> PyResult<SolveOptions> {
    if !weight.is_finite() || weight <= 0.0 {
        return Err(PyValueError::new_err(
            "weight must be a finite float greater than 0.0",
        ));
    }
    if !w_t.is_finite() || !(0.0..=1.0).contains(&w_t) {
        return Err(PyValueError::new_err(
            "w_t must be a finite float in the range [0.0, 1.0]",
        ));
    }
    Ok(SolveOptions {
        strategy: parse_strategy(strategy)?,
        top_c,
        max_movesets_per_group,
        weight,
        restarts,
        lookahead,
        deadlock_policy: parse_deadlock_policy(deadlock_policy)?,
        w_t,
    })
}

// ── Target generator PyO3 types ──

/// Default target generator: moves each control qubit to the CZ blockade
/// partner of its corresponding target qubit.
#[pyclass(
    name = "DefaultTargetGenerator",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyDefaultTargetGenerator;

#[pymethods]
impl PyDefaultTargetGenerator {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> &'static str {
        "DefaultTargetGenerator()"
    }
}

/// Result of a multi-candidate solve via `MoveSolver.solve_with_generator()`.
#[pyclass(
    name = "MultiSolveResult",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyMultiSolveResult {
    inner: MultiSolveResult,
}

fn status_str(s: SolveStatus) -> &'static str {
    match s {
        SolveStatus::Solved => "solved",
        SolveStatus::Unsolvable => "unsolvable",
        SolveStatus::BudgetExceeded => "budget_exceeded",
    }
}

#[pymethods]
impl PyMultiSolveResult {
    /// Status of the winning solve: "solved", "unsolvable", or "budget_exceeded".
    #[getter]
    fn status(&self) -> &'static str {
        status_str(self.inner.result.status)
    }

    /// Index of the candidate that succeeded, or None if all failed.
    #[getter]
    fn candidate_index(&self) -> Option<usize> {
        self.inner.candidate_index
    }

    /// Total nodes expanded across all candidates.
    #[getter]
    fn total_expansions(&self) -> u32 {
        self.inner.total_expansions
    }

    /// Number of candidates actually attempted (excludes validation failures).
    #[getter]
    fn candidates_tried(&self) -> usize {
        self.inner.candidates_tried
    }

    /// Per-candidate attempt details: list of dicts with
    /// `candidate_index`, `status`, `nodes_expanded`.
    #[getter]
    fn attempts(&self) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            self.inner
                .attempts
                .iter()
                .map(|a| {
                    let dict = pyo3::types::PyDict::new(py);
                    dict.set_item("candidate_index", a.candidate_index)?;
                    dict.set_item("status", status_str(a.status))?;
                    dict.set_item("nodes_expanded", a.nodes_expanded)?;
                    Ok(dict.into_any().unbind())
                })
                .collect()
        })
    }

    /// Move layers from the winning candidate (same format as SolveResult.move_layers).
    #[getter]
    #[allow(clippy::type_complexity)]
    fn move_layers(&self) -> Vec<Vec<(u8, u8, u32, u32, u32, u32)>> {
        self.inner
            .result
            .move_layers
            .iter()
            .map(|ms| {
                ms.decode()
                    .into_iter()
                    .map(|lane| {
                        (
                            lane.direction as u8,
                            lane.move_type as u8,
                            lane.zone_id,
                            lane.word_id,
                            lane.site_id,
                            lane.bus_id,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// Goal configuration from the winning candidate.
    #[getter]
    fn goal_config(&self) -> Vec<(u32, u32, u32, u32)> {
        self.inner
            .result
            .goal_config
            .iter()
            .map(|(qid, loc)| (qid, loc.zone_id, loc.word_id, loc.site_id))
            .collect()
    }

    /// Total path cost from the winning candidate.
    #[getter]
    fn cost(&self) -> f64 {
        self.inner.result.cost
    }

    /// Number of deadlocks from the winning candidate.
    #[getter]
    fn deadlocks(&self) -> u32 {
        self.inner.result.deadlocks
    }

    fn __repr__(&self) -> String {
        format!(
            "MultiSolveResult(status='{}', candidate={:?}, tried={}, expansions={})",
            status_str(self.inner.result.status),
            self.inner.candidate_index,
            self.inner.candidates_tried,
            self.inner.total_expansions,
        )
    }
}
