//! PyO3 bindings for the move synthesis solver.
//!
//! Exposes [`PyMoveSolver`] and [`PySolveResult`] to Python. Uses a JSON
//! bridge for `ArchSpec` — the solver is fully decoupled from the bytecode
//! Python bindings.

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_search::DeadlockPolicy;
use bloqade_lanes_search::drivers::entropy::{
    EntropyParams, EntropyTrace, EntropyTraceStep, MovesetMetrics, compute_moveset_metrics,
};
use bloqade_lanes_search::placement::nohome::NoHomeOptions;
use bloqade_lanes_search::placement::receding_horizon::{
    RecedingHorizonOptions, default_weight_grid,
};
use bloqade_lanes_search::placement::target_generator::{DefaultTargetGenerator, TargetGenerator};
use bloqade_lanes_search::primitives::config::Config;
use bloqade_lanes_search::primitives::context::SearchContext;
use bloqade_lanes_search::primitives::distance::DistanceTable;
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::search::solve::{
    EntanglingOptions, EntropyOptions, InnerStrategy, MoveSolver, MultiSolveResult, SolveOptions,
    SolveResult, Strategy,
};

use crate::arch_python::{PyArchSpec, PyLaneAddr, PyLocationAddr};

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
    fn from_rs(s: &Strategy) -> Self {
        match s {
            Strategy::AStar => Self::AStar,
            Strategy::HeuristicDfs => Self::HeuristicDfs,
            Strategy::Bfs => Self::Bfs,
            Strategy::GreedyBestFirst => Self::GreedyBestFirst,
            Strategy::Ids => Self::Ids,
            Strategy::Cascade {
                inner: InnerStrategy::Ids,
            } => Self::CascadeIds,
            Strategy::Cascade {
                inner: InnerStrategy::Dfs,
            } => Self::CascadeDfs,
            Strategy::Cascade {
                inner: InnerStrategy::Entropy,
            } => Self::CascadeEntropy,
            Strategy::Entropy => Self::Entropy,
        }
    }

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
    #[pyo3(name = "ALL_MOVES")]
    AllMoves = 2,
}

#[pymethods]
impl PyDeadlockPolicy {
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::Skip => "SKIP",
            Self::MoveBlockers => "MOVE_BLOCKERS",
            Self::AllMoves => "ALL_MOVES",
        }
    }
}

impl PyDeadlockPolicy {
    fn from_rs(d: &DeadlockPolicy) -> Self {
        match d {
            DeadlockPolicy::Skip => Self::Skip,
            DeadlockPolicy::MoveBlockers => Self::MoveBlockers,
            DeadlockPolicy::AllMoves => Self::AllMoves,
        }
    }

    fn to_rs(self) -> DeadlockPolicy {
        match self {
            Self::Skip => DeadlockPolicy::Skip,
            Self::MoveBlockers => DeadlockPolicy::MoveBlockers,
            Self::AllMoves => DeadlockPolicy::AllMoves,
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
        self.inner.status.as_label()
    }

    /// Move layers: list of move steps, each a list of lane address tuples.
    ///
    /// Each lane is a ``LaneAddress`` with named attributes for direction,
    /// move_type, zone_id, word_id, site_id, bus_id.
    #[getter]
    fn move_layers(&self) -> Vec<Vec<PyLaneAddr>> {
        self.inner
            .move_layers
            .iter()
            .map(|ms| {
                ms.decode()
                    .into_iter()
                    .map(|lane| PyLaneAddr { inner: lane })
                    .collect()
            })
            .collect()
    }

    /// Goal configuration: mapping of qubit_id to LocationAddress.
    #[getter]
    fn goal_config(&self) -> std::collections::HashMap<u32, PyLocationAddr> {
        self.inner
            .goal_config
            .iter()
            .map(|(qid, loc)| (qid, PyLocationAddr { inner: loc }))
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

    /// Optional entropy trace (present when `collect_entropy_trace=True`).
    #[getter]
    fn entropy_trace(&self) -> Option<PyEntropyTrace> {
        self.inner
            .entropy_trace
            .as_ref()
            .map(|trace| PyEntropyTrace {
                inner: trace.clone(),
            })
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveResult(status='{}', steps={}, cost={}, expanded={}, deadlocks={})",
            self.inner.status.as_label(),
            self.inner.move_layers.len(),
            self.inner.cost,
            self.inner.nodes_expanded,
            self.inner.deadlocks,
        )
    }
}

// ── Entropy trace ──

/// Entropy-search trace, recorded when `SolveOptions.collect_entropy_trace=True`.
#[pyclass(
    name = "EntropyTrace",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyEntropyTrace {
    inner: EntropyTrace,
}

#[pymethods]
impl PyEntropyTrace {
    #[getter]
    fn root_node_id(&self) -> u32 {
        self.inner.root_node_id
    }

    #[getter]
    fn best_buffer_size(&self) -> u32 {
        self.inner.best_buffer_size
    }

    #[getter]
    fn steps(&self) -> Vec<PyEntropyTraceStep> {
        self.inner
            .steps
            .iter()
            .map(|s| PyEntropyTraceStep { inner: s.clone() })
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.steps.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "EntropyTrace(root_node_id={}, best_buffer_size={}, steps={})",
            self.inner.root_node_id,
            self.inner.best_buffer_size,
            self.inner.steps.len(),
        )
    }
}

/// One step in an entropy-search trace.
#[pyclass(
    name = "EntropyTraceStep",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyEntropyTraceStep {
    inner: EntropyTraceStep,
}

#[pymethods]
impl PyEntropyTraceStep {
    #[getter]
    fn event(&self) -> String {
        self.inner.event.clone()
    }

    #[getter]
    fn node_id(&self) -> u32 {
        self.inner.node_id
    }

    #[getter]
    fn parent_node_id(&self) -> Option<u32> {
        self.inner.parent_node_id
    }

    #[getter]
    fn depth(&self) -> u32 {
        self.inner.depth
    }

    #[getter]
    fn entropy(&self) -> u32 {
        self.inner.entropy
    }

    #[getter]
    fn unresolved_count(&self) -> u32 {
        self.inner.unresolved_count
    }

    #[getter]
    #[allow(clippy::type_complexity)]
    fn moveset(&self) -> Option<Vec<(u8, u8, u32, u32, u32, u32)>> {
        self.inner.moveset.clone()
    }

    #[getter]
    #[allow(clippy::type_complexity)]
    fn candidate_movesets(&self) -> Vec<Vec<(u8, u8, u32, u32, u32, u32)>> {
        self.inner.candidate_movesets.clone()
    }

    #[getter]
    fn candidate_index(&self) -> Option<u32> {
        self.inner.candidate_index
    }

    #[getter]
    fn reason(&self) -> Option<String> {
        self.inner.reason.clone()
    }

    #[getter]
    fn state_seen_node_id(&self) -> Option<u32> {
        self.inner.state_seen_node_id
    }

    #[getter]
    fn no_valid_moves_qubit(&self) -> Option<u32> {
        self.inner.no_valid_moves_qubit
    }

    #[getter]
    fn trigger_node_id(&self) -> Option<u32> {
        self.inner.trigger_node_id
    }

    #[getter]
    fn configuration(&self) -> Vec<(u32, u32, u32, u32)> {
        self.inner.configuration.clone()
    }

    #[getter]
    fn parent_configuration(&self) -> Option<Vec<(u32, u32, u32, u32)>> {
        self.inner.parent_configuration.clone()
    }

    #[getter]
    fn moveset_score(&self) -> Option<f64> {
        self.inner.moveset_score
    }

    #[getter]
    fn best_buffer_node_ids(&self) -> Vec<u32> {
        self.inner.best_buffer_node_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EntropyTraceStep(event='{}', node_id={}, depth={}, entropy={})",
            self.inner.event, self.inner.node_id, self.inner.depth, self.inner.entropy,
        )
    }
}

// ── Entropy scorer (metrics + score for a single moveset) ──

/// Per-moveset scoring breakdown returned by [`PyEntropyScorer::metrics`].
///
/// Exposes `alpha * distance_progress + beta * arrived + gamma * mobility_gain`
/// and the per-component contributions, plus the qubit ids whose distance to
/// target strictly improved / degraded.
#[pyclass(
    name = "MovesetMetrics",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyMovesetMetrics {
    inner: MovesetMetrics,
    alpha: f64,
    beta: f64,
    gamma: f64,
}

#[pymethods]
impl PyMovesetMetrics {
    #[getter]
    fn distance_progress(&self) -> f64 {
        self.inner.distance_progress
    }

    #[getter]
    fn arrived(&self) -> u32 {
        self.inner.arrived
    }

    #[getter]
    fn mobility_before(&self) -> f64 {
        self.inner.mobility_before
    }

    #[getter]
    fn mobility_after(&self) -> f64 {
        self.inner.mobility_after
    }

    #[getter]
    fn mobility_gain(&self) -> f64 {
        self.inner.mobility_gain()
    }

    #[getter]
    fn closer(&self) -> Vec<u32> {
        self.inner.closer.clone()
    }

    #[getter]
    fn further(&self) -> Vec<u32> {
        self.inner.further.clone()
    }

    /// `alpha * distance_progress + beta * arrived + gamma * mobility_gain`.
    #[getter]
    fn score(&self) -> f64 {
        self.alpha * self.inner.distance_progress
            + self.beta * (self.inner.arrived as f64)
            + self.gamma * self.inner.mobility_gain()
    }

    fn __repr__(&self) -> String {
        format!(
            "MovesetMetrics(score={:.4}, distance_progress={:.3}, arrived={}, mobility_gain={:.3})",
            self.score(),
            self.inner.distance_progress,
            self.inner.arrived,
            self.inner.mobility_gain(),
        )
    }
}

/// Scorer that evaluates candidate movesets for entropy-guided search.
///
/// Build one per (architecture, target, blocked) context to amortize the
/// distance-table precomputation; then call `metrics(current_config, moveset)`
/// or `score_moveset(current_config, moveset)` per candidate.
///
/// This is the Rust-native replacement for the deleted Python
/// `bloqade.lanes.search.scoring.CandidateScorer.score_moveset()` — same
/// `alpha * D + beta * A + gamma * M` formula, same `w_t` blended distance.
#[pyclass(
    name = "EntropyScorer",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyEntropyScorer {
    index: LaneIndex,
    dist_table: DistanceTable,
    blocked: HashSet<u64>,
    targets: Vec<(u32, u64)>,
    params: EntropyParams,
}

impl PyEntropyScorer {
    fn apply_moveset(
        &self,
        config: &Config,
        moveset: &[PyRef<'_, PyLaneAddr>],
    ) -> PyResult<Config> {
        let mut moves: Vec<(u32, LocationAddr)> = Vec::with_capacity(moveset.len());
        for lane_ref in moveset {
            let lane = lane_ref.inner;
            let Some((src, dst)) = self.index.endpoints(&lane) else {
                return Err(PyValueError::new_err(
                    "lane endpoints missing from arch index",
                ));
            };
            let Some(qid) = config.qubit_at(src) else {
                continue;
            };
            moves.push((qid, dst));
        }
        Ok(config.with_moves(&moves))
    }
}

#[pymethods]
impl PyEntropyScorer {
    /// Build a scorer bound to an architecture, target mapping, and params.
    ///
    /// ``alpha``, ``beta``, ``gamma`` weight the distance / arrival / mobility
    /// terms; ``w_t`` blends hop-count (0.0) with move-time distance (1.0).
    #[new]
    #[pyo3(signature = (
        arch_spec,
        target,
        blocked = None,
        alpha = 80.0,
        beta = 3.0,
        gamma = 3.1,
        w_t = 0.05,
    ))]
    fn new(
        arch_spec: &PyArchSpec,
        target: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        blocked: Option<Vec<PyRef<'_, PyLocationAddr>>>,
        alpha: f64,
        beta: f64,
        gamma: f64,
        w_t: f64,
    ) -> PyResult<Self> {
        let index = LaneIndex::from_arch_spec(&arch_spec.inner);

        let targets: Vec<(u32, u64)> = target
            .iter()
            .map(|(q, loc)| (*q, loc.inner.encode()))
            .collect();
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l).collect();
        let dist_table = DistanceTable::new(&target_locs, &index).with_time_distances(&index);

        let blocked_set: HashSet<u64> = blocked
            .unwrap_or_default()
            .iter()
            .map(|p| p.inner.encode())
            .collect();

        let params = EntropyParams {
            alpha,
            beta,
            gamma,
            w_t,
            ..EntropyParams::default()
        };

        Ok(Self {
            index,
            dist_table,
            blocked: blocked_set,
            targets,
            params,
        })
    }

    /// Compute the full metrics breakdown after applying ``moveset``.
    fn metrics(
        &self,
        current_config: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        moveset: Vec<PyRef<'_, PyLaneAddr>>,
    ) -> PyResult<PyMovesetMetrics> {
        let pairs: Vec<(u32, LocationAddr)> = current_config
            .iter()
            .map(|(q, loc)| (*q, loc.inner))
            .collect();
        let old_config = Config::new(pairs)
            .map_err(|e| PyValueError::new_err(format!("invalid current_config: {e}")))?;
        let new_config = self.apply_moveset(&old_config, &moveset)?;

        let mut occupied: HashSet<u64> =
            HashSet::with_capacity(self.blocked.len() + old_config.len());
        occupied.extend(&self.blocked);
        for (_, loc) in old_config.iter() {
            occupied.insert(loc.encode());
        }

        let ctx = SearchContext {
            index: &self.index,
            dist_table: &self.dist_table,
            blocked: &self.blocked,
            targets: &self.targets,
            cz_pairs: None,
        };
        let inner =
            compute_moveset_metrics(&old_config, &new_config, &occupied, &ctx, &self.params);
        Ok(PyMovesetMetrics {
            inner,
            alpha: self.params.alpha,
            beta: self.params.beta,
            gamma: self.params.gamma,
        })
    }

    /// Shorthand for `scorer.metrics(current, moveset).score`.
    fn score_moveset(
        &self,
        current_config: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        moveset: Vec<PyRef<'_, PyLaneAddr>>,
    ) -> PyResult<f64> {
        Ok(self.metrics(current_config, moveset)?.score())
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.params.alpha
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.params.beta
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.params.gamma
    }

    #[getter]
    fn w_t(&self) -> f64 {
        self.params.w_t
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
        Ok(Self {
            inner: MoveSolver::from_arch_spec(&arch.inner),
        })
    }

    /// Solve a move synthesis problem.
    ///
    /// Args:
    ///     initial: Mapping of qubit_id to LocationAddress for starting positions.
    ///     target: Mapping of qubit_id to LocationAddress for desired positions.
    ///     blocked: List of LocationAddress for immovable obstacle locations.
    ///     max_expansions: Optional limit on node expansions.
    ///     options: Search-tuning parameters (SolveOptions). Defaults to SolveOptions().
    ///     entropy_options: Entropy-strategy parameters (EntropyOptions).
    ///         Only consumed when the strategy is entropy.
    ///
    /// Returns:
    ///     SolveResult with status indicating success/failure.
    #[pyo3(signature = (initial, target, blocked, max_expansions=None, options=None, entropy_options=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        target: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        options: Option<&PySolveOptions>,
        entropy_options: Option<&PyEntropyOptions>,
    ) -> PyResult<PySolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let target_pairs: Vec<(u32, LocationAddr)> =
            target.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let entropy_opts = entropy_options.map(|o| o.inner.clone());

        // Release the GIL during search (pure Rust, no Python objects needed).
        let result = py
            .allow_threads(|| {
                self.inner.solve(
                    initial_pairs,
                    target_pairs,
                    blocked_locs,
                    max_expansions,
                    &opts,
                    entropy_opts.as_ref(),
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult { inner: result })
    }

    /// Solve using a target generator: generates candidates, validates each,
    /// and tries them in order with a shared expansion budget.
    ///
    /// Args:
    ///     initial: Mapping of qubit_id to LocationAddress for starting positions.
    ///     blocked: List of LocationAddress for immovable obstacle locations.
    ///     controls: List of control qubit IDs for the CZ gate layer.
    ///     targets: List of target qubit IDs for the CZ gate layer.
    ///     generator: Optional Rust-side target generator (currently must be None).
    ///     max_expansions: Optional limit on total node expansions across all candidates.
    ///     options: Search-tuning parameters (SolveOptions). Defaults to SolveOptions().
    ///
    /// Returns:
    ///     MultiSolveResult with per-candidate debug info.
    #[pyo3(signature = (initial, blocked, controls, targets, generator=None, max_expansions=None, options=None, entropy_options=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve_with_generator(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        generator: Option<&PyDefaultTargetGenerator>,
        max_expansions: Option<u32>,
        options: Option<&PySolveOptions>,
        entropy_options: Option<&PyEntropyOptions>,
    ) -> PyResult<PyMultiSolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let entropy_opts = entropy_options.map(|o| o.inner.clone());

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
                    entropy_opts.as_ref(),
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PyMultiSolveResult { inner: result })
    }

    /// Generate and validate candidate target configurations without solving.
    ///
    /// Returns a list of candidates, each a dict mapping qubit_id to LocationAddress.
    /// Only validated candidates are included.
    #[pyo3(signature = (initial, controls, targets, generator=None))]
    fn generate_candidates(
        &self,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        generator: Option<&PyDefaultTargetGenerator>,
    ) -> PyResult<Vec<std::collections::HashMap<u32, PyLocationAddr>>> {
        if generator.is_some() {
            return Err(PyValueError::new_err(
                "custom generator parameter is not yet supported; pass None or omit",
            ));
        }
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let rust_gen = DefaultTargetGenerator;

        Ok(self
            .inner
            .generate_candidates(&initial_pairs, &controls, &targets, &rust_gen)
            .into_iter()
            .map(|candidate| {
                candidate
                    .into_iter()
                    .map(|(qid, loc)| (qid, PyLocationAddr { inner: loc }))
                    .collect()
            })
            .collect())
    }

    /// Solve a loose-goal entangling placement + routing problem.
    ///
    /// Instead of fixed target locations, the solver receives CZ pair
    /// constraints and simultaneously discovers both the entangling
    /// placement and the routing.
    ///
    /// Args:
    ///     initial: Mapping of qubit_id to LocationAddress for starting positions.
    ///     cz_pairs: List of (qubit_a, qubit_b) tuples that must end up at
    ///         entangling positions.
    ///     blocked: List of LocationAddress for immovable obstacle locations.
    ///     max_expansions: Optional limit on node expansions.
    ///     options: Search-tuning parameters (SolveOptions). Defaults to SolveOptions().
    ///
    /// Returns:
    ///     SolveResult with the discovered entangling placement.
    #[pyo3(signature = (initial, cz_pairs, blocked, max_expansions=None, options=None, entangling_options=None, future_cz_layers=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve_entangling(
        &self,
        py: Python<'_>,
        initial: std::collections::HashMap<u32, PyRef<'_, PyLocationAddr>>,
        cz_pairs: Vec<(u32, u32)>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        options: Option<&PySolveOptions>,
        entangling_options: Option<&PyEntanglingOptions>,
        future_cz_layers: Option<Vec<Vec<(u32, u32)>>>,
    ) -> PyResult<PySolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let ent_opts = entangling_options
            .map(|o| o.inner.clone())
            .unwrap_or_default();
        let future = future_cz_layers.unwrap_or_default();

        let result = py
            .allow_threads(|| {
                self.inner.solve_entangling(
                    initial_pairs,
                    &cz_pairs,
                    blocked_locs,
                    max_expansions,
                    &opts,
                    &ent_opts,
                    &future,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult { inner: result })
    }

    /// Two-phase no-home placement: return assignment + entangling routing.
    ///
    /// Phase 1 assigns displaced qubits to optimal home sites.
    /// Phase 2 routes from home to CZ-staging using solve_entangling.
    ///
    /// Args:
    ///     initial: Mapping of qubit_id to current location.
    ///     cz_pairs: List of (control, target) qubit pairs for the CZ layer.
    ///     blocked: List of immovable obstacle locations.
    ///     max_expansions: Optional node expansion budget.
    ///     options: Search-tuning parameters for routing phases.
    ///     nohome_options: Tuning parameters for the return assignment.
    ///     future_cz_layers: Future CZ layers for lookahead.
    ///
    /// Returns:
    ///     SolveResult with the combined return + entangling placement.
    #[pyo3(signature = (initial, cz_pairs, blocked, max_expansions=None, options=None, nohome_options=None, future_cz_layers=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve_nohome(
        &self,
        py: Python<'_>,
        initial: std::collections::HashMap<u32, PyRef<'_, PyLocationAddr>>,
        cz_pairs: Vec<(u32, u32)>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        options: Option<&PySolveOptions>,
        nohome_options: Option<&PyNoHomeOptions>,
        future_cz_layers: Option<Vec<Vec<(u32, u32)>>>,
    ) -> PyResult<PySolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let nh_opts = nohome_options.map(|o| o.inner.clone()).unwrap_or_default();
        let future = future_cz_layers.unwrap_or_default();

        let result = py
            .allow_threads(|| {
                self.inner.solve_nohome(
                    initial_pairs,
                    &cz_pairs,
                    blocked_locs,
                    max_expansions,
                    &opts,
                    &nh_opts,
                    &future,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult { inner: result })
    }

    /// Receding-horizon (MPC-style) loose-goal entangling solve.
    ///
    /// Like `solve_entangling`, but instead of committing to one Hungarian
    /// assignment up front, the solver generates K diverse candidate
    /// assignments at each stage, rolls each out for `rollout_horizon` move
    /// layers, commits the winning branch's path, and re-plans. Targeted at
    /// high-occupancy regimes where the baseline loose-goal under-uses
    /// parallelism.
    ///
    /// Args:
    ///     initial: Mapping of qubit_id to LocationAddress for starting positions.
    ///     cz_pairs: List of (qubit_a, qubit_b) tuples that must end up at
    ///         entangling positions.
    ///     blocked: List of LocationAddress for immovable obstacle locations.
    ///     max_expansions: Optional limit on total node expansions across all stages.
    ///     options: Search-tuning parameters (SolveOptions).
    ///     entangling_options: Hungarian cost parameters (EntanglingOptions).
    ///     rh_options: Receding-horizon orchestration parameters
    ///         (RecedingHorizonOptions).
    ///     future_cz_layers: Future CZ layers for lookahead (clipped by
    ///         `entangling_options.hungarian_horizon`).
    ///
    /// Returns:
    ///     SolveResult with the committed move-layer trajectory and the
    ///     final entangling-feasible configuration.
    #[pyo3(signature = (initial, cz_pairs, blocked, max_expansions=None, options=None, entangling_options=None, rh_options=None, future_cz_layers=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve_entangling_rh(
        &self,
        py: Python<'_>,
        initial: std::collections::HashMap<u32, PyRef<'_, PyLocationAddr>>,
        cz_pairs: Vec<(u32, u32)>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        options: Option<&PySolveOptions>,
        entangling_options: Option<&PyEntanglingOptions>,
        rh_options: Option<&PyRecedingHorizonOptions>,
        future_cz_layers: Option<Vec<Vec<(u32, u32)>>>,
    ) -> PyResult<PySolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let ent_opts = entangling_options
            .map(|o| o.inner.clone())
            .unwrap_or_default();
        let rh_opts = rh_options.map(|o| o.inner.clone()).unwrap_or_default();
        let future = future_cz_layers.unwrap_or_default();

        let result = py
            .allow_threads(|| {
                self.inner.solve_entangling_rh(
                    initial_pairs,
                    &cz_pairs,
                    blocked_locs,
                    max_expansions,
                    &opts,
                    &ent_opts,
                    &rh_opts,
                    &future,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "MoveSolver(...)".to_string()
    }
}

// ── No-home options ──

/// Tuning parameters for the no-home return assignment.
///
/// Controls how displaced qubits are assigned to available home sites
/// between CZ layers.
#[pyclass(
    name = "NoHomeOptions",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyNoHomeOptions {
    inner: NoHomeOptions,
}

#[pymethods]
impl PyNoHomeOptions {
    #[new]
    #[pyo3(signature = (gamma=0.85, lambda_lookahead=0.5, k_candidates=8, top_bus_signatures=6, bus_reward_rho=1))]
    fn new(
        gamma: f64,
        lambda_lookahead: f64,
        k_candidates: usize,
        top_bus_signatures: usize,
        bus_reward_rho: u32,
    ) -> PyResult<Self> {
        if !gamma.is_finite() || !(0.0..=1.0).contains(&gamma) {
            return Err(PyValueError::new_err(
                "gamma must be a finite float in [0.0, 1.0]",
            ));
        }
        if !lambda_lookahead.is_finite() || lambda_lookahead < 0.0 {
            return Err(PyValueError::new_err(
                "lambda_lookahead must be a non-negative finite float",
            ));
        }
        if k_candidates == 0 {
            return Err(PyValueError::new_err("k_candidates must be >= 1"));
        }
        Ok(Self {
            inner: NoHomeOptions {
                gamma,
                lambda_lookahead,
                k_candidates,
                top_bus_signatures,
                bus_reward_rho,
            },
        })
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    fn lambda_lookahead(&self) -> f64 {
        self.inner.lambda_lookahead
    }

    #[getter]
    fn k_candidates(&self) -> usize {
        self.inner.k_candidates
    }

    #[getter]
    fn top_bus_signatures(&self) -> usize {
        self.inner.top_bus_signatures
    }

    #[getter]
    fn bus_reward_rho(&self) -> u32 {
        self.inner.bus_reward_rho
    }

    fn __repr__(&self) -> String {
        format!(
            "NoHomeOptions(gamma={}, lambda_lookahead={}, k_candidates={}, top_bus_signatures={}, bus_reward_rho={})",
            self.inner.gamma,
            self.inner.lambda_lookahead,
            self.inner.k_candidates,
            self.inner.top_bus_signatures,
            self.inner.bus_reward_rho,
        )
    }
}

// ── Solve options ──

/// Core search-tuning parameters shared by every `MoveSolver` entry point.
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
    #[pyo3(signature = (strategy=PySearchStrategy::AStar, weight=1.0, restarts=1, deadlock_policy=PyDeadlockPolicy::Skip, lookahead=false, top_c=None))]
    fn new(
        strategy: PySearchStrategy,
        weight: f64,
        restarts: u32,
        deadlock_policy: PyDeadlockPolicy,
        lookahead: bool,
        top_c: Option<usize>,
    ) -> PyResult<Self> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(
                "weight must be a finite float greater than 0.0",
            ));
        }
        if matches!(top_c, Some(0)) {
            return Err(PyValueError::new_err(
                "top_c must be None or an integer >= 1",
            ));
        }
        Ok(Self {
            inner: SolveOptions {
                strategy: strategy.to_rs(),
                weight,
                restarts,
                deadlock_policy: deadlock_policy.to_rs(),
                lookahead,
                top_c,
            },
        })
    }

    #[getter]
    fn strategy(&self) -> PySearchStrategy {
        PySearchStrategy::from_rs(&self.inner.strategy)
    }

    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    #[getter]
    fn restarts(&self) -> u32 {
        self.inner.restarts
    }

    #[getter]
    fn deadlock_policy(&self) -> PyDeadlockPolicy {
        PyDeadlockPolicy::from_rs(&self.inner.deadlock_policy)
    }

    #[getter]
    fn lookahead(&self) -> bool {
        self.inner.lookahead
    }

    #[getter]
    fn top_c(&self) -> Option<usize> {
        self.inner.top_c
    }

    fn __repr__(&self) -> String {
        format!(
            "SolveOptions(strategy={}, weight={}, restarts={}, deadlock_policy={}, lookahead={}, top_c={:?})",
            self.strategy().name(),
            self.inner.weight,
            self.inner.restarts,
            self.deadlock_policy().name(),
            self.inner.lookahead,
            self.inner.top_c,
        )
    }
}

// ── Entropy options ──

/// Entropy-strategy-specific parameters.
///
/// Only consumed when the chosen strategy is entropy (or a Cascade variant
/// whose inner is entropy). Pass via the optional `entropy_opts` argument
/// to `MoveSolver.solve`.
#[pyclass(
    name = "EntropyOptions",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyEntropyOptions {
    inner: EntropyOptions,
}

#[pymethods]
impl PyEntropyOptions {
    #[new]
    #[pyo3(signature = (max_movesets_per_group=3, max_goal_candidates=3, w_t=0.05, collect_entropy_trace=false))]
    fn new(
        max_movesets_per_group: usize,
        max_goal_candidates: usize,
        w_t: f64,
        collect_entropy_trace: bool,
    ) -> PyResult<Self> {
        if max_movesets_per_group == 0 {
            return Err(PyValueError::new_err(
                "max_movesets_per_group must be an integer >= 1",
            ));
        }
        if max_goal_candidates == 0 {
            return Err(PyValueError::new_err(
                "max_goal_candidates must be an integer >= 1",
            ));
        }
        if !w_t.is_finite() || !(0.0..=1.0).contains(&w_t) {
            return Err(PyValueError::new_err(
                "w_t must be a finite float in the range [0.0, 1.0]",
            ));
        }
        Ok(Self {
            inner: EntropyOptions {
                max_movesets_per_group,
                max_goal_candidates,
                w_t,
                collect_entropy_trace,
            },
        })
    }

    #[getter]
    fn max_movesets_per_group(&self) -> usize {
        self.inner.max_movesets_per_group
    }

    #[getter]
    fn max_goal_candidates(&self) -> usize {
        self.inner.max_goal_candidates
    }

    #[getter]
    fn w_t(&self) -> f64 {
        self.inner.w_t
    }

    #[getter]
    fn collect_entropy_trace(&self) -> bool {
        self.inner.collect_entropy_trace
    }

    fn __repr__(&self) -> String {
        format!(
            "EntropyOptions(max_movesets_per_group={}, max_goal_candidates={}, w_t={}, collect_entropy_trace={})",
            self.inner.max_movesets_per_group,
            self.inner.max_goal_candidates,
            self.inner.w_t,
            self.inner.collect_entropy_trace,
        )
    }
}

// ── Entangling options ──

/// Loose-goal entangling-search parameters consumed by
/// `MoveSolver.solve_entangling`.
#[pyclass(
    name = "EntanglingOptions",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyEntanglingOptions {
    inner: EntanglingOptions,
}

#[pymethods]
impl PyEntanglingOptions {
    #[new]
    #[pyo3(signature = (congestion_weight=0.0, occupancy_penalty=1.0, hungarian_horizon=Some(4)))]
    fn new(
        congestion_weight: f64,
        occupancy_penalty: f64,
        hungarian_horizon: Option<usize>,
    ) -> PyResult<Self> {
        if !congestion_weight.is_finite() || congestion_weight < 0.0 {
            return Err(PyValueError::new_err(
                "congestion_weight must be a finite non-negative float",
            ));
        }
        if !occupancy_penalty.is_finite() || occupancy_penalty < 0.0 {
            return Err(PyValueError::new_err(
                "occupancy_penalty must be a finite non-negative float",
            ));
        }
        Ok(Self {
            inner: EntanglingOptions {
                congestion_weight,
                occupancy_penalty,
                hungarian_horizon,
            },
        })
    }

    #[getter]
    fn congestion_weight(&self) -> f64 {
        self.inner.congestion_weight
    }

    #[getter]
    fn occupancy_penalty(&self) -> f64 {
        self.inner.occupancy_penalty
    }

    #[getter]
    fn hungarian_horizon(&self) -> Option<usize> {
        self.inner.hungarian_horizon
    }

    fn __repr__(&self) -> String {
        format!(
            "EntanglingOptions(congestion_weight={}, occupancy_penalty={}, hungarian_horizon={:?})",
            self.inner.congestion_weight,
            self.inner.occupancy_penalty,
            self.inner.hungarian_horizon,
        )
    }
}

// ── Receding-horizon options ──

/// Orchestration parameters for `MoveSolver.solve_entangling_rh`.
///
/// Controls how many candidate Hungarian assignments are tried per stage,
/// how far each rollout searches forward, how many layers of the winning
/// branch get committed before re-planning, and other tuning knobs.
#[pyclass(
    name = "RecedingHorizonOptions",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyRecedingHorizonOptions {
    inner: RecedingHorizonOptions,
}

#[pymethods]
impl PyRecedingHorizonOptions {
    #[new]
    #[pyo3(signature = (
        k_candidates = 5,
        rollout_horizon = 5,
        commit_depth = 3,
        tier0_next_h_weight = 0.5,
        weight_grid = None,
        fallback_x_decrement = 1,
        branch_parallel = true,
        max_expansions_per_rollout = 300,
        greedy_first = true,
        inner_beam_width = 2,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        k_candidates: usize,
        rollout_horizon: u32,
        commit_depth: u32,
        tier0_next_h_weight: f64,
        weight_grid: Option<Vec<(f64, f64)>>,
        fallback_x_decrement: u32,
        branch_parallel: bool,
        max_expansions_per_rollout: u32,
        greedy_first: bool,
        inner_beam_width: u32,
    ) -> PyResult<Self> {
        if k_candidates == 0 {
            return Err(PyValueError::new_err("k_candidates must be positive"));
        }
        if rollout_horizon == 0 {
            return Err(PyValueError::new_err("rollout_horizon must be positive"));
        }
        if commit_depth == 0 || commit_depth > rollout_horizon {
            return Err(PyValueError::new_err(
                "commit_depth must satisfy 1 <= commit_depth <= rollout_horizon",
            ));
        }
        if !tier0_next_h_weight.is_finite() || tier0_next_h_weight < 0.0 {
            return Err(PyValueError::new_err(
                "tier0_next_h_weight must be a finite non-negative float",
            ));
        }
        let grid = weight_grid.unwrap_or_else(default_weight_grid);
        if grid.is_empty() {
            return Err(PyValueError::new_err("weight_grid must not be empty"));
        }
        for &(cw, op) in &grid {
            if !cw.is_finite() || cw < 0.0 || !op.is_finite() || op < 0.0 {
                return Err(PyValueError::new_err(
                    "weight_grid entries must be finite non-negative pairs",
                ));
            }
        }
        Ok(Self {
            inner: RecedingHorizonOptions {
                k_candidates,
                rollout_horizon,
                commit_depth,
                tier0_next_h_weight,
                weight_grid: grid,
                fallback_x_decrement: fallback_x_decrement.max(1),
                branch_parallel,
                max_expansions_per_rollout: max_expansions_per_rollout.max(1),
                greedy_first,
                inner_beam_width: inner_beam_width.max(1),
            },
        })
    }

    #[getter]
    fn k_candidates(&self) -> usize {
        self.inner.k_candidates
    }
    #[getter]
    fn rollout_horizon(&self) -> u32 {
        self.inner.rollout_horizon
    }
    #[getter]
    fn commit_depth(&self) -> u32 {
        self.inner.commit_depth
    }
    #[getter]
    fn tier0_next_h_weight(&self) -> f64 {
        self.inner.tier0_next_h_weight
    }
    #[getter]
    fn weight_grid(&self) -> Vec<(f64, f64)> {
        self.inner.weight_grid.clone()
    }
    #[getter]
    fn fallback_x_decrement(&self) -> u32 {
        self.inner.fallback_x_decrement
    }
    #[getter]
    fn branch_parallel(&self) -> bool {
        self.inner.branch_parallel
    }
    #[getter]
    fn max_expansions_per_rollout(&self) -> u32 {
        self.inner.max_expansions_per_rollout
    }
    #[getter]
    fn greedy_first(&self) -> bool {
        self.inner.greedy_first
    }
    #[getter]
    fn inner_beam_width(&self) -> u32 {
        self.inner.inner_beam_width
    }

    fn __repr__(&self) -> String {
        format!(
            "RecedingHorizonOptions(k_candidates={}, rollout_horizon={}, commit_depth={}, tier0_next_h_weight={}, fallback_x_decrement={}, branch_parallel={}, max_expansions_per_rollout={}, greedy_first={}, inner_beam_width={}, weight_grid_len={})",
            self.inner.k_candidates,
            self.inner.rollout_horizon,
            self.inner.commit_depth,
            self.inner.tier0_next_h_weight,
            self.inner.fallback_x_decrement,
            self.inner.branch_parallel,
            self.inner.max_expansions_per_rollout,
            self.inner.greedy_first,
            self.inner.inner_beam_width,
            self.inner.weight_grid.len(),
        )
    }
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

#[pymethods]
impl PyMultiSolveResult {
    /// Status of the winning solve: "solved", "unsolvable", or "budget_exceeded".
    #[getter]
    fn status(&self) -> &'static str {
        self.inner.result.status.as_label()
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
                    dict.set_item("status", a.status.as_label())?;
                    dict.set_item("nodes_expanded", a.nodes_expanded)?;
                    Ok(dict.into_any().unbind())
                })
                .collect()
        })
    }

    /// Move layers from the winning candidate (same format as SolveResult.move_layers).
    #[getter]
    fn move_layers(&self) -> Vec<Vec<PyLaneAddr>> {
        self.inner
            .result
            .move_layers
            .iter()
            .map(|ms| {
                ms.decode()
                    .into_iter()
                    .map(|lane| PyLaneAddr { inner: lane })
                    .collect()
            })
            .collect()
    }

    /// Goal configuration from the winning candidate.
    #[getter]
    fn goal_config(&self) -> std::collections::HashMap<u32, PyLocationAddr> {
        self.inner
            .result
            .goal_config
            .iter()
            .map(|(qid, loc)| (qid, PyLocationAddr { inner: loc }))
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
            self.inner.result.status.as_label(),
            self.inner.candidate_index,
            self.inner.candidates_tried,
            self.inner.total_expansions,
        )
    }
}
