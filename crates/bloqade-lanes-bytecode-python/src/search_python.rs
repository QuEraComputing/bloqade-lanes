//! PyO3 bindings for the move synthesis solver.
//!
//! Exposes [`PyMoveSolver`] and [`PySolveResult`] to Python. Uses a JSON
//! bridge for `ArchSpec` — the solver is fully decoupled from the bytecode
//! Python bindings.

use std::collections::HashSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use bloqade_lanes_search::DeadlockPolicy;
use bloqade_lanes_search::config::Config;
use bloqade_lanes_search::context::SearchContext;
use bloqade_lanes_search::entropy::{
    EntropyParams, EntropyTrace, EntropyTraceStep, MovesetMetrics, compute_moveset_metrics,
};
use bloqade_lanes_search::heuristic::DistanceTable;
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::solve::{
    InnerStrategy, MoveSolver, MultiSolveResult, SolveOptions, SolveResult, SolveStatus, Strategy,
};
use bloqade_lanes_search::target_generator::{DefaultTargetGenerator, TargetGenerator};

use crate::arch_python::{PyArchSpec, PyLocationAddr};

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
///
/// When produced by the Move Policy DSL path (`policy_path` kwarg on
/// `MoveSolver.solve`), the `policy_file`, `policy_params`, and
/// `policy_status` fields are populated; for the strategy-based path
/// they are all `None`.
#[pyclass(
    name = "SolveResult",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PySolveResult {
    inner: SolveResult,
    /// Echo of the `.star` policy path — populated only on the DSL path.
    pub policy_file: Option<String>,
    /// JSON-encoded echo of `policy_params` — populated only on the DSL path.
    pub policy_params: Option<String>,
    /// String representation of `PolicyStatus` — populated only on the DSL path.
    pub policy_status: Option<String>,
}

#[pymethods]
impl PySolveResult {
    /// Status of the solve: "solved", "unsolvable", or "budget_exceeded".
    ///
    /// For results produced via the `policy_path` kwarg, this still reflects
    /// the underlying `SolveResult` status. Use `policy_status` to inspect
    /// the DSL-level terminal state.
    #[getter]
    fn status(&self) -> &'static str {
        match self.inner.status {
            SolveStatus::Solved => "solved",
            SolveStatus::Unsolvable => "unsolvable",
            SolveStatus::BudgetExceeded => "budget_exceeded",
        }
    }

    /// Echo of the `.star` policy file path, or `None` if not a DSL solve.
    #[getter]
    fn policy_file(&self) -> Option<&str> {
        self.policy_file.as_deref()
    }

    /// JSON-encoded echo of `policy_params` dict, or `None` if not a DSL solve.
    /// Use `json.loads(result.policy_params)` to recover the dict.
    #[getter]
    fn policy_params(&self) -> Option<&str> {
        self.policy_params.as_deref()
    }

    /// String representation of the DSL terminal status (e.g. `"solved"`,
    /// `"unsolvable"`, `"budget_exhausted"`, `"timeout"`, etc.), or `None`
    /// if not a DSL solve.
    #[getter]
    fn policy_status(&self) -> Option<&str> {
        self.policy_status.as_deref()
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

impl PySolveResult {
    /// Build a `PySolveResult` from a `PolicyResult` returned by the DSL kernel.
    ///
    /// The inner `SolveResult` is synthesised from the policy result's
    /// `move_layers` and `goal_config`. The strategy-specific fields
    /// (`cost`, `deadlocks`, `entropy_trace`) are zeroed/empty since the
    /// DSL kernel does not track them.
    pub fn from_policy(p: bloqade_lanes_search::move_policy_dsl::PolicyResult) -> Self {
        use bloqade_lanes_search::move_policy_dsl::PolicyStatus;

        let policy_status_str = match &p.status {
            PolicyStatus::Solved => "solved".to_string(),
            PolicyStatus::Unsolvable => "unsolvable".to_string(),
            PolicyStatus::BudgetExhausted => "budget_exhausted".to_string(),
            PolicyStatus::Timeout => "timeout".to_string(),
            PolicyStatus::Fallback(m) => format!("fallback: {m}"),
            PolicyStatus::SyntaxError(m) => format!("syntax_error: {m}"),
            PolicyStatus::RuntimeError(m) => format!("runtime_error: {m}"),
            PolicyStatus::SchemaError(f) => format!("schema_error: {f}"),
            PolicyStatus::BadPolicy(m) => format!("bad_policy: {m}"),
            PolicyStatus::StarlarkBudget => "starlark_budget".to_string(),
            PolicyStatus::StarlarkOOM => "starlark_oom".to_string(),
        };

        let inner_status = match p.status {
            PolicyStatus::Solved => SolveStatus::Solved,
            PolicyStatus::Unsolvable => SolveStatus::Unsolvable,
            _ => SolveStatus::BudgetExceeded,
        };

        let inner = SolveResult {
            status: inner_status,
            move_layers: p.move_layers,
            goal_config: p.goal_config,
            nodes_expanded: p.nodes_expanded,
            cost: 0.0,
            deadlocks: 0,
            entropy_trace: None,
        };

        Self {
            inner,
            policy_file: Some(p.policy_file),
            policy_params: Some(serde_json::to_string(&p.policy_params).unwrap_or_default()),
            policy_status: Some(policy_status_str),
        }
    }
}

// ── JSON helpers for policy_params conversion ──

pub(crate) fn pydict_to_json(d: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let val = pyany_to_json(&v)?;
        obj.insert(key, val);
    }
    Ok(serde_json::Value::Object(obj))
}

fn pyany_to_json(v: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if v.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = v.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = v.extract::<i64>() {
        Ok(serde_json::Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = v.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = v.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = v.downcast::<PyList>() {
        let items: PyResult<Vec<_>> = list.iter().map(|x| pyany_to_json(&x)).collect();
        Ok(serde_json::Value::Array(items?))
    } else if let Ok(dict) = v.downcast::<PyDict>() {
        pydict_to_json(dict)
    } else {
        Err(PyValueError::new_err(format!(
            "policy_params: unsupported value type {}",
            v.get_type().name()?
        )))
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
        moveset: &[(u8, u8, u32, u32, u32, u32)],
    ) -> PyResult<Config> {
        let mut moves: Vec<(u32, LocationAddr)> = Vec::with_capacity(moveset.len());
        for &(dir, mt, zone, word, site, bus) in moveset {
            let direction = match dir {
                0 => Direction::Forward,
                1 => Direction::Backward,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "invalid direction {other} (must be 0 or 1)"
                    )));
                }
            };
            let move_type = match mt {
                0 => MoveType::SiteBus,
                1 => MoveType::WordBus,
                2 => MoveType::ZoneBus,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "invalid move_type {other} (must be 0, 1, or 2)"
                    )));
                }
            };
            let lane = LaneAddr {
                direction,
                move_type,
                zone_id: zone,
                word_id: word,
                site_id: site,
                bus_id: bus,
            };
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
        let json = serde_json::to_string(&arch_spec.inner)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize arch spec: {e}")))?;
        let parsed = serde_json::from_str(&json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec: {e}")))?;
        let index = LaneIndex::new(parsed);

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
    #[allow(clippy::type_complexity)]
    fn metrics(
        &self,
        current_config: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        moveset: Vec<(u8, u8, u32, u32, u32, u32)>,
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
    #[allow(clippy::type_complexity)]
    fn score_moveset(
        &self,
        current_config: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        moveset: Vec<(u8, u8, u32, u32, u32, u32)>,
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
    /// Stored arch spec JSON, used to build a fresh `LaneIndex` for the
    /// Move Policy DSL path (which needs an `Arc<LaneIndex>`).
    arch_spec_json: String,
}

#[pymethods]
impl PyMoveSolver {
    /// Create a solver from an ArchSpec JSON string.
    #[new]
    fn new(arch_spec_json: &str) -> PyResult<Self> {
        let inner = MoveSolver::from_json(arch_spec_json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec JSON: {e}")))?;
        Ok(Self {
            inner,
            arch_spec_json: arch_spec_json.to_owned(),
        })
    }

    /// Create a solver from a native ArchSpec object.
    #[staticmethod]
    fn from_arch_spec(arch: &PyArchSpec) -> PyResult<Self> {
        let json = serde_json::to_string(&arch.inner)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize arch spec: {e}")))?;
        let inner = MoveSolver::from_json(&json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec: {e}")))?;
        Ok(Self {
            inner,
            arch_spec_json: json,
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
    ///     policy_path: Path to a `.star` Move Policy DSL file. When supplied,
    ///         routes through `solve_with_policy` instead of the strategy-based
    ///         search path. The `options` kwarg is ignored on this path.
    ///     policy_params: Free-form dict of parameters echoed into the result's
    ///         `policy_params` field (JSON-encoded string). Only used when
    ///         `policy_path` is supplied.
    ///     timeout_s: Wall-clock time limit (seconds) for the DSL kernel. Only
    ///         used when `policy_path` is supplied.
    ///
    /// Returns:
    ///     SolveResult with status indicating success/failure. When
    ///     `policy_path` is used, also check `policy_status`, `policy_file`,
    ///     and `policy_params` on the returned object.
    #[pyo3(signature = (initial, target, blocked, *, max_expansions=None, options=None, policy_path=None, policy_params=None, timeout_s=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        target: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u64>,
        options: Option<&PySolveOptions>,
        policy_path: Option<&str>,
        policy_params: Option<&Bound<'_, PyDict>>,
        timeout_s: Option<f64>,
    ) -> PyResult<PySolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let target_pairs: Vec<(u32, LocationAddr)> =
            target.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();

        // ── Move Policy DSL path ──────────────────────────────────────────
        if let Some(path) = policy_path {
            let params_json: serde_json::Value = match policy_params {
                Some(d) => pydict_to_json(d)?,
                None => serde_json::Value::Object(Default::default()),
            };
            let policy_opts = bloqade_lanes_search::move_policy_dsl::PolicyOptions {
                policy_path: path.to_string(),
                policy_params: params_json,
                max_expansions: max_expansions.unwrap_or(100_000),
                timeout_s,
                sandbox: bloqade_lanes_dsl_core::sandbox::SandboxConfig::default(),
            };
            let arch_spec: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
                serde_json::from_str(&self.arch_spec_json).map_err(|e| {
                    PyValueError::new_err(format!("failed to re-parse arch spec: {e}"))
                })?;
            let index = std::sync::Arc::new(bloqade_lanes_search::LaneIndex::new(arch_spec));
            let result = py
                .allow_threads(|| {
                    bloqade_lanes_search::move_policy_dsl::solve_with_policy(
                        initial_pairs,
                        target_pairs,
                        blocked_locs,
                        index,
                        policy_opts,
                        &mut bloqade_lanes_search::move_policy_dsl::NoOpMoveObserver,
                    )
                })
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            return Ok(PySolveResult::from_policy(result));
        }

        // ── Strategy-based path (existing) ───────────────────────────────
        let opts = options.map(|o| o.inner.clone()).unwrap_or_default();
        let max_exp_u32 = max_expansions.map(|n| n as u32);

        // Release the GIL during search (pure Rust, no Python objects needed).
        let result = py
            .allow_threads(|| {
                self.inner.solve(
                    initial_pairs,
                    target_pairs,
                    blocked_locs,
                    max_exp_u32,
                    &opts,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult {
            inner: result,
            policy_file: None,
            policy_params: None,
            policy_status: None,
        })
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
    #[pyo3(signature = (initial, blocked, controls, targets, generator=None, max_expansions=None, options=None))]
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
    ) -> PyResult<PyMultiSolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();
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
    #[pyo3(signature = (strategy=PySearchStrategy::AStar, max_movesets_per_group=3, max_goal_candidates=3, weight=1.0, restarts=1, lookahead=false, deadlock_policy=PyDeadlockPolicy::Skip, w_t=0.05, collect_entropy_trace=false))]
    fn new(
        strategy: PySearchStrategy,
        max_movesets_per_group: usize,
        max_goal_candidates: usize,
        weight: f64,
        restarts: u32,
        lookahead: bool,
        deadlock_policy: PyDeadlockPolicy,
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
                max_movesets_per_group,
                max_goal_candidates,
                weight,
                restarts,
                lookahead,
                deadlock_policy: deadlock_policy.to_rs(),
                w_t,
                collect_entropy_trace,
            },
        })
    }

    #[getter]
    fn strategy(&self) -> PySearchStrategy {
        PySearchStrategy::from_rs(&self.inner.strategy)
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
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    #[getter]
    fn restarts(&self) -> u32 {
        self.inner.restarts
    }

    #[getter]
    fn lookahead(&self) -> bool {
        self.inner.lookahead
    }

    #[getter]
    fn deadlock_policy(&self) -> PyDeadlockPolicy {
        PyDeadlockPolicy::from_rs(&self.inner.deadlock_policy)
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
            "SolveOptions(strategy={}, weight={}, restarts={}, deadlock_policy={})",
            self.strategy().name(),
            self.inner.weight,
            self.inner.restarts,
            self.deadlock_policy().name(),
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
            status_str(self.inner.result.status),
            self.inner.candidate_index,
            self.inner.candidates_tried,
            self.inner.total_expansions,
        )
    }
}
