//! PyO3 bindings for the move synthesis solver.
//!
//! Exposes the typed surface to Python: [`PySearchEngine`], [`PyMoveSearch`],
//! [`PyTargetSolver`], [`PySolveResult`], and the four [`CzPlacement`] peers.

use std::collections::HashSet;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_search::DeadlockPolicy;
use bloqade_lanes_search::bounds::BoundStats;
use bloqade_lanes_search::drivers::entropy::{
    EntropyParams, EntropyTrace, EntropyTraceEvent, EntropyTraceStep, MovesetMetrics,
    compute_moveset_metrics,
};
use bloqade_lanes_search::drivers::result::Termination;
use bloqade_lanes_search::observer::EntropyReason;
use bloqade_lanes_search::placement::cz_placement::{
    CandidateAttempt, CzPlacement, CzStage, PlacementBudget, PlacementResult,
};
use bloqade_lanes_search::placement::loose_goal::LooseGoalCzPlacement;
use bloqade_lanes_search::placement::nohome::{NoHomeCzPlacement, NoHomeOptions};
use bloqade_lanes_search::placement::receding_horizon::{
    RecedingHorizonCzPlacement, RecedingHorizonOptions, default_weight_grid,
};
use bloqade_lanes_search::placement::single_heuristic::SingleHeuristicCzPlacement;
use bloqade_lanes_search::placement::target_generator::DefaultTargetGenerator;
use bloqade_lanes_search::primitives::config::Config;
use bloqade_lanes_search::primitives::context::SearchContext;
use bloqade_lanes_search::primitives::distance::DistanceTable;
use bloqade_lanes_search::primitives::graph::MoveSet;
use bloqade_lanes_search::primitives::lane_index::LaneIndex;
use bloqade_lanes_search::search::engine::SearchEngine;
use bloqade_lanes_search::search::move_search::MoveSearch;
use bloqade_lanes_search::search::options::{
    BoundKind, EntanglingOptions, EntropyOptions, InnerStrategy, SolveOptions, Strategy,
};
use bloqade_lanes_search::search::result::{SolveResult, SolveStatus};
use bloqade_lanes_search::search::target_solver::TargetSolver;

use crate::arch_python::{PyArchSpec, PyLaneAddr, PyLocationAddr};

// ── Enum wrappers ──

/// Search strategy for the move solver.
#[pyclass(
    from_py_object,
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
    #[pyo3(name = "PUSH_ROTATE")]
    PushRotate = 9,
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
            Self::PushRotate => "PUSH_ROTATE",
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
            Strategy::PushRotate => Self::PushRotate,
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
            Self::PushRotate => Strategy::PushRotate,
        }
    }
}

/// Deadlock handling policy for the move solver.
#[pyclass(
    from_py_object,
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

// ── Typed result enums ──

/// Python equality for a typed result enum: equal to its own members, and a
/// `TypeError` against a `str`, so code still comparing with the string labels
/// these replaced fails loudly instead of silently reading `False`.
macro_rules! typed_result_enum {
    ($ty:ident, $pyname:literal, [$($variant:ident => $name:literal),+ $(,)?]) => {
        #[pymethods]
        impl $ty {
            /// The member's name.
            #[getter]
            fn name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)+
                }
            }

            fn __hash__(&self) -> u64 {
                *self as u64
            }

            fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
                if other.is_instance_of::<pyo3::types::PyString>() {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        concat!(
                            "cannot compare ", $pyname, " with the string {}: results now ",
                            "report ", $pyname, " members, e.g. ", $pyname, ".{}"
                        ),
                        other.repr()?,
                        self.name(),
                    )));
                }
                let equal = other.extract::<Self>().is_ok_and(|o| o == *self);
                match op {
                    CompareOp::Eq => Ok(equal),
                    CompareOp::Ne => Ok(!equal),
                    _ => Err(pyo3::exceptions::PyTypeError::new_err(concat!(
                        $pyname,
                        " members are not ordered"
                    ))),
                }
            }
        }
    };
}

/// How a solve ended: solved, proven or given up as unsolvable, or out of budget.
///
/// ``UNSOLVABLE`` is a *proof* only when the result's ``proof`` is
/// ``Proof.NO_PLAN``. From a search strategy it usually means the search
/// exhausted the moves its generator offered, which is less than the
/// architecture allows.
#[pyclass(
    from_py_object,
    name = "SolveStatus",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySolveStatus {
    #[pyo3(name = "SOLVED")]
    Solved = 0,
    #[pyo3(name = "UNSOLVABLE")]
    Unsolvable = 1,
    #[pyo3(name = "BUDGET_EXCEEDED")]
    BudgetExceeded = 2,
}

typed_result_enum!(PySolveStatus, "SolveStatus", [
    Solved => "SOLVED",
    Unsolvable => "UNSOLVABLE",
    BudgetExceeded => "BUDGET_EXCEEDED",
]);

impl PySolveStatus {
    fn from_rs(status: SolveStatus) -> Self {
        match status {
            SolveStatus::Solved => Self::Solved,
            SolveStatus::Unsolvable => Self::Unsolvable,
            SolveStatus::BudgetExceeded => Self::BudgetExceeded,
        }
    }
}

/// How the search behind a result ended, as the driver's own account.
///
/// ``BUDGET`` ran out of expansions; ``STOPPED`` ended on a rule of its own,
/// such as collecting its goal quota; ``EXHAUSTED`` drained its space. Whether
/// that is a proof is the result's ``proof``, not this.
#[pyclass(
    from_py_object,
    name = "Termination",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyTermination {
    #[pyo3(name = "BUDGET")]
    Budget = 0,
    #[pyo3(name = "EXHAUSTED")]
    Exhausted = 1,
    #[pyo3(name = "STOPPED")]
    Stopped = 2,
}

typed_result_enum!(PyTermination, "Termination", [
    Budget => "BUDGET",
    Exhausted => "EXHAUSTED",
    Stopped => "STOPPED",
]);

/// What a result proves, when it proves anything.
///
/// ``OPTIMAL``: the plan is optimal (no legal plan is cheaper).
/// ``NO_PLAN``: no plan exists for the instance.
#[pyclass(
    from_py_object,
    name = "Proof",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyProof {
    #[pyo3(name = "OPTIMAL")]
    Optimal = 0,
    #[pyo3(name = "NO_PLAN")]
    NoPlan = 1,
}

typed_result_enum!(PyProof, "Proof", [
    Optimal => "OPTIMAL",
    NoPlan => "NO_PLAN",
]);

/// The proof a result carries: optimal when solved, no plan when unsolvable,
/// and nothing unless the termination is a proof.
fn result_proof(result: &SolveResult) -> Option<PyProof> {
    if !result.proven() {
        return None;
    }
    match result.status {
        SolveStatus::Solved => Some(PyProof::Optimal),
        SolveStatus::Unsolvable => Some(PyProof::NoPlan),
        SolveStatus::BudgetExceeded => None,
    }
}

// ── Solve results ──

/// Branch-and-bound pruning statistics from one solve.
#[pyclass(name = "BoundStats", frozen, module = "bloqade.lanes.bytecode._native")]
pub struct PyBoundStats {
    inner: BoundStats,
}

#[pymethods]
impl PyBoundStats {
    /// Cuts the accumulated cost alone could make.
    #[getter]
    fn cuts_by_g(&self) -> u64 {
        self.inner.cuts_by_g
    }

    /// Cuts only the bound could make.
    #[getter]
    fn cuts_by_h(&self) -> u64 {
        self.inner.cuts_by_h
    }

    /// Branches cut because the bound proved them infeasible.
    #[getter]
    fn cuts_infeasible(&self) -> u64 {
        self.inner.cuts_infeasible
    }

    /// Sum of the depths at which the bound cut.
    #[getter]
    fn cut_depth_sum(&self) -> u64 {
        self.inner.cut_depth_sum
    }

    /// Sum of the depths at which the cost alone would have cut; against
    /// ``cut_depth_sum`` it measures how much earlier the bound fired.
    #[getter]
    fn cut_depth_g_only_sum(&self) -> u64 {
        self.inner.cut_depth_g_only_sum
    }

    /// A certified lower bound on the instance optimum.
    #[getter]
    fn root_lower_bound(&self) -> f64 {
        self.inner.root_lower_bound
    }

    /// Cost of the best plan found, or None if none was.
    #[getter]
    fn incumbent_cost(&self) -> Option<f64> {
        // A `Some` is always a finite `g_score`; the filter keeps a non-finite
        // cost from ever reaching Python as a misleading float.
        self.inner.incumbent_cost.filter(|c| c.is_finite())
    }

    /// ``(incumbent - root_lower_bound) / incumbent``, or None when unsolved.
    #[getter]
    fn optimality_gap(&self) -> Option<f64> {
        self.inner.optimality_gap()
    }

    fn __repr__(&self) -> String {
        format!(
            "BoundStats(cuts_by_g={}, cuts_by_h={}, cuts_infeasible={}, root_lower_bound={}, incumbent_cost={:?})",
            self.inner.cuts_by_g,
            self.inner.cuts_by_h,
            self.inner.cuts_infeasible,
            self.inner.root_lower_bound,
            self.inner.incumbent_cost,
        )
    }
}

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
    /// How the solve ended; see ``SolveStatus``.
    #[getter]
    fn status(&self) -> PySolveStatus {
        PySolveStatus::from_rs(self.inner.status)
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

    /// Number of nodes at which the generator had nothing useful to offer.
    ///
    /// Counts both "nothing scored an improvement" and "nothing was executable"
    /// (every rectangle rejected). **Not** a count of escape moves taken: the
    /// counter is incremented before ``deadlock_policy`` is consulted, so under
    /// the default ``SKIP`` it records nodes where nothing was generated in
    /// response. Read it as "how often the search got stuck".
    #[getter]
    fn deadlocks(&self) -> u32 {
        self.inner.deadlocks
    }

    /// What the result proves, or ``None``.
    ///
    /// ``Proof.OPTIMAL``: the plan is optimal. On the entropy driver that is
    /// the root certificate: the plan's cost reached `h(root)`, a lower bound on
    /// every legal plan, so none is cheaper -- including plans the generator
    /// would never have proposed. ``Proof.NO_PLAN``: no plan exists, as Push
    /// and Rotate's completeness or an infinite root bound shows.
    ///
    /// ``None`` is not "suboptimal" or "solvable", it is "unproven": most
    /// solves end on their expansion budget. Read it to tell a solver giving
    /// up from an instance that is genuinely this hard.
    #[getter]
    fn proof(&self) -> Option<PyProof> {
        result_proof(&self.inner)
    }

    /// How the search ended; see ``Termination``.
    #[getter]
    fn termination(&self) -> PyTermination {
        match self.inner.termination {
            Termination::Budget => PyTermination::Budget,
            Termination::Exhausted { .. } => PyTermination::Exhausted,
            Termination::Stopped => PyTermination::Stopped,
        }
    }

    /// Branch-and-bound pruning statistics, or ``None`` unless
    /// ``EntropyOptions.completion_bound`` was set: an unbounded solve measured
    /// nothing, and zeros would advertise a ``root_lower_bound`` of 0.0 as if
    /// it were a measurement.
    #[getter]
    fn bound_stats(&self) -> Option<PyBoundStats> {
        self.inner
            .bound_stats
            .bound_enabled
            .then_some(PyBoundStats {
                inner: self.inner.bound_stats,
            })
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
            "SolveResult(status=SolveStatus.{}, steps={}, cost={}, expanded={}, deadlocks={}, proof={})",
            PySolveStatus::from_rs(self.inner.status).name(),
            self.inner.move_layers.len(),
            self.inner.cost,
            self.inner.nodes_expanded,
            self.inner.deadlocks,
            result_proof(&self.inner).map_or("None", |p| p.name()),
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

/// The visualizer's label for a trace event.
fn trace_event_label(event: EntropyTraceEvent) -> &'static str {
    match event {
        EntropyTraceEvent::Descend => "descend",
        EntropyTraceEvent::Goal => "goal",
        EntropyTraceEvent::EntropyBump => "entropy_bump",
        EntropyTraceEvent::Revert => "revert",
        EntropyTraceEvent::FallbackStart => "fallback_start",
    }
}

/// The visualizer's label for a trace reason.
fn trace_reason_label(reason: EntropyReason) -> &'static str {
    match reason {
        EntropyReason::NoValidMoves => "no-valid-moves",
        EntropyReason::StateSeen => "state-seen",
        EntropyReason::StateSeenGoal => "state-seen-goal",
        EntropyReason::DeadlockBreaker => "deadlock-breaker",
        EntropyReason::EntropyLimit => "entropy",
    }
}

/// A moveset as the visualizer's lane tuples:
/// `(direction, move_type, zone, word, site, bus)`.
fn trace_moveset(moveset: &MoveSet) -> Vec<(u8, u8, u32, u32, u32, u32)> {
    moveset
        .decode()
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
}

/// A configuration as the visualizer's `(qubit, zone, word, site)` tuples.
fn trace_configuration(config: &Config) -> Vec<(u32, u32, u32, u32)> {
    config
        .iter()
        .map(|(qid, loc)| (qid, loc.zone_id, loc.word_id, loc.site_id))
        .collect()
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
        trace_event_label(self.inner.event).to_string()
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
        self.inner.moveset.as_ref().map(trace_moveset)
    }

    #[getter]
    #[allow(clippy::type_complexity)]
    fn candidate_movesets(&self) -> Vec<Vec<(u8, u8, u32, u32, u32, u32)>> {
        self.inner
            .candidate_movesets
            .iter()
            .map(trace_moveset)
            .collect()
    }

    #[getter]
    fn candidate_index(&self) -> Option<u32> {
        self.inner.candidate_index
    }

    #[getter]
    fn reason(&self) -> Option<String> {
        self.inner
            .reason
            .map(|reason| trace_reason_label(reason).to_string())
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
        trace_configuration(&self.inner.configuration)
    }

    #[getter]
    fn parent_configuration(&self) -> Option<Vec<(u32, u32, u32, u32)>> {
        self.inner
            .parent_configuration
            .as_ref()
            .map(trace_configuration)
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
            trace_event_label(self.inner.event),
            self.inner.node_id,
            self.inner.depth,
            self.inner.entropy,
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
        py: Python<'_>,
    ) -> PyResult<Self> {
        let index = crate::errors::validated_lane_index(py, &arch_spec.inner)?;

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

        let ctx = SearchContext::new(&self.index, &self.dist_table, &self.blocked, &self.targets);
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

// ── No-home options ──

/// Tuning parameters for the no-home return assignment.
///
/// Controls how displaced qubits are assigned to available home sites
/// between CZ layers.
#[pyclass(
    skip_from_py_object,
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

/// Core search-tuning parameters shared by every solver entry point.
#[pyclass(
    skip_from_py_object,
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
    #[pyo3(signature = (strategy=PySearchStrategy::AStar, weight=1.0, restarts=1, deadlock_policy=PyDeadlockPolicy::Skip, lookahead=false, top_c=None, fallback_push_rotate=false, backwards_search=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        strategy: PySearchStrategy,
        weight: f64,
        restarts: u32,
        deadlock_policy: PyDeadlockPolicy,
        lookahead: bool,
        top_c: Option<usize>,
        fallback_push_rotate: bool,
        backwards_search: bool,
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
                fallback_push_rotate,
                backwards_search,
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

    #[getter]
    fn fallback_push_rotate(&self) -> bool {
        self.inner.fallback_push_rotate
    }

    #[getter]
    fn backwards_search(&self) -> bool {
        self.inner.backwards_search
    }

    /// Every constructor field, in constructor order.
    ///
    /// Keep this exhaustive: a `SolveOptions` that prints fewer options than
    /// it carries makes a mis-set flag invisible at exactly the moment
    /// someone is printing the options to find one.
    fn __repr__(&self) -> String {
        format!(
            "SolveOptions(strategy={}, weight={}, restarts={}, deadlock_policy={}, lookahead={}, top_c={:?}, fallback_push_rotate={}, backwards_search={})",
            self.strategy().name(),
            self.inner.weight,
            self.inner.restarts,
            self.deadlock_policy().name(),
            self.inner.lookahead,
            self.inner.top_c,
            self.inner.fallback_push_rotate,
            self.inner.backwards_search,
        )
    }
}

// ── Entropy options ──

/// Entropy-strategy-specific parameters.
///
/// Only consumed when the chosen strategy is entropy (or a Cascade variant
/// whose inner is entropy). Pass via `MoveSearch.with_entropy_options`.
#[pyclass(
    skip_from_py_object,
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
    #[pyo3(signature = (max_movesets_per_group=3, max_goal_candidates=3, w_t=0.05, collect_entropy_trace=false, seed=0, completion_bound=None))]
    fn new(
        max_movesets_per_group: usize,
        max_goal_candidates: usize,
        w_t: f64,
        collect_entropy_trace: bool,
        seed: u64,
        completion_bound: Option<&str>,
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
        let completion_bound = match completion_bound {
            None => None,
            Some("weighted_distance") => Some(BoundKind::WeightedDistance),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "unknown completion_bound '{other}'; expected 'weighted_distance' or None"
                )));
            }
        };
        Ok(Self {
            inner: EntropyOptions {
                max_movesets_per_group,
                max_goal_candidates,
                w_t,
                collect_entropy_trace,
                seed,
                // Rust-only: an A/B measurement knob, not a user setting.
                // Python always lets the bound end a search it has proven.
                bound_terminates: true,
                completion_bound,
            },
        })
    }

    /// Completion bound in use: `"weighted_distance"` or `None`.
    #[getter]
    fn completion_bound(&self) -> Option<&'static str> {
        match self.inner.completion_bound {
            None => None,
            Some(BoundKind::WeightedDistance) => Some("weighted_distance"),
        }
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

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "EntropyOptions(max_movesets_per_group={}, max_goal_candidates={}, w_t={}, collect_entropy_trace={}, seed={})",
            self.inner.max_movesets_per_group,
            self.inner.max_goal_candidates,
            self.inner.w_t,
            self.inner.collect_entropy_trace,
            self.inner.seed,
        )
    }
}

// ── Entangling options ──

/// Loose-goal entangling-search parameters consumed by
/// `LooseGoalCzPlacement`.
#[pyclass(
    skip_from_py_object,
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

/// Orchestration parameters for `RecedingHorizonCzPlacement`.
///
/// Controls how many candidate Hungarian assignments are tried per stage,
/// how far each rollout searches forward, how many layers of the winning
/// branch get committed before re-planning, and other tuning knobs.
#[pyclass(
    skip_from_py_object,
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

/// One candidate a placement tried, in order.
#[pyclass(
    name = "CandidateAttempt",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyCandidateAttempt {
    inner: CandidateAttempt,
}

#[pymethods]
impl PyCandidateAttempt {
    /// Index of the candidate in the order the placement generated them.
    #[getter]
    fn candidate_index(&self) -> usize {
        self.inner.candidate_index
    }

    /// How routing the candidate ended.
    #[getter]
    fn status(&self) -> PySolveStatus {
        PySolveStatus::from_rs(self.inner.status)
    }

    /// Nodes expanded routing it.
    #[getter]
    fn nodes_expanded(&self) -> u32 {
        self.inner.nodes_expanded
    }

    /// The score a candidate evaluator gave it, or None if nothing ranked
    /// the candidates.
    #[getter]
    fn score(&self) -> Option<f64> {
        self.inner.score
    }

    fn __repr__(&self) -> String {
        format!(
            "CandidateAttempt(candidate_index={}, status=SolveStatus.{}, nodes_expanded={})",
            self.inner.candidate_index,
            PySolveStatus::from_rs(self.inner.status).name(),
            self.inner.nodes_expanded,
        )
    }
}

/// The outcome of placing one CZ stage (``CzPlacement.place``).
#[pyclass(
    name = "PlacementResult",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyPlacementResult {
    result: Py<PySolveResult>,
    chosen: Option<usize>,
    attempts: Vec<CandidateAttempt>,
    total_expansions: u32,
}

impl PyPlacementResult {
    fn from_rs(py: Python<'_>, placed: PlacementResult) -> PyResult<Self> {
        Ok(Self {
            result: Py::new(
                py,
                PySolveResult {
                    inner: placed.result,
                },
            )?,
            chosen: placed.chosen,
            attempts: placed.attempts,
            total_expansions: placed.total_expansions,
        })
    }
}

#[pymethods]
impl PyPlacementResult {
    /// The routing result. On success its ``goal_config`` is the chosen
    /// placement. On failure it is usually the stage's starting
    /// configuration, but a placement that commits layers before failing
    /// (``RecedingHorizonCzPlacement``) returns those layers and the
    /// configuration they reach instead: read ``move_layers`` and
    /// ``goal_config`` together.
    #[getter]
    fn result(&self, py: Python<'_>) -> Py<PySolveResult> {
        self.result.clone_ref(py)
    }

    /// Which candidate won, for placements that enumerate candidates; None
    /// when none won or the placement does not enumerate them.
    #[getter]
    fn chosen(&self) -> Option<usize> {
        self.chosen
    }

    /// Every candidate tried, in order. Empty for placements that do not
    /// enumerate candidates.
    #[getter]
    fn attempts(&self) -> Vec<PyCandidateAttempt> {
        self.attempts
            .iter()
            .map(|a| PyCandidateAttempt { inner: a.clone() })
            .collect()
    }

    /// Expansions across every leg and candidate of the placement.
    #[getter]
    fn total_expansions(&self) -> u32 {
        self.total_expansions
    }

    /// Candidates actually routed (validation failures are not counted).
    #[getter]
    fn candidates_tried(&self) -> usize {
        self.attempts.len()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "PlacementResult(status=SolveStatus.{}, chosen={:?}, tried={}, expansions={})",
            PySolveStatus::from_rs(self.result.borrow(py).inner.status).name(),
            self.chosen,
            self.attempts.len(),
            self.total_expansions,
        )
    }
}

/// Place one stage through a `CzPlacement`, with the GIL released.
#[allow(clippy::too_many_arguments)]
fn place_stage(
    py: Python<'_>,
    placement: &(impl CzPlacement + Sync),
    initial: &std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
    pairs: &[(u32, u32)],
    blocked: &[PyRef<'_, PyLocationAddr>],
    max_expansions: Option<u32>,
    future_layers: Option<Vec<Vec<(u32, u32)>>>,
) -> PyResult<PyPlacementResult> {
    let initial: Vec<(u32, LocationAddr)> =
        initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
    let blocked: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();
    let future = future_layers.unwrap_or_default();
    let placed = py
        .detach(|| {
            placement.place(
                &CzStage::new(&initial, pairs, &blocked).with_future_layers(&future),
                &PlacementBudget::new(max_expansions),
            )
        })
        .map_err(|e| crate::errors::config_error_to_py(py, &e))?;
    PyPlacementResult::from_rs(py, placed)
}

// ── New typed surface: SearchEngine / MoveSearch / TargetSolver / CzPlacement peers ──

/// Precomputed search engine bound to an architecture.
///
/// Holds the lane index and (lazily) any cached data that is reused across
/// solves. Construct once per architecture; share across multiple solvers.
#[pyclass(
    name = "SearchEngine",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PySearchEngine {
    pub(crate) inner: Arc<SearchEngine>,
}

#[pymethods]
impl PySearchEngine {
    /// Create a ``SearchEngine`` from a native ``ArchSpec`` object.
    ///
    /// Raises ``ArchSpecError`` (with every individual problem in its
    /// ``errors`` list) for a spec that fails structural validation — the
    /// search assumes per-bus acyclicity rather than checking it.
    #[staticmethod]
    fn from_arch_spec(arch: &PyArchSpec, py: Python<'_>) -> PyResult<Self> {
        let engine = SearchEngine::from_arch_spec(&arch.inner)
            .map_err(|errors| crate::errors::arch_spec_errors_to_py(py, errors))?;
        Ok(Self {
            inner: Arc::new(engine),
        })
    }

    /// Create a ``SearchEngine`` from an ArchSpec JSON string, **without
    /// validating the spec**.
    ///
    /// Prefer :meth:`from_json_validated`. The search layers assume per-bus
    /// acyclicity and endpoint uniqueness rather than checking them, so an
    /// unvalidated spec with a cyclic bus would be routed as if a rotation
    /// were a legal AOD operation.
    #[staticmethod]
    fn from_json(arch_spec_json: &str) -> PyResult<Self> {
        let engine = SearchEngine::from_json(arch_spec_json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec JSON: {e}")))?;
        Ok(Self {
            inner: Arc::new(engine),
        })
    }

    /// Create a ``SearchEngine`` from an ArchSpec JSON string, rejecting a
    /// spec that fails structural validation.
    ///
    /// Raises ``ArchSpecError`` (with every individual problem in its
    /// ``errors`` list) for an invalid spec, or ``ValueError`` for malformed
    /// JSON.
    #[staticmethod]
    fn from_json_validated(arch_spec_json: &str, py: Python<'_>) -> PyResult<Self> {
        let engine = SearchEngine::from_json_validated(arch_spec_json)
            .map_err(|e| crate::errors::arch_spec_load_error_to_py(py, &e))?;
        Ok(Self {
            inner: Arc::new(engine),
        })
    }

    fn __repr__(&self) -> &'static str {
        "SearchEngine(...)"
    }
}

/// Search algorithm configuration bundle.
///
/// Combine a strategy (entropy, A*, IDS, …) with its tuning options.
/// Build via the factory class methods, then pass to ``TargetSolver`` or
/// the ``CzPlacement`` constructors.
#[pyclass(
    skip_from_py_object,
    name = "MoveSearch",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
#[derive(Clone)]
pub struct PyMoveSearch {
    pub(crate) inner: MoveSearch,
}

#[pymethods]
impl PyMoveSearch {
    /// Entropy-guided search. Pass ``options`` to set restarts, lookahead,
    /// etc.; the strategy is always forced to entropy regardless of
    /// ``options.strategy``.
    #[staticmethod]
    #[pyo3(signature = (options = None, entropy_options = None))]
    fn entropy(
        options: Option<&PySolveOptions>,
        entropy_options: Option<&PyEntropyOptions>,
    ) -> Self {
        let mut ms = MoveSearch::entropy();
        if let Some(opts) = options {
            let mut solve_opts = opts.inner.clone();
            solve_opts.strategy = Strategy::Entropy;
            ms = ms.with_options(solve_opts);
        }
        if let Some(eopts) = entropy_options {
            ms = ms.with_entropy_options(eopts.inner.clone());
        }
        Self { inner: ms }
    }

    /// Weighted A* search. Pass ``options`` to set restarts, lookahead, etc.;
    /// the strategy is always AStar and ``weight`` always overrides
    /// ``options.weight``.
    #[staticmethod]
    #[pyo3(signature = (weight = 1.0, options = None))]
    fn astar(weight: f64, options: Option<&PySolveOptions>) -> PyResult<Self> {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(
                "weight must be a finite float greater than 0.0",
            ));
        }
        let mut ms = MoveSearch::astar(weight);
        if let Some(opts) = options {
            let mut solve_opts = opts.inner.clone();
            solve_opts.strategy = Strategy::AStar;
            solve_opts.weight = weight;
            ms = ms.with_options(solve_opts);
        }
        Ok(Self { inner: ms })
    }

    /// Iterative-deepening search. Pass ``options`` to set restarts,
    /// lookahead, etc.; the strategy is always IDS.
    #[staticmethod]
    #[pyo3(signature = (options = None))]
    fn ids(options: Option<&PySolveOptions>) -> Self {
        let mut ms = MoveSearch::ids();
        if let Some(opts) = options {
            let mut solve_opts = opts.inner.clone();
            solve_opts.strategy = Strategy::Ids;
            ms = ms.with_options(solve_opts);
        }
        Self { inner: ms }
    }

    /// Cascade: IDS followed by entropy refinement. Pass ``options`` to set
    /// restarts, lookahead, etc.
    #[staticmethod]
    #[pyo3(signature = (options = None))]
    fn cascade_ids(options: Option<&PySolveOptions>) -> Self {
        let mut ms = MoveSearch::cascade(InnerStrategy::Ids);
        if let Some(opts) = options {
            let mut solve_opts = opts.inner.clone();
            solve_opts.strategy = Strategy::Cascade {
                inner: InnerStrategy::Ids,
            };
            ms = ms.with_options(solve_opts);
        }
        Self { inner: ms }
    }

    /// Cascade: DFS followed by entropy refinement. Pass ``options`` to set
    /// restarts, lookahead, etc.
    #[staticmethod]
    #[pyo3(signature = (options = None))]
    fn cascade_dfs(options: Option<&PySolveOptions>) -> Self {
        let mut ms = MoveSearch::cascade(InnerStrategy::Dfs);
        if let Some(opts) = options {
            let mut solve_opts = opts.inner.clone();
            solve_opts.strategy = Strategy::Cascade {
                inner: InnerStrategy::Dfs,
            };
            ms = ms.with_options(solve_opts);
        }
        Self { inner: ms }
    }

    /// Cascade: entropy followed by a second entropy pass. Pass ``options``
    /// to set restarts, lookahead, etc.
    #[staticmethod]
    #[pyo3(signature = (options = None, entropy_options = None))]
    fn cascade_entropy(
        options: Option<&PySolveOptions>,
        entropy_options: Option<&PyEntropyOptions>,
    ) -> Self {
        let mut ms = MoveSearch::cascade(InnerStrategy::Entropy);
        if let Some(opts) = options {
            let mut solve_opts = opts.inner.clone();
            solve_opts.strategy = Strategy::Cascade {
                inner: InnerStrategy::Entropy,
            };
            ms = ms.with_options(solve_opts);
        }
        if let Some(eopts) = entropy_options {
            ms = ms.with_entropy_options(eopts.inner.clone());
        }
        Self { inner: ms }
    }

    /// Return a copy with replaced ``SolveOptions``.
    fn with_options(&self, options: &PySolveOptions) -> Self {
        Self {
            inner: self.inner.clone().with_options(options.inner.clone()),
        }
    }

    /// Return a copy with replaced ``EntropyOptions``.
    fn with_entropy_options(&self, entropy_options: &PyEntropyOptions) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_entropy_options(entropy_options.inner.clone()),
        }
    }

    #[getter]
    fn strategy(&self) -> PySearchStrategy {
        PySearchStrategy::from_rs(&self.inner.options.strategy)
    }

    /// Whether the carried ``SolveOptions`` requests mirrored solving.
    ///
    /// ``MoveSearch`` deliberately exposes no ``options`` property, so this
    /// getter is the only way to read the flag back off a built search.
    #[getter]
    fn backwards_search(&self) -> bool {
        self.inner.options.backwards_search
    }

    fn __repr__(&self) -> String {
        format!(
            "MoveSearch(strategy={})",
            PySearchStrategy::from_rs(&self.inner.options.strategy).name()
        )
    }
}

/// Single-target fixed-placement solver.
///
/// Takes a ``SearchEngine`` (holds the lane index) and a ``MoveSearch``
/// (holds the strategy + options). Call ``solve()`` to route atoms from
/// an initial to a target configuration.
#[pyclass(
    name = "TargetSolver",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyTargetSolver {
    inner: TargetSolver,
}

#[pymethods]
impl PyTargetSolver {
    #[new]
    fn new(engine: &PySearchEngine, search: &PyMoveSearch) -> Self {
        Self {
            inner: TargetSolver::new(engine.inner.clone(), search.inner.clone()),
        }
    }

    /// Solve a fixed-target routing problem.
    ///
    /// Args:
    ///     initial: Mapping of qubit_id to LocationAddress for starting positions.
    ///     target: Mapping of qubit_id to LocationAddress for desired positions.
    ///     blocked: List of immovable obstacle locations.
    ///     max_expansions: Optional node expansion budget.
    ///
    /// Returns:
    ///     ``SolveResult`` with status, move layers, and search statistics.
    #[pyo3(signature = (initial, target, blocked, max_expansions=None))]
    fn solve(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        target: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
    ) -> PyResult<PySolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let target_pairs: Vec<(u32, LocationAddr)> =
            target.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();

        let result = py
            .detach(|| {
                self.inner
                    .solve(initial_pairs, target_pairs, blocked_locs, max_expansions)
            })
            .map_err(|e| crate::errors::config_error_to_py(py, &e))?;

        Ok(PySolveResult { inner: result })
    }

    fn __repr__(&self) -> &'static str {
        "TargetSolver(...)"
    }
}

/// CZ placement via a target generator + single-heuristic routing.
///
/// Generates candidate target configurations with ``DefaultTargetGenerator``,
/// validates each, then routes from ``initial`` to the first valid candidate
/// within the shared expansion budget.
#[pyclass(
    name = "SingleHeuristicCzPlacement",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PySingleHeuristicCzPlacement {
    inner: SingleHeuristicCzPlacement,
}

#[pymethods]
impl PySingleHeuristicCzPlacement {
    /// Build from an existing ``TargetSolver`` (uses ``DefaultTargetGenerator``).
    #[new]
    fn new(solver: &PyTargetSolver) -> Self {
        Self {
            inner: SingleHeuristicCzPlacement::new(
                TargetSolver::new(solver.inner.engine().clone(), solver.inner.search().clone()),
                Box::new(DefaultTargetGenerator),
            ),
        }
    }

    /// Place and route one CZ stage.
    ///
    /// ``pairs`` are the stage's ``(control, target)`` CZ pairs;
    /// ``future_layers`` are later stages, nearest first, for placements that
    /// look ahead.
    #[pyo3(signature = (initial, pairs, blocked, max_expansions=None, future_layers=None))]
    fn place(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        pairs: Vec<(u32, u32)>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        future_layers: Option<Vec<Vec<(u32, u32)>>>,
    ) -> PyResult<PyPlacementResult> {
        place_stage(
            py,
            &self.inner,
            &initial,
            &pairs,
            &blocked,
            max_expansions,
            future_layers,
        )
    }

    fn __repr__(&self) -> &'static str {
        "SingleHeuristicCzPlacement(...)"
    }
}

/// CZ placement via loose-goal routing.
///
/// Simultaneously discovers the entangling placement and the routing path
/// using ``EntanglingConstraintGoal``. Faster than the two-phase heuristic
/// approach for small atom counts; may need ``solve_pairs`` when future-layer
/// lookahead is required.
#[pyclass(
    name = "LooseGoalCzPlacement",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyLooseGoalCzPlacement {
    inner: LooseGoalCzPlacement,
}

#[pymethods]
impl PyLooseGoalCzPlacement {
    #[new]
    #[pyo3(signature = (engine, search, entangling_options=None))]
    fn new(
        engine: &PySearchEngine,
        search: &PyMoveSearch,
        entangling_options: Option<&PyEntanglingOptions>,
    ) -> Self {
        let ent_opts = entangling_options
            .map(|o| o.inner.clone())
            .unwrap_or_default();
        Self {
            inner: LooseGoalCzPlacement::new(engine.inner.clone(), search.inner.clone(), ent_opts),
        }
    }

    /// Place and route one CZ stage.
    ///
    /// ``pairs`` are the stage's ``(control, target)`` CZ pairs;
    /// ``future_layers`` are later stages, nearest first, for placements that
    /// look ahead.
    #[pyo3(signature = (initial, pairs, blocked, max_expansions=None, future_layers=None))]
    fn place(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        pairs: Vec<(u32, u32)>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        future_layers: Option<Vec<Vec<(u32, u32)>>>,
    ) -> PyResult<PyPlacementResult> {
        place_stage(
            py,
            &self.inner,
            &initial,
            &pairs,
            &blocked,
            max_expansions,
            future_layers,
        )
    }

    fn __repr__(&self) -> &'static str {
        "LooseGoalCzPlacement(...)"
    }
}

/// CZ placement via receding-horizon (MPC-style) loose-goal routing.
///
/// Generates K diverse Hungarian assignments per stage, rolls each out for
/// ``rollout_horizon`` layers, commits the winning branch, and re-plans.
/// Suited for high-occupancy regimes where baseline loose-goal
/// under-exploits parallelism.
#[pyclass(
    name = "RecedingHorizonCzPlacement",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyRecedingHorizonCzPlacement {
    inner: RecedingHorizonCzPlacement,
}

#[pymethods]
impl PyRecedingHorizonCzPlacement {
    #[new]
    #[pyo3(signature = (engine, search, entangling_options=None, rh_options=None))]
    fn new(
        engine: &PySearchEngine,
        search: &PyMoveSearch,
        entangling_options: Option<&PyEntanglingOptions>,
        rh_options: Option<&PyRecedingHorizonOptions>,
    ) -> Self {
        let ent_opts = entangling_options
            .map(|o| o.inner.clone())
            .unwrap_or_default();
        let rh_opts = rh_options.map(|o| o.inner.clone()).unwrap_or_default();
        Self {
            inner: RecedingHorizonCzPlacement::new(
                engine.inner.clone(),
                search.inner.clone(),
                ent_opts,
                rh_opts,
            ),
        }
    }

    /// Place and route one CZ stage.
    ///
    /// ``pairs`` are the stage's ``(control, target)`` CZ pairs;
    /// ``future_layers`` are later stages, nearest first, for placements that
    /// look ahead.
    #[pyo3(signature = (initial, pairs, blocked, max_expansions=None, future_layers=None))]
    fn place(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        pairs: Vec<(u32, u32)>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        future_layers: Option<Vec<Vec<(u32, u32)>>>,
    ) -> PyResult<PyPlacementResult> {
        place_stage(
            py,
            &self.inner,
            &initial,
            &pairs,
            &blocked,
            max_expansions,
            future_layers,
        )
    }

    fn __repr__(&self) -> &'static str {
        "RecedingHorizonCzPlacement(...)"
    }
}

/// Two-phase no-home CZ placement.
///
/// Phase 1 assigns displaced qubits to optimal home sites.
/// Phase 2 routes from home to CZ-staging using loose-goal search.
#[pyclass(
    name = "NoHomeCzPlacement",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyNoHomeCzPlacement {
    inner: NoHomeCzPlacement,
}

#[pymethods]
impl PyNoHomeCzPlacement {
    #[new]
    #[pyo3(signature = (engine, search, nohome_options=None))]
    fn new(
        engine: &PySearchEngine,
        search: &PyMoveSearch,
        nohome_options: Option<&PyNoHomeOptions>,
    ) -> Self {
        let nh_opts = nohome_options.map(|o| o.inner.clone()).unwrap_or_default();
        Self {
            inner: NoHomeCzPlacement::new(engine.inner.clone(), search.inner.clone(), nh_opts),
        }
    }

    /// Place and route one CZ stage.
    ///
    /// ``pairs`` are the stage's ``(control, target)`` CZ pairs;
    /// ``future_layers`` are later stages, nearest first, for placements that
    /// look ahead.
    #[pyo3(signature = (initial, pairs, blocked, max_expansions=None, future_layers=None))]
    fn place(
        &self,
        py: Python<'_>,
        initial: std::collections::BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        pairs: Vec<(u32, u32)>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        max_expansions: Option<u32>,
        future_layers: Option<Vec<Vec<(u32, u32)>>>,
    ) -> PyResult<PyPlacementResult> {
        place_stage(
            py,
            &self.inner,
            &initial,
            &pairs,
            &blocked,
            max_expansions,
            future_layers,
        )
    }

    fn __repr__(&self) -> &'static str {
        "NoHomeCzPlacement(...)"
    }
}
