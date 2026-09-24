//! The `CzPlacement` outermost trait.
//!
//! Every CZ-stage placement strategy (single-heuristic, loose-goal,
//! receding-horizon, no-home, future DSL-driven peers) implements
//! [`CzPlacement::place`] with one signature: take a [`CzStage`] and a
//! [`PlacementBudget`], and return a [`PlacementResult`].
//!
//! The internal composition differs per implementor — `SingleHeuristic`
//! composes a [`TargetSolver`](crate::search::target_solver::TargetSolver)
//! with a [`TargetGenerator`](crate::placement::target_generator::TargetGenerator);
//! `LooseGoal` drives a [`MoveSearch`](crate::search::move_search::MoveSearch)
//! directly against an `EntanglingConstraintGoal`; `RecedingHorizon` and
//! `NoHome` compose with their own options bundles. The trait is the
//! seam that hides those differences behind one call. It stays coarse — one
//! call per CZ stage — because the implementations' internal loops
//! (commit-and-replan, a set-valued goal, one Hungarian assignment) do not
//! share a finer pipeline.

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::primitives::config::ConfigError;
use crate::search::result::{SolveResult, SolveStatus};

/// One CZ stage: where the atoms are, which pairs must end up entangled, and
/// what else is in the way.
///
/// Uses only the address vocabulary; the architecture comes from the
/// placement's own engine.
#[derive(Debug, Clone, Copy)]
pub struct CzStage<'a> {
    /// Starting qubit positions: `(qubit_id, location)`.
    pub initial: &'a [(u32, LocationAddr)],
    /// The stage's CZ pairs, `(control, target)`.
    pub pairs: &'a [(u32, u32)],
    /// Locations held by external atoms, which are immovable obstacles.
    pub blocked: &'a [LocationAddr],
    /// Later CZ stages, nearest first, for placements that look ahead.
    /// Empty means no lookahead.
    pub future_layers: &'a [Vec<(u32, u32)>],
}

impl<'a> CzStage<'a> {
    /// A stage with no lookahead.
    pub fn new(
        initial: &'a [(u32, LocationAddr)],
        pairs: &'a [(u32, u32)],
        blocked: &'a [LocationAddr],
    ) -> Self {
        Self {
            initial,
            pairs,
            blocked,
            future_layers: &[],
        }
    }

    /// Look ahead over `future_layers`, nearest first.
    pub fn with_future_layers(mut self, future_layers: &'a [Vec<(u32, u32)>]) -> Self {
        self.future_layers = future_layers;
        self
    }
}

/// How much work one [`CzPlacement::place`] call may do.
///
/// Holds only an expansion cap for now; `#[non_exhaustive]` so that further
/// dimensions (an evaluation cap, a per-call total) can be added without
/// changing the trait. Each implementation documents the scope its cap
/// applies to, since they differ.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlacementBudget {
    /// Cap on search node expansions; `None` is unlimited.
    pub max_expansions: Option<u32>,
}

impl PlacementBudget {
    /// A budget of at most `max_expansions` expansions (`None`: unlimited).
    pub fn new(max_expansions: Option<u32>) -> Self {
        Self { max_expansions }
    }
}

/// One candidate a placement tried, in order.
#[derive(Debug, Clone)]
pub struct CandidateAttempt {
    /// Index of the candidate in the order the placement generated them.
    pub candidate_index: usize,
    /// How routing that candidate ended.
    pub status: SolveStatus,
    /// Nodes expanded routing it.
    pub nodes_expanded: u32,
    /// The score a candidate evaluator gave it, where one ranked the
    /// candidates before routing. `None` when nothing ranked them.
    pub score: Option<f64>,
}

/// The outcome of placing one CZ stage.
#[derive(Debug)]
pub struct PlacementResult {
    /// The routing result. On success its `goal_config` is the chosen
    /// placement; on failure it is the stage's starting configuration.
    pub result: SolveResult,
    /// Which candidate won, for placements that enumerate candidates;
    /// `None` when none won or the placement does not enumerate them.
    pub chosen: Option<usize>,
    /// Every candidate tried, in order. Empty for placements that do not
    /// enumerate candidates.
    pub attempts: Vec<CandidateAttempt>,
    /// Expansions across every leg and candidate of the placement.
    pub total_expansions: u32,
}

impl PlacementResult {
    /// The result of a placement that routes once rather than choosing
    /// among candidates.
    pub fn single(result: SolveResult) -> Self {
        Self {
            total_expansions: result.nodes_expanded,
            result,
            chosen: None,
            attempts: Vec::new(),
        }
    }

    /// Candidates actually routed (validation failures are not counted).
    pub fn candidates_tried(&self) -> usize {
        self.attempts.len()
    }
}

/// Uniform interface for CZ-stage placement strategies.
///
/// The placement object owns the architecture, the search configuration and
/// any strategy-specific options; `place` takes the per-stage problem.
pub trait CzPlacement {
    /// Place and route one CZ stage.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `stage.initial` is not a valid
    /// configuration (for example, duplicate qubit IDs).
    fn place(
        &self,
        stage: &CzStage<'_>,
        budget: &PlacementBudget,
    ) -> Result<PlacementResult, ConfigError>;
}
