//! Search infrastructure for atom move synthesis.
//!
//! Provides a compact configuration representation, arena-based search graph
//! with transposition table, and multiple search strategies (A*, IDS, Cascade).
//!
//! Module layout (post §6 refactor):
//!
//! - [`primitives`] — `Config`, `SearchGraph`, `SearchContext`, `LaneIndex`,
//!   `DistanceTable`, `MoveSet` — data types reused across every search driver.
//! - [`ops`] — stateless math / arch ops: AOD-grid building (`aod_grid`),
//!   Hungarian assignment + word-pair distances (`entangling`).
//! - [`drivers`] — the search-loop engines: frontier-based (`frontier`),
//!   entropy-guided (`entropy`).
//! - [`search`] — `MoveSearch` facade + result/options types: canonical
//!   `result` / `options` / `restarts` / `engine` / `move_search` /
//!   `target_solver` submodules (no shim layer).
//! - [`placement`] — CZ-placement strategies: `target_generator`,
//!   `nohome`, `receding_horizon` (and forthcoming `single_heuristic` /
//!   `loose_goal` peers).
//! - [`dsl`] — Starlark policy DSL sidecar (Move / Target).
//! - Top-level small modules (`cost`, `goals`, `heuristics`, `scorers`,
//!   `generators`, `observer`, `traits`) — too small to warrant subdirs.

pub mod cost;
pub mod drivers;
pub mod dsl;
pub mod generators;
pub mod goals;
pub mod heuristics;
pub mod observer;
pub mod ops;
pub mod placement;
pub mod primitives;
pub mod scorers;
pub mod search;
#[cfg(test)]
pub(crate) mod test_utils;
pub mod traits;

pub use cost::UniformCost;
pub use drivers::result::SearchResult;
pub use generators::{
    DeadlockPolicy, ExhaustiveGenerator, GreedyGenerator, HeuristicGenerator, LooseTargetGenerator,
};
pub use goals::{AllAtTarget, EntanglingConstraintGoal, PartialPlacementGoal};
pub use heuristics::{MaxHopHeuristic, SumHopHeuristic};
pub use observer::{NoOpObserver, SearchEvent, SearchObserver};
pub use placement::cz_placement::CzPlacement;
pub use placement::loose_goal::LooseGoalCzPlacement;
pub use placement::nohome::NoHomeCzPlacement;
pub use placement::receding_horizon::{
    RecedingHorizonCzPlacement, RecedingHorizonOptions, default_weight_grid,
};
pub use placement::single_heuristic::SingleHeuristicCzPlacement;
pub use placement::target_generator::{
    CandidateError, DefaultTargetGenerator, TargetContext, TargetGenerator,
};
pub use primitives::config::{Config, ConfigError};
pub use primitives::context::{MoveCandidate, SearchContext, SearchState};
pub use primitives::distance::PairDistanceHeuristic;
pub use primitives::graph::{MoveSet, NodeId, SearchGraph};
pub use primitives::lane_index::LaneIndex;
pub use scorers::{DistanceScorer, EntropyScorer};
pub use search::engine::SearchEngine;
pub use search::move_search::MoveSearch;
pub use search::options::{InnerStrategy, SolveOptions, Strategy};
pub use search::result::{CandidateAttempt, MultiSolveResult};
pub use search::target_solver::TargetSolver;
pub use traits::{CandidateScorer, CostFn, Goal, Heuristic, MoveGenerator};
