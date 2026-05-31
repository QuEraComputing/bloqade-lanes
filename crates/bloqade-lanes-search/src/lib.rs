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
//! - [`search`] — `MoveSearch` facade + result/options types (currently
//!   the pre-split `solve.rs` god module; will dissolve into siblings
//!   during the type split).
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
pub use drivers::astar::SearchResult;
pub use generators::{
    DeadlockPolicy, ExhaustiveGenerator, GreedyGenerator, HeuristicGenerator, LooseTargetGenerator,
};
pub use goals::{AllAtTarget, EntanglingConstraintGoal, PartialPlacementGoal};
pub use heuristics::{MaxHopHeuristic, SumHopHeuristic};
pub use observer::{NoOpObserver, SearchEvent, SearchObserver};
pub use placement::receding_horizon::{RecedingHorizonOptions, default_weight_grid};
pub use placement::target_generator::{
    CandidateError, DefaultTargetGenerator, TargetContext, TargetGenerator,
};
pub use primitives::config::{Config, ConfigError};
pub use primitives::context::{MoveCandidate, SearchContext, SearchState};
pub use primitives::distance::PairDistanceHeuristic;
pub use primitives::graph::{MoveSet, NodeId, SearchGraph};
pub use primitives::lane_index::LaneIndex;
pub use scorers::{DistanceScorer, EntropyScorer};
pub use search::solve::{
    CandidateAttempt, InnerStrategy, MultiSolveResult, SolveOptions, Strategy,
};
pub use traits::{CandidateScorer, CostFn, Goal, Heuristic, MoveGenerator};
