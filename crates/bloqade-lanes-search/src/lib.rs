//! Search infrastructure for atom move synthesis.
//!
//! Provides a compact configuration representation, arena-based search graph
//! with transposition table, and multiple search strategies (A*, IDS, Cascade).

pub(crate) mod aod_grid;
pub mod astar;
pub mod config;
pub mod context;
pub mod cost;

pub mod entropy;
pub mod frontier;
pub mod generators;
pub mod goals;
pub mod graph;
pub mod heuristic;
pub mod heuristics;

pub mod lane_index;
pub mod move_policy_dsl;
pub mod observer;
pub(crate) mod ordering;
pub mod scorers;
pub mod solve;
pub mod target_generator;
pub mod target_generator_dsl;
#[cfg(test)]
pub(crate) mod test_utils;
pub mod traits;

pub use astar::SearchResult;
pub use config::{Config, ConfigError};
pub use context::{MoveCandidate, SearchContext, SearchState};
pub use cost::UniformCost;
pub use generators::{DeadlockPolicy, ExhaustiveGenerator, GreedyGenerator, HeuristicGenerator};
pub use goals::{AllAtTarget, PartialPlacementGoal};
pub use graph::{MoveSet, NodeId, SearchGraph};
pub use heuristics::{MaxHopHeuristic, SumHopHeuristic};
pub use lane_index::LaneIndex;
pub use move_policy_dsl::{PolicyOptions, PolicyResult, PolicyStatus};
pub use observer::{NoOpObserver, SearchEvent, SearchObserver};
pub use scorers::{DistanceScorer, EntropyScorer};
pub use solve::{CandidateAttempt, InnerStrategy, MultiSolveResult, SolveOptions, Strategy};
pub use target_generator::{
    CandidateError, DefaultTargetGenerator, TargetContext, TargetGenerator,
};
pub use traits::{CandidateScorer, CostFn, Goal, Heuristic, MoveGenerator};
