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
pub mod scorers;
pub mod solve;
#[cfg(test)]
pub(crate) mod test_utils;
pub mod traits;

pub use astar::{Expander, SearchResult, astar};
pub use config::{Config, ConfigError};
pub use context::{MoveCandidate, SearchContext, SearchState};
pub use cost::UniformCost;
pub use generators::{DeadlockPolicy, ExhaustiveGenerator, HeuristicGenerator};
pub use goals::AllAtTarget;
pub use graph::{MoveSet, NodeId, SearchGraph};
pub use heuristics::{MaxHopHeuristic, SumHopHeuristic};
pub use lane_index::LaneIndex;
pub use scorers::DistanceScorer;
pub use solve::{InnerStrategy, SolveOptions, Strategy};
pub use traits::{CandidateScorer, CostFn, Goal, Heuristic, MoveGenerator};
