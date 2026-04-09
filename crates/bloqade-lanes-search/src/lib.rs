//! Search infrastructure for atom move synthesis.
//!
//! Provides a compact configuration representation, arena-based search graph
//! with transposition table, and multiple search strategies (A*, IDS, Cascade).

pub mod astar;
pub mod config;
pub mod expander;
pub mod frontier;
pub mod graph;
pub mod heuristic;
pub mod heuristic_expander;
pub mod lane_index;
pub mod solve;

pub use astar::{Expander, SearchResult, astar};
pub use config::Config;
pub use graph::{MoveSet, NodeId, SearchGraph};
pub use heuristic_expander::{DeadlockPolicy, FreeRiderPolicy};
pub use lane_index::LaneIndex;
pub use solve::Strategy;
