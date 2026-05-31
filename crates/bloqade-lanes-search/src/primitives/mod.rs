//! Shared primitive types reused by every search driver.
//!
//! Stateless data structures (`Config`, `SearchGraph`, `MoveSet`,
//! `LaneIndex`, `DistanceTable`) plus per-search context types
//! (`SearchContext`, `SearchState`, `MoveCandidate`) and the
//! deterministic tie-break comparators (`ordering`).

pub mod config;
pub mod context;
pub mod distance;
pub mod graph;
pub mod lane_index;
pub(crate) mod ordering;
