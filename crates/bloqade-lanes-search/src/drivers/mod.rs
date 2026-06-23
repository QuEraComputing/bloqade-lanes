//! Search-loop engines.
//!
//! Two driver families:
//!
//! - [`frontier`] — worklist-based loops: A*, BFS, DFS, Greedy, IDS share
//!   the [`frontier::run_search`] generic loop parameterized by a
//!   [`frontier::Frontier`] implementation.
//! - [`entropy`] — single-path DFS with entropy-based backtracking and a
//!   resume buffer. Has its own driver function ([`entropy::entropy_search`])
//!   because the revert-to-best-buffer mechanic doesn't fit the
//!   pop-expand-push frontier abstraction.
//!
//! Both consume the same primitive types (`Config`, `SearchGraph`,
//! `MoveSet`, `LaneIndex`, `DistanceTable`) and the same trait abstractions
//! (`Goal`, `MoveGenerator`, `CandidateScorer`, `CostFn`, `Heuristic`).
//! Observability flows through [`crate::observer::SearchObserver`] for
//! both families.
//!
//! [`result`] holds the shared [`result::SearchResult`] type returned by
//! both driver families.

pub mod entropy;
pub mod frontier;
pub mod result;
