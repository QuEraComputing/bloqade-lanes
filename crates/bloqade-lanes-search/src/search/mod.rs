//! High-level search facade.
//!
//! Slice-1 split of the pre-split `solve.rs` god module:
//!
//! - [`result`] — `SolveResult` / `SolveStatus` and their constructors.
//! - [`options`] — `Strategy` / `InnerStrategy`, `SolveOptions` /
//!   `EntropyOptions` / `EntanglingOptions`, with helpers
//!   (`upgraded_for_entangling`, `clipped_future_layers`).
//! - [`restarts`] — `run_with_components`, `pick_best`, `extract`
//!   (strategy dispatch + restart orchestration).
//! - [`solve`] — `MoveSolver` (the legacy facade) plus
//!   `CandidateAttempt` / `MultiSolveResult`, kept until the Rust-side
//!   cleanup removes them in favour of the `placement::*CzPlacement` peers.
//! - [`move_search`] — `MoveSearch` composition layer.
//! - [`target_solver`] — `TargetSolver` (single-candidate solver wrapping
//!   `SearchEngine` + `MoveSearch`).

pub mod engine;
pub mod move_search;
pub mod options;
pub mod restarts;
pub mod result;
pub mod solve;
pub mod target_solver;
