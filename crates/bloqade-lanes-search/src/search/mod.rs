//! High-level search facade.
//!
//! - [`result`] — `SolveResult` / `SolveStatus` / `CandidateAttempt` /
//!   `MultiSolveResult` and their constructors.
//! - [`options`] — `Strategy` / `InnerStrategy`, `SolveOptions` /
//!   `EntropyOptions` / `EntanglingOptions`, with helpers
//!   (`upgraded_for_entangling`, `clipped_future_layers`).
//! - [`restarts`] — `run_with_components`, `pick_best`, `extract`
//!   (strategy dispatch + restart orchestration).
//! - [`move_search`] — `MoveSearch` composition layer.
//! - [`target_solver`] — `TargetSolver` (single-candidate solver wrapping
//!   `SearchEngine` + `MoveSearch`).

pub mod engine;
pub mod move_search;
pub mod options;
pub mod restarts;
pub mod result;
pub mod target_solver;
