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
//! - [`verify`] — canonical execution-model replay applied to every packaged
//!   plan, so an inexecutable move set fails at its source (issue #866).

pub mod engine;
pub mod move_search;
pub mod options;
pub mod restarts;
pub mod result;
pub mod target_solver;
pub(crate) mod verify;
