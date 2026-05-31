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
//!   `CandidateAttempt` / `MultiSolveResult` until the §6 type split
//!   lifts `solve_with_generator` / `solve_entangling` /
//!   `solve_entangling_rh` / `solve_nohome` into their own
//!   `placement::*Cz*Placement` peers.
//!
//! Forthcoming siblings (per §6 sequencing): `move_search.rs`
//! (`MoveSearch` composition) and `target_solver.rs` (`TargetSolver`).

pub mod engine;
pub mod options;
pub mod restarts;
pub mod result;
pub mod solve;
