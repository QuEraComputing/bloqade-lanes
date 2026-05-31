//! High-level search facade.
//!
//! The pre-split `solve.rs` god module currently lives here under its
//! original filename; it will dissolve into focused siblings during the
//! §6 type split:
//!
//! - `move_search.rs` — the `MoveSearch` strategy + options + observer
//!   composition.
//! - `target_solver.rs` — `TargetSolver`: `solve(initial, target,
//!   blocked) -> moves`.
//! - `result.rs` — `SolveResult` / `SolveStatus` and their constructors.
//! - `options.rs` — `SolveOptions` / `EntropyOptions` /
//!   `EntanglingOptions` and the consolidated helpers
//!   (`upgraded_for_entangling`, `clipped_future_layers`).
//! - `restarts.rs` — `run_with_components`, `pick_best`, `extract`.

pub mod solve;
