//! CZ-placement strategies.
//!
//! The three (or more, with the DSL) ways to turn a (controls, targets)
//! CZ-pair specification into a routed solve:
//!
//! - [`cz_placement`] — the [`CzPlacement`](cz_placement::CzPlacement) trait
//!   shared by all four strategies below.
//! - [`single_heuristic`] — [`SingleHeuristicCzPlacement`](single_heuristic::SingleHeuristicCzPlacement):
//!   composes a `TargetGenerator` with a `TargetSolver`.
//! - [`loose_goal`] — [`LooseGoalCzPlacement`](loose_goal::LooseGoalCzPlacement):
//!   drives `MoveSearch` directly against an entangling-constraint goal.
//! - [`receding_horizon`] — [`RecedingHorizonCzPlacement`](receding_horizon::RecedingHorizonCzPlacement):
//!   MPC-style outer loop on top of loose-goal IDS rollouts + Hungarian compass.
//! - [`nohome`] — [`NoHomeCzPlacement`](nohome::NoHomeCzPlacement):
//!   two-phase return assignment + entangling routing via Hungarian assignment.
//! - [`target_generator`] — fixed-target plugin trait + `DefaultTargetGenerator`,
//!   consumed by `SingleHeuristicCzPlacement`.
//!
//! The legacy `MoveSolver::solve_entangling`, `solve_entangling_rh`, and
//! `solve_nohome` entry points in [`crate::search::solve`] are kept until
//! the Rust-side cleanup (tracked in issue #706) removes them.

pub mod cz_placement;
pub mod loose_goal;
pub mod nohome;
pub mod receding_horizon;
pub mod single_heuristic;
pub mod target_generator;
