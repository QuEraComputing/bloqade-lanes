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
//! Each driver also exposes a free `solve_*` function ([`loose_goal::solve_loose_goal`],
//! [`receding_horizon::solve_receding_horizon`], [`nohome::solve_nohome`],
//! [`single_heuristic::solve_single_heuristic`]) sharing the same search core.

pub mod cz_placement;
pub mod loose_goal;
pub mod nohome;
pub mod receding_horizon;
pub mod single_heuristic;
pub mod target_generator;
