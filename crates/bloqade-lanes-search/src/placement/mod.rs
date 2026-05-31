//! CZ-placement strategies.
//!
//! The three (or more, with the DSL) ways to turn a (controls, targets)
//! CZ-pair specification into a routed solve:
//!
//! - [`target_generator`] — fixed-target plugin trait + `DefaultTargetGenerator`.
//!   Consumed by the upcoming `SingleHeuristicCzPlacement`, which composes
//!   a `TargetGenerator` with a `TargetSolver`.
//! - [`receding_horizon`] — MPC-style outer loop on top of loose-goal IDS
//!   rollouts + Hungarian compass.
//! - [`nohome`] — Hungarian post-pass to assign return locations after the
//!   main route completes.
//!
//! Currently the `MoveSolver::solve_entangling`, `solve_entangling_rh`,
//! and `solve_nohome` entry points (in [`crate::search::solve`]) dispatch
//! to these helpers. The §6 type split will lift each into its own
//! `CzPlacement` struct (`SingleHeuristicCzPlacement`,
//! `LooseGoalCzPlacement`, `RecedingHorizonCzPlacement`, plus
//! `NoHomeCzPlacement`) sharing a `solve(initial, controls, targets, …)`
//! interface.

pub mod nohome;
pub mod receding_horizon;
pub mod target_generator;
