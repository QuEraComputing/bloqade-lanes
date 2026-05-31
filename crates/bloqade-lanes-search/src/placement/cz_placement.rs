//! The `CzPlacement` outermost trait.
//!
//! Every CZ-stage placement strategy (single-heuristic, loose-goal,
//! receding-horizon, no-home, future DSL-driven peers) implements
//! [`CzPlacement::solve`] with a uniform signature: take an
//! `(initial, controls, targets, blocked)` problem and return a
//! [`SolveResult`].
//!
//! The internal composition differs per implementor — `SingleHeuristic`
//! composes a [`TargetSolver`](crate::search::target_solver::TargetSolver)
//! with a [`TargetGenerator`](crate::placement::target_generator::TargetGenerator);
//! `LooseGoal` drives a [`MoveSearch`](crate::search::move_search::MoveSearch)
//! directly against an `EntanglingConstraintGoal`; the upcoming
//! receding-horizon and no-home variants compose with their own
//! options bundles. The trait is the user-facing seam that hides
//! the composition differences behind one call site.
//!
//! Implementors with richer outcome data (e.g. `SingleHeuristic`'s
//! per-candidate attempt list) expose those on the concrete type via
//! `solve_with_attempts` or similar — `CzPlacement::solve` returns
//! the trimmed-to-a-single-`SolveResult` view.

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::primitives::config::ConfigError;
use crate::search::result::SolveResult;

/// Uniform interface for CZ-stage placement strategies.
///
/// `solve` takes the per-call problem (positions + CZ pair lists +
/// blocked locations + budget); the placement object itself owns the
/// arch, search algorithm, and any strategy-specific options.
pub trait CzPlacement {
    /// Solve a CZ-stage placement.
    ///
    /// # Arguments
    ///
    /// * `initial` — Starting qubit positions: `(qubit_id, location)` pairs.
    /// * `controls` — Control-qubit IDs of the CZ pairs at this layer.
    /// * `targets` — Target-qubit IDs of the CZ pairs at this layer.
    ///   `controls[i]` and `targets[i]` are partnered for the same CZ.
    /// * `blocked` — Locations occupied by external atoms (immovable
    ///   obstacles).
    /// * `max_expansions` — Optional limit on node expansions.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError`] if `initial` contains duplicate qubit IDs.
    fn solve(
        &self,
        initial: &[(u32, LocationAddr)],
        controls: &[u32],
        targets: &[u32],
        blocked: &[LocationAddr],
        max_expansions: Option<u32>,
    ) -> Result<SolveResult, ConfigError>;
}
