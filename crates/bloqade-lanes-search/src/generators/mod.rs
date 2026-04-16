//! Move generator wrappers that implement [`MoveGenerator`] by delegating
//! to the existing [`Expander`]-based implementations.

pub mod exhaustive;
pub mod heuristic;

pub use exhaustive::ExhaustiveGenerator;
pub use heuristic::HeuristicGenerator;
