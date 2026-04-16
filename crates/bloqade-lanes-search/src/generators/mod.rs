//! Move generator wrappers that implement [`MoveGenerator`] by delegating
//! to the existing [`Expander`]-based implementations.

pub mod entropy;
pub mod exhaustive;
pub mod heuristic;

pub use entropy::EntropyGenerator;
pub use exhaustive::ExhaustiveGenerator;
pub use heuristic::{DeadlockPolicy, HeuristicGenerator};
