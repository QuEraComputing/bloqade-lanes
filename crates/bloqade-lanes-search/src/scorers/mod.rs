//! Candidate scoring strategies for search.

pub mod distance;
pub mod entropy;

pub use distance::DistanceScorer;
pub use entropy::EntropyScorer;
