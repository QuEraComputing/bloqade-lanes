//! Stateless math and arch operations: AOD-grid building, Hungarian
//! assignment + word-pair distance precomputes.
//!
//! These are *primitives over `LaneIndex`* rather than search drivers —
//! they consume an architecture and return derived data without
//! maintaining per-search state.

pub(crate) mod aod_grid;
pub mod entangling;
