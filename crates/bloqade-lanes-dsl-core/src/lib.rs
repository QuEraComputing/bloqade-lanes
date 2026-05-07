//! Shared substrate for Starlark-hosted policy DSLs.
//!
//! Hosts the starlark-rust adapter, sandbox configuration, and shared
//! primitives (`LocationAddress`, `LaneAddr`, `Config`, `MoveSet`,
//! `ArchSpec` wrapper, utilities) used by both the Move Policy DSL and
//! the Target Generator DSL.

pub mod adapter;
pub mod errors;
pub mod policy_trait;
pub mod primitives;
pub mod sandbox;

pub use errors::DslError;
pub use policy_trait::{Policy, StepResult};
