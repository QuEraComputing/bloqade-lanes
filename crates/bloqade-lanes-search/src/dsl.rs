//! DSL umbrella module.
//!
//! Holds every Starlark Move Policy / Target Generator DSL submodule plus
//! the shared fixture loader and the AOD candidate pipeline used by the
//! DSL kernel. Kept under a single `pub mod dsl;` declaration in `lib.rs`
//! so adding new DSL modules does not require editing the crate root.

pub mod fixture;
pub mod move_policy_dsl;
pub mod pipeline;
pub mod target_generator_dsl;
