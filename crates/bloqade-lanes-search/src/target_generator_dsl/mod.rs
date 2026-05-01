//! Target Generator DSL: Starlark-hosted adapter for `TargetGeneratorABC`.
//!
//! Plan B of #597. Uses the shared substrate in `bloqade-lanes-dsl-core`
//! (parser, sandbox, primitives) and reuses [`crate::target_generator::
//! validate_candidate`] for safety enforcement.

pub mod ctx_handle;
pub mod kernel;
pub mod lib_target;

pub use ctx_handle::StarlarkTargetContext;
pub use kernel::{TargetPolicyError, TargetPolicyRunner, run_target_policy};
pub use lib_target::StarlarkLibTarget;
