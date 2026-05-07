//! Move Policy DSL: kernel + actions + handles for Starlark policies.

pub mod actions;
pub mod adapter_impl;
pub(super) mod builtins;
pub mod graph_handle;
pub mod kernel;
pub mod lib_move;
pub mod observer;

pub use kernel::{PolicyOptions, PolicyResult, PolicyStatus, solve_with_policy};
pub use observer::{
    GraphDelta, JsonMoveTraceObserver, MoveKernelObserver, NoOpMoveObserver, PolicyGraphSnapshot,
};
