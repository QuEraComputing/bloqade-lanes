//! Generic `Policy` trait — in-crate documentation of the seam between a
//! kernel and a Starlark-hosted algorithm.
//!
//! The Move DSL kernel and the Target DSL adapter each supply their own
//! private struct that satisfies this shape *without* the trait bound; the
//! trait is therefore presently unused. It is kept `pub(crate)` so the
//! shape stays documented and future kernels have a reference to conform
//! to. Promote back to `pub` (and wire `MovePolicy` / `TargetPolicy` to
//! implement it) once there are external consumers or a second in-crate
//! kernel that benefits from generic dispatch.
#![allow(dead_code)]

use crate::errors::DslError;

/// Outcome of a single `step()` invocation.
pub(crate) enum StepResult<A> {
    /// Apply these actions, in order, then call `step()` again.
    Continue(Vec<A>),
    /// Halt with this status. Kernel finalises the result.
    Halt(HaltStatus),
}

/// Halt reasons surfaced by `step()`.
#[derive(Debug, Clone)]
pub(crate) enum HaltStatus {
    Solved,
    Unsolvable,
    BudgetExhausted,
    Fallback(String),
    Error(String),
}

/// A hosted algorithm. Type parameters keep this generic enough to host
/// future placement DSLs (§11 of the spec) without modification.
pub(crate) trait Policy {
    type Handle;
    type Action;
    type InitArg;
    type NodeState;
    type GlobalState;

    /// One-time setup. Called after the kernel inserts the root node.
    fn init(&mut self, arg: Self::InitArg) -> Result<Self::GlobalState, DslError>;

    /// One tick. Returns actions to apply atomically, or a halt signal.
    fn step(
        &mut self,
        handle: &mut Self::Handle,
        gs: &mut Self::GlobalState,
    ) -> Result<StepResult<Self::Action>, DslError>;
}
