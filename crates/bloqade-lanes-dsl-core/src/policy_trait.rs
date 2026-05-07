//! Generic `Policy` trait — the seam between the kernel and a Starlark-hosted
//! algorithm. The Move DSL kernel and (later) the Target DSL adapter each
//! supply an implementation backed by a `LoadedPolicy`.

use crate::errors::DslError;

/// Outcome of a single `step()` invocation.
pub enum StepResult<A> {
    /// Apply these actions, in order, then call `step()` again.
    Continue(Vec<A>),
    /// Halt with this status. Kernel finalises the result.
    Halt(HaltStatus),
}

/// Halt reasons surfaced by `step()`.
#[derive(Debug, Clone)]
pub enum HaltStatus {
    Solved,
    Unsolvable,
    BudgetExhausted,
    Fallback(String),
    Error(String),
}

/// A hosted algorithm. Type parameters keep this generic enough to host
/// future placement DSLs (§11 of the spec) without modification.
pub trait Policy {
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
