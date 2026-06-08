//! `eval-policy` and `trace-policy` CLI subcommands.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §4.

pub mod eval;
pub mod output;
pub mod trace;

pub use eval::run_eval_policy;
pub use trace::run_trace_policy;
