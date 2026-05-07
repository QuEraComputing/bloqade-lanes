//! Sandbox configuration for Starlark evaluators.
//!
//! Disables non-deterministic / IO / network builtins, caps per-tick
//! step count, and caps per-solve heap usage. Disables `load()` to
//! prevent silent file-system access: each policy is a single
//! self-contained `.star` file.

use starlark::environment::{Globals, GlobalsBuilder, Module};
use starlark::eval::Evaluator;

/// Tunable resource limits applied to every policy evaluator.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Per-`step()` Starlark instruction budget. Default 1_000_000.
    pub starlark_steps_per_tick: u64,
    /// Per-solve Starlark heap cap, in bytes. Default 128 MiB.
    pub starlark_memory_per_solve_bytes: u64,
    /// Max number of `.star` files that may be loaded. We disable
    /// `load()` outright; this is set to 1 (the policy file).
    pub max_loaded_files: u32,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            starlark_steps_per_tick: 1_000_000,
            starlark_memory_per_solve_bytes: 128 * 1024 * 1024,
            max_loaded_files: 1,
        }
    }
}

/// Build the `Globals` for a sandboxed evaluator.
///
/// Adds nothing beyond Starlark's standard library minus IO/time. The
/// caller (DSL-specific glue) extends this with `lib`, `actions`, etc.
pub fn build_globals(_cfg: &SandboxConfig) -> Globals {
    GlobalsBuilder::standard()
        // Standard Starlark already excludes time/IO/network. Recursion
        // is forbidden by the language spec. `load()` is rejected by
        // not registering a `FileLoader` on the evaluator (see
        // `make_evaluator` below).
        .with(crate::primitives::utilities::register_utilities)
        .build()
}

/// Create an `Evaluator` bound to `module`, using `globals`, with all
/// sandbox limits applied.
///
/// `load()` is rejected because no `FileLoader` is registered on the
/// `Evaluator`; starlark-rust 0.13 rejects `load()` by default when no
/// loader is present.
///
/// # Caps wired in starlark-0.13
/// - `set_max_callstack_size(64)` — bounds recursion depth.
/// - Per-tick step budget and per-solve heap cap: starlark-0.13 does not
///   expose `set_step_limit` or `set_heap_limit` on `Evaluator`, so those
///   caps are tracked at the kernel level instead.
///   TODO(0.13): wire step + memory caps if a future starlark release
///   exposes `Evaluator::set_step_limit` / `Evaluator::set_heap_limit`.
pub fn make_evaluator<'a, 'v>(
    module: &'v Module,
    globals: &'a Globals,
    cfg: &SandboxConfig,
) -> Evaluator<'v, 'a, 'a> {
    // `globals` is passed in but not bound to the evaluator here; the
    // caller passes it to `eval_module(ast, &globals)` at call time.
    let _ = globals;
    let _ = cfg;
    let mut eval = Evaluator::new(module);
    // Cap recursion depth. `set_max_callstack_size` returns a Result in
    // some versions and () in others; `.ok()` silences both cases.
    let _ = eval.set_max_callstack_size(64);
    // No FileLoader is registered, so `load()` statements are rejected
    // by the starlark-rust runtime with an evaluation error.
    eval
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_finite_and_positive() {
        let cfg = SandboxConfig::default();
        assert!(cfg.starlark_steps_per_tick >= 10_000);
        assert!(cfg.starlark_memory_per_solve_bytes >= 16 * 1024 * 1024);
        assert_eq!(cfg.max_loaded_files, 1);
    }

    #[test]
    fn evaluator_rejects_load_statement() {
        let cfg = SandboxConfig::default();
        let module = starlark::environment::Module::new();
        let globals = build_globals(&cfg);
        let mut eval = make_evaluator(&module, &globals, &cfg);
        let src = r#"load("other.star", "x")"#;
        let ast = starlark::syntax::AstModule::parse(
            "test.star",
            src.to_owned(),
            &starlark::syntax::Dialect::Standard,
        )
        .expect("parse");
        let result = eval.eval_module(ast, &globals);
        assert!(result.is_err(), "load() must be rejected by sandbox");
    }
}
