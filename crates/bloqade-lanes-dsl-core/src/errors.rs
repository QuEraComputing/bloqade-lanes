//! `DslError` — error type surfaced from policy load and execution.

use thiserror::Error;

/// Errors raised by the DSL adapter or kernel.
///
/// These map 1:1 to the status codes documented in the spec §5.10
/// (Move DSL error model). The hosting kernel (e.g.
/// `move_policy_dsl::kernel`) is responsible for converting these into
/// public `SolveResult` statuses.
#[derive(Debug, Error)]
pub enum DslError {
    /// `.star` file failed to parse.
    #[error("{path}: parse error: {message}")]
    Parse { path: String, message: String },

    /// Starlark runtime error during `init`/`step`/`generate`.
    #[error("starlark runtime error:\n{traceback}")]
    Runtime { traceback: String },

    /// `update_node_state` / `update_global_state` named a field not in
    /// the declared schema.
    #[error("schema error on field `{field}`: {message}")]
    Schema { field: String, message: String },

    /// `step()` returned something that wasn't an `Action` or
    /// `list[Action]`.
    #[error("policy returned an invalid value: {0}")]
    BadPolicy(String),

    /// Per-`step()` Starlark step budget exceeded.
    #[error("starlark step budget exceeded")]
    StepBudget,

    /// Per-solve Starlark memory cap exceeded.
    #[error("starlark memory cap exceeded")]
    MemoryBudget,

    /// Wrapper over arbitrary IO errors (e.g. opening the .star file).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_displays_with_location() {
        let err = DslError::Parse {
            path: "foo.star".into(),
            message: "unexpected EOF".into(),
        };
        let s = format!("{err}");
        assert!(s.contains("foo.star"), "missing path in display: {s}");
        assert!(s.contains("unexpected EOF"));
    }

    #[test]
    fn variants_round_trip_through_display() {
        let cases = [
            DslError::Runtime {
                traceback: "x".into(),
            },
            DslError::Schema {
                field: "entropy".into(),
                message: "type".into(),
            },
            DslError::BadPolicy("not an action".into()),
            DslError::StepBudget,
            DslError::MemoryBudget,
        ];
        for c in &cases {
            let _ = format!("{c}"); // must not panic
        }
    }
}
