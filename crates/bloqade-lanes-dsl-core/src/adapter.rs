//! Starlark-rust adapter: parse, freeze, invoke named exports.

use std::path::Path;

use starlark::environment::{FrozenModule, Globals, Module};
use starlark::syntax::{AstModule, Dialect};

use crate::errors::DslError;
use crate::sandbox::{SandboxConfig, build_globals, make_evaluator};

/// A loaded, frozen Starlark policy module.
///
/// Re-usable across multiple solve invocations: `init`/`step`/`generate`
/// are pure functions over arguments; the module itself holds no
/// mutable state.
pub struct LoadedPolicy {
    pub frozen: FrozenModule,
    pub globals: Globals,
    pub source_path: String,
}

impl LoadedPolicy {
    /// Parse and freeze a policy from a file path.
    pub fn from_path(path: impl AsRef<Path>, cfg: &SandboxConfig) -> Result<Self, DslError> {
        let path_ref = path.as_ref();
        let src = std::fs::read_to_string(path_ref)?;
        Self::from_source(path_ref.to_string_lossy().into_owned(), src, cfg)
    }

    /// Parse and freeze a policy from in-memory source. Used in tests.
    pub fn from_source(
        source_path: String,
        source: String,
        cfg: &SandboxConfig,
    ) -> Result<Self, DslError> {
        let ast = AstModule::parse(&source_path, source, &Dialect::Standard).map_err(|e| {
            DslError::Parse {
                path: source_path.clone(),
                message: format!("{e}"),
            }
        })?;

        let module = Module::new();
        let globals = build_globals(cfg);
        {
            let mut eval = make_evaluator(&module, &globals, cfg);
            eval.eval_module(ast, &globals)
                .map_err(|e| DslError::Runtime {
                    traceback: format!("{e:?}"),
                })?;
        }
        let frozen = module.freeze().map_err(|e| DslError::Runtime {
            traceback: format!("{e:?}"),
        })?;
        Ok(Self {
            frozen,
            globals,
            source_path,
        })
    }

    /// Look up a top-level binding by name. Returns None if absent.
    pub fn get(&self, name: &str) -> Option<starlark::values::OwnedFrozenValue> {
        self.frozen.get(name).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_a_simple_policy_from_source() {
        let cfg = SandboxConfig::default();
        let src = r#"
PARAMS = {"answer": 42}
def hello(x):
    return x + 1
"#;
        let p = LoadedPolicy::from_source("inline.star".into(), src.into(), &cfg).expect("load");
        assert!(p.get("PARAMS").is_some(), "PARAMS not exported");
        assert!(p.get("hello").is_some(), "hello not exported");
    }

    #[test]
    fn parse_error_returns_parse_variant() {
        let cfg = SandboxConfig::default();
        let src = "def broken(:\n";
        let err = LoadedPolicy::from_source("bad.star".into(), src.into(), &cfg)
            .err()
            .expect("must fail");
        assert!(matches!(err, DslError::Parse { .. }), "got {err:?}");
    }

    #[test]
    fn load_statement_is_rejected() {
        let cfg = SandboxConfig::default();
        let src = r#"load("other.star", "x")"#;
        let err = LoadedPolicy::from_source("loader.star".into(), src.into(), &cfg)
            .err()
            .expect("must fail");
        // Either Parse or Runtime is acceptable; the contract is "must fail".
        assert!(matches!(
            err,
            DslError::Parse { .. } | DslError::Runtime { .. }
        ));
    }
}
