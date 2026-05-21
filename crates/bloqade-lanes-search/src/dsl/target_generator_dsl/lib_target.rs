//! `lib` handle exposed to target generator policies.
//!
//! Minimal v1 surface (per plan §"Lib scope: minimal"):
//! - `lib.arch_spec` (attr) — alias of `ctx.arch_spec`.
//! - `lib.cz_partner(loc)` — shortcut for `ctx.arch_spec.get_cz_partner(loc)`.
//!
//! Stable_sort/argmax/normalize are already global builtins via
//! `dsl_core::sandbox::build_globals`.

use bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec;
use bloqade_lanes_dsl_core::primitives::types::StarlarkLocation;
use starlark::starlark_module;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkLibTarget {
    pub arch_spec: StarlarkArchSpec,
}

impl StarlarkLibTarget {
    pub fn new(arch_spec: StarlarkArchSpec) -> Self {
        Self { arch_spec }
    }
}

impl allocative::Allocative for StarlarkLibTarget {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

starlark::starlark_simple_value!(StarlarkLibTarget);

impl std::fmt::Display for StarlarkLibTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "lib_target")
    }
}

#[starlark::values::starlark_value(type = "LibTarget")]
impl<'v> StarlarkValue<'v> for StarlarkLibTarget {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "arch_spec" => Some(heap.alloc(self.arch_spec.clone())),
            _ => None,
        }
    }

    fn get_methods() -> Option<&'static starlark::environment::Methods> {
        static METHODS: starlark::environment::MethodsStatic =
            starlark::environment::MethodsStatic::new();
        METHODS.methods(register_lib_target_methods)
    }
}

#[starlark_module]
fn register_lib_target_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// CZ blockade partner of `loc`, or `None`.
    ///
    /// Equivalent to `ctx.arch_spec.get_cz_partner(loc)`; provided as a
    /// shortcut so policies don't need to thread the arch_spec through helpers.
    fn cz_partner<'v>(
        this: &StarlarkLibTarget,
        loc: &StarlarkLocation,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        match this.arch_spec.0.get_cz_partner(&loc.0) {
            Some(partner) => Ok(heap.alloc(StarlarkLocation(partner))),
            None => Ok(Value::new_none()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use bloqade_lanes_bytecode_core::version::Version;

    use super::*;

    fn empty_arch() -> Arc<ArchSpec> {
        Arc::new(ArchSpec {
            version: Version::new(2, 0),
            words: vec![],
            zones: vec![],
            zone_buses: vec![],
            modes: vec![],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
            blockade_radius: None,
        })
    }

    #[test]
    fn constructs_with_arch_spec() {
        let lib = StarlarkLibTarget::new(StarlarkArchSpec(empty_arch()));
        assert_eq!(format!("{lib}"), "lib_target");
    }
}
