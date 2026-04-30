//! Starlark wrapper for `ArchSpec`.
//!
//! # API adaptation notes
//!
//! - All query methods (`get_cz_partner`, `lane_endpoints`, `location_position`,
//!   `check_locations`) live on `ArchSpec` via the `impl` blocks in
//!   `bloqade_lanes_bytecode_core::arch::query`, not in `types`. They are
//!   re-exported through the module so the import path is just `arch::types::ArchSpec`.
//! - `check_location_group` from the plan → `check_locations` in the actual API.
//!   Returns `Vec<LocationGroupError>` (each implements `Display`).
//! - `num_locations()` does not exist. Computed as `words.len() * sites_per_word()`,
//!   where both are on the `ArchSpec` struct / method respectively.
//! - `ArchSpec` has no `name` field or `name()` method. The `Display` impl
//!   for `StarlarkArchSpec` emits `"ArchSpec(<num_words>w x <sites_per_word>s)"`.
//! - `heap.alloc((a, b))` for tuples: starlark-0.13 does not implement `AllocValue`
//!   for `(A, B)` tuples directly.  We build a `Vec<Value>` and call
//!   `heap.alloc(AllocTuple(vec))` instead.

use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use starlark::starlark_module;
use starlark::values::list::AllocList;
use starlark::values::list::ListRef;
use starlark::values::tuple::AllocTuple;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, UnpackValue, Value};

use crate::primitives::types::{StarlarkLane, StarlarkLocation};

// ─── StarlarkArchSpec ─────────────────────────────────────────────────────────

/// Read-only Starlark handle exposing the architecture spec to a policy.
///
/// Holds an `Arc<ArchSpec>` so the wrapper can be cloned cheaply across
/// the Rust↔Starlark boundary.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkArchSpec(pub Arc<ArchSpec>);

// Manual Allocative impl: ArchSpec contains heap data, but the shared Arc is
// not duplicated per Starlark value. We treat the wrapper as a simple sized
// leaf, matching the convention established in `primitives::types`.
impl allocative::Allocative for StarlarkArchSpec {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

starlark::starlark_simple_value!(StarlarkArchSpec);

impl std::fmt::Display for StarlarkArchSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // ArchSpec has no name field; summarise shape instead.
        write!(
            f,
            "ArchSpec({}w x {}s)",
            self.0.words.len(),
            self.0.sites_per_word()
        )
    }
}

#[starlark::values::starlark_value(type = "ArchSpec")]
impl<'v> StarlarkValue<'v> for StarlarkArchSpec {
    fn get_methods() -> Option<&'static starlark::environment::Methods> {
        static METHODS: starlark::environment::MethodsStatic =
            starlark::environment::MethodsStatic::new();
        METHODS.methods(register_arch_spec_methods)
    }
}

// ─── Method registration ──────────────────────────────────────────────────────

#[starlark_module]
fn register_arch_spec_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Return the CZ-blockade partner of `loc`, or `None`.
    ///
    /// Searches `zones[loc.zone_id].entangling_pairs` for a pair containing
    /// `loc.word_id`. Returns the partner location (same zone, same site,
    /// partner word). Returns `None` if the word is not in any entangling pair.
    fn get_cz_partner<'v>(
        this: &StarlarkArchSpec,
        loc: &StarlarkLocation,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        match this.0.get_cz_partner(&loc.0) {
            Some(partner) => Ok(heap.alloc(StarlarkLocation(partner))),
            None => Ok(Value::new_none()),
        }
    }

    /// Validate a candidate location group; returns a list of error-message strings.
    ///
    /// API adaptation: plan called this `check_location_group`; the actual
    /// bytecode-core method is `check_locations`. Each `LocationGroupError`
    /// implements `Display` and is converted to a `String`.
    ///
    /// Parameter: accepts a Starlark `list[Location]`. `Vec<&StarlarkLocation>`
    /// does not implement `UnpackValue` in starlark-0.13; we accept
    /// `&ListRef<'v>` and unpack each element individually.
    fn check_location_group<'v>(
        this: &StarlarkArchSpec,
        locs: &ListRef<'v>,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let mut raw = Vec::with_capacity(locs.len());
        for v in locs.iter() {
            // unpack_value returns Result<Option<T>, starlark::Error> in 0.13.
            let maybe = <&StarlarkLocation>::unpack_value(v)?;
            let loc = maybe.ok_or_else(|| {
                // Wrap error string in a type that implements std::error::Error
                // so it satisfies `Into<anyhow::Error>` required by `new_other`.
                #[derive(Debug)]
                struct Msg(String);
                impl std::fmt::Display for Msg {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str(&self.0)
                    }
                }
                impl std::error::Error for Msg {}
                starlark::Error::new_other(Msg(format!(
                    "check_location_group: expected Location, got {v}"
                )))
            })?;
            raw.push(loc.0);
        }
        let errors: Vec<String> = this
            .0
            .check_locations(&raw)
            .into_iter()
            .map(|e| e.to_string())
            .collect();
        Ok(heap.alloc(AllocList(errors.into_iter())))
    }

    /// Total number of atom sites in this architecture (`words × sites_per_word`).
    ///
    /// API adaptation: `num_locations()` does not exist on `ArchSpec`.
    /// Derived as `words.len() * sites_per_word()`.
    fn num_locations(this: &StarlarkArchSpec) -> starlark::Result<i32> {
        let n = this.0.words.len() * this.0.sites_per_word();
        Ok(n as i32)
    }

    /// `(x, y)` physical position of `loc` in micrometers, or `None` if unknown.
    fn position<'v>(
        this: &StarlarkArchSpec,
        loc: &StarlarkLocation,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        match this.0.location_position(&loc.0) {
            Some((x, y)) => {
                // starlark-0.13 does not implement AllocValue for (A, B) tuples
                // directly.  Build a 2-element AllocTuple from a Vec<Value>.
                let xv = heap.alloc(x);
                let yv = heap.alloc(y);
                Ok(heap.alloc(AllocTuple(vec![xv, yv])))
            }
            None => Ok(Value::new_none()),
        }
    }

    /// Endpoints of a lane as `(src, dst)` Location pair, or `None`.
    fn lane_endpoints<'v>(
        this: &StarlarkArchSpec,
        lane: &StarlarkLane,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        match this.0.lane_endpoints(&lane.0) {
            Some((src, dst)) => {
                let sv = heap.alloc(StarlarkLocation(src));
                let dv = heap.alloc(StarlarkLocation(dst));
                Ok(heap.alloc(AllocTuple(vec![sv, dv])))
            }
            None => Ok(Value::new_none()),
        }
    }

    /// Triplet `(move_type, bus_id, direction)` of a lane, as integers.
    ///
    /// The values are derived from the `LaneAddr` fields directly (no arch
    /// lookup needed). `move_type` and `direction` are `#[repr(u8)]` enums
    /// cast to `i32`.
    fn lane_triplet<'v>(
        this: &StarlarkArchSpec,
        lane: &StarlarkLane,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let _ = this; // values come from the lane address itself
        let mt = heap.alloc(lane.0.move_type as i32);
        let bus = heap.alloc(lane.0.bus_id as i32);
        let dir = heap.alloc(lane.0.direction as i32);
        Ok(heap.alloc(AllocTuple(vec![mt, bus, dir])))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use bloqade_lanes_bytecode_core::version::Version;

    use super::*;

    fn minimal_arch_spec() -> Arc<ArchSpec> {
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
    fn display_shows_shape() {
        let spec = StarlarkArchSpec(minimal_arch_spec());
        let s = format!("{spec}");
        assert!(s.contains("ArchSpec"), "got: {s}");
    }

    #[test]
    fn num_locations_empty() {
        let spec = StarlarkArchSpec(minimal_arch_spec());
        // 0 words × 0 sites_per_word = 0
        assert_eq!(spec.0.words.len() * spec.0.sites_per_word(), 0);
    }
}
