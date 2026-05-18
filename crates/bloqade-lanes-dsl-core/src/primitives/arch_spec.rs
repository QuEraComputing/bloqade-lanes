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

use crate::primitives::move_set::{MoveSet, StarlarkMoveSet};
use crate::primitives::types::{
    StarlarkLane, StarlarkLocation, decode_direction, decode_move_type,
};
use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};

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

    /// Validated `Location` constructor: build a `Location` from
    /// `(zone_id, word_id, site_id)` and verify it corresponds to a real
    /// site on this arch. Raises a Starlark error if the location is
    /// out of bounds.
    ///
    /// Use this when a policy NEEDS the resulting `Location` to be a
    /// valid arch site (e.g., to compare against a qubit's current
    /// position). For pure value-object construction with no validation,
    /// use the free `Location(...)` global instead.
    fn location<'v>(
        this: &StarlarkArchSpec,
        zone_id: u32,
        word_id: u32,
        site_id: u32,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let loc = LocationAddr {
            zone_id,
            word_id,
            site_id,
        };
        if let Some(msg) = this.0.check_location(&loc) {
            return Err(starlark::Error::new_other(ArchValidationError(format!(
                "arch_spec.location({zone_id}, {word_id}, {site_id}): {msg}"
            ))));
        }
        Ok(heap.alloc(StarlarkLocation(loc)))
    }

    /// Validated `Lane` constructor: build a `Lane` from its components
    /// and verify it exists in the active arch's lane index. Raises a
    /// Starlark error if the lane is not registered (i.e.,
    /// `arch_spec.lane_endpoints(...)` would return `None`).
    ///
    /// Use this when a policy wants to refer to a SPECIFIC arch lane it
    /// can pass to `insert_child` without an `aod_invalid` surprise at
    /// action-dispatch time. For pure value-object construction with no
    /// validation, use the free `Lane(...)` global instead.
    #[allow(clippy::too_many_arguments)] // mirrors LaneAddr field count
    fn lane<'v>(
        this: &StarlarkArchSpec,
        direction: i32,
        move_type: i32,
        zone_id: u32,
        word_id: u32,
        site_id: u32,
        bus_id: u32,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let direction = decode_direction(direction)?;
        let move_type = decode_move_type(move_type)?;
        let lane = LaneAddr {
            direction,
            move_type,
            zone_id,
            word_id,
            site_id,
            bus_id,
        };
        // `lane_endpoints` is the canonical "is this lane registered?"
        // check — same one the kernel uses at action-dispatch time when
        // it emits `aod_invalid: lane <id> not in index`. Returning
        // `None` here means the constructed `LaneAddr` is not a real
        // lane on the arch.
        if this.0.lane_endpoints(&lane).is_none() {
            return Err(starlark::Error::new_other(ArchValidationError(format!(
                "arch_spec.lane(direction={}, move_type={}, zone_id={}, word_id={}, \
                 site_id={}, bus_id={}): not registered on the active arch (encoded={})",
                lane.direction as i32,
                lane.move_type as i32,
                lane.zone_id,
                lane.word_id,
                lane.site_id,
                lane.bus_id,
                lane.encode_u64(),
            ))));
        }
        Ok(heap.alloc(StarlarkLane(lane)))
    }

    /// Validated `MoveSet` constructor: take a list of `Lane` values
    /// and verify each one is registered on this arch's lane index.
    /// Returns a `MoveSet` (sorted, deduplicated) on success, or a
    /// Starlark error on the first lane that doesn't exist.
    ///
    /// This is the validated-at-source counterpart of passing a raw
    /// `[lane_a, lane_b, ...]` list into `insert_child` and letting
    /// the kernel detect `aod_invalid: lane <id> not in index` at
    /// dispatch time. Useful when the policy is hand-constructing
    /// lanes via the free `Lane(...)` global and wants to fail fast.
    fn move_set<'v>(
        this: &StarlarkArchSpec,
        lanes: &ListRef<'v>,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let mut decoded: Vec<LaneAddr> = Vec::with_capacity(lanes.len());
        for (i, v) in lanes.iter().enumerate() {
            let lane = <&StarlarkLane>::unpack_value(v)?.ok_or_else(|| {
                starlark::Error::new_other(ArchValidationError(format!(
                    "arch_spec.move_set: lanes[{i}] must be a Lane, got {v}"
                )))
            })?;
            if this.0.lane_endpoints(&lane.0).is_none() {
                return Err(starlark::Error::new_other(ArchValidationError(format!(
                    "arch_spec.move_set: lanes[{i}] (encoded={}) is not registered on the \
                     active arch",
                    lane.0.encode_u64(),
                ))));
            }
            decoded.push(lane.0);
        }
        Ok(heap.alloc(StarlarkMoveSet(MoveSet::new(decoded))))
    }
}

#[derive(Debug)]
struct ArchValidationError(String);
impl std::fmt::Display for ArchValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ArchValidationError {}

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

    /// `arch_spec.location(zone, word, site)` on an empty arch must
    /// surface a `check_location`-derived error (no real site exists).
    #[test]
    fn arch_spec_location_rejects_out_of_bounds() {
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        let globals = GlobalsBuilder::standard().build();
        let module = Module::new();
        let spec_val = module.heap().alloc(StarlarkArchSpec(minimal_arch_spec()));
        module.set("arch", spec_val);

        let ast = AstModule::parse(
            "test",
            "arch.location(zone_id=99, word_id=99, site_id=99)".to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        let err = eval
            .eval_module(ast, &globals)
            .expect_err("out-of-bounds location must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("arch_spec.location"),
            "expected error to mention arch_spec.location, got: {msg}"
        );
    }

    /// `arch_spec.lane(...)` on an empty arch must surface a
    /// "not registered on the active arch" error (no lanes in the
    /// index). Pinning the `lane_endpoints`-based validation.
    #[test]
    fn arch_spec_lane_rejects_unregistered() {
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        let globals = GlobalsBuilder::standard().build();
        let module = Module::new();
        let spec_val = module.heap().alloc(StarlarkArchSpec(minimal_arch_spec()));
        module.set("arch", spec_val);

        let ast = AstModule::parse(
            "test",
            "arch.lane(direction=0, move_type=0, zone_id=0, word_id=0, site_id=0, bus_id=0)"
                .to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        let err = eval
            .eval_module(ast, &globals)
            .expect_err("unregistered lane must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("not registered on the active arch"),
            "expected 'not registered' error, got: {msg}"
        );
    }

    /// `arch_spec.move_set([...])` rejects the first unregistered lane
    /// and reports its index in the input list.
    #[test]
    fn arch_spec_move_set_rejects_unregistered_lane() {
        use crate::primitives::types::register_address_constructors;
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        // Need the free `Lane(...)` constructor in scope to build a
        // hand-crafted lane that can't possibly be on the empty arch.
        let globals = GlobalsBuilder::standard()
            .with(register_address_constructors)
            .build();
        let module = Module::new();
        let spec_val = module.heap().alloc(StarlarkArchSpec(minimal_arch_spec()));
        module.set("arch", spec_val);

        let ast = AstModule::parse(
            "test",
            "arch.move_set([Lane(direction=0, move_type=0, zone_id=0, \
             word_id=0, site_id=0, bus_id=0)])"
                .to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        let err = eval
            .eval_module(ast, &globals)
            .expect_err("unregistered lane in move_set must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("arch_spec.move_set") && msg.contains("lanes[0]"),
            "expected 'arch_spec.move_set' + 'lanes[0]' in error, got: {msg}"
        );
    }

    /// `arch_spec.move_set([])` on an empty list is allowed — returns
    /// an empty `MoveSet`. (No lanes to validate against.)
    #[test]
    fn arch_spec_move_set_empty_list_is_ok() {
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        let globals = GlobalsBuilder::standard().build();
        let module = Module::new();
        let spec_val = module.heap().alloc(StarlarkArchSpec(minimal_arch_spec()));
        module.set("arch", spec_val);

        let ast = AstModule::parse(
            "test",
            "ms = arch.move_set([])\nout = ms.len".to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        eval.eval_module(ast, &globals).unwrap();
        let out = module.get("out").unwrap();
        assert_eq!(format!("{out}"), "0");
    }
}
