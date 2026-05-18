//! Starlark wrappers for the bytecode core address types.
//!
//! # API adaptation notes
//!
//! - `LocationAddr` exposes `zone_id`, `word_id`, `site_id` as **public
//!   fields** (not methods). There is no `LocationAddr::new()` constructor;
//!   construction uses struct-literal syntax.
//! - `LaneAddr` exposes `direction`, `move_type`, `zone_id`, `word_id`,
//!   `site_id`, `bus_id` as **public fields** (not methods).
//! - `MoveType` and `Direction` are `#[repr(u8)]` enums, so casting via
//!   `as i32` produces their integer discriminant.
//! - `LaneAddr::encode_u64(&self) -> u64` exists and is used for the
//!   `encoded` attribute.
//! - `LocationAddr` and `LaneAddr` do not implement `allocative::Allocative`;
//!   we use a manual `Allocative` impl that treats the inner type as a leaf
//!   (skipping its fields), which is sound for these small Copy types.
//! - `Value::new_int` is `pub(crate)` in starlark-0.13; integer `Value`s
//!   must be allocated via `heap.alloc(n)` where `n: i32`.

use std::fmt;

use allocative::Allocative;
use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use starlark::starlark_module;
use starlark::starlark_simple_value;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

// ─── StarlarkLocation ────────────────────────────────────────────────────────

/// Starlark-visible wrapper around [`LocationAddr`].
///
/// Exposed attributes:
/// - `word_id` — 32-bit word index, returned as Starlark integer.
/// - `site_id` — 32-bit site index, returned as Starlark integer.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkLocation(pub LocationAddr);

impl Allocative for StarlarkLocation {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

starlark_simple_value!(StarlarkLocation);

impl fmt::Display for StarlarkLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Fields are public on LocationAddr.
        write!(f, "Loc({}, {})", self.0.word_id, self.0.site_id)
    }
}

#[starlark::values::starlark_value(type = "Location")]
impl<'v> StarlarkValue<'v> for StarlarkLocation {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            // LocationAddr fields are public u32. Architecture sizes are
            // bounded in practice (zone < 256, word/site < 65536 by the
            // bit-packing in `LocationAddr::encode`), so casting to i32
            // is safe — values well under i32::MAX.
            "zone_id" => Some(heap.alloc(self.0.zone_id as i32)),
            "word_id" => Some(heap.alloc(self.0.word_id as i32)),
            "site_id" => Some(heap.alloc(self.0.site_id as i32)),
            _ => None,
        }
    }
}

// ─── StarlarkLane ─────────────────────────────────────────────────────────────

/// Starlark-visible wrapper around [`LaneAddr`].
///
/// `.encoded` is the canonical stable identifier (the `encode_u64()` result).
/// Structural fields (`move_type`, `bus_id`, `direction`, `zone_id`) are
/// exposed for tie-breaking and diagnostics.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkLane(pub LaneAddr);

impl Allocative for StarlarkLane {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

starlark_simple_value!(StarlarkLane);

impl fmt::Display for StarlarkLane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Lane({})", self.0.encode_u64())
    }
}

#[starlark::values::starlark_value(type = "Lane")]
impl<'v> StarlarkValue<'v> for StarlarkLane {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            // Physical-Gemini lane encodings routinely exceed `i32::MAX` (the
            // `LaneAddr::encode_u64` packs zone/bus/site bits into the high
            // half of a `u64`). Casting to `i32` here silently wraps the
            // value — typically to `0` for the high-bit-only encodings —
            // and the resulting `lanes: [0]` makes every `insert_child`
            // call fail with `aod_invalid: lane 0 has no qubit at src`,
            // silently corrupting every DSL search policy that reads
            // `lane.encoded` to forward to `insert_child`.
            "encoded" => Some(heap.alloc(self.0.encode_u64() as i64)),
            // direction and move_type are #[repr(u8)] enums; cast to i32.
            "direction" => Some(heap.alloc(self.0.direction as i32)),
            "move_type" => Some(heap.alloc(self.0.move_type as i32)),
            // The remaining u32 fields. Arch sizes bound them well under
            // i32::MAX in practice, same as `StarlarkLocation`.
            "bus_id" => Some(heap.alloc(self.0.bus_id as i32)),
            "zone_id" => Some(heap.alloc(self.0.zone_id as i32)),
            "word_id" => Some(heap.alloc(self.0.word_id as i32)),
            "site_id" => Some(heap.alloc(self.0.site_id as i32)),
            _ => None,
        }
    }
}

// ─── Free constructors and enum constants ────────────────────────────────────

/// Decode a `Direction` discriminant int. Used by `Lane(...)` and by
/// `ArchSpec.lane(...)`; rejected discriminants surface as a
/// policy-author-visible Starlark error.
pub(crate) fn decode_direction(d: i32) -> starlark::Result<Direction> {
    match d {
        0 => Ok(Direction::Forward),
        1 => Ok(Direction::Backward),
        other => Err(starlark::Error::new_other(BadDiscriminant(format!(
            "invalid direction discriminant {other}; expected 0 (FORWARD) or 1 (BACKWARD)"
        )))),
    }
}

/// Decode a `MoveType` discriminant int. See `decode_direction`.
pub(crate) fn decode_move_type(m: i32) -> starlark::Result<MoveType> {
    match m {
        0 => Ok(MoveType::SiteBus),
        1 => Ok(MoveType::WordBus),
        2 => Ok(MoveType::ZoneBus),
        other => Err(starlark::Error::new_other(BadDiscriminant(format!(
            "invalid move_type discriminant {other}; expected 0 (SITE_BUS), 1 (WORD_BUS), or 2 (ZONE_BUS)"
        )))),
    }
}

#[derive(Debug)]
struct BadDiscriminant(String);
impl fmt::Display for BadDiscriminant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for BadDiscriminant {}

/// Register Starlark globals for constructing `Location` and `Lane` values
/// directly (no arch validation), plus the enum-discriminant constants.
///
/// These are the FREE-CONSTRUCTION verbs — they let policy code synthesize
/// addresses for algorithmic reasoning (e.g. "what would the candidate
/// destination look like if I bumped site_id by 1?") without going through
/// the arch's lane index. If the policy actually wants to USE the resulting
/// address in an `insert_child` action, the kernel will validate it against
/// the lane index at action-dispatch time and emit `aod_invalid` if it
/// doesn't exist. For validated construction at the source, use
/// `arch_spec.location(...)` / `arch_spec.lane(...)` instead.
///
/// Constants registered:
/// - `DIR_FORWARD = 0`, `DIR_BACKWARD = 1`
/// - `MT_SITE_BUS = 0`, `MT_WORD_BUS = 1`, `MT_ZONE_BUS = 2`
#[starlark_module]
pub fn register_address_constructors(builder: &mut starlark::environment::GlobalsBuilder) {
    /// Direction discriminant for `LaneAddr.direction`: forward.
    const DIR_FORWARD: i32 = 0;
    /// Direction discriminant for `LaneAddr.direction`: backward.
    const DIR_BACKWARD: i32 = 1;
    /// MoveType discriminant for `LaneAddr.move_type`: within-word site bus.
    const MT_SITE_BUS: i32 = 0;
    /// MoveType discriminant for `LaneAddr.move_type`: cross-word word bus.
    const MT_WORD_BUS: i32 = 1;
    /// MoveType discriminant for `LaneAddr.move_type`: cross-zone zone bus.
    const MT_ZONE_BUS: i32 = 2;

    /// Construct a `Location` from its `(zone_id, word_id, site_id)`
    /// components. NO architecture validation — the returned `Location`
    /// is a pure value object that may or may not correspond to a real
    /// site on the active `ArchSpec`. Use `arch_spec.location(...)` for
    /// the validated variant.
    fn Location<'v>(
        zone_id: u32,
        word_id: u32,
        site_id: u32,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        Ok(heap.alloc(StarlarkLocation(LocationAddr {
            zone_id,
            word_id,
            site_id,
        })))
    }

    /// Construct a `Lane` from its component fields, matching the
    /// `LaneAddr` Rust struct layout. NO architecture validation — the
    /// returned `Lane` is a pure value object; the kernel rejects it as
    /// `aod_invalid: lane <id> not in index` if you pass it to
    /// `insert_child` and it doesn't exist on the active arch. Use
    /// `arch_spec.lane(...)` for the validated variant.
    ///
    /// `direction` and `move_type` take their discriminant ints — use
    /// the `DIR_*` and `MT_*` constants to avoid magic numbers.
    fn Lane<'v>(
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
        Ok(heap.alloc(StarlarkLane(LaneAddr {
            direction,
            move_type,
            zone_id,
            word_id,
            site_id,
            bus_id,
        })))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

    use super::*;

    #[test]
    fn location_attrs_are_readable() {
        // LocationAddr has no ::new() constructor; use struct-literal syntax.
        let loc = LocationAddr {
            zone_id: 0,
            word_id: 3,
            site_id: 7,
        };
        let s = StarlarkLocation(loc);
        let display = format!("{s}");
        assert!(
            display.contains("3, 7"),
            "expected '3, 7' in display, got: {display}"
        );
    }

    #[test]
    fn lane_display_contains_encoded_u64() {
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 1,
            site_id: 0,
            bus_id: 0,
        };
        let expected_encoded = lane.encode_u64();
        let s = StarlarkLane(lane);
        let display = format!("{s}");
        assert!(
            display.contains(&expected_encoded.to_string()),
            "expected encoded value in display, got: {display}"
        );
    }

    /// Pin the constructor + readback round-trip end-to-end via a real
    /// Starlark evaluator. If a policy writes
    /// `Location(zone_id=2, word_id=3, site_id=7).word_id`, the result must
    /// be `3`. This is the property that lets the LLM write algorithms
    /// over `loc.zone_id` / `loc.word_id` / `loc.site_id` against
    /// freshly-constructed `Location` values.
    #[test]
    fn location_constructor_roundtrips_all_three_fields() {
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        let globals = GlobalsBuilder::standard()
            .with(register_address_constructors)
            .build();
        let module = Module::new();
        let ast = AstModule::parse(
            "test",
            r#"
loc = Location(zone_id=2, word_id=3, site_id=7)
out = [loc.zone_id, loc.word_id, loc.site_id]
"#
            .to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        eval.eval_module(ast, &globals).unwrap();
        let out = module.get("out").unwrap();
        // Out is `[i32, i32, i32]`. Format-and-compare avoids fiddling with
        // the starlark-list iteration API.
        assert_eq!(format!("{out}"), "[2, 3, 7]");
    }

    /// Same pin for `Lane(...)` — all six components must round-trip,
    /// including the `DIR_*` and `MT_*` discriminants.
    #[test]
    fn lane_constructor_roundtrips_all_six_fields() {
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        let globals = GlobalsBuilder::standard()
            .with(register_address_constructors)
            .build();
        let module = Module::new();
        let ast = AstModule::parse(
            "test",
            r#"
lane = Lane(
    direction=DIR_BACKWARD,
    move_type=MT_WORD_BUS,
    zone_id=1,
    word_id=2,
    site_id=3,
    bus_id=4,
)
out = [
    lane.direction,
    lane.move_type,
    lane.zone_id,
    lane.word_id,
    lane.site_id,
    lane.bus_id,
]
"#
            .to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        eval.eval_module(ast, &globals).unwrap();
        let out = module.get("out").unwrap();
        // DIR_BACKWARD=1, MT_WORD_BUS=1, then the four u32 fields.
        assert_eq!(format!("{out}"), "[1, 1, 1, 2, 3, 4]");
    }

    /// Reject invalid discriminants instead of silently wrapping.
    /// `direction=42` would otherwise cast `as u8` and produce
    /// undefined behaviour for the `Direction` enum.
    #[test]
    fn lane_constructor_rejects_invalid_direction() {
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        let globals = GlobalsBuilder::standard()
            .with(register_address_constructors)
            .build();
        let module = Module::new();
        let ast = AstModule::parse(
            "test",
            "Lane(direction=42, move_type=0, zone_id=0, word_id=0, site_id=0, bus_id=0)".to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        let err = eval
            .eval_module(ast, &globals)
            .expect_err("invalid discriminant must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("direction") && msg.contains("42"),
            "expected error to mention 'direction' and the bad value, got: {msg}"
        );
    }

    #[test]
    fn lane_constructor_rejects_invalid_move_type() {
        use starlark::environment::{GlobalsBuilder, Module};
        use starlark::eval::Evaluator;
        use starlark::syntax::{AstModule, Dialect};

        let globals = GlobalsBuilder::standard()
            .with(register_address_constructors)
            .build();
        let module = Module::new();
        let ast = AstModule::parse(
            "test",
            "Lane(direction=0, move_type=99, zone_id=0, word_id=0, site_id=0, bus_id=0)".to_owned(),
            &Dialect::Standard,
        )
        .unwrap();
        let mut eval = Evaluator::new(&module);
        let err = eval
            .eval_module(ast, &globals)
            .expect_err("invalid discriminant must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("move_type") && msg.contains("99"),
            "expected error to mention 'move_type' and the bad value, got: {msg}"
        );
    }
}
