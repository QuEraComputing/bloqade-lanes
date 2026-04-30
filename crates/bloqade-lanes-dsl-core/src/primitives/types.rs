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
use bloqade_lanes_bytecode_core::arch::addr::{LaneAddr, LocationAddr};
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
            // LocationAddr.word_id and .site_id are public u32 fields.
            // Starlark Value::new_int is pub(crate); use heap.alloc(n: i32).
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
            // encode_u64 returns u64; Gemini lane IDs fit in i32 in practice.
            "encoded" => Some(heap.alloc(self.0.encode_u64() as i32)),
            // direction and move_type are #[repr(u8)] enums; cast to i32.
            "direction" => Some(heap.alloc(self.0.direction as i32)),
            "move_type" => Some(heap.alloc(self.0.move_type as i32)),
            // bus_id and zone_id are public u32 fields on LaneAddr.
            "bus_id" => Some(heap.alloc(self.0.bus_id as i32)),
            "zone_id" => Some(heap.alloc(self.0.zone_id as i32)),
            _ => None,
        }
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
}
