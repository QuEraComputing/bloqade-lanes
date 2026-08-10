//! `MoveSet` and its Starlark wrapper.
//!
//! Hosted in `dsl-core` (rather than `search/graph.rs` where it
//! originally lived) so the dsl-core `ArchSpec` methods can return a
//! validated `StarlarkMoveSet` without `dsl-core` having to depend on
//! the `search` crate. `search` re-exports both types so existing
//! imports (`crate::graph::MoveSet`,
//! `crate::move_policy_dsl::lib_move::StarlarkMoveSet`) keep working.

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr};
use starlark::starlark_simple_value;
use starlark::values::list::AllocList;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

/// A set of lanes applied simultaneously in one move step.
///
/// Stored as a sorted, deduplicated `Vec<u64>` of
/// [`LaneAddr::encode_u64()`] values, making it order-independent
/// (analogous to Python's `frozenset[LaneAddress]`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize)]
pub struct MoveSet {
    lanes: Vec<u64>,
}

impl MoveSet {
    /// Create from an iterator of lane addresses.
    ///
    /// The lanes are encoded, sorted, and deduplicated.
    pub fn new(lanes: impl IntoIterator<Item = LaneAddr>) -> Self {
        let mut encoded: Vec<u64> = lanes.into_iter().map(|l| l.encode_u64()).collect();
        encoded.sort_unstable();
        encoded.dedup();
        Self { lanes: encoded }
    }

    /// Create from pre-encoded lane u64 values. Sorts and deduplicates.
    pub fn from_encoded(mut encoded: Vec<u64>) -> Self {
        encoded.sort_unstable();
        encoded.dedup();
        Self { lanes: encoded }
    }

    /// Decode back to `LaneAddr` values.
    pub fn decode(&self) -> Vec<LaneAddr> {
        self.lanes
            .iter()
            .map(|&bits| LaneAddr::decode_u64(bits))
            .collect()
    }

    /// Number of lanes in this move set.
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Returns `true` if the move set contains no lanes.
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// Return the encoded lane values (sorted, deduplicated).
    pub fn encoded_lanes(&self) -> &[u64] {
        &self.lanes
    }

    /// Return the move set that undoes this one.
    ///
    /// Lane identity (`move_type`, `zone_id`, `word_id`, `site_id`, `bus_id`)
    /// is direction-*independent*: `site_id` and `word_id` always encode the
    /// forward-direction source, and `Direction` only selects which endpoint
    /// `ArchSpec::lane_endpoints` returns as src versus dst. Flipping it
    /// therefore maps every lane's `(src, dst)` to `(dst, src)`.
    ///
    /// The flipped lane always exists in the architecture, because
    /// `LaneIndex::new` registers every lane in both directions.
    pub fn inverse(&self) -> Self {
        Self::new(self.decode().into_iter().map(|mut lane| {
            lane.direction = match lane.direction {
                Direction::Forward => Direction::Backward,
                Direction::Backward => Direction::Forward,
            };
            lane
        }))
    }
}

/// Starlark-visible wrapper around [`MoveSet`].
///
/// Exposed attributes:
/// - `len`     — number of encoded lanes.
/// - `encoded` — list of encoded lane integers (`list[int]`).
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkMoveSet(pub MoveSet);

impl allocative::Allocative for StarlarkMoveSet {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let v = visitor.enter_self_sized::<Self>();
        v.exit();
    }
}

starlark_simple_value!(StarlarkMoveSet);

impl std::fmt::Display for StarlarkMoveSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MoveSet(len={})", self.0.len())
    }
}

#[starlark::values::starlark_value(type = "MoveSet")]
impl<'v> StarlarkValue<'v> for StarlarkMoveSet {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "len" => Some(heap.alloc(self.0.len() as i32)),
            "encoded" => {
                // Expose the encoded lane vec as a Starlark list of ints.
                let lanes: Vec<i64> = self.0.encoded_lanes().iter().map(|&l| l as i64).collect();
                Some(heap.alloc(AllocList(lanes.into_iter())))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, MoveType};

    fn lane(direction: Direction, site_id: u32) -> LaneAddr {
        LaneAddr {
            direction,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id,
            bus_id: 0,
        }
    }

    #[test]
    fn inverse_flips_every_lane_direction() {
        let ms = MoveSet::new([lane(Direction::Forward, 0), lane(Direction::Forward, 1)]);
        let inv = ms.inverse();
        let dirs: Vec<Direction> = inv.decode().into_iter().map(|l| l.direction).collect();
        assert_eq!(dirs, vec![Direction::Backward, Direction::Backward]);
    }

    #[test]
    fn inverse_is_an_involution() {
        let ms = MoveSet::new([lane(Direction::Forward, 0), lane(Direction::Backward, 3)]);
        assert_eq!(ms.inverse().inverse(), ms);
    }

    #[test]
    fn inverse_preserves_lane_identity() {
        let ms = MoveSet::new([lane(Direction::Forward, 7)]);
        let original = ms.decode();
        let inverted = ms.inverse().decode();
        assert_eq!(original.len(), 1);
        assert_eq!(inverted.len(), 1);
        assert_eq!(inverted[0].move_type, original[0].move_type);
        assert_eq!(inverted[0].zone_id, original[0].zone_id);
        assert_eq!(inverted[0].word_id, original[0].word_id);
        assert_eq!(inverted[0].site_id, original[0].site_id);
        assert_eq!(inverted[0].bus_id, original[0].bus_id);
    }

    #[test]
    fn inverse_of_empty_is_empty() {
        assert!(MoveSet::new([]).inverse().is_empty());
    }
}
