//! Occupancy-independent AOD-grid lookup maps for one bus group.
//!
//! Lives in the `primitives` layer alongside [`LaneIndex`], which owns the
//! per-bus-group cache of these maps. `ops::aod_grid` borrows them to build
//! rectangular AOD grids without rebuilding the maps on every call. Keeping the
//! type here (rather than in `ops`) means `primitives` never depends on `ops`.

use std::collections::HashMap;
use std::hash::Hash;

use bloqade_lanes_bytecode_core::arch::addr::LaneAddr;

use crate::primitives::lane_index::LaneIndex;

/// Occupancy-independent lookup maps for one bus group.
///
/// These are a pure function of the architecture ([`LaneIndex`]) and the bus
/// group `(move_type, bus_id, direction)` — they do **not** depend on which
/// locations are currently occupied. [`LaneIndex`] precomputes and caches one
/// of these per bus group so `BusGridContext::new` can borrow it instead of
/// rebuilding all four maps (an all-lanes scan) on every call. That scan sits
/// in the entropy driver's hottest loop (`generate_candidates` builds a context
/// per bus-triplet group, thousands of times per solve).
///
/// The maps are not reducible to bare lane-address arithmetic: a lane encodes
/// its *forward* source `(zone, word, site)`, but for a backward lane the actual
/// source is the forward destination (see [`crate`]'s `lane_endpoints`), so
/// `lane.{zone,word,site} != src.{zone,word,site}` in general. Both the
/// source→lane and source→dst directions therefore require a stored map.
#[derive(Debug, Clone, Default)]
pub(crate) struct BusGridMaps {
    /// `(x_bits, y_bits) → encoded source location` for ALL bus positions.
    pub(crate) pos_to_src: HashMap<(u64, u64), u64>,
    /// `encoded source → encoded lane address` for ALL bus lanes.
    pub(crate) src_to_lane: HashMap<u64, u64>,
    /// `encoded source → encoded destination location` for ALL bus lanes.
    pub(crate) src_to_dst: HashMap<u64, u64>,
    /// `encoded source → (x_bits, y_bits)` reverse lookup.
    pub(crate) src_to_pos: HashMap<u64, (u64, u64)>,
}

impl BusGridMaps {
    /// Build the maps for one bus group from the given lanes.
    ///
    /// Built by the [`LaneIndex`] per-group precompute (one entry per
    /// `(move_type, bus_id, zone_id, direction)`) and by tests. Lanes whose
    /// endpoints or source position are unknown are skipped (matches the
    /// legacy behaviour).
    ///
    /// The source-keyed maps (`src_to_lane`, `src_to_dst`, `src_to_pos`) are
    /// guarded by [`insert_unique`]: within one group a source has one lane, so
    /// a collision is a bug rather than an ambiguity to resolve by insertion
    /// order. (Groups used to merge every zone's lanes under a zone-less key,
    /// where a backward zone-bus lane could share a source key with another
    /// zone's lane; per-zone grouping removed that hazard, and the guard stays
    /// as a cheap invariant.) `pos_to_src` is deliberately *not* guarded:
    /// distinct sources can share a physical position in some geometries (the
    /// `full.json` fixture stacks every word at identical grid coordinates),
    /// which is the P1 precondition the exhaustive generator checks and the
    /// grid builder tolerates by last-insert-wins.
    pub(crate) fn from_lanes(index: &LaneIndex, lanes: impl IntoIterator<Item = LaneAddr>) -> Self {
        let mut maps = Self::default();
        for lane in lanes {
            let Some((src, dst)) = index.endpoints(&lane) else {
                continue;
            };
            let Some((x, y)) = index.position(src) else {
                continue;
            };
            let src_enc = src.encode();
            let pos = (x.to_bits(), y.to_bits());
            // Not guarded — distinct sources may share a position (see above).
            maps.pos_to_src.insert(pos, src_enc);
            insert_unique(&mut maps.src_to_lane, src_enc, lane.encode_u64());
            insert_unique(&mut maps.src_to_dst, src_enc, dst.encode());
            insert_unique(&mut maps.src_to_pos, src_enc, pos);
        }
        maps
    }
}

/// Insert `key → value`, debug-asserting we never overwrite an existing key
/// with a *different* value.
///
/// Re-inserting an identical mapping (e.g. a lane appearing twice) is fine; a
/// conflicting overwrite on a source-keyed map means the all-zones merge
/// collapsed two semantically distinct lanes onto one source — see
/// [`BusGridMaps::from_lanes`] for why that cannot happen for SiteBus/WordBus
/// and would indicate a non-injective (many-to-one) zone bus.
fn insert_unique<K: Eq + Hash, V: PartialEq>(map: &mut HashMap<K, V>, key: K, value: V) {
    if let Some(existing) = map.get(&key) {
        debug_assert!(
            *existing == value,
            "BusGridMaps merge conflict: a source key resolved to two different \
             values (unexpected for SiteBus/WordBus; indicates a non-injective zone bus)"
        );
    }
    map.insert(key, value);
}
