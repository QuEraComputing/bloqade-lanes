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
    /// Shared by the [`LaneIndex`] all-zones precompute and the per-zone
    /// fallback in `BusGridContext::new`. Lanes whose endpoints or source
    /// position are unknown are skipped (matches the legacy behaviour).
    ///
    /// The all-zones precompute merges every zone's lanes for a
    /// `(move_type, bus_id, direction)` group, so dropping `zone_id` from the
    /// group key could in principle collapse two lanes that share a source
    /// location but point at different destinations. That is the *source*-keyed
    /// direction (`src_to_lane`, `src_to_dst`, `src_to_pos`):
    ///
    /// * SiteBus / WordBus never collide, because a source's *encoded* location
    ///   already carries its `zone_id` (and `word_id`/`site_id`), so lanes from
    ///   different zones map to distinct source keys.
    /// * ZoneBus (registered since #846) is the interesting case: a *backward*
    ///   ZoneBus lane's source is the forward destination, which lives in the
    ///   destination zone, so two lanes whose forward moves target the same word
    ///   would share a source key. That only happens for a non-injective
    ///   (many-to-one) zone bus; the AOD-rectangle zone buses in the current
    ///   specs are injective, so no collision occurs.
    ///
    /// [`insert_unique`] pins this with a `debug_assert!` on the source-keyed
    /// maps, so a non-injective zone bus surfaces loudly instead of resolving to
    /// a silent last-insert-wins over `HashMap` order. The search test suite —
    /// including the ZoneBus coverage added by #846 — runs with these
    /// assertions active.
    ///
    /// `pos_to_src` is deliberately *not* guarded: distinct sources can
    /// legitimately share a physical position in some geometries (e.g. the
    /// `full.json` test fixture stacks every word at identical grid
    /// coordinates), so a position collision is a pre-existing, tolerated
    /// property of the arch — unrelated to the zone-merge hazard — that the old
    /// per-call `BusGridContext::new` also resolved by last-insert-wins.
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
