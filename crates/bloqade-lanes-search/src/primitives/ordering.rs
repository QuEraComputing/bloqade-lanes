use std::cmp::Ordering;

use bloqade_lanes_bytecode_core::arch::addr::{Direction, MoveType};

use crate::primitives::config::Config;
use crate::primitives::graph::MoveSet;

/// Deterministic sort/group key for a bus triplet: `(move_type, bus_id,
/// direction)`.
///
/// Derived `Ord` compares the fields in declaration order, exactly as the
/// former `(u8, u32, u8)` tuple did. `MoveType`/`Direction` declare their
/// variants in ascending discriminant order, so their derived `Ord` matches
/// the numeric `#[repr(u8)]` values — i.e. the prior `as u8` casts — and
/// `BTreeMap`/`sort` iteration order is preserved. Keeping the enums typed
/// removes the encode-as-`u8` / decode-by-`match` round-trip at the use sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct TripletKey {
    pub(crate) move_type: MoveType,
    pub(crate) bus_id: u32,
    pub(crate) direction: Direction,
}

impl TripletKey {
    pub(crate) fn new(move_type: MoveType, bus_id: u32, direction: Direction) -> Self {
        Self {
            move_type,
            bus_id,
            direction,
        }
    }
}

/// Shared deterministic tie-breaker for triplet-scored entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmp_triplet_entry_tiebreak(
    a_key: &TripletKey,
    a_qubit: u32,
    a_lane: u64,
    a_dst: u64,
    b_key: &TripletKey,
    b_qubit: u32,
    b_lane: u64,
    b_dst: u64,
) -> Ordering {
    a_key
        .cmp(b_key)
        .then_with(|| a_qubit.cmp(&b_qubit))
        .then_with(|| a_lane.cmp(&b_lane))
        .then_with(|| a_dst.cmp(&b_dst))
}

/// Shared deterministic tie-breaker for score-group entries.
pub(crate) fn cmp_qubit_lane_dst_tiebreak(
    a_qubit: u32,
    a_lane: u64,
    a_dst: u64,
    b_qubit: u32,
    b_lane: u64,
    b_dst: u64,
) -> Ordering {
    a_qubit
        .cmp(&b_qubit)
        .then_with(|| a_lane.cmp(&b_lane))
        .then_with(|| a_dst.cmp(&b_dst))
}

/// Shared deterministic tie-breaker for candidate ordering.
pub(crate) fn cmp_moveset_config_tiebreak(
    a_ms: &MoveSet,
    a_cfg: &Config,
    b_ms: &MoveSet,
    b_cfg: &Config,
) -> Ordering {
    a_ms.encoded_lanes()
        .cmp(b_ms.encoded_lanes())
        .then_with(|| a_cfg.as_entries().cmp(b_cfg.as_entries()))
}
