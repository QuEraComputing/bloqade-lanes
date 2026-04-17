use std::cmp::Ordering;

use crate::config::Config;
use crate::graph::MoveSet;

pub(crate) type TripletKey = (u8, u32, u8);

/// Shared deterministic tie-breaker for triplet-scored entries.
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
