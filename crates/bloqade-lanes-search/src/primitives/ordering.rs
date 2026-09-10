use std::cmp::Ordering;
use std::fmt;

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, MoveType};

use crate::primitives::config::Config;
use crate::primitives::graph::MoveSet;

/// A bus group: the four fields every lane of one shot must share
/// (`move_type`, `bus_id`, `zone_id`, `direction`), which is also the
/// validator's one-bus-group rule (S3 in the branch-and-bound spec).
///
/// Derived `Ord` compares the fields in declaration order. `MoveType` and
/// `Direction` declare their variants in ascending discriminant order, so the
/// derived `Ord` matches the numeric `#[repr(u8)]` values and `BTreeMap` /
/// `sort` iteration order is deterministic. Every generator groups its
/// candidate lanes by this key and builds one AOD rectangle per group: a
/// rectangle spanning two zones is never a valid shot, so grouping without
/// the zone (as the key did before the two-zone fixtures existed) could hand
/// the grid builder a cross-zone product that `check_lanes` rejects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GroupKey {
    pub move_type: MoveType,
    pub bus_id: u32,
    pub zone_id: u32,
    pub direction: Direction,
}

impl GroupKey {
    pub fn new(move_type: MoveType, bus_id: u32, zone_id: u32, direction: Direction) -> Self {
        Self {
            move_type,
            bus_id,
            zone_id,
            direction,
        }
    }

    /// The group a lane belongs to.
    ///
    /// A lane address carries its *forward* source zone, so a zone-bus lane
    /// and its backward twin report the same `zone_id` even though the
    /// backward lane's source physically lies in the destination zone. They
    /// are still different groups: `direction` is part of the key, because one
    /// shot drives one direction.
    pub fn of(lane: &LaneAddr) -> Self {
        Self::new(lane.move_type, lane.bus_id, lane.zone_id, lane.direction)
    }
}

impl fmt::Display for GroupKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} bus {} zone {} {:?}",
            self.move_type, self.bus_id, self.zone_id, self.direction
        )
    }
}

/// Shared deterministic tie-breaker for triplet-scored entries.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmp_triplet_entry_tiebreak(
    a_key: &GroupKey,
    a_qubit: u32,
    a_lane: u64,
    a_dst: u64,
    b_key: &GroupKey,
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
