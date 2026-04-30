//! Pure-Rust pipeline helpers for the Move Policy DSL candidate pipeline.
//!
//! Mirrors stages 2–4 of [`HeuristicGenerator::generate`] but operates on
//! `f64` scores supplied by Starlark policy code (vs. the inline `i32`
//! distance-improvement scores used inside `HeuristicGenerator`).
//!
//! Stage 1 (per-lane scoring) is intentionally not extracted here — for the
//! DSL the score comes from a user-provided Starlark callable and is driven
//! directly from `lib_move::score_lanes`.
//!
//! These helpers are `pub(crate)` and intended for use by the DSL primitives
//! in `move_policy_dsl::lib_move`. The companion file is
//! [`super::heuristic`], whose pipeline remains untouched (the DSL needs
//! `f64` semantics; mixing them would fight existing tie-breakers).
//!
//! Determinism contract:
//! * `top_c_per_qubit`: per-qubit sort by `(score desc, lane.encoded asc)`,
//!   then take top `c`. If no entry has a positive score for any qubit, the
//!   single best (highest-score, then lowest-lane) entry across all qubits
//!   is retained.
//! * `group_by_triplet`: groups by `(move_type, bus_id, direction)`, sorted
//!   ascending on the triplet key.
//! * `pack_aod_rectangles`: AOD rectangles built per group via
//!   [`crate::aod_grid::BusGridContext`], lifted to candidates and sorted
//!   by `score_sum desc` (ties broken by encoded lane vector then config
//!   entries, mirroring `cmp_moveset_config_tiebreak`).

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};

use crate::aod_grid::BusGridContext;
use crate::config::Config;
use crate::graph::MoveSet;
use crate::lane_index::LaneIndex;
use crate::ordering::cmp_moveset_config_tiebreak;

/// One scored `(qubit, lane)` pair produced by Stage 1 of the pipeline.
#[derive(Debug, Clone)]
pub(crate) struct ScoredLane {
    pub qid: u32,
    pub lane: LaneAddr,
    pub score: f64,
}

/// Triplet key `(move_type, bus_id, direction)` as encoded `u8`/`u32`.
pub(crate) type TripletKey = (u8, u32, u8);

/// Group of [`ScoredLane`]s sharing the same triplet key.
#[derive(Debug, Clone)]
pub(crate) struct TripletGroup {
    pub key: TripletKey,
    pub entries: Vec<ScoredLane>,
}

/// Final candidate produced by Stage 4 (AOD packing).
#[derive(Debug, Clone)]
pub(crate) struct PackedCandidate {
    pub move_set: MoveSet,
    pub new_config: Config,
    pub score_sum: f64,
}

/// Stage 2: for each qubit, retain the top-`c` scored lanes by
/// `(score desc, lane.encoded asc)`. Falls back to the single best entry
/// (across all qubits) when no qubit has any positive-scoring lane.
pub(crate) fn top_c_per_qubit(scored: Vec<ScoredLane>, c: usize) -> Vec<ScoredLane> {
    if scored.is_empty() || c == 0 {
        return Vec::new();
    }

    // Group by qid (BTreeMap for deterministic qid order).
    let mut per_qubit: BTreeMap<u32, Vec<ScoredLane>> = BTreeMap::new();
    for entry in scored {
        per_qubit.entry(entry.qid).or_default().push(entry);
    }

    // Sort each per-qubit bucket by (score desc, lane.encoded asc).
    let mut has_positive = false;
    for entries in per_qubit.values_mut() {
        entries.sort_by(cmp_scored_lane);
        if entries.iter().any(|e| e.score > 0.0) {
            has_positive = true;
        }
    }

    if has_positive {
        let mut out: Vec<ScoredLane> = Vec::new();
        for entries in per_qubit.into_values() {
            out.extend(entries.into_iter().take(c));
        }
        out
    } else {
        // Fallback: single best entry across all qubits.
        let mut all: Vec<ScoredLane> = per_qubit.into_values().flatten().collect();
        all.sort_by(cmp_scored_lane);
        all.into_iter().take(1).collect()
    }
}

/// Sort key for `ScoredLane`: highest score first, ties broken by lower
/// `lane.encoded` first, then lower `qid`.
fn cmp_scored_lane(a: &ScoredLane, b: &ScoredLane) -> Ordering {
    b.score
        .partial_cmp(&a.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.lane.encode_u64().cmp(&b.lane.encode_u64()))
        .then_with(|| a.qid.cmp(&b.qid))
}

/// Stage 3: group entries by `(move_type, bus_id, direction)`, sorted by
/// triplet key ascending. Within each group, entries preserve the input
/// order (which the caller is expected to have made deterministic — e.g.
/// via [`top_c_per_qubit`]).
pub(crate) fn group_by_triplet(scored: Vec<ScoredLane>) -> Vec<TripletGroup> {
    let mut groups: BTreeMap<TripletKey, Vec<ScoredLane>> = BTreeMap::new();
    for entry in scored {
        let key = (
            entry.lane.move_type as u8,
            entry.lane.bus_id,
            entry.lane.direction as u8,
        );
        groups.entry(key).or_default().push(entry);
    }
    groups
        .into_iter()
        .map(|(key, entries)| TripletGroup { key, entries })
        .collect()
}

/// Stage 4: per group, build AOD-compatible rectangular grids via
/// [`BusGridContext::build_aod_grids`] and lift to [`PackedCandidate`]s.
/// Returns candidates sorted by `score_sum desc`, with deterministic
/// tie-breakers on `(MoveSet encoded lanes, Config entries)`.
pub(crate) fn pack_aod_rectangles(
    groups: Vec<TripletGroup>,
    config: &Config,
    index: &LaneIndex,
    blocked: &HashSet<u64>,
) -> Vec<PackedCandidate> {
    // Build occupied set: blocked locations + qubits already in this config.
    let mut occupied: HashSet<u64> = HashSet::with_capacity(blocked.len() + config.len());
    occupied.extend(blocked);
    for (_, loc) in config.iter() {
        occupied.insert(loc.encode());
    }

    let mut candidates: Vec<PackedCandidate> = Vec::new();

    for group in groups {
        let TripletGroup { key, entries } = group;
        if entries.is_empty() {
            continue;
        }
        let (mt_u8, bus_id, dir_u8) = key;
        let Some(mt) = decode_move_type(mt_u8) else {
            continue;
        };
        let Some(dir) = decode_direction(dir_u8) else {
            continue;
        };

        // Build the grid context across all zones for the bus.
        let grid_ctx = BusGridContext::new(index, mt, bus_id, None, dir, &occupied);

        // Build src→lane entries and lane→entry lookup for score lifting
        // and destination derivation.
        let mut grid_entries: HashMap<u64, u64> = HashMap::with_capacity(entries.len());
        let mut entry_by_lane: HashMap<u64, &ScoredLane> = HashMap::with_capacity(entries.len());
        for e in &entries {
            if let Some((src, _)) = index.endpoints(&e.lane) {
                let src_enc = src.encode();
                let lane_enc = e.lane.encode_u64();
                grid_entries.insert(src_enc, lane_enc);
                entry_by_lane.insert(lane_enc, e);
            }
        }

        let grids = grid_ctx.build_aod_grids(&grid_entries);

        for grid_lanes in grids {
            let mut score_sum: f64 = 0.0;
            let mut moves: Vec<(u32, LocationAddr)> = Vec::with_capacity(grid_lanes.len());

            for &lane_enc in &grid_lanes {
                if let Some(e) = entry_by_lane.get(&lane_enc) {
                    score_sum += e.score;
                    if let Some((_, dst)) = index.endpoints(&e.lane) {
                        moves.push((e.qid, dst));
                    }
                }
            }

            if moves.is_empty() {
                continue;
            }

            let move_set = MoveSet::from_encoded(grid_lanes);
            let new_config = config.with_moves(&moves);

            // Deduplicate identical movesets (e.g., distinct grid orderings
            // covering the same lane set).
            if candidates.iter().any(|c| c.move_set == move_set) {
                continue;
            }

            candidates.push(PackedCandidate {
                move_set,
                new_config,
                score_sum,
            });
        }
    }

    // Sort by score_sum desc, then by (moveset, config) for determinism.
    candidates.sort_by(|a, b| {
        b.score_sum
            .partial_cmp(&a.score_sum)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                cmp_moveset_config_tiebreak(&a.move_set, &a.new_config, &b.move_set, &b.new_config)
            })
    });

    candidates
}

fn decode_move_type(v: u8) -> Option<MoveType> {
    if v == MoveType::SiteBus as u8 {
        Some(MoveType::SiteBus)
    } else if v == MoveType::WordBus as u8 {
        Some(MoveType::WordBus)
    } else if v == MoveType::ZoneBus as u8 {
        Some(MoveType::ZoneBus)
    } else {
        None
    }
}

fn decode_direction(v: u8) -> Option<Direction> {
    if v == Direction::Forward as u8 {
        Some(Direction::Forward)
    } else if v == Direction::Backward as u8 {
        Some(Direction::Backward)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn first_outgoing(index: &LaneIndex, src: LocationAddr) -> LaneAddr {
        *index
            .outgoing_lanes(src)
            .first()
            .expect("at least one outgoing lane")
    }

    #[test]
    fn top_c_per_qubit_keeps_best_per_qubit() {
        let index = make_index();
        let lane_a = first_outgoing(&index, loc(0, 0));
        let lane_b = first_outgoing(&index, loc(0, 1));

        let scored = vec![
            ScoredLane {
                qid: 0,
                lane: lane_a,
                score: 1.0,
            },
            ScoredLane {
                qid: 0,
                lane: lane_b,
                score: 5.0,
            },
            ScoredLane {
                qid: 1,
                lane: lane_a,
                score: 2.0,
            },
        ];

        let topped = top_c_per_qubit(scored, 1);
        assert_eq!(topped.len(), 2);
        // qubit 0's best score is 5.0
        let q0 = topped.iter().find(|e| e.qid == 0).unwrap();
        assert_eq!(q0.score, 5.0);
        // qubit 1 has only one entry, score 2.0
        let q1 = topped.iter().find(|e| e.qid == 1).unwrap();
        assert_eq!(q1.score, 2.0);
    }

    #[test]
    fn top_c_per_qubit_fallback_when_no_positive() {
        let index = make_index();
        let lane_a = first_outgoing(&index, loc(0, 0));
        let lane_b = first_outgoing(&index, loc(0, 1));

        let scored = vec![
            ScoredLane {
                qid: 0,
                lane: lane_a,
                score: -1.0,
            },
            ScoredLane {
                qid: 1,
                lane: lane_b,
                score: -3.0,
            },
        ];

        let topped = top_c_per_qubit(scored, 5);
        assert_eq!(topped.len(), 1, "fallback retains exactly one entry");
        // The retained entry should be the one with highest score (-1.0).
        assert_eq!(topped[0].score, -1.0);
        assert_eq!(topped[0].qid, 0);
    }

    #[test]
    fn top_c_per_qubit_lane_tiebreak_ascending() {
        let index = make_index();
        let lane_a = first_outgoing(&index, loc(0, 0));
        let lane_b = first_outgoing(&index, loc(0, 1));
        // Same qubit, same score; lane with smaller encoded comes first.
        let (smaller, larger) = if lane_a.encode_u64() < lane_b.encode_u64() {
            (lane_a, lane_b)
        } else {
            (lane_b, lane_a)
        };
        let scored = vec![
            ScoredLane {
                qid: 0,
                lane: larger,
                score: 1.0,
            },
            ScoredLane {
                qid: 0,
                lane: smaller,
                score: 1.0,
            },
        ];
        let topped = top_c_per_qubit(scored, 1);
        assert_eq!(topped.len(), 1);
        assert_eq!(topped[0].lane.encode_u64(), smaller.encode_u64());
    }

    #[test]
    fn group_by_triplet_groups_and_sorts() {
        let index = make_index();
        let lane_a = first_outgoing(&index, loc(0, 0));
        let lane_b = first_outgoing(&index, loc(0, 1));

        let scored = vec![
            ScoredLane {
                qid: 0,
                lane: lane_a,
                score: 1.0,
            },
            ScoredLane {
                qid: 1,
                lane: lane_b,
                score: 2.0,
            },
        ];
        let groups = group_by_triplet(scored);
        // The number of groups depends on whether lanes share triplet
        // keys. Just check the invariant: groups are sorted by key.
        for w in groups.windows(2) {
            assert!(w[0].key <= w[1].key, "groups must be sorted by key");
        }
        // Total entries preserved.
        let total: usize = groups.iter().map(|g| g.entries.len()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn pack_aod_rectangles_produces_candidate() {
        let index = make_index();
        let config = Config::new([(0u32, loc(0, 0))]).unwrap();
        let lane = first_outgoing(&index, loc(0, 0));
        let scored = vec![ScoredLane {
            qid: 0,
            lane,
            score: 3.0,
        }];
        let groups = group_by_triplet(scored);
        let blocked = HashSet::new();
        let candidates = pack_aod_rectangles(groups, &config, &index, &blocked);
        assert!(
            !candidates.is_empty(),
            "should produce at least one candidate"
        );
        // All candidates must have positive score_sum (we fed in a +3.0).
        for c in &candidates {
            assert!(c.score_sum > 0.0);
        }
        // First candidate has the highest score_sum.
        let first = &candidates[0];
        for rest in &candidates[1..] {
            assert!(first.score_sum >= rest.score_sum);
        }
    }

    #[test]
    fn pack_aod_rectangles_skips_blocked_destinations() {
        let index = make_index();
        let config = Config::new([(0u32, loc(0, 0))]).unwrap();
        let lane = first_outgoing(&index, loc(0, 0));
        let (_, dst) = index.endpoints(&lane).expect("endpoints");
        // Block the destination.
        let blocked: HashSet<u64> = [dst.encode()].into_iter().collect();

        let scored = vec![ScoredLane {
            qid: 0,
            lane,
            score: 3.0,
        }];
        let groups = group_by_triplet(scored);
        let candidates = pack_aod_rectangles(groups, &config, &index, &blocked);
        // No candidate should land qubit 0 on the blocked destination.
        for c in &candidates {
            assert_ne!(c.new_config.location_of(0), Some(dst));
        }
    }
}
