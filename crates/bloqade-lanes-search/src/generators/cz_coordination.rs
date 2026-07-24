//! Explicit CZ-coordination policy for [`HeuristicGenerator`].
//!
//! Replaces the implicit `ctx.cz_pairs.is_some()` branch with two named
//! policies:
//!
//! - [`FixedTargetCoordination`] — the legacy single-target default used by
//!   plain fixed-target callers (A*/IDS/cascade on main): no contested
//!   penalty, fallback width 1, and no pair-coordinated boosting.
//! - [`EntanglingCoordination`] — the loose-goal / no-home / receding-horizon
//!   path: subtracts a contested-destination penalty, widens the
//!   no-positive-score fallback for branch diversity, and boosts CZ-pair
//!   entries that share a bus triplet so they land in the same AOD grid.
//!
//! The policy is selected once at the top of
//! [`HeuristicGenerator::generate`](crate::generators::heuristic::HeuristicGenerator)
//! from `ctx.cz_pairs`, making the previously-implicit two-mode behavior
//! explicit while preserving it exactly.
//!
//! [`HeuristicGenerator`]: crate::generators::heuristic::HeuristicGenerator

use std::collections::{HashMap, HashSet};

use crate::generators::heuristic::ScoredTriple;
use crate::primitives::ordering::TripletKey;

/// Policy controlling how CZ pairing influences candidate scoring/selection.
///
/// Each method mirrors exactly one of the three formerly-inline
/// `ctx.cz_pairs.is_some()` decision sites in `HeuristicGenerator::generate`.
pub(crate) trait CzCoordination {
    /// Penalty subtracted from a candidate that routes to a *contested*
    /// destination (another qubit's still-unresolved target), gated on the
    /// candidate currently having a positive score.
    ///
    /// Default `0` (fixed-target: no penalty). [`EntanglingCoordination`]
    /// returns `1`.
    fn contested_penalty(&self) -> i32 {
        0
    }

    /// Number of fallback entries to keep when *no* candidate scores positive.
    ///
    /// Default `1` (fixed-target: legacy `truncate(1)`).
    /// [`EntanglingCoordination`] widens to `3` to give the search more
    /// branches to explore on jump-back (non-monotonic routes).
    fn fallback_width(&self) -> usize {
        1
    }

    /// Boost coordinated CZ-pair entries that share a bus triplet so they are
    /// more likely to end up in the same AOD grid (coordinated pair movement).
    ///
    /// Default: no-op (fixed-target). [`EntanglingCoordination`] applies the
    /// `+1` boost.
    fn boost_coordinated_pairs(&self, _selected: &mut Vec<(TripletKey, ScoredTriple)>) {}
}

/// Legacy single-target coordination: no penalty, width 1, no boost.
pub(crate) struct FixedTargetCoordination;

impl CzCoordination for FixedTargetCoordination {}

/// Entangling coordination driven by the active CZ pairs.
pub(crate) struct EntanglingCoordination<'a> {
    /// The CZ pairs to coordinate (`(qubit_a, qubit_b)`).
    pub pairs: &'a [(u32, u32)],
}

impl CzCoordination for EntanglingCoordination<'_> {
    fn contested_penalty(&self) -> i32 {
        1
    }

    fn fallback_width(&self) -> usize {
        3
    }

    fn boost_coordinated_pairs(&self, selected: &mut Vec<(TripletKey, ScoredTriple)>) {
        let pairs = self.pairs;

        // Build qubit → set of selected triplet keys.
        let mut keys_by_qubit: HashMap<u32, HashSet<TripletKey>> = HashMap::new();
        for entry in selected.iter() {
            keys_by_qubit
                .entry(entry.1.qubit_id)
                .or_default()
                .insert(entry.0);
        }

        // Find shared triplet keys for each CZ pair.
        let mut boost_set: HashSet<(TripletKey, u32)> = HashSet::new();
        for &(qa, qb) in pairs {
            if let (Some(keys_a), Some(keys_b)) = (keys_by_qubit.get(&qa), keys_by_qubit.get(&qb)) {
                for key in keys_a.intersection(keys_b) {
                    boost_set.insert((*key, qa));
                    boost_set.insert((*key, qb));
                }
            }
        }

        // Apply +1 boost to coordinated entries.
        if !boost_set.is_empty() {
            for entry in selected.iter_mut() {
                if boost_set.contains(&(entry.0, entry.1.qubit_id)) {
                    entry.1.score += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, MoveType};

    use super::*;

    // Two distinct bus triplets for boost tests; the specific variants are
    // irrelevant — only triplet equality/inequality matters.
    fn key_a() -> TripletKey {
        TripletKey::new(MoveType::WordBus, 1, Direction::Backward)
    }
    fn key_b() -> TripletKey {
        TripletKey::new(MoveType::ZoneBus, 2, Direction::Backward)
    }

    fn triple(qubit_id: u32, score: i32) -> ScoredTriple {
        ScoredTriple {
            qubit_id,
            score,
            lane_encoded: 0,
            dst_encoded: 0,
        }
    }

    // -- contested_penalty pins --

    #[test]
    fn fixed_target_penalty_is_zero() {
        assert_eq!(FixedTargetCoordination.contested_penalty(), 0);
    }

    #[test]
    fn entangling_penalty_is_one() {
        let pairs = [(0u32, 1u32)];
        assert_eq!(
            EntanglingCoordination { pairs: &pairs }.contested_penalty(),
            1
        );
    }

    // -- fallback_width pins --

    #[test]
    fn fixed_target_fallback_width_is_one() {
        assert_eq!(FixedTargetCoordination.fallback_width(), 1);
    }

    #[test]
    fn entangling_fallback_width_is_three() {
        let pairs = [(0u32, 1u32)];
        assert_eq!(EntanglingCoordination { pairs: &pairs }.fallback_width(), 3);
    }

    // -- boost_coordinated_pairs pins --

    #[test]
    fn fixed_target_boost_is_noop() {
        // key_a shared by qubits 0 and 1 — would be boosted in entangling
        // mode, must be untouched in fixed mode.
        let mut selected = vec![(key_a(), triple(0, 5)), (key_a(), triple(1, 5))];
        let before = selected.iter().map(|e| e.1.score).collect::<Vec<_>>();
        FixedTargetCoordination.boost_coordinated_pairs(&mut selected);
        let after = selected.iter().map(|e| e.1.score).collect::<Vec<_>>();
        assert_eq!(before, after, "fixed-target boost must be a no-op");
    }

    #[test]
    fn entangling_boosts_shared_triplet_for_both_pair_members() {
        // qubits 0 and 1 form a CZ pair and both have an entry on triplet
        // key_a: both get +1. The key_b entry for qubit 0 is not
        // shared and must NOT be boosted.
        let pairs = [(0u32, 1u32)];
        let policy = EntanglingCoordination { pairs: &pairs };
        let mut selected = vec![
            (key_a(), triple(0, 5)),
            (key_a(), triple(1, 7)),
            (key_b(), triple(0, 3)),
        ];
        policy.boost_coordinated_pairs(&mut selected);
        assert_eq!(selected[0].1.score, 6, "shared entry for q0 boosted");
        assert_eq!(selected[1].1.score, 8, "shared entry for q1 boosted");
        assert_eq!(selected[2].1.score, 3, "non-shared entry unchanged");
    }

    #[test]
    fn entangling_no_boost_when_triplet_not_shared() {
        // qubits 0 and 1 are paired but sit on different triplets: no boost.
        let pairs = [(0u32, 1u32)];
        let policy = EntanglingCoordination { pairs: &pairs };
        let mut selected = vec![(key_a(), triple(0, 5)), (key_b(), triple(1, 7))];
        policy.boost_coordinated_pairs(&mut selected);
        assert_eq!(selected[0].1.score, 5);
        assert_eq!(selected[1].1.score, 7);
    }

    #[test]
    fn entangling_no_boost_when_qubit_not_in_pair() {
        // qubit 2 is unpaired; shares a triplet with qubit 0 but 0's partner
        // (1) is absent, so nothing is boosted.
        let pairs = [(0u32, 1u32)];
        let policy = EntanglingCoordination { pairs: &pairs };
        let mut selected = vec![(key_a(), triple(0, 5)), (key_a(), triple(2, 7))];
        policy.boost_coordinated_pairs(&mut selected);
        assert_eq!(selected[0].1.score, 5);
        assert_eq!(selected[1].1.score, 7);
    }
}
