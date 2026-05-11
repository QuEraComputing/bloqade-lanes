//! Trait-based wrappers around [`HopDistanceHeuristic`].
//!
//! [`MaxHopHeuristic`] is admissible (max over qubits).
//! [`SumHopHeuristic`] is not admissible but gives better ordering for IDS/DFS.

use crate::config::Config;
use crate::heuristic::HopDistanceHeuristic;
use crate::traits::Heuristic;

// ── MaxHopHeuristic ────────────────────────────────────────────────

/// Admissible heuristic: maximum hop distance over all qubits.
///
/// Delegates to [`HopDistanceHeuristic::estimate_max`].
pub struct MaxHopHeuristic<'a> {
    inner: HopDistanceHeuristic<'a>,
}

impl<'a> MaxHopHeuristic<'a> {
    pub fn new(inner: HopDistanceHeuristic<'a>) -> Self {
        Self { inner }
    }
}

impl Heuristic for MaxHopHeuristic<'_> {
    fn estimate(&self, config: &Config) -> f64 {
        self.inner.estimate_max(config)
    }
}

// ── SumHopHeuristic ────────────────────────────────────────────────

/// Non-admissible heuristic: sum of hop distances over all qubits.
///
/// Delegates to [`HopDistanceHeuristic::estimate_sum`].
pub struct SumHopHeuristic<'a> {
    inner: HopDistanceHeuristic<'a>,
}

impl<'a> SumHopHeuristic<'a> {
    pub fn new(inner: HopDistanceHeuristic<'a>) -> Self {
        Self { inner }
    }
}

impl Heuristic for SumHopHeuristic<'_> {
    fn estimate(&self, config: &Config) -> f64 {
        self.inner.estimate_sum(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::heuristic::{DistanceTable, HopDistanceHeuristic};
    use crate::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    fn make_index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        LaneIndex::new(spec)
    }

    fn make_table(
        targets: &[(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)],
        index: &LaneIndex,
    ) -> DistanceTable {
        let target_locs: Vec<u64> = targets.iter().map(|&(_, l)| l.encode()).collect();
        DistanceTable::new(&target_locs, index)
    }

    #[test]
    fn max_hop_matches_estimate_max() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(1, 5))];
        let table = make_table(&targets, &index);

        let inner = HopDistanceHeuristic::new(targets, &table);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]).unwrap();

        let expected = inner.estimate_max(&config);

        // Rebuild inner for the wrapper (consumed by first call is fine — estimate_max borrows).
        let inner2 = HopDistanceHeuristic::new(targets, &table);
        let heuristic = MaxHopHeuristic::new(inner2);
        assert_eq!(heuristic.estimate(&config), expected);
    }

    #[test]
    fn sum_hop_matches_estimate_sum() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(1, 5))];
        let table = make_table(&targets, &index);

        let inner = HopDistanceHeuristic::new(targets, &table);
        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]).unwrap();

        let expected = inner.estimate_sum(&config);

        let inner2 = HopDistanceHeuristic::new(targets, &table);
        let heuristic = SumHopHeuristic::new(inner2);
        assert_eq!(heuristic.estimate(&config), expected);
    }

    #[test]
    fn max_hop_at_target_is_zero() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let heuristic = MaxHopHeuristic::new(HopDistanceHeuristic::new(targets, &table));
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert_eq!(heuristic.estimate(&config), 0.0);
    }

    #[test]
    fn sum_hop_at_target_is_zero() {
        let index = make_index();
        let targets = [(0, loc(0, 5))];
        let table = make_table(&targets, &index);
        let heuristic = SumHopHeuristic::new(HopDistanceHeuristic::new(targets, &table));
        let config = Config::new([(0, loc(0, 5))]).unwrap();
        assert_eq!(heuristic.estimate(&config), 0.0);
    }

    #[test]
    fn sum_greater_or_equal_max() {
        let index = make_index();
        let targets = [(0, loc(0, 5)), (1, loc(1, 5))];
        let table = make_table(&targets, &index);

        let config = Config::new([(0, loc(0, 0)), (1, loc(0, 0))]).unwrap();

        let max_h = MaxHopHeuristic::new(HopDistanceHeuristic::new(targets, &table));
        let sum_h = SumHopHeuristic::new(HopDistanceHeuristic::new(targets, &table));

        assert!(sum_h.estimate(&config) >= max_h.estimate(&config));
    }
}
