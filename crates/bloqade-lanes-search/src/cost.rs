//! Edge cost implementations.

use std::collections::HashMap;

use bloqade_lanes_bytecode_core::arch::addr::LaneAddr;

use crate::primitives::config::Config;
use crate::primitives::graph::MoveSet;
use crate::primitives::lane_index::LaneIndex;
use crate::traits::{CostFn, Objective, ObjectiveId};

/// Uniform edge cost: every move step costs 1.0.
///
/// As an [`Objective`] this is "minimize the number of movesets" — the
/// objective every search driver in this crate currently accumulates `g` for.
/// Because each shot costs exactly `1.0`, `g(node) == depth(node)` for every
/// node, which several call sites rely on (see
/// [`Objective::min_shot_cost`] for the sanctioned way to depend on it).
pub struct UniformCost;

impl CostFn for UniformCost {
    fn edge_cost(&self, _move_set: &MoveSet, _from: &Config, _to: &Config) -> f64 {
        1.0
    }
}

impl Objective for UniformCost {
    /// A shot costs `1.0` and contains at least one lane, so `1.0` is the
    /// tightest per-lane floor satisfying C3 with equality.
    fn lane_weight(&self, _lane: LaneAddr) -> f64 {
        1.0
    }

    fn min_shot_cost(&self) -> f64 {
        1.0
    }

    fn id(&self) -> ObjectiveId {
        ObjectiveId {
            kind: "uniform",
            params: 0,
        }
    }
}

/// Duration-weighted objective: a shot costs `1 + dur(shot) / tau`.
///
/// A shot's duration is the **max** over its lanes, not the sum — a moveset's
/// lanes transport in parallel, which is also how
/// [`approx_layer_time_us`](crate::drivers::entropy) prices a layer. Filler
/// lanes that AOD-rectangle completion adds are included, which only raises a
/// shot's cost and so cannot break C3.
///
/// Both terms are additive per shot, satisfying C1. C3 holds because the max
/// over a shot's lanes dominates each individual lane's own weight.
///
/// # `tau`
///
/// There is no default. `tau` normalizes a duration into "moveset-equivalents"
/// and thereby sets the trade between plan length and plan time, which is a
/// deliberate objective-policy choice rather than something this type should
/// pick for a caller. `LaneIndex::fastest_lane_duration_us()` is the natural
/// arch-derived value if you want one.
///
/// This objective is **not** wired into any production path: the solver
/// pipeline runs [`UniformCost`], and `SolveResult::cost` is read as a moveset
/// count downstream. It exists so the driver's objective-swappability is
/// exercised by tests.
pub struct WeightedDuration {
    /// `lane encoding → dur(lane) / tau`, precomputed so `edge_cost` is a
    /// lookup rather than a per-expansion recomputation, and so the type owns
    /// its arch data (no lifetime parameter reaching the bound types).
    norm_dur: HashMap<u64, f64>,
    /// Fallback for lanes with no transport path, matching the convention
    /// `approx_layer_time_us` uses. Must be identical in `edge_cost` and
    /// `lane_weight` or C3 breaks.
    missing_norm_dur: f64,
    tau: f64,
    min_shot: f64,
}

impl WeightedDuration {
    /// Build from an architecture. Panics if `tau` is not positive and finite.
    pub fn new(index: &LaneIndex, tau: f64) -> Self {
        assert!(
            tau > 0.0 && tau.is_finite(),
            "tau must be positive and finite, got {tau}"
        );
        let missing_norm_dur = MISSING_LANE_DURATION_US / tau;
        let mut norm_dur = HashMap::new();
        let mut min_norm = f64::INFINITY;
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let d = index
                    .lane_duration_us(&lane)
                    .unwrap_or(MISSING_LANE_DURATION_US)
                    / tau;
                norm_dur.insert(lane.encode_u64(), d);
                min_norm = min_norm.min(d);
            }
        }
        let min_shot = 1.0 + if min_norm.is_finite() { min_norm } else { 0.0 };
        Self {
            norm_dur,
            missing_norm_dur,
            tau,
            min_shot,
        }
    }

    pub fn tau(&self) -> f64 {
        self.tau
    }

    #[inline]
    fn norm_dur_of(&self, lane: LaneAddr) -> f64 {
        self.norm_dur
            .get(&lane.encode_u64())
            .copied()
            .unwrap_or(self.missing_norm_dur)
    }
}

/// Duration assumed for a lane with no transport-path data, matching the
/// entropy driver's `approx_layer_time_us` convention.
const MISSING_LANE_DURATION_US: f64 = 1.0;

impl CostFn for WeightedDuration {
    fn edge_cost(&self, move_set: &MoveSet, _from: &Config, _to: &Config) -> f64 {
        let slowest = move_set
            .decode()
            .into_iter()
            .map(|lane| self.norm_dur_of(lane))
            .fold(0.0_f64, f64::max);
        1.0 + slowest
    }
}

impl Objective for WeightedDuration {
    fn lane_weight(&self, lane: LaneAddr) -> f64 {
        1.0 + self.norm_dur_of(lane)
    }

    fn min_shot_cost(&self) -> f64 {
        self.min_shot
    }

    fn id(&self) -> ObjectiveId {
        ObjectiveId {
            kind: "weighted-duration",
            // `tau` is the only parameter, and it fully determines both
            // `edge_cost` and `lane_weight` for a given architecture.
            params: self.tau.to_bits(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::loc;

    #[test]
    fn uniform_cost_always_returns_one() {
        let cost = UniformCost;
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let ms = MoveSet::from_encoded(vec![]);
        assert_eq!(cost.edge_cost(&ms, &config, &config), 1.0);
    }
}
