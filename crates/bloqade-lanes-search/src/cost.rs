//! Edge cost implementations.

use bloqade_lanes_bytecode_core::arch::addr::LaneAddr;

use crate::primitives::config::Config;
use crate::primitives::graph::MoveSet;
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
