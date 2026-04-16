//! Edge cost implementations.

use crate::config::Config;
use crate::graph::MoveSet;
use crate::traits::CostFn;

/// Uniform edge cost: every move step costs 1.0.
pub struct UniformCost;

impl CostFn for UniformCost {
    fn edge_cost(&self, _move_set: &MoveSet, _from: &Config, _to: &Config) -> f64 {
        1.0
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
