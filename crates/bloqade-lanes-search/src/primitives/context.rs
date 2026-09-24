//! Shared context and mutable state for search invocations.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::primitives::config::Config;
use crate::primitives::distance::DistanceTable;
use crate::primitives::graph::{MoveSet, NodeId};
use crate::primitives::lane_index::LaneIndex;

/// A candidate move produced by a generator, before scoring or cost computation.
#[derive(Clone)]
pub struct MoveCandidate {
    pub move_set: MoveSet,
    pub new_config: Config,
}

/// The AOD capacity type lives with the architecture spec; re-exported here,
/// where the search crate has always named it.
pub use bloqade_lanes_bytecode_core::arch::AodCapacity;

/// Read-only context built once per solve() invocation.
pub struct SearchContext<'a> {
    pub index: &'a LaneIndex,
    pub dist_table: &'a DistanceTable,
    pub blocked: &'a HashSet<u64>,
    pub targets: &'a [(u32, u64)],
    /// CZ pairs for loose-goal search. `None` for fixed-target solves.
    /// Used by the heuristic generator to coordinate pair moves.
    pub cz_pairs: Option<&'a [(u32, u32)]>,
}

/// Per-node state for entropy-guided search.
#[derive(Debug, Clone)]
pub struct EntropyNodeState {
    pub entropy: u32,
    pub candidates_tried: u32,
}

/// Mutable state for a single search run.
#[derive(Default)]
pub struct SearchState {
    pub entropy_map: HashMap<NodeId, EntropyNodeState>,
}

#[cfg(test)]
mod tests {
    use super::AodCapacity;

    fn cap(x: usize, y: usize) -> AodCapacity {
        AodCapacity::new(x, y).expect("non-zero")
    }

    /// A zero on either axis admits nothing at all, so it is not a capacity.
    #[test]
    fn a_zero_axis_is_refused() {
        assert!(AodCapacity::new(0, 1).is_none());
        assert!(AodCapacity::new(1, 0).is_none());
        assert!(AodCapacity::new(0, 0).is_none());
        let one = AodCapacity::new(1, 1).expect("one by one is a capacity");
        assert!(one.admits(1, 1));
        assert_eq!((one.x(), one.y()), (1, 1));
    }

    #[test]
    fn tighten_truth_table() {
        assert_eq!(AodCapacity::tighten(None, None), None);
        assert_eq!(AodCapacity::tighten(Some(cap(2, 5)), None), Some(cap(2, 5)));
        assert_eq!(AodCapacity::tighten(None, Some(cap(2, 5))), Some(cap(2, 5)));
        assert_eq!(
            AodCapacity::tighten(Some(cap(2, 5)), Some(cap(3, 4))),
            Some(cap(2, 4))
        );
        // Symmetric.
        assert_eq!(
            AodCapacity::tighten(Some(cap(2, 5)), Some(cap(3, 4))),
            AodCapacity::tighten(Some(cap(3, 4)), Some(cap(2, 5)))
        );
    }

    #[test]
    fn admits_is_componentwise_and_inclusive() {
        assert!(cap(2, 5).admits(2, 5));
        assert!(cap(2, 5).admits(0, 0));
        assert!(!cap(2, 5).admits(3, 1));
        assert!(!cap(2, 5).admits(1, 6));
    }

    /// The tightened cap admits a rectangle exactly when both inputs do.
    #[test]
    fn tighten_admits_the_intersection() {
        let t = AodCapacity::tighten(Some(cap(2, 5)), Some(cap(3, 4))).unwrap();
        for nx in 0..5 {
            for ny in 0..7 {
                assert_eq!(
                    t.admits(nx, ny),
                    cap(2, 5).admits(nx, ny) && cap(3, 4).admits(nx, ny)
                );
            }
        }
    }
}
