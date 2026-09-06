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

/// The largest AOD rectangle a single shot may drive, as a tone count per
/// axis: at most `x` distinct source columns and `y` distinct source rows.
///
/// This is a hardware parameter — the number of tones the AOD can hold on each
/// axis — and, once known, belongs on the architecture spec. Until that field
/// exists it travels per solve on [`SearchContext::capacity`], where `None`
/// means unlimited, which is what every path that predates the field assumes.
/// Generators that own a cap of their own combine the two with
/// [`AodCapacity::tighten`]; the shot space at a smaller cap is always a
/// subset of the shot space at a larger one, so tightening never admits a
/// shot the looser cap would have refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AodCapacity {
    /// Maximum number of distinct source columns (x tones) in one shot.
    pub x: usize,
    /// Maximum number of distinct source rows (y tones) in one shot.
    pub y: usize,
}

impl AodCapacity {
    /// Whether a rectangle spanning `nx` source columns and `ny` source rows
    /// fits within this capacity.
    #[inline]
    pub fn admits(self, nx: usize, ny: usize) -> bool {
        nx <= self.x && ny <= self.y
    }

    /// Componentwise minimum of two optional caps, where `None` is
    /// "unlimited" on either side: the result admits a rectangle exactly
    /// when both inputs do.
    pub fn tighten(a: Option<Self>, b: Option<Self>) -> Option<Self> {
        match (a, b) {
            (None, other) | (other, None) => other,
            (Some(a), Some(b)) => Some(Self {
                x: a.x.min(b.x),
                y: a.y.min(b.y),
            }),
        }
    }
}

/// Read-only context built once per solve() invocation.
pub struct SearchContext<'a> {
    pub index: &'a LaneIndex,
    pub dist_table: &'a DistanceTable,
    pub blocked: &'a HashSet<u64>,
    pub targets: &'a [(u32, u64)],
    /// CZ pairs for loose-goal search. `None` for fixed-target solves.
    /// Used by the heuristic generator to coordinate pair moves.
    pub cz_pairs: Option<&'a [(u32, u32)]>,
    /// The solve's effective AOD tone limit per axis; `None` is unlimited.
    ///
    /// Read by everything that assembles a shot's AOD rectangle, so that no
    /// generator emits a shot the hardware cannot drive. The authoritative
    /// home for the value is the architecture spec once it carries one; today
    /// it comes from `SolveOptions::aod_capacity`, and every path that does
    /// not set it passes `None`, which reproduces the uncapped behaviour that
    /// predates the field.
    pub capacity: Option<AodCapacity>,
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

    const A: AodCapacity = AodCapacity { x: 2, y: 5 };
    const B: AodCapacity = AodCapacity { x: 3, y: 4 };

    #[test]
    fn tighten_truth_table() {
        assert_eq!(AodCapacity::tighten(None, None), None);
        assert_eq!(AodCapacity::tighten(Some(A), None), Some(A));
        assert_eq!(AodCapacity::tighten(None, Some(A)), Some(A));
        assert_eq!(
            AodCapacity::tighten(Some(A), Some(B)),
            Some(AodCapacity { x: 2, y: 4 })
        );
        // Symmetric.
        assert_eq!(
            AodCapacity::tighten(Some(A), Some(B)),
            AodCapacity::tighten(Some(B), Some(A))
        );
    }

    #[test]
    fn admits_is_componentwise_and_inclusive() {
        assert!(A.admits(2, 5));
        assert!(A.admits(0, 0));
        assert!(!A.admits(3, 1));
        assert!(!A.admits(1, 6));
    }

    /// The tightened cap admits a rectangle exactly when both inputs do.
    #[test]
    fn tighten_admits_the_intersection() {
        let t = AodCapacity::tighten(Some(A), Some(B)).unwrap();
        for nx in 0..5 {
            for ny in 0..7 {
                assert_eq!(t.admits(nx, ny), A.admits(nx, ny) && B.admits(nx, ny));
            }
        }
    }
}
