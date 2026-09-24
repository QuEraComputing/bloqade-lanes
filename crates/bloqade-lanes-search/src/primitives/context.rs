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
/// Both axes are at least one, and the fields are private so that stays true.
/// A zero on either axis would admit no rectangle at all, not even a single
/// atom, so every shot any generator could propose would violate it — while
/// the paths that emit a lone single-lane shot without consulting the cap (the
/// deadlock escapes, and both entropy fallbacks) would keep emitting it. There
/// is no useful behaviour to define for such a cap, so it is refused at
/// construction rather than defended against at every use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AodCapacity {
    x: usize,
    y: usize,
}

impl AodCapacity {
    /// A capacity of `x` source columns by `y` source rows, or `None` when
    /// either axis is zero. See the type's docs for why zero is refused.
    pub const fn new(x: usize, y: usize) -> Option<Self> {
        if x == 0 || y == 0 {
            return None;
        }
        Some(Self { x, y })
    }

    /// Maximum number of distinct source columns (x tones) in one shot.
    #[inline]
    pub fn x(self) -> usize {
        self.x
    }

    /// Maximum number of distinct source rows (y tones) in one shot.
    #[inline]
    pub fn y(self) -> usize {
        self.y
    }

    /// Whether a rectangle spanning `nx` source columns and `ny` source rows
    /// fits within this capacity.
    #[inline]
    pub fn admits(self, nx: usize, ny: usize) -> bool {
        nx <= self.x && ny <= self.y
    }

    /// Componentwise minimum of two optional caps, where `None` is
    /// "unlimited" on either side: the result admits a rectangle exactly
    /// when both inputs do.
    ///
    /// Total: both inputs are already non-zero on both axes, so their
    /// componentwise minimum is too.
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
