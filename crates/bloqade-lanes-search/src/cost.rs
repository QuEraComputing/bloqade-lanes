//! Edge cost implementations.

use std::collections::HashMap;

use bloqade_lanes_bytecode_core::arch::addr::LaneAddr;

use crate::primitives::config::Config;
use crate::primitives::graph::MoveSet;
use crate::primitives::lane_index::LaneIndex;
use crate::search::options::ObjectiveKind;
use crate::traits::{CostFn, Objective, ObjectiveId};

/// The objective a solve was asked for, as one type the dispatch can hold and
/// hand to a driver that is generic over [`Objective`].
///
/// Delegates every method to the wrapped objective, `id()` included, so the
/// pairing assertion between an objective and its completion bound sees the
/// wrapped instance's identity. `Uniform` is therefore indistinguishable from
/// a bare [`UniformCost`] to every consumer.
pub enum SolveObjective {
    Uniform(UniformCost),
    WeightedDuration(WeightedDuration),
}

impl SolveObjective {
    /// Resolve an [`ObjectiveKind`] against the architecture. For
    /// `WeightedDuration { tau: None }` the normalizer is the fastest lane
    /// duration the index knows, or `1.0` when the spec carries no transport
    /// paths ([`WeightedDuration::new`] requires a positive finite `tau`).
    pub fn from_kind(kind: ObjectiveKind, index: &LaneIndex) -> Self {
        match kind {
            ObjectiveKind::Uniform => Self::Uniform(UniformCost),
            ObjectiveKind::WeightedDuration { tau } => {
                let tau = tau
                    .or_else(|| index.fastest_lane_duration_us())
                    .unwrap_or(1.0);
                Self::WeightedDuration(WeightedDuration::new(index, tau))
            }
        }
    }

    /// The kind this objective resolved to, with `tau` filled in.
    pub fn kind(&self) -> ObjectiveKind {
        match self {
            Self::Uniform(_) => ObjectiveKind::Uniform,
            Self::WeightedDuration(w) => ObjectiveKind::WeightedDuration { tau: Some(w.tau()) },
        }
    }
}

impl CostFn for SolveObjective {
    #[inline]
    fn edge_cost(&self, move_set: &MoveSet, from: &Config, to: &Config) -> f64 {
        match self {
            Self::Uniform(o) => o.edge_cost(move_set, from, to),
            Self::WeightedDuration(o) => o.edge_cost(move_set, from, to),
        }
    }
}

impl Objective for SolveObjective {
    #[inline]
    fn lane_weight(&self, lane: LaneAddr) -> f64 {
        match self {
            Self::Uniform(o) => o.lane_weight(lane),
            Self::WeightedDuration(o) => o.lane_weight(lane),
        }
    }

    fn min_shot_cost(&self) -> f64 {
        match self {
            Self::Uniform(o) => o.min_shot_cost(),
            Self::WeightedDuration(o) => o.min_shot_cost(),
        }
    }

    fn id(&self) -> ObjectiveId {
        match self {
            Self::Uniform(o) => o.id(),
            Self::WeightedDuration(o) => o.id(),
        }
    }
}

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
    /// Smallest value `norm_dur_of` can return, over the stored lanes *and*
    /// [`Self::missing_norm_dur`]. Doubles as the `edge_cost` fold seed, which
    /// is what makes `min_shot_cost` a floor on every shot including an empty
    /// one — see [`CostFn::edge_cost`].
    min_norm: f64,
    /// Order-independent digest of the `(lane, normalized duration)` pairs this
    /// instance was built from. Folded into [`Objective::id`] so two instances
    /// built from *different* architectures never compare equal: `tau` alone
    /// does not determine `lane_weight`, and `ObjectiveId` promises that equal
    /// ids agree on every input.
    arch_digest: u64,
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
        // Seeded with the fallback, because `norm_dur_of` returns it for any
        // lane absent from the map — one reached through a different accessor
        // than the one enumerated here. Seeding means `min_norm` bounds every
        // value `norm_dur_of` can produce, by construction, rather than by
        // assuming the two lane sets coincide.
        let mut min_norm = missing_norm_dur;
        let mut arch_digest = 0_u64;
        for (mt, bus_id, zone_id, dir) in index.bus_groups() {
            for &lane in index.lanes_for(mt, bus_id, zone_id, dir) {
                let d = index
                    .lane_duration_us(&lane)
                    .unwrap_or(MISSING_LANE_DURATION_US)
                    / tau;
                let enc = lane.encode_u64();
                norm_dur.insert(enc, d);
                min_norm = min_norm.min(d);
                // Order-independent (`wrapping_add` is commutative), so the
                // digest does not depend on `bus_groups` iteration order.
                arch_digest = arch_digest.wrapping_add(mix_lane_digest(enc, d));
            }
        }
        Self {
            norm_dur,
            missing_norm_dur,
            tau,
            min_norm,
            arch_digest,
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

/// Mix one `(lane, normalized duration)` pair into a digest value.
///
/// Only needs to separate architectures that differ in any lane weight, not to
/// resist collisions adversarially, so a cheap multiply-xor suffices.
#[inline]
fn mix_lane_digest(lane_enc: u64, norm_dur: f64) -> u64 {
    const ODD: u64 = 0x9E37_79B9_7F4A_7C15;
    let mixed = lane_enc.wrapping_mul(ODD).rotate_left(31) ^ norm_dur.to_bits().wrapping_mul(ODD);
    mixed ^ (mixed >> 29)
}

impl CostFn for WeightedDuration {
    /// `1 + max` over the shot's lanes, with the fold seeded at
    /// [`Self::min_norm`] rather than `0.0`.
    ///
    /// The seed is what makes C4 (`edge_cost >= min_shot_cost`) hold for *every*
    /// shot rather than only non-empty ones. A `MoveSet` with no lanes would
    /// otherwise fold to `0.0` and cost `1.0`, below the advertised floor of
    /// `1 + min_norm`; since `min_norm` bounds every value `norm_dur_of` can
    /// return, seeding with it leaves the result unchanged for any shot that
    /// does have lanes.
    fn edge_cost(&self, move_set: &MoveSet, _from: &Config, _to: &Config) -> f64 {
        let slowest = move_set
            .decode()
            .into_iter()
            .map(|lane| self.norm_dur_of(lane))
            .fold(self.min_norm, f64::max);
        1.0 + slowest
    }
}

impl Objective for WeightedDuration {
    fn lane_weight(&self, lane: LaneAddr) -> f64 {
        1.0 + self.norm_dur_of(lane)
    }

    fn min_shot_cost(&self) -> f64 {
        1.0 + self.min_norm
    }

    fn id(&self) -> ObjectiveId {
        ObjectiveId {
            kind: "weighted-duration",
            // `tau` alone is not enough: it determines the weights only *for a
            // fixed architecture*, so the arch digest has to participate or two
            // instances over different lane graphs would compare equal and a
            // bound built against one could prune a search accumulating the
            // other.
            params: self.arch_digest.rotate_left(1) ^ self.tau.to_bits(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::loc;

    /// C4 must hold for *every* shot the type can be handed, including a
    /// lane-less one. Regression: `edge_cost` folded from `0.0`, so an empty
    /// `MoveSet` cost `1.0` while `min_shot_cost` advertised `1 + min_norm`,
    /// breaking the floor that `floor(cost / min_shot_cost)` relies on to turn
    /// a cost budget into a depth budget.
    #[test]
    fn weighted_duration_floor_holds_for_an_empty_shot() {
        use crate::primitives::lane_index::LaneIndex;
        use crate::test_utils::example_arch_json;
        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        for tau in [0.5, 1.0, 10.0] {
            let objective = WeightedDuration::new(&index, tau);
            let empty = MoveSet::from_encoded(vec![]);
            let cost = objective.edge_cost(&empty, &config, &config);
            assert!(
                cost >= objective.min_shot_cost(),
                "tau={tau}: empty-shot cost {cost} below min_shot_cost {}",
                objective.min_shot_cost()
            );
        }
    }

    fn physical_index() -> crate::primitives::lane_index::LaneIndex {
        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
        let spec: ArchSpec = serde_json::from_str(include_str!(
            "../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json"
        ))
        .unwrap();
        crate::primitives::lane_index::LaneIndex::new(spec)
    }

    /// Both kinds satisfy the objective contract through the delegating
    /// wrapper, and `Uniform` carries `UniformCost`'s identity.
    #[test]
    fn solve_objective_delegates_and_satisfies_the_contract() {
        use crate::bounds::assert_objective_contract;
        use crate::primitives::lane_index::LaneIndex;
        use crate::test_utils::example_arch_json;
        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = LaneIndex::new(spec);
        let uniform = SolveObjective::from_kind(ObjectiveKind::Uniform, &index);
        assert_objective_contract(&uniform, &index);
        assert_eq!(uniform.id(), UniformCost.id());
        assert_eq!(uniform.kind(), ObjectiveKind::Uniform);

        let weighted =
            SolveObjective::from_kind(ObjectiveKind::WeightedDuration { tau: Some(2.0) }, &index);
        assert_objective_contract(&weighted, &index);
        assert_eq!(weighted.id(), WeightedDuration::new(&index, 2.0).id());
        assert_eq!(
            weighted.kind(),
            ObjectiveKind::WeightedDuration { tau: Some(2.0) }
        );
    }

    /// `tau: None` resolves to the fastest lane on a spec with paths and to
    /// `1.0` on one without.
    #[test]
    fn weighted_duration_tau_defaults_to_the_fastest_lane() {
        use crate::primitives::lane_index::LaneIndex;
        use crate::test_utils::example_arch_json;
        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let no_paths = LaneIndex::new(spec);
        assert_eq!(no_paths.fastest_lane_duration_us(), None);
        let o = SolveObjective::from_kind(ObjectiveKind::WeightedDuration { tau: None }, &no_paths);
        assert_eq!(o.kind(), ObjectiveKind::WeightedDuration { tau: Some(1.0) });

        let physical = physical_index();
        let fastest = physical
            .fastest_lane_duration_us()
            .expect("the physical spec carries transport paths");
        let o = SolveObjective::from_kind(ObjectiveKind::WeightedDuration { tau: None }, &physical);
        assert_eq!(
            o.kind(),
            ObjectiveKind::WeightedDuration { tau: Some(fastest) }
        );
    }

    #[test]
    fn uniform_cost_always_returns_one() {
        let cost = UniformCost;
        let config = Config::new([(0, loc(0, 0))]).unwrap();
        let ms = MoveSet::from_encoded(vec![]);
        assert_eq!(cost.edge_cost(&ms, &config, &config), 1.0);
    }
}
