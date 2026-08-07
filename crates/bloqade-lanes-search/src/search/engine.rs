//! Arch-bound state shared by every solver entry point.
//!
//! [`SearchEngine`] is the data layer below `MoveSearch` /
//! `TargetSolver` / the `CzPlacement` peers: it owns the [`LaneIndex`]
//! and the lazy-initialized architecture-derived caches
//! ([`EntanglingCache`] for Hungarian word-pair distances,
//! [`NoHomeCache`] for home-site precomputes). Build it once per
//! architecture, share it via [`std::sync::Arc`] across the
//! composition layers above.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use bloqade_lanes_bytecode_core::arch::query::ArchSpecLoadError;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_bytecode_core::arch::validate::ArchSpecError;

use crate::drivers::entropy::BlendedColumnCache;
use crate::ops::entangling::{self, WordPairDistances};
use crate::primitives::distance::DistanceTable;
use crate::primitives::lane_index::LaneIndex;

/// Cached architecture-dependent data for the entangling solver paths.
///
/// All fields depend only on the architecture (lane index), not on
/// per-call data (initial positions, CZ pairs). Built once on first
/// access via [`SearchEngine::entangling_cache`] and reused for all
/// subsequent calls.
pub(crate) struct EntanglingCache {
    pub ent_set: HashSet<(u64, u64)>,
    pub partner_map: HashMap<u64, u64>,
    pub dist_table: Arc<DistanceTable>,
    pub wpd: WordPairDistances,
}

/// Cached architecture-dependent data for the no-home solver path.
///
/// All fields depend only on the architecture (lane index). Built once
/// on first access via [`SearchEngine::nohome_cache`] and reused for
/// all subsequent calls.
pub(crate) struct NoHomeCache {
    pub home_locs: Vec<u64>,
    pub home_set: HashSet<u64>,
    pub dist_table: Arc<DistanceTable>,
}

/// Arch-bound state for the search-crate composition layer.
///
/// Construct once per architecture (it precomputes the
/// [`LaneIndex`]). The lazy caches initialize on first use and are
/// safe to share across threads via [`Arc<SearchEngine>`].
pub struct SearchEngine {
    index: LaneIndex,
    entangling_cache: OnceLock<EntanglingCache>,
    nohome_cache: OnceLock<NoHomeCache>,
    /// Cross-solve cache of entropy blended-distance columns; see
    /// [`BlendedColumnCache`]. Remove alongside the entropy driver.
    blended_cache: OnceLock<BlendedColumnCache>,
}

impl std::fmt::Debug for SearchEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchEngine")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl SearchEngine {
    /// Construct from an [`ArchSpec`] JSON string **without validating it**.
    ///
    /// Parses the JSON, builds the lane index (precomputes all lane
    /// lookups, endpoints, and positions).
    ///
    /// Prefer [`Self::from_json_validated`]: this constructor performs no
    /// structural validation, so it will happily build an engine on a spec
    /// whose buses are cyclic or have duplicate endpoints — invariants the
    /// search relies on rather than checks (see
    /// [`crate::ops::aod_grid`], which cannot tell a rotation from a chain).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let arch_spec = serde_json::from_str(json)?;
        Ok(Self::from_index(LaneIndex::new(arch_spec)))
    }

    /// Construct from an [`ArchSpec`] JSON string, rejecting a spec that does
    /// not satisfy [`ArchSpec::validate`].
    ///
    /// This is the loader every caller should use. The search layers treat
    /// per-bus acyclicity and endpoint uniqueness (#874) as *given*: rectangle
    /// growth exempts an occupied destination whose occupant moves in the same
    /// shot, which is what makes conveyor chains legal — and, on a cyclic bus,
    /// what would make a physically impossible rotation look legal too.
    /// Validating at load is what keeps that assumption true.
    pub fn from_json_validated(json: &str) -> Result<Self, ArchSpecLoadError> {
        let arch_spec = ArchSpec::from_json_validated(json)?;
        Ok(Self::from_index(LaneIndex::new(arch_spec)))
    }

    /// Construct from a borrowed [`ArchSpec`], rejecting a spec that does not
    /// satisfy [`ArchSpec::validate`]. Avoids the JSON round-trip that callers
    /// holding a wrapper around an `ArchSpec` would otherwise pay to
    /// materialize an owned spec.
    ///
    /// Validates for the same reason [`Self::from_json_validated`] does, and
    /// it matters more here: this is the constructor the Python placement
    /// layers use, so an unchecked version would let a user-supplied spec
    /// reach the search with a cyclic bus and have a physically impossible
    /// rotation routed as a legal AOD operation.
    ///
    /// [`Self::from_index`] remains the unchecked escape hatch, for callers
    /// that have already validated or are building a fixture by hand.
    pub fn from_arch_spec(arch_spec: &ArchSpec) -> Result<Self, Vec<ArchSpecError>> {
        arch_spec.validate()?;
        Ok(Self::from_index(LaneIndex::from_arch_spec(arch_spec)))
    }

    /// Construct from an existing [`LaneIndex`].
    pub fn from_index(index: LaneIndex) -> Self {
        Self {
            index,
            entangling_cache: OnceLock::new(),
            nohome_cache: OnceLock::new(),
            blended_cache: OnceLock::new(),
        }
    }

    /// Get or build the cross-solve entropy blended-column cache.
    pub(crate) fn blended_cache(&self) -> &BlendedColumnCache {
        self.blended_cache
            .get_or_init(|| BlendedColumnCache::new(&self.index))
    }

    /// Access the underlying lane index.
    pub fn index(&self) -> &LaneIndex {
        &self.index
    }

    /// Get or build the cached entangling precomputation.
    pub(crate) fn entangling_cache(&self) -> &EntanglingCache {
        self.entangling_cache.get_or_init(|| {
            let arch = self.index.arch_spec();
            let word_pairs = entangling::enumerate_word_pairs(arch);
            let ent_locs = entangling::all_entangling_locations(arch);
            let ent_set = entangling::build_entangling_set(arch);
            let partner_map = entangling::build_partner_map(&ent_set);
            // Always include time distances — callers with w_t=0.0 just
            // ignore them (hop-count fields are separate).
            let dist_table = Arc::new(
                DistanceTable::new(&ent_locs, &self.index).with_time_distances(&self.index),
            );
            let wpd =
                entangling::WordPairDistances::from_dist_table(&word_pairs, arch, &dist_table);
            EntanglingCache {
                ent_set,
                partner_map,
                dist_table,
                wpd,
            }
        })
    }

    /// Get or build the cached no-home precomputation.
    pub(crate) fn nohome_cache(&self) -> &NoHomeCache {
        self.nohome_cache.get_or_init(|| {
            let arch = self.index.arch_spec();
            let home_locs = entangling::home_sites(arch);
            let home_set: HashSet<u64> = home_locs.iter().copied().collect();
            let dist_table = Arc::new(
                DistanceTable::new(&home_locs, &self.index).with_time_distances(&self.index),
            );
            NoHomeCache {
                home_locs,
                home_set,
                dist_table,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::example_arch_json;
    use bloqade_lanes_bytecode_core::arch::addr::SiteRef;

    /// A spec whose bus is a rotation must be refused at load. Nothing below
    /// this point can catch it: rectangle growth exempts an occupied
    /// destination whose occupant moves in the same shot, which is exactly
    /// what a rotation looks like locally, so the search would route it as a
    /// legal AOD operation (see `ops::aod_grid`'s
    /// `the_grid_layer_relies_on_arch_level_acyclicity`).
    #[test]
    fn from_json_validated_rejects_a_cyclic_bus() {
        let mut spec: ArchSpec =
            serde_json::from_str(example_arch_json()).expect("arch json parses");
        let bus = &mut spec.zones[0].site_buses[0];
        bus.src = vec![SiteRef(0), SiteRef(1)];
        bus.dst = vec![SiteRef(1), SiteRef(0)];
        let json = serde_json::to_string(&spec).expect("spec serializes");

        let err =
            SearchEngine::from_json_validated(&json).expect_err("a cyclic bus must be rejected");
        assert!(
            matches!(err, ArchSpecLoadError::Validation(_)),
            "expected a validation error, got {err:?}"
        );

        // The unvalidated constructor still accepts it — which is precisely
        // why callers must use the validating one.
        assert!(SearchEngine::from_json(&json).is_ok());
    }

    #[test]
    fn from_json_validated_accepts_a_legal_spec() {
        assert!(SearchEngine::from_json_validated(example_arch_json()).is_ok());
    }
}
