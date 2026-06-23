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

use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

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
}

impl std::fmt::Debug for SearchEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchEngine")
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl SearchEngine {
    /// Construct from an [`ArchSpec`] JSON string.
    ///
    /// Parses the JSON, builds the lane index (precomputes all lane
    /// lookups, endpoints, and positions).
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let arch_spec = serde_json::from_str(json)?;
        Ok(Self::from_index(LaneIndex::new(arch_spec)))
    }

    /// Construct from a borrowed [`ArchSpec`]. Avoids the JSON
    /// round-trip that callers holding a wrapper around an `ArchSpec`
    /// would otherwise pay to materialize an owned spec.
    pub fn from_arch_spec(arch_spec: &ArchSpec) -> Self {
        Self::from_index(LaneIndex::from_arch_spec(arch_spec))
    }

    /// Construct from an existing [`LaneIndex`].
    pub fn from_index(index: LaneIndex) -> Self {
        Self {
            index,
            entangling_cache: OnceLock::new(),
            nohome_cache: OnceLock::new(),
        }
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
