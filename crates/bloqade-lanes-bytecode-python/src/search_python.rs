//! PyO3 bindings for the move synthesis solver.
//!
//! Exposes [`PyMoveSolver`] and [`PySolveResult`] to Python. Uses a JSON
//! bridge for `ArchSpec` — the solver is fully decoupled from the bytecode
//! Python bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_search::DeadlockPolicy;
use bloqade_lanes_search::solve::{
    InnerStrategy, MoveSolver, SolveOptions, SolveResult, SolveStatus, Strategy,
};

use crate::arch_python::PyArchSpec;

/// Result of a move synthesis solve.
///
/// Contains the sequence of move steps, the final qubit configuration,
/// and search statistics.
#[pyclass(
    name = "SolveResult",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PySolveResult {
    inner: SolveResult,
}

#[pymethods]
impl PySolveResult {
    /// Status of the solve: "solved", "unsolvable", or "budget_exceeded".
    #[getter]
    fn status(&self) -> &'static str {
        match self.inner.status {
            SolveStatus::Solved => "solved",
            SolveStatus::Unsolvable => "unsolvable",
            SolveStatus::BudgetExceeded => "budget_exceeded",
        }
    }

    /// Move layers: list of move steps, each a list of lane address tuples.
    ///
    /// Each lane is represented as `(direction, move_type, zone_id, word_id, site_id, bus_id)`
    /// where direction is 0=Forward/1=Backward and move_type is 0=SiteBus/1=WordBus/2=ZoneBus.
    #[getter]
    #[allow(clippy::type_complexity)]
    fn move_layers(&self) -> Vec<Vec<(u8, u8, u32, u32, u32, u32)>> {
        self.inner
            .move_layers
            .iter()
            .map(|ms| {
                ms.decode()
                    .into_iter()
                    .map(|lane| {
                        (
                            lane.direction as u8,
                            lane.move_type as u8,
                            lane.zone_id,
                            lane.word_id,
                            lane.site_id,
                            lane.bus_id,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// Goal configuration: list of (qubit_id, zone_id, word_id, site_id) tuples.
    #[getter]
    fn goal_config(&self) -> Vec<(u32, u32, u32, u32)> {
        self.inner
            .goal_config
            .iter()
            .map(|(qid, loc)| (qid, loc.zone_id, loc.word_id, loc.site_id))
            .collect()
    }

    /// Number of nodes expanded during search.
    #[getter]
    fn nodes_expanded(&self) -> u32 {
        self.inner.nodes_expanded
    }

    /// Total path cost.
    #[getter]
    fn cost(&self) -> f64 {
        self.inner.cost
    }

    /// Number of deadlocks encountered during search.
    #[getter]
    fn deadlocks(&self) -> u32 {
        self.inner.deadlocks
    }

    fn __repr__(&self) -> String {
        let status = match self.inner.status {
            SolveStatus::Solved => "solved",
            SolveStatus::Unsolvable => "unsolvable",
            SolveStatus::BudgetExceeded => "budget_exceeded",
        };
        format!(
            "SolveResult(status='{}', steps={}, cost={}, expanded={}, deadlocks={})",
            status,
            self.inner.move_layers.len(),
            self.inner.cost,
            self.inner.nodes_expanded,
            self.inner.deadlocks,
        )
    }
}

// ── parity_oracle: distance_table_lookup ───────────────────────────────────

/// Compute the blended distance table for a set of target locations.
///
/// Returns a Python dict keyed by `(src_encoded, tgt_encoded)` tuples with
/// blended float distances.  Intended for parity testing only; compiled only
/// when the `parity_oracle` feature is enabled.
#[cfg(feature = "parity_oracle")]
#[pyfunction]
pub fn distance_table_lookup(
    py: Python<'_>,
    arch_json: &str,
    targets: Vec<u64>,
    w_t: f64,
) -> PyResult<PyObject> {
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use bloqade_lanes_search::{DistanceTable, LaneIndex};
    use pyo3::types::PyDict;

    let arch: ArchSpec = serde_json::from_str(arch_json)
        .map_err(|e| PyValueError::new_err(format!("bad arch json: {e}")))?;
    let index = LaneIndex::new(arch);

    let table = if w_t > 0.0 {
        DistanceTable::new(&targets, &index).with_time_distances(&index)
    } else {
        DistanceTable::new(&targets, &index)
    };

    let fastest_opt = table.fastest_lane_us();

    let out = PyDict::new(py);
    for &target in &targets {
        for src in index.all_location_encodings() {
            let Some(hop) = table.distance(src, target) else {
                continue;
            };
            let blended = if w_t <= 0.0 || fastest_opt.is_none() {
                hop as f64
            } else {
                let fastest = fastest_opt.unwrap();
                let Some(time_us) = table.time_distance(src, target) else {
                    continue;
                };
                (1.0 - w_t) * hop as f64 + w_t * (time_us / fastest)
            };
            out.set_item((src, target), blended)?;
        }
    }
    Ok(out.into())
}

/// Score a moveset using the Rust entropy scorer.
///
/// Calls `bloqade_lanes_search::score_moveset` directly and returns the
/// scalar score. Intended for parity testing only; compiled only when the
/// `parity_oracle` feature is enabled.
///
/// Args:
///     arch_json: Architecture spec as a JSON string.
///     old_config: List of (qubit_id, encoded_location) pairs for pre-move config.
///     new_config: List of (qubit_id, encoded_location) pairs for post-move config.
///     targets:    List of (qubit_id, encoded_location) pairs for target positions.
///     blocked:    List of encoded location values that are immovable obstacles.
///     alpha, beta, gamma: Moveset scoring weights.
///     w_t:        Time-distance blend weight (0.0 = hop-count only).
#[cfg(feature = "parity_oracle")]
#[pyfunction]
#[pyo3(signature = (arch_json, old_config, new_config, targets, blocked, alpha, beta, gamma, w_t))]
pub fn entropy_score_moveset(
    arch_json: &str,
    old_config: Vec<(u32, u64)>,
    new_config: Vec<(u32, u64)>,
    targets: Vec<(u32, u64)>,
    blocked: Vec<u64>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    w_t: f64,
) -> PyResult<f64> {
    use std::collections::HashSet;

    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use bloqade_lanes_search::{
        Config, DistanceTable, EntropyParams, LaneIndex, SearchContext, score_moveset,
    };

    let arch: ArchSpec = serde_json::from_str(arch_json)
        .map_err(|e| PyValueError::new_err(format!("bad arch json: {e}")))?;
    let index = LaneIndex::new(arch);

    // Decode encoded locations into LocationAddr pairs for Config::new().
    let old_pairs: Vec<(u32, LocationAddr)> = old_config
        .iter()
        .map(|&(qid, enc)| (qid, LocationAddr::decode(enc)))
        .collect();
    let new_pairs: Vec<(u32, LocationAddr)> = new_config
        .iter()
        .map(|&(qid, enc)| (qid, LocationAddr::decode(enc)))
        .collect();

    let old_cfg = Config::new(old_pairs)
        .map_err(|e| PyValueError::new_err(format!("old_config error: {e}")))?;
    let new_cfg = Config::new(new_pairs)
        .map_err(|e| PyValueError::new_err(format!("new_config error: {e}")))?;

    // Build distance table from target location encodings.
    let target_encs: Vec<u64> = targets.iter().map(|&(_, enc)| enc).collect();
    let dist_table = if w_t > 0.0 {
        DistanceTable::new(&target_encs, &index).with_time_distances(&index)
    } else {
        DistanceTable::new(&target_encs, &index)
    };

    // Build blocked set.
    let blocked_set: HashSet<u64> = blocked.into_iter().collect();

    // Build old occupied set: locations from old_config + blocked.
    let mut occupied: HashSet<u64> = old_cfg.iter().map(|(_, loc)| loc.encode()).collect();
    occupied.extend(&blocked_set);

    let ctx = SearchContext {
        index: &index,
        dist_table: &dist_table,
        blocked: &blocked_set,
        targets: &targets,
    };

    let params = EntropyParams {
        alpha,
        beta,
        gamma,
        w_t,
        ..EntropyParams::default()
    };

    Ok(score_moveset(&old_cfg, &new_cfg, &occupied, &ctx, &params))
}

/// Generate ranked candidate movesets using entropy-weighted scoring.
///
/// Mirrors the Rust `generate_candidates` function and the Python
/// `HeuristicMoveGenerator.generate()` + `CandidateScorer` combo.
/// Intended for parity testing only; compiled only when the `parity_oracle`
/// feature is enabled.
///
/// Args:
///     arch_json:              Architecture spec as a JSON string.
///     config:                 List of (qubit_id, encoded_location) pairs for current config.
///     entropy:                Entropy level (controls distance vs. mobility weighting).
///     alpha, beta, gamma:     Moveset scoring weights.
///     w_d, w_m, w_t:          Per-qubit scoring weights (distance, mobility, time-blend).
///     e_max:                  Maximum entropy before forced backtrack/fallback.
///     max_candidates:         Maximum number of candidates to return.
///     max_movesets_per_group: Max movesets retained per bus group.
///     targets:                List of (qubit_id, encoded_location) pairs for target positions.
///     blocked:                List of encoded location values that are immovable obstacles.
///     seed:                   RNG seed (0 = no perturbation).
///
/// Returns:
///     List of (lane_encodings, new_config_pairs, score) triples where
///     lane_encodings is a list of u64 encoded lane addresses,
///     new_config_pairs is a list of (qubit_id, encoded_location) pairs, and
///     score is the moveset score (always 1.0 in current Rust implementation).
#[cfg(feature = "parity_oracle")]
#[pyfunction]
#[pyo3(signature = (
    arch_json, config, entropy,
    alpha, beta, gamma, w_d, w_m, w_t,
    e_max, max_candidates, max_movesets_per_group,
    targets, blocked, seed,
))]
#[allow(clippy::too_many_arguments)]
pub fn entropy_generate_candidates(
    arch_json: &str,
    config: Vec<(u32, u64)>,
    entropy: u32,
    alpha: f64,
    beta: f64,
    gamma: f64,
    w_d: f64,
    w_m: f64,
    w_t: f64,
    e_max: u32,
    max_candidates: usize,
    max_movesets_per_group: usize,
    targets: Vec<(u32, u64)>,
    blocked: Vec<u64>,
    seed: u64,
) -> PyResult<Vec<(Vec<u64>, Vec<(u32, u64)>, f64)>> {
    use std::collections::HashSet;

    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use bloqade_lanes_search::{
        Config, DistanceTable, EntropyParams, LaneIndex, SearchContext, generate_candidates,
    };

    let arch: ArchSpec = serde_json::from_str(arch_json)
        .map_err(|e| PyValueError::new_err(format!("bad arch json: {e}")))?;
    let index = LaneIndex::new(arch);

    let target_encs: Vec<u64> = targets.iter().map(|(_, t)| *t).collect();
    let dist_table = if w_t > 0.0 {
        DistanceTable::new(&target_encs, &index).with_time_distances(&index)
    } else {
        DistanceTable::new(&target_encs, &index)
    };

    let blocked_set: HashSet<u64> = blocked.into_iter().collect();

    let ctx = SearchContext {
        index: &index,
        dist_table: &dist_table,
        targets: &targets,
        blocked: &blocked_set,
    };

    let cfg_pairs: Vec<(u32, LocationAddr)> = config
        .into_iter()
        .map(|(q, e)| (q, LocationAddr::decode(e)))
        .collect();
    let cfg =
        Config::new(cfg_pairs).map_err(|e| PyValueError::new_err(format!("config error: {e}")))?;

    let params = EntropyParams {
        alpha,
        beta,
        gamma,
        w_d,
        w_m,
        w_t,
        e_max,
        max_candidates,
        max_movesets_per_group,
        ..EntropyParams::default()
    };

    let out = generate_candidates(&cfg, entropy, &params, &ctx, seed);

    Ok(out
        .into_iter()
        .map(|(ms, new_cfg, score)| {
            let lanes: Vec<u64> = ms.encoded_lanes().to_vec();
            let cfg_pairs: Vec<(u32, u64)> = new_cfg
                .as_entries()
                .iter()
                .map(|&(q, enc)| (q, enc))
                .collect();
            (lanes, cfg_pairs, score)
        })
        .collect())
}

/// Reusable move synthesis solver.
///
/// Constructed once from an architecture specification JSON string.
/// The constructor parses the JSON and precomputes lane indexes.
/// Then `solve()` can be called multiple times with different placements.
///
/// Works for both physical and logical architectures.
#[pyclass(name = "MoveSolver", frozen, module = "bloqade.lanes.bytecode._native")]
pub struct PyMoveSolver {
    inner: MoveSolver,
}

#[pymethods]
impl PyMoveSolver {
    /// Create a solver from an ArchSpec JSON string.
    #[new]
    fn new(arch_spec_json: &str) -> PyResult<Self> {
        let inner = MoveSolver::from_json(arch_spec_json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec JSON: {e}")))?;
        Ok(Self { inner })
    }

    /// Create a solver from a native ArchSpec object.
    #[staticmethod]
    fn from_arch_spec(arch: &PyArchSpec) -> PyResult<Self> {
        let json = serde_json::to_string(&arch.inner)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize arch spec: {e}")))?;
        let inner = MoveSolver::from_json(&json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec: {e}")))?;
        Ok(Self { inner })
    }

    /// Solve a move synthesis problem.
    ///
    /// Args:
    ///     initial: List of (qubit_id, zone_id, word_id, site_id) tuples for starting positions.
    ///     target: List of (qubit_id, zone_id, word_id, site_id) tuples for desired positions.
    ///     blocked: List of (zone_id, word_id, site_id) tuples for immovable obstacle locations.
    ///     max_expansions: Optional limit on node expansions.
    ///     strategy: Search strategy string.
    ///     top_c: Top bus options per qubit in the heuristic expander (default 3).
    ///     max_movesets_per_group: Max movesets per bus group (default 3).
    ///     max_goal_candidates: Number of goal candidates to collect in entropy search (default 3).
    ///     weight: Heuristic weight for A* (1.0 = standard, >1.0 = bounded suboptimal).
    ///     restarts: Number of parallel restarts with perturbed scoring (1 = no restarts).
    ///     lookahead: Enable 2-step lookahead scoring.
    ///     deadlock_policy: Deadlock handling: "skip" or "move_blockers".
    ///     w_t: Time-distance blend weight (0.0 = hop-count only, 1.0 = time only). Affects entropy strategy.
    ///
    /// Returns:
    ///     SolveResult with status indicating success/failure.
    #[pyo3(signature = (initial, target, blocked, max_expansions=None, strategy="astar", top_c=3, max_movesets_per_group=3, max_goal_candidates=3, weight=1.0, restarts=1, lookahead=false, deadlock_policy="skip", w_t=0.05))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        initial: Vec<(u32, u32, u32, u32)>,
        target: Vec<(u32, u32, u32, u32)>,
        blocked: Vec<(u32, u32, u32)>,
        max_expansions: Option<u32>,
        strategy: &str,
        top_c: usize,
        max_movesets_per_group: usize,
        max_goal_candidates: usize,
        weight: f64,
        restarts: u32,
        lookahead: bool,
        deadlock_policy: &str,
        w_t: f64,
    ) -> PyResult<PySolveResult> {
        // Validate: check for duplicate qubit IDs in target.
        {
            let mut seen = std::collections::HashSet::new();
            for &(qid, _, _, _) in &target {
                if !seen.insert(qid) {
                    return Err(PyValueError::new_err(format!(
                        "duplicate qubit_id {qid} in target placement"
                    )));
                }
            }
        }

        let initial_pairs: Vec<_> = initial
            .into_iter()
            .map(|(qid, zone_id, word_id, site_id)| {
                (
                    qid,
                    LocationAddr {
                        zone_id,
                        word_id,
                        site_id,
                    },
                )
            })
            .collect();

        let target_pairs: Vec<_> = target
            .into_iter()
            .map(|(qid, zone_id, word_id, site_id)| {
                (
                    qid,
                    LocationAddr {
                        zone_id,
                        word_id,
                        site_id,
                    },
                )
            })
            .collect();

        let blocked_locs: Vec<_> = blocked
            .into_iter()
            .map(|(zone_id, word_id, site_id)| LocationAddr {
                zone_id,
                word_id,
                site_id,
            })
            .collect();

        let strat = match strategy {
            "astar" => Strategy::AStar,
            "dfs" => Strategy::HeuristicDfs,
            "bfs" => Strategy::Bfs,
            "greedy" => Strategy::GreedyBestFirst,
            "ids" => Strategy::Ids,
            "cascade" | "cascade-ids" => Strategy::Cascade {
                inner: InnerStrategy::Ids,
            },
            "cascade-dfs" => Strategy::Cascade {
                inner: InnerStrategy::Dfs,
            },
            "cascade-entropy" => Strategy::Cascade {
                inner: InnerStrategy::Entropy,
            },
            "entropy" => Strategy::Entropy,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown strategy '{strategy}', expected: astar, dfs, bfs, greedy, ids, cascade, cascade-ids, cascade-dfs, cascade-entropy, entropy"
                )));
            }
        };

        let dl_policy = match deadlock_policy {
            "skip" => DeadlockPolicy::Skip,
            "move_blockers" => DeadlockPolicy::MoveBlockers,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unknown deadlock_policy '{deadlock_policy}', expected: skip, move_blockers"
                )));
            }
        };

        // Validate numeric parameters.
        if !weight.is_finite() || weight <= 0.0 {
            return Err(PyValueError::new_err(
                "weight must be a finite float greater than 0.0",
            ));
        }
        if !w_t.is_finite() || !(0.0..=1.0).contains(&w_t) {
            return Err(PyValueError::new_err(
                "w_t must be a finite float in the range [0.0, 1.0]",
            ));
        }

        let opts = SolveOptions {
            strategy: strat,
            top_c,
            max_movesets_per_group,
            max_goal_candidates,
            weight,
            restarts,
            lookahead,
            deadlock_policy: dl_policy,
            w_t,
        };

        // Release the GIL during search (pure Rust, no Python objects needed).
        let result = py
            .allow_threads(|| {
                self.inner.solve(
                    initial_pairs,
                    target_pairs,
                    blocked_locs,
                    max_expansions,
                    &opts,
                )
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(PySolveResult { inner: result })
    }

    fn __repr__(&self) -> String {
        "MoveSolver(...)".to_string()
    }
}
