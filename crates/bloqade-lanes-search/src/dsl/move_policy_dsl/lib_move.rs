//! Move Policy DSL `lib.*` primitives.
//!
//! Distance/mobility helpers, plus `Config` and `MoveSet` Starlark
//! wrappers (placed here rather than in dsl-core to avoid a
//! search→dsl-core→search dep cycle — see plan §6 / Task 6).
//!
//! The move-query primitives expose legal raw lanes and a Rust-backed AOD
//! packer. Search policy choices such as lane scoring, top-c pruning, and
//! branch ordering are intentionally left to Starlark policy code.
//!
//! The `Ctx` handle (exposed to policies as `ctx`) lands in Task 16.

use std::collections::HashSet;
use std::sync::Arc;

use allocative::Allocative;
use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_dsl_core::primitives::types::{StarlarkLane, StarlarkLocation};
use starlark::starlark_module;
use starlark::values::dict::AllocDict;
use starlark::values::float::StarlarkFloat;
use starlark::values::list::{AllocList, ListRef};
use starlark::values::tuple::AllocTuple;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, UnpackValue, Value};

use crate::config::Config;
use crate::dsl::pipeline::{
    PackedCandidate, ScoredLane, group_by_triplet as pipeline_group_by_triplet,
    pack_aod_rectangles as pipeline_pack_aod_rectangles,
};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;

// ── StarlarkConfig ─────────────────────────────────────────────────────

/// Starlark-visible wrapper around [`Config`].
///
/// Exposed attributes:
/// - `len`  — number of qubits in the configuration.
/// - `hash` — FNV-1a hash of the configuration; stable and deterministic
///   across runs and platforms (see `Config::cached_hash`).
///
/// Exposed methods (via [`register_config_methods`]):
/// - `get(qid) -> Location | None`
/// - `iter() -> list[(qid, Location)]`
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkConfig(pub Config);

impl allocative::Allocative for StarlarkConfig {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let v = visitor.enter_self_sized::<Self>();
        v.exit();
    }
}

starlark::starlark_simple_value!(StarlarkConfig);

impl std::fmt::Display for StarlarkConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Config(len={})", self.0.len())
    }
}

#[starlark::values::starlark_value(type = "Config")]
impl<'v> StarlarkValue<'v> for StarlarkConfig {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "len" => Some(heap.alloc(self.0.len() as i32)),
            // FNV-1a hash — deterministic across runs and platforms.
            // `Config::cached_hash()` exposes the private field via a
            // dedicated accessor added to config.rs (Task 14 scope).
            "hash" => Some(heap.alloc(self.0.cached_hash() as i64)),
            _ => None,
        }
    }

    fn get_methods() -> Option<&'static starlark::environment::Methods> {
        static METHODS: starlark::environment::MethodsStatic =
            starlark::environment::MethodsStatic::new();
        METHODS.methods(register_config_methods)
    }
}

#[starlark_module]
fn register_config_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Get the location of qubit `qid`. Returns `None` if not in this config.
    fn get<'v>(this: &StarlarkConfig, qid: i32, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        match this.0.location_of(qid as u32) {
            Some(loc) => Ok(heap.alloc(StarlarkLocation(loc))),
            None => Ok(Value::new_none()),
        }
    }

    /// Iterate `(qid, loc)` pairs in qubit-id order. Returns a list of 2-tuples.
    fn iter<'v>(this: &StarlarkConfig, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        let pairs: Vec<Value<'v>> = this
            .0
            .iter()
            .map(|(qid, loc)| {
                heap.alloc(AllocTuple(vec![
                    heap.alloc(qid as i32),
                    heap.alloc(StarlarkLocation(loc)),
                ]))
            })
            .collect();
        Ok(heap.alloc(AllocList(pairs.into_iter())))
    }
}

// ── StarlarkMoveSet ────────────────────────────────────────────────────

// `StarlarkMoveSet` was moved to
// `bloqade_lanes_dsl_core::primitives::move_set` alongside its
// inner `MoveSet`, so the dsl-core `ArchSpec` methods can construct it
// directly without dsl-core depending on this crate. Re-exported so
// existing `crate::dsl::move_policy_dsl::lib_move::StarlarkMoveSet` imports
// keep working.
pub use bloqade_lanes_dsl_core::primitives::move_set::StarlarkMoveSet;

// ── Candidate wrapper types (ScoredLane / PackedCandidate) ────────────────
//
// These mirror the pure-Rust types in `crate::dsl::pipeline`. They
// expose `.qid`, `.lane`, `.score` (etc.) attributes so policy code reads
// them like records. We chose typed wrappers over dicts to avoid lossy
// round-trips (LaneAddr ⇄ encoded u64 ⇄ LaneAddr) before AOD packing.

/// Starlark-visible wrapper around [`ScoredLane`].
///
/// Exposed attributes:
/// - `qid`   — qubit id (`int`).
/// - `lane`  — `Lane` value.
/// - `score` — `float` score returned by the user's `score_fn`.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub(crate) struct StarlarkScoredLane(pub(crate) ScoredLane);

impl Allocative for StarlarkScoredLane {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let v = visitor.enter_self_sized::<Self>();
        v.exit();
    }
}

starlark::starlark_simple_value!(StarlarkScoredLane);

impl std::fmt::Display for StarlarkScoredLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ScoredLane(qid={}, lane={}, score={})",
            self.0.qid,
            self.0.lane.encode_u64(),
            self.0.score
        )
    }
}

#[starlark::values::starlark_value(type = "ScoredLane")]
impl<'v> StarlarkValue<'v> for StarlarkScoredLane {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "qid" => Some(heap.alloc(self.0.qid as i32)),
            "lane" => Some(heap.alloc(StarlarkLane(self.0.lane))),
            "score" => Some(heap.alloc(self.0.score)),
            _ => None,
        }
    }
}

/// Starlark-visible wrapper around [`PackedCandidate`].
///
/// Exposed attributes:
/// - `move_set`   — `MoveSet` value.
/// - `new_config` — `Config` value.
/// - `score_sum`  — `float` sum of contributing lane scores.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub(crate) struct StarlarkPackedCandidate(pub(crate) PackedCandidate);

impl Allocative for StarlarkPackedCandidate {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let v = visitor.enter_self_sized::<Self>();
        v.exit();
    }
}

starlark::starlark_simple_value!(StarlarkPackedCandidate);

impl std::fmt::Display for StarlarkPackedCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PackedCandidate(score_sum={}, lanes={})",
            self.0.score_sum,
            self.0.move_set.len()
        )
    }
}

#[starlark::values::starlark_value(type = "PackedCandidate")]
impl<'v> StarlarkValue<'v> for StarlarkPackedCandidate {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "move_set" => Some(heap.alloc(StarlarkMoveSet(self.0.move_set.clone()))),
            "new_config" => Some(heap.alloc(StarlarkConfig(self.0.new_config.clone()))),
            "score_sum" => Some(heap.alloc(self.0.score_sum)),
            _ => None,
        }
    }
}

// ── Ctx handle ──────────────────────────────────────────────────────────

/// Read-only Starlark context exposing per-solve targets, blocked
/// locations, and the architecture spec to a policy. Bound as the
/// `ctx` global by the kernel (Task 17). The fields mirror the
/// arguments the kernel receives: `targets` is the list of
/// `(qid, target_loc)` pairs, `blocked` is the set of immovable
/// obstacles, `arch_spec` is a shared `Arc<ArchSpec>` wrapper.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct Ctx {
    pub(super) targets: Vec<(u32, u64)>,
    pub(super) blocked: std::collections::HashSet<u64>,
    pub(super) arch_spec: bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec,
}

impl allocative::Allocative for Ctx {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let v = visitor.enter_self_sized::<Self>();
        v.exit();
    }
}

starlark::starlark_simple_value!(Ctx);

impl std::fmt::Display for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ctx(targets={}, blocked={})",
            self.targets.len(),
            self.blocked.len()
        )
    }
}

#[starlark::values::starlark_value(type = "Ctx")]
impl<'v> StarlarkValue<'v> for Ctx {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "arch_spec" => Some(heap.alloc(self.arch_spec.clone())),
            "targets" => Some(self.alloc_targets(heap)),
            "blocked" => Some(self.alloc_blocked(heap)),
            _ => None,
        }
    }
}

impl Ctx {
    /// Construct a `Ctx` from the kernel's per-solve inputs. Used by
    /// the kernel (Task 17) and by tests in this file.
    pub fn new(
        targets: Vec<(u32, u64)>,
        blocked: std::collections::HashSet<u64>,
        arch_spec: bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec,
    ) -> Self {
        Self {
            targets,
            blocked,
            arch_spec,
        }
    }

    fn alloc_targets<'v>(&self, heap: &'v Heap) -> Value<'v> {
        // Each target is a (qid: int, location: Location) 2-tuple.
        let pairs: Vec<Value<'v>> = self
            .targets
            .iter()
            .map(|&(qid, loc)| {
                let l = bloqade_lanes_dsl_core::primitives::types::StarlarkLocation(
                    bloqade_lanes_bytecode_core::arch::addr::LocationAddr::decode(loc),
                );
                heap.alloc(AllocTuple(vec![heap.alloc(qid as i32), heap.alloc(l)]))
            })
            .collect();
        heap.alloc(AllocList(pairs.into_iter()))
    }

    fn alloc_blocked<'v>(&self, heap: &'v Heap) -> Value<'v> {
        // Stable iteration order: sort the encoded values to keep
        // policy-side iteration deterministic across runs.
        let mut sorted: Vec<u64> = self.blocked.iter().copied().collect();
        sorted.sort_unstable();
        let locs: Vec<Value<'v>> = sorted
            .into_iter()
            .map(|loc_enc| {
                let l = bloqade_lanes_dsl_core::primitives::types::StarlarkLocation(
                    bloqade_lanes_bytecode_core::arch::addr::LocationAddr::decode(loc_enc),
                );
                heap.alloc(l)
            })
            .collect();
        heap.alloc(AllocList(locs.into_iter()))
    }
}

// ── LibMove handle ─────────────────────────────────────────────────────

/// Move-DSL `lib` handle. Exposes distance, mobility, legal-lane queries,
/// and the AOD packing primitive. The kernel constructs one `LibMove` per
/// solve and binds it as a Starlark global.
///
/// Starlark methods (registered via [`register_lib_methods`]):
/// - `hop_distance(from, to) -> int | None`
/// - `time_distance(from, to) -> float | None`
/// - `blended_distance(from, to, w_t) -> float`
/// - `fastest_lane_us() -> float`
/// - `mobility(loc, targets) -> float`
/// - `unresolved_qubits(config) -> [dict]`
/// - `legal_lanes(config, qid) -> [Lane]`
/// - `scored_lane(qid, lane, score) -> ScoredLane`
/// - `pack_aod_rectangles(scored, config, ctx) -> [PackedCandidate]`
#[derive(Clone, ProvidesStaticType, NoSerialize)]
pub struct LibMove {
    pub(super) index: Arc<LaneIndex>,
    pub(super) dist_table: Arc<DistanceTable>,
    /// Targets as `(qid, encoded_loc)` pairs. Used by query helpers to
    /// derive each qubit's target.
    pub(super) targets: Vec<(u32, u64)>,
    /// Locations blocked from use as move destinations.
    pub(super) blocked: HashSet<u64>,
}

impl allocative::Allocative for LibMove {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let v = visitor.enter_self_sized::<Self>();
        v.exit();
    }
}

impl std::fmt::Debug for LibMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LibMove").finish()
    }
}

impl std::fmt::Display for LibMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<lib>")
    }
}

starlark::starlark_simple_value!(LibMove);

#[starlark::values::starlark_value(type = "Lib")]
impl<'v> StarlarkValue<'v> for LibMove {
    fn get_methods() -> Option<&'static starlark::environment::Methods> {
        static METHODS: starlark::environment::MethodsStatic =
            starlark::environment::MethodsStatic::new();
        METHODS.methods(register_lib_methods)
    }
}

impl LibMove {
    fn occupied_locations(&self, config: &Config) -> HashSet<u64> {
        let mut occupied = HashSet::with_capacity(self.blocked.len() + config.len());
        occupied.extend(&self.blocked);
        for (_, loc) in config.iter() {
            occupied.insert(loc.encode());
        }
        occupied
    }

    fn unresolved_qubits_raw(&self, config: &Config) -> Vec<(u32, LocationAddr, LocationAddr)> {
        self.targets
            .iter()
            .filter_map(|&(qid, target_enc)| {
                let cur = config.location_of(qid)?;
                if cur.encode() == target_enc {
                    None
                } else {
                    Some((qid, cur, LocationAddr::decode(target_enc)))
                }
            })
            .collect()
    }
}

#[starlark_module]
fn register_lib_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Lane-hop distance from `from_loc` to `target_loc`. `None` if unreachable.
    fn hop_distance<'v>(
        this: &LibMove,
        from_loc: &StarlarkLocation,
        target_loc: &StarlarkLocation,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        match this
            .dist_table
            .distance(from_loc.0.encode(), target_loc.0.encode())
        {
            Some(d) => Ok(heap.alloc(d as i32)),
            None => Ok(Value::new_none()),
        }
    }

    /// Time-distance (µs). `None` if no time table or unreachable.
    fn time_distance<'v>(
        this: &LibMove,
        from_loc: &StarlarkLocation,
        target_loc: &StarlarkLocation,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        match this
            .dist_table
            .time_distance(from_loc.0.encode(), target_loc.0.encode())
        {
            Some(t) => Ok(heap.alloc(t)),
            None => Ok(Value::new_none()),
        }
    }

    /// Convex blend of hop and time distance with weight `w_t` ∈ [0, 1].
    ///
    /// Returns `(1 - w_t) * hop_distance + w_t * time_distance`.
    /// Unreachable locations contribute `INFINITY`.
    ///
    /// `w_t` accepts both integer and float Starlark values (via `StarlarkFloat`).
    fn blended_distance(
        this: &LibMove,
        from_loc: &StarlarkLocation,
        target_loc: &StarlarkLocation,
        w_t: StarlarkFloat,
    ) -> starlark::Result<f64> {
        let w_t = w_t.0;
        let h = this
            .dist_table
            .distance(from_loc.0.encode(), target_loc.0.encode())
            .map(|d| d as f64)
            .unwrap_or(f64::INFINITY);
        let t = this
            .dist_table
            .time_distance(from_loc.0.encode(), target_loc.0.encode())
            .unwrap_or(f64::INFINITY);
        Ok((1.0 - w_t) * h + w_t * t)
    }

    /// Fastest single-lane time across the architecture (µs).
    ///
    /// Returns `0.0` if no time data has been loaded.
    fn fastest_lane_us(this: &LibMove) -> starlark::Result<f64> {
        Ok(this.dist_table.fastest_lane_us().unwrap_or(0.0))
    }

    /// Return the mobility score of `loc`: Σ 1/(1+hop_distance) over outgoing lanes × targets.
    ///
    /// `targets` is a list of `(qid, target_loc)` 2-tuples (e.g., `ctx.targets`).
    /// Lanes that reach unreachable targets (no path) contribute nothing.
    fn mobility<'v>(
        this: &LibMove,
        loc: &StarlarkLocation,
        targets: &'v ListRef<'v>,
        heap: &'v Heap,
    ) -> starlark::Result<f64> {
        // Extract the target location (second element) from each (qid, loc) tuple.
        // In starlark-0.13, `Value::iterate(heap) -> Result<StarlarkIterator<'v>>`
        // works on any iterable value (list, tuple, etc.).
        let mut target_locs: Vec<u64> = Vec::with_capacity(targets.len());
        for tv in targets.iter() {
            // Collect tuple elements via iterate().
            let iter = tv
                .iterate(heap)
                .map_err(|_| mk_error("mobility: target entry must be iterable"))?;
            let elems: Vec<Value<'v>> = iter.collect();
            if elems.len() < 2 {
                return Err(mk_error("mobility: target tuple needs >= 2 elements"));
            }
            let target_v = elems[1];
            let loc_w = <&StarlarkLocation>::unpack_value(target_v)?.ok_or_else(|| {
                mk_error("mobility: target tuple second element must be a Location")
            })?;
            target_locs.push(loc_w.0.encode());
        }

        let mut sum = 0.0_f64;
        for &lane in this.index.outgoing_lanes(loc.0).iter() {
            let Some((_, dst)) = this.index.endpoints(&lane) else {
                continue;
            };
            let dst_enc = dst.encode();
            for &target_enc in &target_locs {
                if let Some(d) = this.dist_table.distance(dst_enc, target_enc) {
                    sum += 1.0 / (1.0 + d as f64);
                }
            }
        }
        Ok(sum)
    }

    // ── Strategy-neutral candidate helpers ────────────────────────────

    /// Return unresolved qubits as dicts: {"qid", "current", "target"}.
    fn unresolved_qubits<'v>(
        this: &LibMove,
        config: &StarlarkConfig,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let items: Vec<Value<'v>> = this
            .unresolved_qubits_raw(&config.0)
            .into_iter()
            .map(|(qid, cur, target)| {
                heap.alloc(AllocDict([
                    ("qid", heap.alloc(qid as i32)),
                    ("current", heap.alloc(StarlarkLocation(cur))),
                    ("target", heap.alloc(StarlarkLocation(target))),
                ]))
            })
            .collect();
        Ok(heap.alloc(AllocList(items.into_iter())))
    }

    /// Return outgoing legal lanes for `qid`, skipping occupied or blocked destinations.
    fn legal_lanes<'v>(
        this: &LibMove,
        config: &StarlarkConfig,
        qid: i32,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        if qid < 0 {
            return Ok(heap.alloc(AllocList(std::iter::empty::<Value<'v>>())));
        }
        let Some(cur) = config.0.location_of(qid as u32) else {
            return Ok(heap.alloc(AllocList(std::iter::empty::<Value<'v>>())));
        };
        let occupied = this.occupied_locations(&config.0);
        let mut lanes: Vec<_> = this
            .index
            .outgoing_lanes(cur)
            .iter()
            .copied()
            .filter(|lane| {
                let Some((_, dst)) = this.index.endpoints(lane) else {
                    return false;
                };
                !occupied.contains(&dst.encode())
            })
            .collect();
        lanes.sort_unstable_by_key(|lane| lane.encode_u64());
        let items = lanes.into_iter().map(StarlarkLane);
        Ok(heap.alloc(AllocList(items)))
    }

    /// Construct a `ScoredLane` record from policy-owned scoring logic.
    fn scored_lane<'v>(
        this: &LibMove,
        qid: i32,
        lane: &StarlarkLane,
        score: Value<'v>,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let _ = this;
        if qid < 0 {
            return Err(mk_error(format!(
                "scored_lane: qid must be non-negative, got {qid}"
            )));
        }
        let score = unpack_score(score)?;
        Ok(heap.alloc(StarlarkScoredLane(ScoredLane {
            qid: qid as u32,
            lane: lane.0,
            score,
        })))
    }

    /// Pack scored lanes into AOD-compatible `PackedCandidate` rectangles.
    ///
    /// Starlark owns which lanes to score and keep. Rust groups by
    /// `(move_type, bus_id, direction)` internally and applies the AOD
    /// rectangle packer, preserving physical legality.
    ///
    /// `ctx` is currently unused (reserved for the Task 16 `Ctx` wrapper).
    fn pack_aod_rectangles<'v>(
        this: &LibMove,
        scored: &'v ListRef<'v>,
        config: &StarlarkConfig,
        ctx: Value<'v>,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let _ = ctx;
        let raw_scored = unpack_scored_list(scored)?;
        let raw_groups = pipeline_group_by_triplet(raw_scored);

        let candidates =
            pipeline_pack_aod_rectangles(raw_groups, &config.0, &this.index, &this.blocked);
        let items: Vec<Value<'v>> = candidates
            .into_iter()
            .map(|c| heap.alloc(StarlarkPackedCandidate(c)))
            .collect();
        Ok(heap.alloc(AllocList(items.into_iter())))
    }
}

// ── pipeline helper functions ─────────────────────────────────────────

/// Unpack a Starlark `Value` as an `f64` score. Accepts `int` or `float`.
fn unpack_score(v: Value<'_>) -> starlark::Result<f64> {
    if let Some(sf) = <StarlarkFloat as UnpackValue>::unpack_value(v)? {
        Ok(sf.0)
    } else if let Some(i) = <i64 as UnpackValue>::unpack_value(v)? {
        Ok(i as f64)
    } else {
        Err(mk_error(format!("score must be a number, got {v}")))
    }
}

/// Unpack a `ListRef<StarlarkScoredLane>` to a `Vec<ScoredLane>`.
fn unpack_scored_list<'v>(list: &ListRef<'v>) -> starlark::Result<Vec<ScoredLane>> {
    list.iter()
        .map(|v| {
            <&StarlarkScoredLane>::unpack_value(v)?
                .map(|s| s.0.clone())
                .ok_or_else(|| mk_error("expected list of ScoredLane values"))
        })
        .collect()
}

// ── error helper ───────────────────────────────────────────────────────

/// Construct a `starlark::Error` with an "other" (generic) error payload.
///
/// Mirrors the inline pattern in `arch_spec.rs` / `check_location_group`.
fn mk_error(msg: impl Into<String>) -> starlark::Error {
    #[derive(Debug)]
    struct Msg(String);
    impl std::fmt::Display for Msg {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl std::error::Error for Msg {}
    starlark::Error::new_other(Msg(msg.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::graph::MoveSet;
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

    #[test]
    fn config_wrapper_reports_len() {
        let cfg = Config::new([(
            0u32,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )])
        .unwrap();
        let w = StarlarkConfig(cfg);
        assert_eq!(format!("{w}"), "Config(len=1)");
    }

    #[test]
    fn moveset_wrapper_reports_len() {
        let ms = MoveSet::from_encoded(vec![1, 2, 3]);
        let w = StarlarkMoveSet(ms);
        assert_eq!(format!("{w}"), "MoveSet(len=3)");
    }

    #[test]
    fn scored_lane_wrapper_display_includes_fields() {
        use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, MoveType};
        let lane = LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id: 1,
            site_id: 0,
            bus_id: 0,
        };
        let w = StarlarkScoredLane(ScoredLane {
            qid: 7,
            lane,
            score: 1.5,
        });
        let display = format!("{w}");
        assert!(display.contains("qid=7"), "display: {display}");
        assert!(display.contains("score=1.5"), "display: {display}");
    }

    #[test]
    fn packed_candidate_wrapper_display_includes_score() {
        let cfg = Config::new([(
            0u32,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )])
        .unwrap();
        let pc = PackedCandidate {
            move_set: MoveSet::from_encoded(vec![1, 2]),
            new_config: cfg,
            score_sum: 4.5,
        };
        let w = StarlarkPackedCandidate(pc);
        let display = format!("{w}");
        assert!(display.contains("score_sum=4.5"), "display: {display}");
        assert!(display.contains("lanes=2"), "display: {display}");
    }

    #[test]
    fn ctx_display_shows_target_and_blocked_counts() {
        use std::collections::HashSet;
        use std::sync::Arc;

        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
        use bloqade_lanes_bytecode_core::version::Version;
        use bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec;

        let arch = Arc::new(ArchSpec {
            version: Version::new(2, 0),
            words: vec![],
            zones: vec![],
            zone_buses: vec![],
            modes: vec![],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
            blockade_radius: None,
        });
        let arch_wrap = StarlarkArchSpec(arch);
        let targets = vec![
            (
                0u32,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 0,
                }
                .encode(),
            ),
            (
                1u32,
                LocationAddr {
                    zone_id: 0,
                    word_id: 0,
                    site_id: 1,
                }
                .encode(),
            ),
        ];
        let blocked: HashSet<u64> = HashSet::new();
        let ctx = Ctx::new(targets, blocked, arch_wrap);
        let s = format!("{ctx}");
        assert!(s.contains("targets=2"), "display: {s}");
        assert!(s.contains("blocked=0"), "display: {s}");
    }
}
