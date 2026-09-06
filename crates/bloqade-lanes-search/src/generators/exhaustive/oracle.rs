//! Naive oracle for [`ExhaustiveGenerator`]: the acceptance test that decides
//! whether the generator may be called exhaustive.
//!
//! The generator is a pruned enumeration, and a pruned enumeration can drop
//! cases silently. So the reference here is deliberately naive and shares no
//! code with it: lanes are read from the [`ArchSpec`] directly rather than
//! through `LaneIndex`, every lane subset of a bus group is tried by brute
//! force, and a subset is kept iff the *validator* accepts it — `check_lanes`
//! for the static rules (S1–S5) and `validate_moves` for the state-dependent
//! ones (D1–D3), against a state that holds the blocked sites as phantom
//! atoms — plus the three search-model rules stated in the spec: no blocked
//! source (B1), at least one mover (B2), within capacity (B3). Surviving
//! subsets are mapped to their mover-tight representative, which is
//! re-validated rather than trusted, and the generator's output must equal
//! that set of representatives per group: nothing missing, nothing extra, no
//! duplicates, one shot per child configuration.
//!
//! See the spec's *Acceptance criterion and tests*.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, LocationAddr, MoveType};
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_bytecode_core::atom_state::AtomStateData;
use rand::rngs::SmallRng;
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};

use super::{ExhaustiveGenerator, ExhaustivePrecondition, SeedPolicy};
use crate::primitives::config::Config;
use crate::primitives::context::{AodCapacity, MoveCandidate, SearchContext, SearchState};
use crate::primitives::distance::DistanceTable;
use crate::primitives::graph::NodeId;
use crate::primitives::lane_index::LaneIndex;
use crate::traits::MoveGenerator;

/// A bus group as the oracle names it: the four fields every lane of a shot
/// must share (S3). Ordered by field, so reports are stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Group {
    pub(super) move_type: MoveType,
    pub(super) bus_id: u32,
    pub(super) zone_id: u32,
    pub(super) direction: Direction,
}

impl Group {
    fn of(lane: &LaneAddr) -> Self {
        Self {
            move_type: lane.move_type,
            bus_id: lane.bus_id,
            zone_id: lane.zone_id,
            direction: lane.direction,
        }
    }
}

/// A shot as the oracle stores it: sorted encoded lanes.
type Shot = Vec<u64>;

/// Groups larger than this are not power-set enumerated; the caller supplies
/// a subset-size cap instead.
const POWER_SET_LIMIT: usize = 12;

/// One lane of a group with what the oracle needs about it, resolved through
/// the spec's own endpoint and position queries.
#[derive(Debug, Clone, Copy)]
struct OracleLane {
    lane: LaneAddr,
    src: LocationAddr,
    dst: LocationAddr,
    /// Source position bits, the coordinates the search builds rectangles on.
    src_x: u64,
    src_y: u64,
}

/// The spec's bus groups and their lanes, enumerated once per spec.
pub(super) struct Oracle {
    spec: ArchSpec,
    groups: BTreeMap<Group, Vec<OracleLane>>,
}

impl Oracle {
    pub(super) fn new(spec: ArchSpec) -> Self {
        let mut groups = BTreeMap::new();
        for group in group_keys(&spec) {
            let lanes: Vec<OracleLane> = group_lanes(&spec, group)
                .into_iter()
                .map(|lane| {
                    let (src, dst) = spec
                        .lane_endpoints(&lane)
                        .expect("check_lane accepted the lane, so it resolves");
                    let (x, y) = spec
                        .location_position(&src)
                        .expect("a lane source is a location with a position");
                    OracleLane {
                        lane,
                        src,
                        dst,
                        src_x: x.to_bits(),
                        src_y: y.to_bits(),
                    }
                })
                .collect();
            if !lanes.is_empty() {
                groups.insert(group, lanes);
            }
        }
        Self { spec, groups }
    }

    /// Every location that is a lane endpoint, sorted: the sites a sweep may
    /// place atoms and blocked markers on.
    pub(super) fn endpoints(&self) -> Vec<LocationAddr> {
        let mut seen: BTreeMap<u64, LocationAddr> = BTreeMap::new();
        for lanes in self.groups.values() {
            for l in lanes {
                seen.insert(l.src.encode(), l.src);
                seen.insert(l.dst.encode(), l.dst);
            }
        }
        seen.into_values().collect()
    }

    /// The lanes of each group as the oracle enumerated them, encoded.
    pub(super) fn lane_sets(&self) -> BTreeMap<Group, BTreeSet<u64>> {
        self.groups
            .iter()
            .map(|(g, lanes)| (*g, lanes.iter().map(|l| l.lane.encode_u64()).collect()))
            .collect()
    }

    /// The validator's verdict on one lane subset at `config`, with `blocked`
    /// held by phantom atoms, plus the search-model rules B1–B3.
    ///
    /// Returns the child configuration the execution model lands on when the
    /// shot is accepted, `Err(reason)` otherwise. The reason is a short tag
    /// used to classify the generator's extras.
    fn accept(
        &self,
        subset: &[OracleLane],
        config: &Config,
        blocked: &HashSet<u64>,
        cap: Option<AodCapacity>,
    ) -> Result<Config, &'static str> {
        let lanes: Vec<LaneAddr> = subset.iter().map(|l| l.lane).collect();
        if !self.spec.check_lanes(&lanes).is_empty() {
            return Err("check_lanes");
        }
        // B1, source half: a blocked site is immovable, so it may not be a
        // lane source even as a filler.
        if subset.iter().any(|l| blocked.contains(&l.src.encode())) {
            return Err("blocked source (B1)");
        }
        // B2: at least one lane source holds an atom.
        if !subset.iter().any(|l| config.is_occupied(l.src)) {
            return Err("no mover (B2)");
        }
        // B3: unique source columns and rows within capacity.
        if let Some(cap) = cap {
            let xs: BTreeSet<u64> = subset.iter().map(|l| l.src_x).collect();
            let ys: BTreeSet<u64> = subset.iter().map(|l| l.src_y).collect();
            if !cap.admits(xs.len(), ys.len()) {
                return Err("over capacity (B3)");
            }
        }
        // D1–D3 through the execution model, with phantoms on blocked sites
        // so the destination half of B1 is D2.
        let mut atoms: Vec<(u32, LocationAddr)> = config.iter().collect();
        let mut phantom_sites: Vec<u64> = blocked
            .iter()
            .copied()
            .filter(|enc| !config.is_occupied(LocationAddr::decode(*enc)))
            .collect();
        phantom_sites.sort_unstable();
        let first_phantom = u32::MAX - phantom_sites.len() as u32;
        atoms.extend(
            phantom_sites
                .iter()
                .enumerate()
                .map(|(i, &enc)| (u32::MAX - i as u32, LocationAddr::decode(enc))),
        );
        let state = AtomStateData::from_locations(&atoms);
        let validated = match state.validate_moves(&lanes, &self.spec) {
            Ok(v) => v,
            Err(_) => return Err("validate_moves (D2/D3)"),
        };
        let after = state
            .apply_validated(&validated)
            .expect("a freshly validated token applies");
        let child = Config::new(
            after
                .qubit_to_locations
                .into_iter()
                .filter(|&(q, _)| q <= first_phantom),
        )
        .expect("the execution model never duplicates a qubit");
        Ok(child)
    }

    /// The mover-tight representative of an accepted subset: its lanes inside
    /// the product of the columns and rows that hold a mover.
    fn tight(&self, subset: &[OracleLane], config: &Config) -> Vec<OracleLane> {
        let movers: Vec<&OracleLane> = subset
            .iter()
            .filter(|l| config.is_occupied(l.src))
            .collect();
        let xs: BTreeSet<u64> = movers.iter().map(|l| l.src_x).collect();
        let ys: BTreeSet<u64> = movers.iter().map(|l| l.src_y).collect();
        subset
            .iter()
            .filter(|l| xs.contains(&l.src_x) && ys.contains(&l.src_y))
            .copied()
            .collect()
    }

    /// The set of valid shots at `config`, one tight representative per child
    /// configuration per group, with that child.
    ///
    /// `max_subset` bounds the subset size when a group is too large for its
    /// power set; `None` means the power set (and panics on a group over
    /// [`POWER_SET_LIMIT`], which is a misuse of the oracle, not a finding).
    pub(super) fn accepted_shots(
        &self,
        config: &Config,
        blocked: &HashSet<u64>,
        cap: Option<AodCapacity>,
        max_subset: Option<usize>,
    ) -> BTreeMap<Group, BTreeMap<Shot, Config>> {
        let mut out: BTreeMap<Group, BTreeMap<Shot, Config>> = BTreeMap::new();
        for (group, lanes) in &self.groups {
            // B2 lets us skip a group no atom can move on: every subset would
            // fail it. This is a shortcut on the *oracle's* side only.
            if !lanes.iter().any(|l| config.is_occupied(l.src)) {
                continue;
            }
            let k = match max_subset {
                Some(k) => k.min(lanes.len()),
                None => {
                    assert!(
                        lanes.len() <= POWER_SET_LIMIT,
                        "group {group:?} has {} lanes; pass a subset-size cap",
                        lanes.len()
                    );
                    lanes.len()
                }
            };
            let shots = out.entry(*group).or_default();
            for subset in subsets_up_to(lanes, k) {
                let Ok(child) = self.accept(&subset, config, blocked, cap) else {
                    continue;
                };
                let rep = self.tight(&subset, config);
                let rep_lanes: Vec<LaneAddr> = rep.iter().map(|l| l.lane).collect();
                assert!(
                    self.spec.check_lanes(&rep_lanes).is_empty(),
                    "tight representative of an accepted shot is not a valid lane group: \
                     {rep_lanes:?} (from {subset:?}) — a P2 violation or an oracle bug"
                );
                let mut shot: Shot = rep.iter().map(|l| l.lane.encode_u64()).collect();
                shot.sort_unstable();
                if let Some(prev) = shots.insert(shot, child.clone()) {
                    assert_eq!(
                        prev, child,
                        "two supersets of one tight representative disagree on the child"
                    );
                }
            }
        }
        out
    }
}

/// Every group key the spec's bus tables can carry lanes for.
fn group_keys(spec: &ArchSpec) -> Vec<Group> {
    let mut keys = BTreeSet::new();
    for (zone_id, zone) in spec.zones.iter().enumerate() {
        for bus_id in 0..zone.site_buses.len() {
            for direction in [Direction::Forward, Direction::Backward] {
                keys.insert(Group {
                    move_type: MoveType::SiteBus,
                    bus_id: bus_id as u32,
                    zone_id: zone_id as u32,
                    direction,
                });
            }
        }
        for bus_id in 0..zone.word_buses.len() {
            for direction in [Direction::Forward, Direction::Backward] {
                keys.insert(Group {
                    move_type: MoveType::WordBus,
                    bus_id: bus_id as u32,
                    zone_id: zone_id as u32,
                    direction,
                });
            }
        }
    }
    for (bus_id, bus) in spec.zone_buses.iter().enumerate() {
        for src in &bus.src {
            for direction in [Direction::Forward, Direction::Backward] {
                keys.insert(Group {
                    move_type: MoveType::ZoneBus,
                    bus_id: bus_id as u32,
                    zone_id: src.zone_id as u32,
                    direction,
                });
            }
        }
    }
    keys.into_iter().collect()
}

/// The lanes of a group by brute force: every address with the group's four
/// fields and in-range word and site ids that `check_lane` accepts.
fn group_lanes(spec: &ArchSpec, group: Group) -> Vec<LaneAddr> {
    let mut lanes = Vec::new();
    for word_id in 0..spec.words.len() as u32 {
        for site_id in 0..spec.sites_per_word() as u32 {
            let lane = LaneAddr {
                direction: group.direction,
                move_type: group.move_type,
                zone_id: group.zone_id,
                word_id,
                site_id,
                bus_id: group.bus_id,
            };
            if spec.check_lane(&lane).is_empty() {
                lanes.push(lane);
            }
        }
    }
    lanes
}

/// Every non-empty subset of `items` with at most `k` elements, in
/// lexicographic index order.
fn subsets_up_to<T: Copy>(items: &[T], k: usize) -> Vec<Vec<T>> {
    let n = items.len();
    let mut out = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    fn rec<T: Copy>(
        items: &[T],
        start: usize,
        k: usize,
        current: &mut Vec<usize>,
        out: &mut Vec<Vec<T>>,
    ) {
        if !current.is_empty() {
            out.push(current.iter().map(|&i| items[i]).collect());
        }
        if current.len() == k {
            return;
        }
        for i in start..items.len() {
            current.push(i);
            rec(items, i + 1, k, current, out);
            current.pop();
        }
    }
    if n > 0 {
        rec(items, 0, k, &mut current, &mut out);
    }
    out
}

// ── Generator side ──

/// Run the generator on one configuration at `SeedPolicy::Any` and `cap`.
pub(super) fn generator_output(
    index: &LaneIndex,
    config: &Config,
    blocked: &HashSet<u64>,
    cap: Option<AodCapacity>,
) -> Vec<MoveCandidate> {
    let dist_table = DistanceTable::new(&[], index);
    let ctx = SearchContext {
        index,
        dist_table: &dist_table,
        blocked,
        targets: &[],
        cz_pairs: None,
        capacity: None,
    };
    let generator =
        ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, cap).expect("preconditions hold");
    let mut out = Vec::new();
    generator.generate(
        config,
        NodeId(0),
        &ctx,
        &mut SearchState::default(),
        &mut out,
    );
    out
}

/// Where the generator and the oracle disagree, counted by kind, with a few
/// examples of each so a failure names concrete shots.
#[derive(Default)]
pub(super) struct Summary {
    pub(super) configs: usize,
    pub(super) emitted: usize,
    pub(super) expected: usize,
    /// The same move set emitted more than once for one configuration.
    pub(super) duplicates: usize,
    /// Emitted shots the oracle rejects, by the first rule they fail.
    pub(super) extras: BTreeMap<&'static str, usize>,
    /// Oracle representatives the generator never emitted.
    pub(super) missing: usize,
    /// Emitted shots whose `new_config` differs from the validator's replay.
    pub(super) wrong_child: usize,
    /// Two distinct emitted shots of one group with one child configuration.
    pub(super) shared_child: usize,
    examples: Vec<String>,
}

impl Summary {
    pub(super) fn is_clean(&self) -> bool {
        self.duplicates == 0
            && self.extras.is_empty()
            && self.missing == 0
            && self.wrong_child == 0
            && self.shared_child == 0
    }

    fn example(&mut self, text: String) {
        if self.examples.len() < 12 {
            self.examples.push(text);
        }
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} configurations, {} shots emitted, {} expected",
            self.configs, self.emitted, self.expected
        )?;
        writeln!(f, "  duplicates:   {}", self.duplicates)?;
        writeln!(f, "  missing:      {}", self.missing)?;
        writeln!(f, "  wrong child:  {}", self.wrong_child)?;
        writeln!(f, "  shared child: {}", self.shared_child)?;
        for (why, n) in &self.extras {
            writeln!(f, "  extra ({why}): {n}")?;
        }
        for e in &self.examples {
            writeln!(f, "    e.g. {e}")?;
        }
        Ok(())
    }
}

/// Compare the generator against the oracle on one configuration.
///
/// `equality` requests the full comparison (extras *and* missing, using the
/// oracle's enumeration with `max_subset`); without it only soundness is
/// checked — every emitted shot must be validator-accepted and tight — which
/// is all that is affordable on specs too large to enumerate.
pub(super) fn compare(
    oracle: &Oracle,
    index: &LaneIndex,
    config: &Config,
    blocked: &HashSet<u64>,
    cap: Option<AodCapacity>,
    equality: Option<Option<usize>>,
    summary: &mut Summary,
) {
    let emitted = generator_output(index, config, blocked, cap);
    summary.configs += 1;
    summary.emitted += emitted.len();

    // Partition by the group of the first lane; duplicates across the whole
    // output are counted once per repeat.
    let mut seen: HashMap<Shot, usize> = HashMap::new();
    let mut by_group: BTreeMap<Group, Vec<(Shot, Config)>> = BTreeMap::new();
    for cand in &emitted {
        let shot: Shot = cand.move_set.encoded_lanes().to_vec();
        let n = seen.entry(shot.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            summary.duplicates += 1;
            summary.example(format!("duplicate {shot:#x?}"));
            continue;
        }
        let Some(&first) = shot.first() else {
            *summary.extras.entry("empty shot").or_default() += 1;
            continue;
        };
        by_group
            .entry(Group::of(&LaneAddr::decode_u64(first)))
            .or_default()
            .push((shot, cand.new_config.clone()));
    }

    let expected: BTreeMap<Group, BTreeMap<Shot, Config>> = match equality {
        Some(max_subset) => oracle.accepted_shots(config, blocked, cap, max_subset),
        None => BTreeMap::new(),
    };
    summary.expected += expected.values().map(|m| m.len()).sum::<usize>();

    let all_groups: BTreeSet<Group> = by_group.keys().chain(expected.keys()).copied().collect();
    for group in all_groups {
        let got = by_group.get(&group).cloned().unwrap_or_default();
        let want = expected.get(&group).cloned().unwrap_or_default();

        let mut children: HashMap<Vec<(u32, u64)>, Shot> = HashMap::new();
        for (shot, child) in &got {
            if let Some(other) = children.insert(child.as_entries().to_vec(), shot.clone()) {
                summary.shared_child += 1;
                summary.example(format!(
                    "{group:?}: {shot:#x?} and {other:#x?} reach the same child"
                ));
            }
            match want.get(shot) {
                Some(expected_child) => {
                    if expected_child != child {
                        summary.wrong_child += 1;
                        summary.example(format!("{group:?}: {shot:#x?} misreports its child"));
                    }
                }
                None => {
                    // Not an expected representative: say why the validator
                    // rejects it, or that it is a non-tight duplicate of one.
                    // In soundness-only mode nothing was enumerated, so an
                    // accepted, tight shot is simply correct.
                    let lanes: Vec<OracleLane> = shot
                        .iter()
                        .map(|&enc| oracle_lane(oracle, LaneAddr::decode_u64(enc)))
                        .collect();
                    let why = match oracle.accept(&lanes, config, blocked, cap) {
                        Err(why) => Some(why),
                        Ok(expected_child) => {
                            if expected_child != *child {
                                summary.wrong_child += 1;
                                summary
                                    .example(format!("{group:?}: {shot:#x?} misreports its child"));
                            }
                            let tight = oracle.tight(&lanes, config);
                            if tight.len() != lanes.len() {
                                Some("not mover-tight")
                            } else if equality.is_some() {
                                Some("accepted but not enumerated by the oracle")
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(why) = why {
                        *summary.extras.entry(why).or_default() += 1;
                        summary.example(format!("{group:?}: extra {shot:#x?}: {why}"));
                    }
                }
            }
        }
        if equality.is_some() {
            let got_shots: HashSet<&Shot> = got.iter().map(|(s, _)| s).collect();
            for shot in want.keys() {
                if !got_shots.contains(shot) {
                    summary.missing += 1;
                    summary.example(format!("{group:?}: missing {shot:#x?}"));
                }
            }
        }
    }
}

/// Resolve a lane the generator emitted through the oracle's own tables; a
/// lane the oracle never enumerated is itself a finding (S1).
fn oracle_lane(oracle: &Oracle, lane: LaneAddr) -> OracleLane {
    oracle
        .groups
        .get(&Group::of(&lane))
        .and_then(|lanes| lanes.iter().find(|l| l.lane == lane))
        .copied()
        .unwrap_or_else(|| panic!("generator emitted a lane the spec does not have: {lane:?}"))
}

/// A seeded sweep of random configurations on one spec.
///
/// Atoms and blocked sites are drawn from lane endpoints (1–4 atoms, 0–2
/// blocked sites, disjoint), the only places a shot can touch.
pub(super) fn sweep(
    oracle: &Oracle,
    index: &LaneIndex,
    seed: u64,
    configs: usize,
    cap: Option<AodCapacity>,
    equality: Option<Option<usize>>,
) -> Summary {
    let endpoints = oracle.endpoints();
    assert!(
        endpoints.len() >= 2,
        "a spec with lanes has at least two endpoints"
    );
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut summary = Summary::default();
    for _ in 0..configs {
        let n_atoms = rng.random_range(1..=4usize.min(endpoints.len() - 1));
        let n_blocked = rng.random_range(0..=2usize.min(endpoints.len() - n_atoms));
        let picked: Vec<LocationAddr> = endpoints
            .choose_multiple(&mut rng, n_atoms + n_blocked)
            .copied()
            .collect();
        let config = Config::new(
            picked[..n_atoms]
                .iter()
                .enumerate()
                .map(|(q, &l)| (q as u32, l)),
        )
        .expect("distinct locations");
        let blocked: HashSet<u64> = picked[n_atoms..].iter().map(|l| l.encode()).collect();
        compare(
            oracle,
            index,
            &config,
            &blocked,
            cap,
            equality,
            &mut summary,
        );
    }
    summary
}

// ── Fixtures ──

pub(super) fn physical_spec_json() -> &'static str {
    include_str!("../../../../../python/bloqade/lanes/arch/gemini/physical/_physical_spec.json")
}

pub(super) fn logical_spec_json() -> &'static str {
    include_str!("../../../../../python/bloqade/lanes/arch/gemini/logical/_logical_spec.json")
}

pub(super) fn load(json: &str) -> (Oracle, LaneIndex) {
    let spec: ArchSpec = serde_json::from_str(json).expect("spec json parses");
    assert!(spec.validate().is_ok(), "{:?}", spec.validate().err());
    (Oracle::new(spec.clone()), LaneIndex::new(spec))
}

/// The fixtures small enough for the power set, by name.
pub(super) fn small_fixtures() -> Vec<(&'static str, String)> {
    use crate::test_utils::*;
    vec![
        ("example", example_arch_json().to_string()),
        ("chain", chain_arch_json()),
        (
            "chain with siding",
            chain_with_siding_arch_json().to_string(),
        ),
        ("two-zone bus", two_zone_bus_arch_json().to_string()),
        (
            "two-zone aligned site buses",
            two_zone_aligned_site_bus_arch_json().to_string(),
        ),
    ]
}

const SEED: u64 = 0x0E4A_C7E5_0AC1_E001;

/// Run `f` over the small fixtures and fail with every fixture's summary if
/// any is not clean.
fn assert_clean_on_small_fixtures(cap: Option<AodCapacity>, configs: usize) {
    let mut report = String::new();
    for (name, json) in small_fixtures() {
        let (oracle, index) = load(&json);
        let summary = sweep(&oracle, &index, SEED, configs, cap, Some(None));
        if !summary.is_clean() {
            report.push_str(&format!("\n[{name}] cap {cap:?}\n{summary}"));
        }
    }
    assert!(
        report.is_empty(),
        "generator differs from the oracle:{report}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The oracle's brute-force lane enumeration agrees with `LaneIndex`'s
    /// registration on every fixture and both shipped specs. A disagreement
    /// here is a registration bug, visible to the oracle because it reads
    /// the spec rather than the index.
    #[test]
    fn oracle_lanes_match_the_lane_index() {
        let mut specs = small_fixtures();
        specs.push(("physical", physical_spec_json().to_string()));
        specs.push(("logical", logical_spec_json().to_string()));
        specs.push((
            "non-separable",
            crate::test_utils::non_separable_bus_arch_json().to_string(),
        ));
        for (name, json) in specs {
            let (oracle, index) = load(&json);
            let from_index: BTreeMap<Group, BTreeSet<u64>> = index
                .bus_groups()
                .map(|(mt, bus_id, zone_id, dir)| {
                    let group = Group {
                        move_type: mt,
                        bus_id,
                        zone_id,
                        direction: dir,
                    };
                    let lanes = index
                        .lanes_for(mt, bus_id, zone_id, dir)
                        .iter()
                        .map(|l| l.encode_u64())
                        .collect();
                    (group, lanes)
                })
                .collect();
            assert_eq!(oracle.lane_sets(), from_index, "{name}");
        }
    }

    /// **The acceptance criterion.** On every fixture small enough for the
    /// power set, the generator's output equals the oracle's representatives
    /// per group, with no duplicates and one shot per child, uncapped.
    #[test]
    fn generator_equals_the_oracle_on_small_fixtures() {
        assert_clean_on_small_fixtures(None, 200);
    }

    /// The same under a 2×2 capacity: B3 must hold and nothing within the cap
    /// may go missing.
    #[test]
    fn generator_equals_the_oracle_at_capacity_two() {
        assert_clean_on_small_fixtures(Some(AodCapacity { x: 2, y: 2 }), 200);
    }

    /// On the shipped specs the power set is out of reach, but at unit
    /// capacity every shot is one lane, so equality is affordable: each
    /// single-lane subset is checked.
    #[test]
    fn generator_equals_the_oracle_on_shipped_specs_at_unit_capacity() {
        let cap = Some(AodCapacity { x: 1, y: 1 });
        let mut report = String::new();
        for (name, json) in [
            ("physical", physical_spec_json()),
            ("logical", logical_spec_json()),
        ] {
            let (oracle, index) = load(json);
            let summary = sweep(&oracle, &index, SEED, 40, cap, Some(Some(1)));
            if !summary.is_clean() {
                report.push_str(&format!("\n[{name}] cap {cap:?}\n{summary}"));
            }
        }
        assert!(
            report.is_empty(),
            "generator differs from the oracle:{report}"
        );
    }

    /// Soundness on the shipped specs: every emitted shot is
    /// validator-accepted and mover-tight, with no duplicates and one shot
    /// per child. Equality is not checked (no enumeration at this size), so
    /// `missing` is not meaningful here.
    fn assert_sound_on_shipped_specs(cap: Option<AodCapacity>, configs: usize) {
        let mut report = String::new();
        for (name, json) in [
            ("physical", physical_spec_json()),
            ("logical", logical_spec_json()),
        ] {
            let (oracle, index) = load(json);
            let summary = sweep(&oracle, &index, SEED, configs, cap, None);
            if !summary.is_clean() {
                report.push_str(&format!("\n[{name}] cap {cap:?}\n{summary}"));
            }
        }
        assert!(
            report.is_empty(),
            "generator emits shots the validator rejects:{report}"
        );
    }

    #[test]
    fn generator_is_sound_on_shipped_specs_at_capacity_two() {
        assert_sound_on_shipped_specs(Some(AodCapacity { x: 2, y: 2 }), 30);
    }

    /// Uncapped: the tight enumeration is anchored on the atoms, so even with
    /// no capacity the sweep is cheap (the previous geometry-anchored
    /// enumeration took about a minute here in debug).
    #[test]
    fn generator_is_sound_on_shipped_specs_uncapped() {
        assert_sound_on_shipped_specs(None, 30);
    }

    /// `full.json` violates P1 (coincident words): a legal spec on which the
    /// enumeration model is not well defined. The precondition check rejects
    /// it before any enumeration runs, so the oracle never has to compare
    /// against output produced on it.
    #[test]
    fn full_json_is_rejected_by_the_precondition() {
        let (_oracle, index) = load(crate::test_utils::full_arch_json());
        let dist_table = DistanceTable::new(&[], &index);
        let blocked = HashSet::new();
        let ctx = SearchContext {
            index: &index,
            dist_table: &dist_table,
            blocked: &blocked,
            targets: &[],
            cz_pairs: None,
            capacity: None,
        };
        let err = ExhaustiveGenerator::for_solve(&ctx, SeedPolicy::Any, None)
            .expect_err("full.json must be rejected");
        assert!(
            matches!(err, ExhaustivePrecondition::PositionCollision { .. }),
            "{err}"
        );
    }

    /// The subset enumerator is what the oracle's completeness rests on.
    #[test]
    fn subsets_up_to_enumerates_every_subset_once() {
        let items = [1u8, 2, 3, 4];
        let all = subsets_up_to(&items, 4);
        assert_eq!(all.len(), 15);
        let distinct: BTreeSet<Vec<u8>> = all.iter().cloned().collect();
        assert_eq!(distinct.len(), 15);
        let small = subsets_up_to(&items, 2);
        assert_eq!(small.len(), 4 + 6);
        assert!(small.iter().all(|s| s.len() <= 2));
    }
}
