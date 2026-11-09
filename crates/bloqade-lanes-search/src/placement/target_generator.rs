//! Target generator plugin system for move synthesis.
//!
//! Provides a trait-based abstraction for generating candidate target
//! configurations for CZ gate placements.  The solver tries each
//! candidate in order with a shared expansion budget.

use std::collections::{HashMap, HashSet};
use std::fmt;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::primitives::lane_index::LaneIndex;

/// Read-only context for target generation, analogous to Python's `TargetContext`.
pub struct TargetContext<'a> {
    /// Current qubit positions: `(qubit_id, location)` pairs.
    pub placement: &'a [(u32, LocationAddr)],
    /// Control qubit IDs for the CZ gate layer.
    pub controls: &'a [u32],
    /// Target qubit IDs for the CZ gate layer.
    pub targets: &'a [u32],
    /// Architecture lane index (provides arch spec + CZ partner lookups).
    pub index: &'a LaneIndex,
}

/// Generates candidate target configurations for move synthesis.
///
/// Each candidate is a full placement: `Vec<(qubit_id, LocationAddr)>`.
/// Candidates are tried in order by
/// [`solve_single_heuristic`](crate::placement::single_heuristic::solve_single_heuristic);
/// the first successful solve wins.
pub trait TargetGenerator: Send + Sync {
    /// Generate an ordered list of candidate target configurations.
    ///
    /// Each candidate maps every qubit to its desired location.
    /// The solver tries them in order with a shared expansion budget.
    fn generate(&self, ctx: &TargetContext) -> Vec<Vec<(u32, LocationAddr)>>;
}

/// A fixed list of candidate placements, offered in order whatever the
/// context.
///
/// This is how a caller supplies its own candidates — for example ones a
/// Python generator produced — to
/// [`SingleHeuristicCzPlacement`](crate::placement::single_heuristic::SingleHeuristicCzPlacement).
/// Each candidate still goes through [`validate_candidate`] before it is
/// routed.
#[derive(Debug, Clone, Default)]
pub struct CandidateList(pub Vec<Vec<(u32, LocationAddr)>>);

impl TargetGenerator for CandidateList {
    fn generate(&self, _ctx: &TargetContext) -> Vec<Vec<(u32, LocationAddr)>> {
        self.0.clone()
    }
}

/// Default target generator: keeps target qubits fixed, moves each control
/// qubit to its CZ blockade partner location.
///
/// Mirrors the Python `DefaultTargetGenerator` / `_target_from_stage_controls_only`.
/// Always produces exactly one candidate (or zero if a partner lookup fails).
#[derive(Debug, Clone, Copy)]
pub struct DefaultTargetGenerator;

impl TargetGenerator for DefaultTargetGenerator {
    fn generate(&self, ctx: &TargetContext) -> Vec<Vec<(u32, LocationAddr)>> {
        let placement_map: HashMap<u32, LocationAddr> = ctx.placement.iter().copied().collect();

        let mut target = placement_map.clone();

        for (&control_qid, &target_qid) in ctx.controls.iter().zip(ctx.targets.iter()) {
            let target_loc = match placement_map.get(&target_qid) {
                Some(loc) => *loc,
                None => return vec![], // missing qubit
            };
            let partner = match ctx.index.cz_partner(&target_loc) {
                Some(p) => p,
                None => return vec![], // no CZ partner
            };
            target.insert(control_qid, partner);
        }

        let candidate: Vec<(u32, LocationAddr)> = target.into_iter().collect();
        vec![candidate]
    }
}

// ── Validation ──

/// Validation errors for a target candidate.
#[derive(Debug, Clone)]
pub enum CandidateError {
    /// A qubit in controls/targets is missing from the candidate.
    MissingQubit(u32),
    /// A location in the candidate is not a valid architecture location.
    InvalidLocation(LocationAddr),
    /// A (control, target) pair is not at CZ partner locations.
    NotCzPair { control: u32, target: u32 },
    /// Candidate contains duplicate qubit IDs.
    DuplicateQubit(u32),
    /// A qubit in the candidate is not in the stage's placement.
    UnexpectedQubit(u32),
    /// Two qubits in the candidate share a location.
    DuplicateLocation(LocationAddr),
    /// Controls and targets have different lengths.
    LengthMismatch { controls: usize, targets: usize },
}

impl fmt::Display for CandidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingQubit(qid) => write!(f, "qubit {qid} missing from candidate"),
            Self::InvalidLocation(loc) => {
                write!(
                    f,
                    "location ({}, {}, {}) is not valid in the architecture",
                    loc.zone_id, loc.word_id, loc.site_id
                )
            }
            Self::NotCzPair { control, target } => {
                write!(
                    f,
                    "qubits ({control}, {target}) are not at CZ partner locations"
                )
            }
            Self::DuplicateQubit(qid) => {
                write!(f, "duplicate qubit {qid} in candidate")
            }
            Self::UnexpectedQubit(qid) => {
                write!(f, "qubit {qid} in candidate is not in the placement")
            }
            Self::DuplicateLocation(loc) => {
                write!(
                    f,
                    "two qubits share location ({}, {}, {})",
                    loc.zone_id, loc.word_id, loc.site_id
                )
            }
            Self::LengthMismatch { controls, targets } => {
                write!(
                    f,
                    "controls length ({controls}) != targets length ({targets})"
                )
            }
        }
    }
}

impl std::error::Error for CandidateError {}

/// Validate a candidate target configuration against the stage's
/// `placement`.
///
/// Checks:
/// 1. Controls and targets have the same length.
/// 2. No duplicate qubit IDs in the candidate.
/// 3. The candidate places exactly the placement's qubits: every one of
///    them, and no others.
/// 4. All locations are valid positions in the architecture.
/// 5. No two qubits share a location.
/// 6. Each (control, target) pair sits at CZ partner locations, in either
///    direction. On a validated spec the partner relation is symmetric, so
///    one direction would do; but `ArchSpec::validate` is what rules out a
///    word in two entangling pairs, and on a spec loaded without it
///    [`LaneIndex::cz_partner`] reports only the first pair, so checking one
///    direction would reject a valid pair.
pub fn validate_candidate(
    candidate: &[(u32, LocationAddr)],
    placement: &[(u32, LocationAddr)],
    controls: &[u32],
    targets: &[u32],
    index: &LaneIndex,
) -> Result<(), CandidateError> {
    // Check controls/targets length match.
    if controls.len() != targets.len() {
        return Err(CandidateError::LengthMismatch {
            controls: controls.len(),
            targets: targets.len(),
        });
    }

    // Check for duplicate qubit IDs.
    let mut seen = std::collections::HashSet::new();
    for &(qid, _) in candidate {
        if !seen.insert(qid) {
            return Err(CandidateError::DuplicateQubit(qid));
        }
    }

    let candidate_map: HashMap<u32, LocationAddr> = candidate.iter().copied().collect();

    // Check the candidate places exactly the placement's qubits, and that
    // the CZ qubits are among them.
    let placed: HashSet<u32> = placement.iter().map(|&(qid, _)| qid).collect();
    for &qid in placed.iter().chain(controls).chain(targets) {
        if !candidate_map.contains_key(&qid) {
            return Err(CandidateError::MissingQubit(qid));
        }
    }
    if let Some(&(qid, _)) = candidate.iter().find(|(qid, _)| !placed.contains(qid)) {
        return Err(CandidateError::UnexpectedQubit(qid));
    }

    // Check all locations are valid, and no two qubits share one.
    let mut occupied = HashSet::with_capacity(candidate.len());
    for &(_, loc) in candidate {
        if index.position(loc).is_none() {
            return Err(CandidateError::InvalidLocation(loc));
        }
        if !occupied.insert(loc.encode()) {
            return Err(CandidateError::DuplicateLocation(loc));
        }
    }

    // Check CZ pair validity, in either direction.
    for (&cqid, &tqid) in controls.iter().zip(targets.iter()) {
        let c_loc = candidate_map[&cqid];
        let t_loc = candidate_map[&tqid];
        if index.cz_partner(&t_loc) != Some(c_loc) && index.cz_partner(&c_loc) != Some(t_loc) {
            return Err(CandidateError::NotCzPair {
                control: cqid,
                target: tqid,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::example_arch_json;

    fn make_index() -> LaneIndex {
        let arch_spec =
            bloqade_lanes_bytecode_core::arch::types::ArchSpec::from_json(example_arch_json())
                .unwrap();
        LaneIndex::new(arch_spec)
    }

    fn loc(zone: u32, word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            zone_id: zone,
            word_id: word,
            site_id: site,
        }
    }

    #[test]
    fn default_generator_produces_one_candidate() {
        let index = make_index();
        // Place qubit 0 at word 0 site 0, qubit 1 at word 1 site 0.
        // CZ pair: word 0 ↔ word 1.
        // Control (qubit 0) should move to CZ partner of qubit 1's location.
        let placement = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0))];
        let controls = [0];
        let targets = [1];
        let ctx = TargetContext {
            placement: &placement,
            controls: &controls,
            targets: &targets,
            index: &index,
        };

        let generator = DefaultTargetGenerator;
        let candidates = generator.generate(&ctx);
        assert_eq!(candidates.len(), 1);

        let candidate_map: HashMap<u32, LocationAddr> = candidates[0].iter().copied().collect();
        // Qubit 1 stays at word 1 site 0.
        assert_eq!(candidate_map[&1], loc(0, 1, 0));
        // Qubit 0 should be at CZ partner of (word 1, site 0) = (word 0, site 0).
        assert_eq!(candidate_map[&0], loc(0, 0, 0));
    }

    #[test]
    fn default_generator_returns_empty_for_missing_qubit() {
        let index = make_index();
        let placement = vec![(0, loc(0, 0, 0))]; // qubit 1 missing
        let controls = [0];
        let targets = [1];
        let ctx = TargetContext {
            placement: &placement,
            controls: &controls,
            targets: &targets,
            index: &index,
        };

        let candidates = DefaultTargetGenerator.generate(&ctx);
        assert!(candidates.is_empty());
    }

    /// A stage placement of qubits 0 and 1; only the qubit IDs matter to
    /// `validate_candidate`.
    const PLACED: [(u32, LocationAddr); 2] = [
        (
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        ),
        (
            1,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1,
            },
        ),
    ];

    #[test]
    fn validate_accepts_valid_candidate() {
        let index = make_index();
        // Word 0 ↔ Word 1 are CZ partners.
        // Control at word 0, target at word 1: valid CZ pair.
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0))];
        let controls = [0];
        let targets = [1];
        assert!(validate_candidate(&candidate, &PLACED, &controls, &targets, &index).is_ok());
    }

    #[test]
    fn validate_rejects_missing_qubit() {
        let index = make_index();
        let candidate = vec![(0, loc(0, 0, 0))]; // qubit 1 missing
        let controls = [0];
        let targets = [1];
        let err = validate_candidate(&candidate, &PLACED, &controls, &targets, &index).unwrap_err();
        assert!(matches!(err, CandidateError::MissingQubit(1)));
    }

    #[test]
    fn validate_rejects_non_cz_pair() {
        let index = make_index();
        // Both qubits at word 0 — not a CZ pair.
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 0, 1))];
        let controls = [0];
        let targets = [1];
        let err = validate_candidate(&candidate, &PLACED, &controls, &targets, &index).unwrap_err();
        assert!(matches!(
            err,
            CandidateError::NotCzPair {
                control: 0,
                target: 1
            }
        ));
    }

    #[test]
    fn validate_rejects_duplicate_qubit() {
        let index = make_index();
        let candidate = vec![(0, loc(0, 0, 0)), (0, loc(0, 1, 0))];
        let controls = [0];
        let targets = [1];
        let err = validate_candidate(&candidate, &PLACED, &controls, &targets, &index).unwrap_err();
        assert!(matches!(err, CandidateError::DuplicateQubit(0)));
    }

    #[test]
    fn validate_rejects_a_candidate_missing_a_spectator() {
        let index = make_index();
        // Qubit 2 is a spectator in the placement, absent from the candidate.
        let placed = [PLACED[0], PLACED[1], (2, loc(0, 0, 2))];
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0))];
        let err = validate_candidate(&candidate, &placed, &[0], &[1], &index).unwrap_err();
        assert!(matches!(err, CandidateError::MissingQubit(2)));
    }

    #[test]
    fn validate_rejects_a_qubit_not_in_the_placement() {
        let index = make_index();
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0)), (7, loc(0, 0, 3))];
        let err = validate_candidate(&candidate, &PLACED, &[0], &[1], &index).unwrap_err();
        assert!(matches!(err, CandidateError::UnexpectedQubit(7)));
    }

    #[test]
    fn validate_rejects_two_qubits_on_one_location() {
        let index = make_index();
        let placed = [PLACED[0], PLACED[1], (2, loc(0, 0, 2))];
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0)), (2, loc(0, 1, 0))];
        let err = validate_candidate(&candidate, &placed, &[0], &[1], &index).unwrap_err();
        assert!(matches!(err, CandidateError::DuplicateLocation(l) if l == loc(0, 1, 0)));
    }

    /// Word 1 belongs to two entangling pairs, `[0, 1]` and `[1, 2]`, so
    /// `cz_partner` of word 1 reports word 0 only. A pair with its control
    /// on word 2 and its target on word 1 is still a CZ pair. Validation
    /// rejects such a spec, so this covers callers that load one unvalidated
    /// (`ArchSpec::from_json` does not validate).
    #[test]
    fn validate_accepts_a_pair_in_either_direction() {
        let arch = bloqade_lanes_bytecode_core::arch::types::ArchSpec::from_json(
            r#"{
                "version": "2.0",
                "words": [{ "sites": [[0, 0]] }, { "sites": [[0, 1]] }, { "sites": [[0, 2]] }],
                "zones": [{
                    "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [], "y_spacing": [2.0, 2.0] },
                    "site_buses": [],
                    "word_buses": [{ "src": [0, 1], "dst": [1, 2] }],
                    "words_with_site_buses": [], "sites_with_word_buses": [0],
                    "entangling_pairs": [[0, 1], [1, 2]]
                }],
                "zone_buses": [],
                "modes": [{ "name": "default", "zones": [0], "bitstring_order": [] }]
            }"#,
        )
        .unwrap();
        let index = LaneIndex::new(arch);
        assert_eq!(index.cz_partner(&loc(0, 1, 0)), Some(loc(0, 0, 0)));

        let placed = [(0, loc(0, 2, 0)), (1, loc(0, 1, 0))];
        let result = validate_candidate(&placed, &placed, &[0], &[1], &index);
        assert!(result.is_ok(), "{result:?}");
    }

    #[test]
    fn validate_rejects_length_mismatch() {
        let index = make_index();
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0))];
        let controls = [0, 1];
        let targets = [1];
        let err = validate_candidate(&candidate, &PLACED, &controls, &targets, &index).unwrap_err();
        assert!(matches!(
            err,
            CandidateError::LengthMismatch {
                controls: 2,
                targets: 1
            }
        ));
    }
}
