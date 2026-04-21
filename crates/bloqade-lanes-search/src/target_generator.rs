//! Target generator plugin system for move synthesis.
//!
//! Provides a trait-based abstraction for generating candidate target
//! configurations for CZ gate placements.  The solver tries each
//! candidate in order with a shared expansion budget.

use std::collections::HashMap;
use std::fmt;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::lane_index::LaneIndex;

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
/// Candidates are tried in order by [`MoveSolver::solve_with_generator`];
/// the first successful solve wins.
///
/// [`MoveSolver::solve_with_generator`]: crate::solve::MoveSolver::solve_with_generator
pub trait TargetGenerator: Send + Sync {
    /// Generate an ordered list of candidate target configurations.
    ///
    /// Each candidate maps every qubit to its desired location.
    /// The solver tries them in order with a shared expansion budget.
    fn generate(&self, ctx: &TargetContext) -> Vec<Vec<(u32, LocationAddr)>>;
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
        let arch_spec = ctx.index.arch_spec();
        let placement_map: HashMap<u32, LocationAddr> = ctx.placement.iter().copied().collect();

        let mut target = placement_map.clone();

        for (&control_qid, &target_qid) in ctx.controls.iter().zip(ctx.targets.iter()) {
            let target_loc = match placement_map.get(&target_qid) {
                Some(loc) => *loc,
                None => return vec![], // missing qubit
            };
            let partner = match arch_spec.get_cz_partner(&target_loc) {
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

/// Validate a candidate target configuration.
///
/// Checks:
/// 1. Controls and targets have the same length.
/// 2. No duplicate qubit IDs in the candidate.
/// 3. All control and target qubits are present in the candidate.
/// 4. All locations are valid positions in the architecture.
/// 5. Each (control, target) pair sits at CZ partner locations.
pub fn validate_candidate(
    candidate: &[(u32, LocationAddr)],
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
    let arch_spec = index.arch_spec();

    // Check all control/target qubits are present.
    for &qid in controls.iter().chain(targets.iter()) {
        if !candidate_map.contains_key(&qid) {
            return Err(CandidateError::MissingQubit(qid));
        }
    }

    // Check all locations are valid.
    for &(_, loc) in candidate {
        if index.position(loc).is_none() {
            return Err(CandidateError::InvalidLocation(loc));
        }
    }

    // Check CZ pair validity.
    for (&cqid, &tqid) in controls.iter().zip(targets.iter()) {
        let c_loc = candidate_map[&cqid];
        let t_loc = candidate_map[&tqid];
        match arch_spec.get_cz_partner(&t_loc) {
            Some(partner) if partner == c_loc => {}
            _ => {
                return Err(CandidateError::NotCzPair {
                    control: cqid,
                    target: tqid,
                });
            }
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

    #[test]
    fn validate_accepts_valid_candidate() {
        let index = make_index();
        // Word 0 ↔ Word 1 are CZ partners.
        // Control at word 0, target at word 1: valid CZ pair.
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0))];
        let controls = [0];
        let targets = [1];
        assert!(validate_candidate(&candidate, &controls, &targets, &index).is_ok());
    }

    #[test]
    fn validate_rejects_missing_qubit() {
        let index = make_index();
        let candidate = vec![(0, loc(0, 0, 0))]; // qubit 1 missing
        let controls = [0];
        let targets = [1];
        let err = validate_candidate(&candidate, &controls, &targets, &index).unwrap_err();
        assert!(matches!(err, CandidateError::MissingQubit(1)));
    }

    #[test]
    fn validate_rejects_non_cz_pair() {
        let index = make_index();
        // Both qubits at word 0 — not a CZ pair.
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 0, 1))];
        let controls = [0];
        let targets = [1];
        let err = validate_candidate(&candidate, &controls, &targets, &index).unwrap_err();
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
        let err = validate_candidate(&candidate, &controls, &targets, &index).unwrap_err();
        assert!(matches!(err, CandidateError::DuplicateQubit(0)));
    }

    #[test]
    fn validate_rejects_length_mismatch() {
        let index = make_index();
        let candidate = vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0))];
        let controls = [0, 1];
        let targets = [1];
        let err = validate_candidate(&candidate, &controls, &targets, &index).unwrap_err();
        assert!(matches!(
            err,
            CandidateError::LengthMismatch {
                controls: 2,
                targets: 1
            }
        ));
    }
}
