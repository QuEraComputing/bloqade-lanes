//! Canonical execution-model replay for packaged solve results.
//!
//! The search crate carries its own compact state representation
//! ([`Config`]) and trusts generators to produce executable move sets:
//! [`Config::with_moves`] is a documented no-validation per-qubit overwrite,
//! so a generator bug — or a policy that routes an atom onto an occupied site
//! — yields a plan that silently corrupts atom state downstream rather than
//! failing here.
//!
//! This module closes that gap at the one place every plan passes through:
//! before a [`SolveResult`](crate::search::result::SolveResult) is handed
//! back, its layers are replayed from the root configuration through the
//! canonical execution model in `bloqade-lanes-bytecode-core`
//! ([`AtomStateData::validate_moves`] + [`AtomStateData::apply_validated`]) —
//! the same code the IR analysis and bytecode validator use. A plan that
//! cannot execute is therefore caught at its source, with the offending layer
//! named, instead of surfacing as a confusing IR-level error later.
//!
//! The replay is O(layers × lanes) per solve, negligible next to the search
//! that produced the plan, and runs in every build: the invariant it protects
//! (issue #866) is a correctness property, not a debugging aid.

use std::collections::HashMap;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_bytecode_core::atom_state::AtomStateData;

use crate::primitives::config::Config;
use crate::primitives::graph::MoveSet;

/// Replay `layers` from `root` through the canonical execution model.
///
/// Returns the placement the plan actually lands on — `qubit → location`, as
/// resolved by lane endpoints rather than by [`Config::with_moves`] — or a
/// diagnostic naming the first layer that cannot execute.
pub(crate) fn replay_move_layers(
    root: &Config,
    layers: &[MoveSet],
    arch: &ArchSpec,
) -> Result<HashMap<u32, LocationAddr>, String> {
    let atoms: Vec<_> = root.iter().collect();
    let mut state = AtomStateData::from_locations(&atoms);

    for (layer_idx, move_set) in layers.iter().enumerate() {
        let lanes = move_set.decode();
        let validated = state
            .validate_moves(&lanes, arch)
            .map_err(|errors| format_layer_error(layer_idx, layers.len(), &errors))?;
        state = state
            .apply_validated(&validated)
            .map_err(|errors| format_layer_error(layer_idx, layers.len(), &errors))?;
    }

    Ok(state.qubit_to_locations)
}

/// Replay `layers` from `root` and check the result against the placement the
/// solver reports as its goal.
///
/// Executability alone is not enough: `Config::with_moves` overwrites a
/// qubit's location with whatever the generator passed, without checking that
/// the qubit was at the lane's source or that the lane leads where the
/// generator claims. A plan can therefore be perfectly executable while the
/// reported `goal_config` is wrong — a teleported qubit, or a move credited to
/// the wrong atom. Comparing the replayed placement against `expected_goal`
/// closes that gap, and is the only check that actually exercises
/// lane-endpoint semantics end to end.
pub(crate) fn verify_move_layers(
    root: &Config,
    layers: &[MoveSet],
    arch: &ArchSpec,
    expected_goal: &Config,
) -> Result<(), String> {
    let replayed = replay_move_layers(root, layers, arch)?;

    let claimed: HashMap<u32, LocationAddr> = expected_goal.iter().collect();
    if replayed == claimed {
        return Ok(());
    }

    // Report every qubit that disagrees, sorted for a stable diagnostic.
    let mut qubits: Vec<u32> = replayed.keys().chain(claimed.keys()).copied().collect();
    qubits.sort_unstable();
    qubits.dedup();
    let mut details = String::new();
    for qubit in qubits {
        let (got, want) = (replayed.get(&qubit), claimed.get(&qubit));
        if got != want {
            details.push_str(&format!(
                "\n  - qubit {qubit}: plan lands on {got:?}, solver reports {want:?}"
            ));
        }
    }
    Err(format!(
        "replaying the plan does not reproduce the reported goal placement:{details}"
    ))
}

fn format_layer_error<E: std::fmt::Display>(idx: usize, total: usize, errors: &[E]) -> String {
    let details = errors
        .iter()
        .map(|e| format!("\n  - {e}"))
        .collect::<Vec<_>>()
        .join("");
    format!("move layer {idx} of {total} cannot execute:{details}")
}

/// Panic with a full diagnostic when a packaged plan is not executable.
///
/// Reaching this is a bug in the generator, policy, or router that produced
/// the plan — not a user error — so it fails loudly rather than returning a
/// plan that would corrupt atom state (or, since #877, be rejected with a
/// much vaguer message once it reaches the IR analysis).
pub(crate) fn assert_move_layers_executable(
    root: &Config,
    layers: &[MoveSet],
    arch: &ArchSpec,
    expected_goal: &Config,
) {
    if let Err(diagnostic) = verify_move_layers(root, layers, arch, expected_goal) {
        panic!(
            "solver produced an invalid plan (this is a bug in the move \
             generator, not in the request): {diagnostic}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::lane_index::LaneIndex;
    use crate::test_utils::{example_arch_json, loc};
    use bloqade_lanes_bytecode_core::arch::addr::{Direction, LaneAddr, MoveType};

    fn index() -> LaneIndex {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).expect("arch json parses");
        LaneIndex::new(spec)
    }

    /// Site bus 0 on the example arch maps sites 0..5 → 5..10 within a word.
    fn site_lane(word_id: u32, site_id: u32) -> LaneAddr {
        LaneAddr {
            direction: Direction::Forward,
            move_type: MoveType::SiteBus,
            zone_id: 0,
            word_id,
            site_id,
            bus_id: 0,
        }
    }

    #[test]
    fn accepts_an_executable_plan_reaching_the_reported_goal() {
        let index = index();
        let root = Config::new([(0, loc(0, 0))]).expect("config");
        let layers = vec![MoveSet::new(vec![site_lane(0, 0)])];
        // Site bus 0 maps site 0 → site 5.
        let goal = Config::new([(0, loc(0, 5))]).expect("config");
        assert_eq!(
            verify_move_layers(&root, &layers, index.arch_spec(), &goal),
            Ok(())
        );
    }

    #[test]
    fn empty_plan_must_report_the_root_placement() {
        let index = index();
        let root = Config::new([(0, loc(0, 0))]).expect("config");
        assert_eq!(
            verify_move_layers(&root, &[], index.arch_spec(), &root),
            Ok(())
        );
    }

    #[test]
    fn rejects_a_move_onto_a_stationary_atom() {
        let index = index();
        // Qubit 1 sits on site 5 — the destination of site 0's lane — and has
        // no lane of its own in the group, so the layer cannot execute.
        let root = Config::new([(0, loc(0, 0)), (1, loc(0, 5))]).expect("config");
        let layers = vec![MoveSet::new(vec![site_lane(0, 0)])];
        // Executability is checked before the placement comparison, so the
        // expected goal passed here is irrelevant — use the root.
        let err = verify_move_layers(&root, &layers, index.arch_spec(), &root)
            .expect_err("landing on a stationary atom must be rejected");
        assert!(err.contains("move layer 0 of 1"), "{err}");
        assert!(err.contains("occupied by qubit 1"), "{err}");
    }

    #[test]
    fn reports_the_offending_layer_index() {
        let index = index();
        // Layer 0 is fine (0 → 5); layer 1 then drives site 1 → 6 while qubit
        // 1 still occupies site 6.
        let root = Config::new([(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 6))]).expect("config");
        let layers = vec![
            MoveSet::new(vec![site_lane(0, 0)]),
            MoveSet::new(vec![site_lane(0, 1)]),
        ];
        let err = verify_move_layers(&root, &layers, index.arch_spec(), &root)
            .expect_err("second layer must be rejected");
        assert!(err.contains("move layer 1 of 2"), "{err}");
    }

    /// The check Copilot asked for on #888: a plan can be perfectly executable
    /// while the placement the solver *reports* is wrong, because
    /// `Config::with_moves` overwrites a qubit's location without consulting
    /// lane endpoints. Here the lane moves qubit 0 to site 5, but the reported
    /// goal claims site 6.
    #[test]
    fn rejects_an_executable_plan_that_misreports_the_goal() {
        let index = index();
        let root = Config::new([(0, loc(0, 0))]).expect("config");
        let layers = vec![MoveSet::new(vec![site_lane(0, 0)])];
        let wrong_goal = Config::new([(0, loc(0, 6))]).expect("config");

        let err = verify_move_layers(&root, &layers, index.arch_spec(), &wrong_goal)
            .expect_err("a misreported goal placement must be rejected");
        assert!(
            err.contains("does not reproduce the reported goal"),
            "{err}"
        );
        assert!(err.contains("qubit 0"), "{err}");
    }

    /// A plan that silently drops an atom (reports fewer qubits than it moves)
    /// is caught by the same comparison.
    #[test]
    fn rejects_a_goal_missing_a_qubit() {
        let index = index();
        let root = Config::new([(0, loc(0, 0)), (1, loc(0, 1))]).expect("config");
        let layers = vec![MoveSet::new(vec![site_lane(0, 0)])];
        let goal = Config::new([(0, loc(0, 5))]).expect("config");

        let err = verify_move_layers(&root, &layers, index.arch_spec(), &goal)
            .expect_err("a dropped qubit must be rejected");
        assert!(err.contains("qubit 1"), "{err}");
    }
}
