import pytest
from tests._validation_squin_kernels import (
    KernelSpec,
    select_kernels,
)

from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.compile import squin_to_move
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical_placement import PhysicalPlacementStrategy
from bloqade.lanes.validation.address import Validation


def _collect_move_nodes(move_kernel) -> list[move.StatefulStatement]:
    return [
        stmt
        for stmt in move_kernel.callable_region.walk()
        if isinstance(stmt, move.StatefulStatement)
    ]


def _compile_physical_move(kernel):
    return squin_to_move(
        kernel,
        PhysicalLayoutHeuristicGraphPartitionCenterOut(),
        PhysicalPlacementStrategy(),
        logical_initialize=False,
        no_raise=False,
    )


def _assert_cz_pairs_are_blockaded_neighbors(move_kernel) -> None:
    arch_spec = get_physical_arch_spec()
    atom_interp = atom.AtomInterpreter(move_kernel.dialects, arch_spec=arch_spec)
    frame, _ = atom_interp.run(move_kernel)

    for stmt in move_kernel.callable_region.walk():
        if not isinstance(stmt, move.CZ):
            continue

        state = frame.get(stmt.current_state)
        assert isinstance(
            state, atom.AtomState
        ), "Expected concrete atom state at move.CZ"

        controls, targets, _ = state.data.get_qubit_pairing(
            stmt.zone_address, arch_spec
        )
        assert len(controls) == len(
            targets
        ), "Mismatched control/target pairing at move.CZ"

        for control_id, target_id in zip(controls, targets, strict=True):
            control_loc = state.data.qubit_to_locations[control_id]
            target_loc = state.data.qubit_to_locations[target_id]

            assert (
                arch_spec.get_cz_partner(control_loc) == target_loc
                or arch_spec.get_cz_partner(target_loc) == control_loc
            ), (
                "CZ pair is not blockaded neighbors: "
                f"control={control_id}@{control_loc}, target={target_id}@{target_loc}"
            )


@pytest.mark.parametrize("spec", select_kernels(), ids=lambda spec: spec.name)
def test_kernel_catalog_compiles_to_valid_physical_move(spec: KernelSpec):
    """Ensure every catalog kernel compiles and passes physical address validation."""
    kernel = spec.build_kernel()
    move_kernel = _compile_physical_move(kernel)

    _, validation_errors = Validation(arch_spec=get_physical_arch_spec()).run(
        move_kernel
    )
    assert (
        not validation_errors
    ), f"{spec.name} produced invalid physical addresses: {validation_errors}"

    # Physical compilation should not include logical initialization artifacts.
    assert all(
        not isinstance(stmt, move.LogicalInitialize)
        for stmt in move_kernel.callable_region.walk()
    )

    _assert_cz_pairs_are_blockaded_neighbors(move_kernel)


@pytest.mark.parametrize("spec", select_kernels(), ids=lambda spec: spec.name)
def test_tagged_kernel_features_appear_in_compiled_move_ir(spec: KernelSpec):
    """Check that tag-labeled behaviors are present in compiled move operations."""
    kernel = spec.build_kernel()
    move_kernel = _compile_physical_move(kernel)
    nodes = _collect_move_nodes(move_kernel)

    if "cz" in spec.tags:
        assert any(
            isinstance(stmt, move.CZ) for stmt in nodes
        ), f"{spec.name} is tagged 'cz' but compiled without move.CZ"

    if "movement" in spec.tags:
        assert any(
            isinstance(stmt, move.Move) and len(stmt.lanes) > 0 for stmt in nodes
        ), f"{spec.name} is tagged 'movement' but compiled without lane movement"

    if "layering" in spec.tags:
        assert (
            len(nodes) >= 2
        ), f"{spec.name} is tagged 'layering' but produced too few stateful move ops"


def test_movement_cz_interaction_has_nontrivial_physical_artifacts():
    """Guard a known movement+CZ kernel against trivialized physical compilation."""
    spec = next(
        s
        for s in select_kernels(tags={"movement", "cz"})
        if s.name == "curated_chain_cz_sweep"
    )
    kernel = spec.build_kernel()
    move_kernel = _compile_physical_move(kernel)
    nodes = _collect_move_nodes(move_kernel)

    move_count = sum(
        isinstance(stmt, move.Move) and len(stmt.lanes) > 0 for stmt in nodes
    )
    cz_count = sum(isinstance(stmt, move.CZ) for stmt in nodes)

    assert move_count >= 1
    assert cz_count >= 2
