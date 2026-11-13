"""Slot allocation stays unresolved until the physical compiler is configured."""

from typing import Any

import pytest
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.lanes.analysis.code_blocks import CodeBlockTag
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.dialects import code_block, move, place
from bloqade.lanes.rewrite.allocate_slots import has_slot_allocations
from bloqade.lanes.transform import PhysicalNativeToPlace, PhysicalPipeline

from .._code_block_kernel import kernel


@kernel
def two_blocks():
    blocks = code_block.qalloc_at(ilist.IList([1, 0]))
    squin.h(blocks[0][0])
    return squin.broadcast.measure(blocks[0] + blocks[1])


SLOTS = [[(0, 0, 0), (0, 0, 4)], [(0, 0, 2), (0, 0, 6)]]


def test_slot_allocation_preserves_membership_and_address_order():
    arch = get_arch_spec()
    out = PhysicalNativeToPlace(arch_spec=arch, slots=SLOTS).emit(
        two_blocks.similar(), no_raise=False
    )
    allocations = [
        s for s in out.callable_region.walk() if isinstance(s, place.NewPinnedQubit)
    ]
    assert [s.location_address for s in allocations] == [
        arch.location_at(*coord) for slot in [SLOTS[1], SLOTS[0]] for coord in slot
    ]
    assert [s.code_block for s in allocations] == [
        CodeBlockTag(0, 0),
        CodeBlockTag(0, 1),
        CodeBlockTag(1, 0),
        CodeBlockTag(1, 1),
    ]


def test_slot_allocation_through_physical_pipeline():
    out = PhysicalPipeline(slots=SLOTS).emit(two_blocks.similar(), no_raise=False)
    fill = next(s for s in out.callable_region.walk() if isinstance(s, move.Fill))
    assert len(fill.location_addresses) == 4


def test_missing_slots_is_an_error():
    with pytest.raises(ValueError, match="requires compiler slots"):
        PhysicalPipeline().emit(two_blocks.similar(), no_raise=False)


@kernel
def one_block():
    blocks = code_block.qalloc_at(ilist.IList([0]))
    return squin.broadcast.measure(blocks[0])


@pytest.mark.parametrize(
    "slot",
    [
        [(0, 0, 8), (0, 0, 2), (0, 1, 0)],
        [(0, 0, 4 * i) for i in range(8)] + [(0, 1, 0)],
    ],
)
def test_arbitrary_pinned_block_shapes(slot):
    arch = get_arch_spec()
    out = PhysicalPipeline(arch_spec=arch, slots=[slot]).emit(
        one_block.similar(), no_raise=False
    )
    fill = next(s for s in out.callable_region.walk() if isinstance(s, move.Fill))
    assert set(fill.location_addresses) == {arch.location_at(*coord) for coord in slot}


@pytest.mark.parametrize(
    "slots, message",
    [
        ([], "out of range"),
        ([[]], "is empty"),
        ([[(0, 0)]], "expected.*zone, row, col"),
        ([[(0, 0, 999)]], "no architecture location"),
    ],
)
def test_invalid_slots(slots, message):
    with pytest.raises(ValueError, match=message):
        PhysicalPipeline(slots=slots).emit(one_block.similar(), no_raise=False)


def test_duplicate_addresses_are_rejected():
    with pytest.raises(ValidationErrorGroup, match="pinned by two"):
        PhysicalPipeline(slots=[[(0, 0, 0), (0, 0, 0)]]).emit(
            one_block.similar(), no_raise=False
        )


def test_kernel_can_be_recompiled_with_different_slots():
    for slot in [[(0, 0, 0)], [(0, 1, 0), (0, 1, 2)]]:
        out = PhysicalPipeline(slots=[slot]).emit(one_block.similar(), no_raise=False)
        fill = next(s for s in out.callable_region.walk() if isinstance(s, move.Fill))
        assert len(fill.location_addresses) == len(slot)
    assert has_slot_allocations(one_block)


def test_helper_allocation_and_nested_list_iteration():
    @kernel
    def allocate():
        return code_block.qalloc_at(ilist.IList([1, 0]))

    @kernel(fold=False)
    def program():
        blocks = allocate()
        ilist.for_each(squin.broadcast.x, blocks)
        return squin.broadcast.measure(blocks[0] + blocks[1])

    out = PhysicalPipeline(slots=SLOTS).emit(program, no_raise=False)
    assert not has_slot_allocations(out)


@pytest.mark.parametrize("indices", [[-1], [2], [True]])
def test_invalid_slot_indices(indices):
    ids = ilist.IList(indices)

    @kernel
    def program():
        blocks = code_block.qalloc_at(ids)
        return squin.broadcast.measure(blocks[0])

    with pytest.raises(ValueError, match="out of range"):
        PhysicalPipeline(slots=SLOTS).emit(program, no_raise=False)


def test_runtime_slot_indices_are_rejected():
    @kernel
    def program(indices: ilist.IList[int, Any]):
        blocks = code_block.qalloc_at(indices)
        return squin.broadcast.measure(blocks[0])

    with pytest.raises(TypeError, match="compile-time constant IList"):
        PhysicalPipeline(slots=SLOTS).emit(program, no_raise=False)


def test_terminal_validation_runs_after_expansion():
    @kernel
    def partial_measure():
        blocks = code_block.qalloc_at(ilist.IList([0, 1]))
        return squin.broadcast.measure(blocks[0])

    with pytest.raises(ValidationErrorGroup, match="all qubits must be measured"):
        PhysicalPipeline(slots=SLOTS).emit(partial_measure, no_raise=False)


def test_repeated_slot_is_rejected():
    @kernel
    def program():
        blocks = code_block.qalloc_at(ilist.IList([0, 0]))
        return squin.broadcast.measure(blocks[0] + blocks[1])

    with pytest.raises(ValidationErrorGroup, match="pinned by two"):
        PhysicalPipeline(slots=SLOTS).emit(program, no_raise=False)


def test_serialized_kernel_resolves_slots_when_compiled():
    decoded = kernel.decode_json(kernel.encode_json(one_block))
    out = PhysicalPipeline(slots=[SLOTS[0]]).emit(decoded, no_raise=False)
    assert not has_slot_allocations(out)
