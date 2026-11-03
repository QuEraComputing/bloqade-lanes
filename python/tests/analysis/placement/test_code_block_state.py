"""Code blocks reach layout and placement analysis, and survive every strategy."""

import pytest
from bloqade.analysis import address

from bloqade import squin
from bloqade.gemini.physical import kernel
from bloqade.lanes import code_block
from bloqade.lanes.analysis.code_blocks import (
    CodeBlock,
    CodeBlockWarning,
    LocalCodeBlock,
    localize_code_blocks,
)
from bloqade.lanes.analysis.layout import (
    CodeBlockLayoutHeuristicABC,
    LayoutAnalysis,
)
from bloqade.lanes.analysis.placement import ConcreteState, PlacementAnalysis
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.physical.movement import (
    make_physical_placement_strategy,
)
from bloqade.lanes.heuristics.simple_layout import PhysicalLayoutHeuristicFixed
from bloqade.lanes.passes import SequentialPlacePass
from bloqade.lanes.transform.native_to_place import PhysicalNativeToPlace


@kernel
def two_blocks_and_ancilla():
    def new_block():
        q = squin.qalloc(4)
        code_block.register(q)
        return q

    a = new_block()
    b = new_block()
    anc = squin.qalloc(1)
    squin.broadcast.cz(a, b)
    squin.cz(anc[0], a[1])
    squin.broadcast.measure(a + b + anc)


EXPECTED_BLOCKS = (CodeBlock(0, (0, 1, 2, 3)), CodeBlock(1, (4, 5, 6, 7)))


def _place(mt):
    arch = get_arch_spec()
    out = PhysicalNativeToPlace(arch_spec=arch).emit(mt, no_raise=False)
    SequentialPlacePass(out.dialects)(out)
    return out


def _layout(out, heuristic):
    address_analysis = address.AddressAnalysis(out.dialects)
    frame, _ = address_analysis.run(out)
    analysis = LayoutAnalysis(
        out.dialects,
        heuristic,
        frame.entries,
        tuple(range(address_analysis.next_address)),
    )
    return analysis, analysis.get_layout(out), frame.entries


def test_layout_analysis_collects_blocks():
    out = _place(two_blocks_and_ancilla)
    heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=get_arch_spec()
    )
    analysis, layout, _ = _layout(out, heuristic)
    assert analysis.code_blocks == EXPECTED_BLOCKS
    for block in EXPECTED_BLOCKS:
        words = {layout[q].word_id for q in block.qids}
        sites = [layout[q].site_id for q in block.qids]
        assert len(words) == 1
        assert sites == list(range(sites[0], sites[0] + len(sites)))


def test_non_aware_heuristic_warns_and_ignores_blocks():
    out = _place(two_blocks_and_ancilla)
    heuristic = PhysicalLayoutHeuristicFixed(arch_spec=get_arch_spec())
    with pytest.warns(CodeBlockWarning, match="not block-aware"):
        analysis, layout, _ = _layout(out, heuristic)
    assert analysis.code_blocks == ()
    stages = list(analysis.stages)
    assert layout == heuristic.compute_layout(tuple(range(9)), stages, {})


def test_layout_post_condition_rejects_broken_blocks():
    class Broken(CodeBlockLayoutHeuristicABC):
        def __init__(self):
            self.arch_spec = get_arch_spec()
            self.inner = PhysicalLayoutHeuristicFixed(arch_spec=self.arch_spec)

        def compute_layout(self, all_qubits, stages, pinned=None):
            return self.inner.compute_layout(all_qubits, stages, pinned)

        def compute_layout_with_blocks(self, all_qubits, stages, pinned, code_blocks):
            layout = list(self.compute_layout(all_qubits, stages, pinned))
            layout[0], layout[2] = layout[2], layout[0]
            return tuple(layout)

    out = _place(two_blocks_and_ancilla)
    with pytest.raises(RuntimeError, match="broke code block 0"):
        _layout(out, Broken())


# Built the way the pipeline builds them. A bare ``PhysicalPlacementStrategy()``
# cannot route this kernel's ancilla CZ with or without blocks.
STRATEGIES = {
    "factory_entropy": lambda arch: make_physical_placement_strategy(arch_spec=arch),
    "factory_no_return": lambda arch: make_physical_placement_strategy(
        arch_spec=arch, return_moves=False
    ),
    "factory_greedy": lambda arch: make_physical_placement_strategy(
        arch_spec=arch, strategy="greedy"
    ),
    "factory_astar": lambda arch: make_physical_placement_strategy(
        arch_spec=arch, strategy="astar"
    ),
}


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_code_blocks_survive_strategy(name: str):
    arch = get_arch_spec()
    out = _place(two_blocks_and_ancilla)
    analysis, layout, entries = _layout(
        out, PhysicalLayoutHeuristicGraphPartitionCenterOut(arch_spec=arch)
    )
    placement = PlacementAnalysis(
        out.dialects,
        layout,
        entries,
        STRATEGIES[name](arch),
        code_blocks=analysis.code_blocks,
    )
    frame, _ = placement.run(out)
    states = [s for s in frame.entries.values() if isinstance(s, ConcreteState)]
    assert states
    for state in states:
        assert state.code_blocks == (
            LocalCodeBlock(0, (0, 1, 2, 3)),
            LocalCodeBlock(1, (4, 5, 6, 7)),
        )


def test_code_blocks_do_not_affect_state_equality():
    loc = LocationAddress(0, 0, 0)
    plain = ConcreteState(occupied=frozenset(), layout=(loc,), move_count=(0,))
    tagged = ConcreteState(
        occupied=frozenset(),
        layout=(loc,),
        move_count=(0,),
        code_blocks=(LocalCodeBlock(0, (0,)),),
    )
    assert plain == tagged
    assert plain.is_subseteq(tagged) and tagged.is_subseteq(plain)


def test_localize_code_blocks_handles_partial_and_absent_blocks():
    blocks = (CodeBlock(0, (3, 4, 5)), CodeBlock(1, (6, 7)), CodeBlock(2, (8,)))
    local = localize_code_blocks(blocks, [5, 9, 3, 6])
    assert local == (
        LocalCodeBlock(0, (2, None, 0)),
        LocalCodeBlock(1, (3, None)),
    )
