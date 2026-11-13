"""Block-aware layout in PhysicalLayoutHeuristicGraphPartitionCenterOut."""

import pytest

from bloqade.lanes.analysis.code_blocks import (
    CodeBlock,
    CodeBlockPlacementError,
    block_shape_error,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
    _BlockLayout,
    _to_cz_layers,
)


def _heuristic(**kwargs) -> PhysicalLayoutHeuristicGraphPartitionCenterOut:
    return PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=get_arch_spec(), **kwargs
    )


def _blocks(size: int, count: int) -> tuple[CodeBlock, ...]:
    return tuple(
        CodeBlock(i, tuple(range(size * i, size * (i + 1)))) for i in range(count)
    )


def _transversal(a: CodeBlock, b: CodeBlock) -> tuple[tuple[int, int], ...]:
    return tuple(zip(a.qids, b.qids))


def _assert_shape(layout, blocks):
    for block in blocks:
        assert block_shape_error([layout[q] for q in block.qids], 8) is None


def test_every_block_lands_in_one_word_on_contiguous_sites():
    blocks = _blocks(4, 4)
    stages = [
        _transversal(blocks[0], blocks[1]) + _transversal(blocks[2], blocks[3]),
        _transversal(blocks[0], blocks[2]) + _transversal(blocks[1], blocks[3]),
        _transversal(blocks[0], blocks[3]) + _transversal(blocks[1], blocks[2]),
    ]
    layout = _heuristic().compute_layout_with_blocks(
        tuple(range(16)), stages, {}, blocks
    )
    _assert_shape(layout, blocks)
    words = [layout[b.qids[0]].word_id for b in blocks]
    assert sorted(words.count(w) for w in set(words)) == [2, 2]


def test_transversal_partners_share_offset_in_different_words():
    a, b = _blocks(7, 2)
    layout = _heuristic().compute_layout_with_blocks(
        tuple(range(14)), [_transversal(a, b)], {}, (a, b)
    )
    _assert_shape(layout, (a, b))
    assert layout[a.qids[0]].site_id == layout[b.qids[0]].site_id
    assert layout[a.qids[0]].word_id != layout[b.qids[0]].word_id


def test_blocks_share_a_word_when_capacity_requires_it():
    blocks = _blocks(4, 2)
    layout = _heuristic(max_words=1).compute_layout_with_blocks(
        tuple(range(8)), [], {}, blocks
    )
    _assert_shape(layout, blocks)
    assert {layout[q].word_id for q in range(8)} == {0}


def test_blocks_are_ranked_by_entanglement_weight():
    blocks = _blocks(2, 3)
    # Block 2 has the most CZs leaving it, then block 1; block 0 has none.
    stages = [((4, 6), (5, 7)), ((4, 6),), ((2, 6),)]
    qubits = tuple(range(8))
    ranker = _BlockLayout(_heuristic(), qubits, _to_cz_layers(stages), {}, blocks)
    assert [b.block_id for b in ranker._rank()] == [2, 1, 0]


def test_unblocked_qubits_are_pulled_toward_their_block_partners():
    (block,) = _blocks(4, 1)
    layout = _heuristic().compute_layout_with_blocks(
        tuple(range(5)), [((4, 2),)], {}, (block,)
    )
    _assert_shape(layout, (block,))
    partner = layout[2]
    free_sites = sorted({s for s in range(8)} - {layout[q].site_id for q in block.qids})
    nearest = min(free_sites, key=lambda s: abs(s - partner.site_id))
    assert layout[4].site_id == nearest


def test_pinned_blocks_stay_at_their_pins():
    pins = {q: LocationAddress(2, 4 + q, 0) for q in range(4)}
    blocks = _blocks(4, 2)
    layout = _heuristic().compute_layout_with_blocks(
        tuple(range(8)), [_transversal(*blocks)], pins, blocks
    )
    assert all(layout[q] == pins[q] for q in range(4))
    _assert_shape(layout, blocks)
    assert layout[4].word_id != 2 and layout[4].site_id == 4


def test_unpackable_blocks_raise():
    with pytest.raises(CodeBlockPlacementError, match="does not fit"):
        _heuristic(max_words=1).compute_layout_with_blocks(
            tuple(range(10)), [], {}, _blocks(5, 2)
        )


def test_oversized_and_partly_pinned_blocks_raise():
    with pytest.raises(CodeBlockPlacementError, match="only 8 sites"):
        _heuristic().compute_layout_with_blocks(tuple(range(9)), [], {}, _blocks(9, 1))
    with pytest.raises(CodeBlockPlacementError, match="partly pinned"):
        _heuristic().compute_layout_with_blocks(
            tuple(range(2)), [], {0: LocationAddress(0, 0, 0)}, _blocks(2, 1)
        )


@pytest.mark.parametrize(
    "pinned",
    [{}, {1: LocationAddress(0, 3, 0), 6: LocationAddress(2, 0, 0)}],
)
def test_no_blocks_matches_compute_layout(pinned):
    qubits = tuple(range(12))
    stages = [((0, 5), (2, 7)), ((1, 6), (3, 11)), ((0, 9),), ((4, 10), (8, 2))]
    heuristic = _heuristic()
    assert heuristic.compute_layout_with_blocks(
        qubits, stages, pinned, ()
    ) == heuristic.compute_layout(qubits, stages, pinned)
