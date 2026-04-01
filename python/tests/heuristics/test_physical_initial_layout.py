import pytest

from bloqade.lanes.arch import (
    AllToAllSiteTopology,
    ArchBlueprint,
    DeviceLayout,
    HypercubeWordTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)
from bloqade.lanes.heuristics.simple_layout import PhysicalLayoutHeuristicFixed

pymetis = pytest.importorskip("pymetis")


def _make_arch(dims: int = 1, sites_per_word: int = 10):
    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=1,
                num_cols=2 * 2**dims,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=AllToAllSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=sites_per_word),
    )
    return build_arch(bp).arch


def _weighted_stages(
    edge_counts: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], ...]]:
    stages: list[tuple[tuple[int, int], ...]] = []
    for (u, v), count in sorted(edge_counts.items()):
        for _ in range(count):
            stages.append(((u, v),))
    return stages


def _cut_weight(
    qubits: tuple[int, ...],
    q_to_word: dict[int, int],
    edge_counts: dict[tuple[int, int], int],
) -> int:
    _ = qubits
    total = 0
    for (u, v), weight in edge_counts.items():
        if q_to_word[u] != q_to_word[v]:
            total += weight
    return total


def _layout_affinity_cost(
    locations: dict[int, int],
    edge_counts: dict[tuple[int, int], int],
) -> int:
    total = 0
    for (u, v), weight in edge_counts.items():
        total += weight * abs(locations[u] - locations[v])
    return total


def test_initial_layout_is_left_only_and_full_word_first():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(1),
        max_words=2,
    )
    qubits = tuple(range(8))
    stages = _weighted_stages(
        {
            (0, 1): 4,
            (1, 2): 4,
            (2, 3): 4,
            (4, 5): 4,
            (5, 6): 4,
            (6, 7): 4,
        }
    )
    layout_out = strategy.compute_layout(qubits, stages)
    assert len(layout_out) == len(qubits)
    assert len(set(layout_out)) == len(layout_out)
    assert all(addr.site_id < strategy.left_site_count for addr in layout_out)
    per_word = {}
    for addr in layout_out:
        per_word[addr.word_id] = per_word.get(addr.word_id, 0) + 1
    assert per_word.get(0, 0) == 5
    assert per_word.get(1, 0) == 3


def test_fill_capacity_enforcement_avoids_balanced_split_when_target_is_5_3():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(1),
        max_words=2,
    )
    qubits = tuple(range(8))
    stages = [
        ((0, 4), (1, 5), (2, 6), (3, 7)),
        ((0, 1), (2, 3), (4, 5), (6, 7)),
    ]
    layout_out = strategy.compute_layout(qubits, stages)
    per_word = {}
    for addr in layout_out:
        per_word[addr.word_id] = per_word.get(addr.word_id, 0) + 1
    assert per_word.get(0, 0) == 5
    assert per_word.get(1, 0) == 3


def test_initial_layout_is_deterministic():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(1),
        max_words=2,
    )
    qubits = tuple(range(8))
    stages = _weighted_stages(
        {
            (0, 1): 3,
            (1, 2): 3,
            (2, 3): 3,
            (4, 5): 3,
            (5, 6): 3,
            (6, 7): 3,
            (1, 6): 1,
        }
    )
    first = strategy.compute_layout(qubits, stages)
    second = strategy.compute_layout(qubits, stages)
    third = strategy.compute_layout(qubits, stages)
    assert first == second == third


def test_partition_word_fill_is_full_then_partial_left_to_right():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(1),
        max_words=2,
    )
    qubits = tuple(range(6))
    edge_counts = {
        (0, 1): 8,
        (1, 2): 8,
        (0, 2): 8,
        (3, 4): 8,
        (4, 5): 8,
        (3, 5): 8,
        (2, 3): 1,
    }
    stages = _weighted_stages(edge_counts)
    layout_out = strategy.compute_layout(qubits, stages)
    per_word = {}
    for addr in layout_out:
        per_word[addr.word_id] = per_word.get(addr.word_id, 0) + 1
    assert sorted(per_word.keys()) == [0, 1]
    assert per_word[0] == 5
    assert per_word[1] == 1


def test_within_word_affinity_cost_beats_naive_ordering():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(1),
        max_words=1,
    )
    qubits = tuple(range(5))
    # Star-like affinity where center-out should place qubit 0 near the center.
    edge_counts = {
        (0, 1): 10,
        (0, 2): 10,
        (0, 3): 10,
        (0, 4): 10,
    }
    stages = _weighted_stages(edge_counts)
    layout_out = strategy.compute_layout(qubits, stages)

    strategy_sites = {
        qid: addr.site_id for qid, addr in zip(qubits, layout_out, strict=True)
    }
    strategy_cost = _layout_affinity_cost(strategy_sites, edge_counts)

    naive_sites = {qid: idx for idx, qid in enumerate(sorted(qubits))}
    naive_cost = _layout_affinity_cost(naive_sites, edge_counts)
    assert strategy_cost <= naive_cost


def test_word_assignment_overflow_expands_to_next_word():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(4),
        max_words=4,
    )
    qubits = tuple(range(8))
    edge_counts = {
        (0, 1): 3,
        (2, 3): 3,
        (4, 5): 3,
        (6, 7): 3,
    }
    stages = _weighted_stages(edge_counts)
    layout_out = strategy.compute_layout(qubits, stages)

    used_words = sorted({addr.word_id for addr in layout_out})
    assert len(used_words) == 2
    assert used_words == [0, 1]


def test_final_partial_word_places_from_lowest_site_up():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(1, sites_per_word=14),
        max_words=1,
    )
    qubits = (0,)
    stages = _weighted_stages({})
    layout_out = strategy.compute_layout(qubits, stages)
    assert layout_out[0].word_id == 0
    assert layout_out[0].site_id == 0


def test_relabel_words_fill_left_to_right():
    strategy = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=_make_arch(2),
        max_words=4,
    )
    # Partition ids are arbitrary METIS labels; largest block should map to leftmost word.
    q_to_word = {
        0: 10,
        1: 10,
        2: 11,
        3: 12,
        4: 13,
    }
    relabeled = strategy._left_to_right_relabel_words(q_to_word)
    assert relabeled[0] == 0
    assert relabeled[1] == 0


def test_fixed_baseline_fill_order():
    strategy = PhysicalLayoutHeuristicFixed(
        arch_spec=_make_arch(1),
    )
    qubits = tuple(range(6))
    out = strategy.compute_layout(qubits, _weighted_stages({}))
    coords = tuple((addr.word_id, addr.site_id) for addr in out)
    assert coords == ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0))
