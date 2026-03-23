from bloqade.lanes.heuristics import physical_layout as physical_layout_module
from bloqade.lanes.heuristics.physical_layout import (
    PhysicalLayoutHeuristicGraphPartitionCenterOut,
)


def _weighted_stages(
    edge_counts: dict[tuple[int, int], int],
) -> list[tuple[tuple[int, int], ...]]:
    stages: list[tuple[tuple[int, int], ...]] = []
    for (u, v), count in sorted(edge_counts.items()):
        for _ in range(count):
            stages.append(((u, v),))
    return stages


def test_partition_words_uses_kahip_backend(monkeypatch):
    strategy = object.__new__(PhysicalLayoutHeuristicGraphPartitionCenterOut)
    strategy.u_factor = 1
    strategy.partitioner_seed = 0
    qubits = tuple(range(6))
    stages = _weighted_stages(
        {
            (0, 1): 3,
            (1, 2): 3,
            (3, 4): 3,
            (4, 5): 3,
        }
    )
    cz_layers = physical_layout_module._to_cz_layers(stages)
    target_sizes = (3, 3)

    class FakeKaHIP:
        def __init__(self):
            self.called = False

        def kaffpa(
            self,
            vwgt,
            xadj,
            adjcwgt,
            adjncy,
            nblocks,
            imbalance,
            suppress_output,
            seed,
            mode,
        ):
            self.called = True
            assert len(vwgt) == len(qubits)
            assert xadj[0] == 0
            assert len(adjncy) == len(adjcwgt)
            assert nblocks == 2
            assert imbalance > 0
            assert suppress_output in (0, False)
            assert seed == strategy.partitioner_seed
            assert isinstance(mode, int)
            return 0, [0, 0, 0, 1, 1, 1]

    fake = FakeKaHIP()
    monkeypatch.setattr(physical_layout_module, "kahip", fake, raising=False)

    q_to_word = strategy._partition_words(
        qubits=qubits,
        cz_layers=cz_layers,
        k_words=2,
        target_sizes=target_sizes,
    )
    assert fake.called is True
    assert q_to_word == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
