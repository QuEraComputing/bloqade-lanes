from dataclasses import dataclass, field
from itertools import chain, combinations

from kirin import interp

from bloqade.lanes import layout
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini.logical import get_arch_spec


@dataclass
class LogicalLayoutHeuristic(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_arch_spec, init=False)

    def score_parallelism(
        self,
        edges: dict[tuple[int, int], int],
        qubit_map: dict[int, layout.LocationAddress],
    ) -> int:
        move_weights = {}
        for n, m in combinations(qubit_map.keys(), 2):
            n, m = (min(n, m), max(n, m))
            edge_weight = edges.get((n, m))
            if edge_weight is None:
                continue

            addr_n = qubit_map[n]
            addr_m = qubit_map[m]
            site_diff = (addr_n.site_id - addr_m.site_id) // 2
            word_diff = addr_n.word_id - addr_m.word_id
            if word_diff != 0:
                edge_weight *= 2

            move_weights[(word_diff, site_diff)] = (
                move_weights.get((word_diff, site_diff), 0) + edge_weight
            )

        all_moves = list(move_weights.keys())
        score = 0
        for i, move_i in enumerate(all_moves):
            for move_j in all_moves[i + 1 :]:
                score += move_weights[move_i] + move_weights[move_j]

        return score

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
    ) -> tuple[layout.LocationAddress, ...]:

        if len(all_qubits) > self.arch_spec.max_qubits:
            raise interp.InterpreterError(
                f"Number of qubits in circuit ({len(all_qubits)}) exceeds maximum supported by logical architecture ({self.arch_spec.max_qubits})"
            )

        edges = {}

        for control, target in chain.from_iterable(stages):
            n, m = min(control, target), max(control, target)
            edge_weight = edges.get((n, m), 0)
            edges[(n, m)] = edge_weight + 1

        available_addresses = set(
            [
                layout.LocationAddress(word_id, site_id)
                for word_id in range(len(self.arch_spec.words))
                for site_id in range(5)
            ]
        )

        qubit_map: dict[int, layout.LocationAddress] = {}
        layout_map: dict[layout.LocationAddress, int] = {}
        for qubit in sorted(all_qubits):

            scores: dict[layout.LocationAddress, int] = {}
            for addr in available_addresses:
                qubit_map = qubit_map.copy()
                qubit_map[qubit] = addr
                scores[addr] = self.score_parallelism(edges, qubit_map)

            best_addr = min(
                scores.keys(), key=lambda x: (scores[x], x.word_id, x.site_id)
            )
            available_addresses.remove(best_addr)
            qubit_map[qubit] = best_addr
            layout_map[best_addr] = qubit

        # invert layout
        final_layout = list(layout_map.keys())
        final_layout.sort(key=lambda x: layout_map[x])
        return tuple(final_layout)
