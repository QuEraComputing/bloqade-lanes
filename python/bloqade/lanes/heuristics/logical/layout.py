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
        edges: dict[tuple[int, int], float],
        qubit_map: dict[int, layout.LocationAddress],
    ) -> float:
        move_weights: dict[tuple[int, int], float] = {}
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
        score = 0.0
        for i, move_i in enumerate(all_moves):
            for move_j in all_moves[i + 1 :]:
                score += move_weights[move_i] + move_weights[move_j]

        return score

    def _validate_pinned(
        self,
        all_qubits: tuple[int, ...],
        pinned: dict[int, layout.LocationAddress],
    ) -> None:
        if len(set(pinned.values())) < len(pinned):
            raise ValueError(
                "pinned addresses must be unique; two qubit IDs share the same address"
            )
        extra_keys = set(pinned) - set(all_qubits)
        if extra_keys:
            raise ValueError(
                f"pinned contains qubit IDs not in all_qubits: {sorted(extra_keys)}"
            )
        self._validate_pinned_in_arch(pinned, self.arch_spec)

    def _compute_layout_from_weighted_edges(
        self,
        all_qubits: tuple[int, ...],
        edges: dict[tuple[int, int], float],
        pinned: dict[int, layout.LocationAddress],
    ) -> tuple[layout.LocationAddress, ...]:
        if len(all_qubits) > self.arch_spec.max_qubits:
            raise interp.InterpreterError(
                f"Number of qubits in circuit ({len(all_qubits)}) exceeds "
                f"maximum supported by logical architecture ({self.arch_spec.max_qubits})"
            )

        pinned_addresses = set(pinned.values())
        available_addresses = set(self.arch_spec.home_sites) - pinned_addresses

        unpinned_qubits = [q for q in sorted(all_qubits) if q not in pinned]

        if len(unpinned_qubits) > len(available_addresses):
            raise ValueError(
                f"layout heuristic cannot place {len(unpinned_qubits)} un-pinned qubits: "
                f"arch provides {len(self.arch_spec.home_sites)} total sites, "
                f"{len(pinned_addresses)} are pinned, leaving {len(available_addresses)} available; "
                "no legal positions remain"
            )

        # Pre-seed pinned qubits so score_parallelism sees them as context.
        qubit_map: dict[int, layout.LocationAddress] = dict(pinned)
        layout_map: dict[layout.LocationAddress, int] = {
            addr: q for q, addr in pinned.items()
        }

        for qubit in unpinned_qubits:
            scores: dict[layout.LocationAddress, float] = {}
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

        final_layout = list(layout_map.keys())
        final_layout.sort(key=lambda x: layout_map[x])
        return tuple(final_layout)

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, layout.LocationAddress] | None = None,
    ) -> tuple[layout.LocationAddress, ...]:
        pinned = pinned or {}
        self._validate_pinned(all_qubits, pinned)
        edges: dict[tuple[int, int], float] = {}

        for control, target in chain.from_iterable(stages):
            n, m = min(control, target), max(control, target)
            edge_weight = edges.get((n, m), 0)
            edges[(n, m)] = edge_weight + 1

        return self._compute_layout_from_weighted_edges(all_qubits, edges, pinned)


@dataclass
class LogicalLayoutHeuristicRecencyWeighted(LogicalLayoutHeuristic):
    layout_lookahead_layers: int | None = None
    layout_decay_gamma: float = 0.85

    def __post_init__(self):
        if not 0.0 < self.layout_decay_gamma <= 1.0:
            raise ValueError("layout_decay_gamma must be in the interval (0, 1].")

        if (
            self.layout_lookahead_layers is not None
            and self.layout_lookahead_layers < 0
        ):
            raise ValueError("layout_lookahead_layers must be non-negative or None.")

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, layout.LocationAddress] | None = None,
    ) -> tuple[layout.LocationAddress, ...]:
        pinned = pinned or {}
        self._validate_pinned(all_qubits, pinned)
        if self.layout_lookahead_layers is None:
            considered_layers = stages
        else:
            considered_layers = stages[: self.layout_lookahead_layers]

        edges: dict[tuple[int, int], float] = {}
        for depth, layer in enumerate(considered_layers):
            decay_weight = self.layout_decay_gamma**depth
            for control, target in layer:
                n, m = min(control, target), max(control, target)
                edge_weight = edges.get((n, m), 0.0)
                edges[(n, m)] = edge_weight + decay_weight

        return self._compute_layout_from_weighted_edges(all_qubits, edges, pinned)
