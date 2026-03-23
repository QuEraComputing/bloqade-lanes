from dataclasses import dataclass

from bloqade.lanes import layout
from bloqade.lanes.analysis.layout import LayoutHeuristicABC


@dataclass
class PhysicalLayoutHeuristicFixed(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec

    @property
    def left_site_count(self) -> int:
        return len(self.arch_spec.words[0].site_indices) // 2

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
    ) -> tuple[layout.LocationAddress, ...]:
        _ = stages
        qubits = tuple(sorted(all_qubits))
        sites: list[layout.LocationAddress] = []
        for word_id in range(len(self.arch_spec.words)):
            for site_id in range(self.left_site_count):
                sites.append(layout.LocationAddress(word_id, site_id))
        return tuple(sites[: len(qubits)])
