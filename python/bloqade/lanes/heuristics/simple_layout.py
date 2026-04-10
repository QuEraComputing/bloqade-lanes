from dataclasses import dataclass, field

from bloqade.lanes import layout
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini.physical import (
    get_physical_layout_arch_spec,
)


@dataclass
class PhysicalLayoutHeuristicFixed(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_physical_layout_arch_spec)

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
                # NOTE: assumes single-zone architecture (zone_id=0).
                sites.append(layout.LocationAddress(word_id, site_id, 0))
        return tuple(sites[: len(qubits)])
