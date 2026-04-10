from dataclasses import dataclass, field

from bloqade.lanes import layout
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini.physical import (
    get_physical_layout_arch_spec,
)


@dataclass
class PhysicalLayoutHeuristicFixed(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_physical_layout_arch_spec)

    def _validate_single_zone(self) -> None:
        if len(self.arch_spec.zones) != 1:
            raise ValueError(
                "PhysicalLayoutHeuristicFixed expects exactly one entangling "
                f"zone, got {len(self.arch_spec.zones)}."
            )

    @property
    def home_word_ids(self) -> tuple[int, ...]:
        self._validate_single_zone()
        return tuple(sorted(self.arch_spec._home_words))

    @property
    def sites_per_home_word(self) -> int:
        return len(self.arch_spec.words[0].site_indices)

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
    ) -> tuple[layout.LocationAddress, ...]:
        _ = stages
        qubits = tuple(sorted(all_qubits))
        sites: list[layout.LocationAddress] = []
        for word_id in self.home_word_ids:
            for site_id in range(self.sites_per_home_word):
                sites.append(layout.LocationAddress(word_id, site_id))
        return tuple(sites[: len(qubits)])
