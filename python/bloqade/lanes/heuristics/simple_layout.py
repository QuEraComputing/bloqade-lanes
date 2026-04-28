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
        pinned: dict[int, layout.LocationAddress] | None = None,
    ) -> tuple[layout.LocationAddress, ...]:
        _ = stages
        pinned = pinned or {}
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
        qubits = tuple(sorted(all_qubits))

        pinned_addresses: set[layout.LocationAddress] = set(pinned.values())

        candidates: list[layout.LocationAddress] = []
        for word_id in self.home_word_ids:
            for site_id in range(self.sites_per_home_word):
                addr = layout.LocationAddress(word_id, site_id)
                if addr not in pinned_addresses:
                    candidates.append(addr)

        unpinned_qubits = [q for q in qubits if q not in pinned]
        if len(unpinned_qubits) > len(candidates):
            raise ValueError(
                f"layout heuristic cannot place {len(unpinned_qubits)} un-pinned qubits: "
                f"arch provides {len(candidates) + len(pinned_addresses)} total sites, "
                f"{len(pinned_addresses)} are pinned, leaving {len(candidates)} available; "
                "no legal positions remain"
            )

        result = dict(zip(unpinned_qubits, candidates)) | pinned
        return tuple(result[q] for q in qubits)
