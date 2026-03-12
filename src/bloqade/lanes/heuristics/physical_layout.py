from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import pymetis

from bloqade.lanes import layout
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec


def _to_cz_layers(
    stages: list[tuple[tuple[int, int], ...]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    layers = []
    for stage in stages:
        controls = tuple(pair[0] for pair in stage)
        targets = tuple(pair[1] for pair in stage)
        layers.append((controls, targets))
    return tuple(layers)


@dataclass
class PhysicalLayoutHeuristicFixed(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)

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


@dataclass
class PhysicalLayoutHeuristicGraphPartitionCenterOut(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)
    preferred_fill: int | None = None
    max_words: int | None = None
    metis_ubvec: float = 1.001
    metis_recursive: bool = False

    @property
    def left_site_count(self) -> int:
        return len(self.arch_spec.words[0].site_indices) // 2

    def _effective_fill(self) -> int:
        _ = self.preferred_fill
        # Pack words by physical capacity first: full words, then one partial word.
        return self.left_site_count

    def _word_count(self, qubit_count: int) -> int:
        max_words = (
            len(self.arch_spec.words) if self.max_words is None else self.max_words
        )
        f_eff = self._effective_fill()
        return min(max_words, math.ceil(qubit_count / f_eff))

    def _target_block_sizes(self, qubit_count: int, k_words: int) -> tuple[int, ...]:
        sizes: list[int] = []
        remaining = qubit_count
        for _ in range(k_words):
            size = min(self.left_site_count, remaining)
            sizes.append(size)
            remaining -= size
        return tuple(sizes)

    def _build_weighted_graph(
        self,
        qubits: tuple[int, ...],
        cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
        qids = tuple(sorted(set(qubits)))
        q_to_node = {qid: idx for idx, qid in enumerate(qids)}

        edge_weights: dict[tuple[int, int], int] = defaultdict(int)
        for controls, targets in cz_layers:
            for c, t in zip(controls, targets):
                if c == t:
                    continue
                if c not in q_to_node or t not in q_to_node:
                    continue
                u = q_to_node[c]
                v = q_to_node[t]
                if u > v:
                    u, v = v, u
                edge_weights[(u, v)] += 1

        return q_to_node, edge_weights

    def _partition_words(
        self,
        qubits: tuple[int, ...],
        cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        k_words: int,
        target_sizes: tuple[int, ...],
    ) -> dict[int, int]:
        q_to_node, edge_weights = self._build_weighted_graph(qubits, cz_layers)
        qids = tuple(sorted(set(qubits)))
        n = len(qids)
        if k_words == 1:
            return {qid: 0 for qid in qids}

        tpwgts = [size / float(n) for size in target_sizes]
        adjacency: list[list[int]] = [[] for _ in range(n)]
        adjacency_w: list[list[int]] = [[] for _ in range(n)]
        for (u, v), w in sorted(edge_weights.items()):
            adjacency[u].append(v)
            adjacency[v].append(u)
            adjacency_w[u].append(w)
            adjacency_w[v].append(w)
        xadj = [0]
        adjncy: list[int] = []
        eweights: list[int] = []
        for node in range(n):
            adjncy.extend(adjacency[node])
            eweights.extend(adjacency_w[node])
            xadj.append(len(adjncy))

        ufactor = max(1, int(round((self.metis_ubvec - 1.0) * 1000.0)))
        options = pymetis.Options(ufactor=ufactor)
        csr_adjacency = pymetis.CSRAdjacency(xadj, adjncy)
        graph_partition = pymetis.part_graph(
            nparts=k_words,
            adjacency=csr_adjacency,
            eweights=eweights,
            tpwgts=tpwgts,
            recursive=self.metis_recursive,
            options=options,
        )
        if hasattr(graph_partition, "vertex_part"):
            parts = graph_partition.vertex_part
        else:
            _, parts = graph_partition
        if len(parts) != n:
            raise RuntimeError("METIS returned unexpected partition size.")
        q_to_word = {qid: int(parts[q_to_node[qid]]) for qid in qids}
        return q_to_word

    def _left_to_right_relabel_words(
        self,
        q_to_word: dict[int, int],
    ) -> dict[int, int]:
        members: dict[int, list[int]] = defaultdict(list)
        for qid, wid in q_to_word.items():
            members[wid].append(qid)
        old_words = sorted(members.keys(), key=lambda wid: min(members[wid]))
        desired_word_ids = list(range(len(old_words)))
        old_words_sorted = sorted(
            old_words,
            key=lambda wid: (-len(members[wid]), min(members[wid])),
        )
        remap = {
            old_wid: new_wid
            for old_wid, new_wid in zip(old_words_sorted, desired_word_ids)
        }
        return {qid: remap[wid] for qid, wid in q_to_word.items()}

    def _left_sites_center_out(self, word_id: int) -> list[layout.LocationAddress]:
        left_sites = [
            layout.LocationAddress(word_id, site_id)
            for site_id in range(self.left_site_count)
        ]
        center_site = (self.left_site_count - 1) / 2.0
        return sorted(
            left_sites,
            key=lambda addr: (abs(addr.site_id - center_site), addr.site_id),
        )

    def _within_word_placement(
        self,
        members: list[int],
        q_to_node: dict[int, int],
        weighted_edges: dict[tuple[int, int], int],
        sites: list[layout.LocationAddress],
    ) -> dict[int, layout.LocationAddress]:
        def edge_weight(q0: int, q1: int) -> int:
            u = q_to_node[q0]
            v = q_to_node[q1]
            if u > v:
                u, v = v, u
            return weighted_edges.get((u, v), 0)

        def local_weighted_degree(q: int) -> int:
            return sum(edge_weight(q, other) for other in members if other != q)

        def local_unweighted_degree(q: int) -> int:
            return sum(
                1 for other in members if other != q and edge_weight(q, other) > 0
            )

        if len(members) == 0:
            return {}

        placed: dict[int, layout.LocationAddress] = {}
        remaining = set(members)
        center_qubit = max(
            sorted(remaining),
            key=lambda q: (
                local_weighted_degree(q),
                local_unweighted_degree(q),
                -q,
            ),
        )
        placed[center_qubit] = sites[0]
        remaining.remove(center_qubit)

        for site in sites[1 : len(members)]:
            best_q = None
            best_key = None
            for q in sorted(remaining):
                score = 0.0
                for p, p_site in placed.items():
                    distance = abs(site.site_id - p_site.site_id)
                    score += edge_weight(q, p) / float(1 + distance)
                key = (
                    score,
                    local_weighted_degree(q),
                    local_unweighted_degree(q),
                    -q,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_q = q
            assert best_q is not None
            placed[best_q] = site
            remaining.remove(best_q)

        return placed

    def _compute_layout_from_cz_layers(
        self,
        qubits: tuple[int, ...],
        cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> tuple[layout.LocationAddress, ...]:
        k_words = self._word_count(len(qubits))
        target_sizes = self._target_block_sizes(len(qubits), k_words)
        q_to_node, weighted_edges = self._build_weighted_graph(qubits, cz_layers)

        q_to_word_raw = self._partition_words(qubits, cz_layers, k_words, target_sizes)
        q_to_word = self._left_to_right_relabel_words(q_to_word_raw)

        members_by_word: dict[int, list[int]] = defaultdict(list)
        for qid in sorted(qubits):
            members_by_word[q_to_word[qid]].append(qid)

        # Enforce left-to-right target capacities after METIS assignment.
        capacities = {word_id: target_sizes[word_id] for word_id in range(k_words)}
        for word_id in range(k_words):
            members_by_word.setdefault(word_id, [])

        for src_word in range(k_words):
            while len(members_by_word[src_word]) > capacities[src_word]:
                # Move deterministically: spill highest-id qubits first.
                spill_qid = max(members_by_word[src_word])
                members_by_word[src_word].remove(spill_qid)
                for dst_word in range(src_word + 1, k_words):
                    if len(members_by_word[dst_word]) < capacities[dst_word]:
                        members_by_word[dst_word].append(spill_qid)
                        break
                else:
                    members_by_word[src_word].append(spill_qid)
                    break

        q_to_location: dict[int, layout.LocationAddress] = {}
        for word_id in range(k_words):
            members = members_by_word[word_id]
            sites = self._left_sites_center_out(word_id)
            assigned = self._within_word_placement(
                members,
                q_to_node=q_to_node,
                weighted_edges=weighted_edges,
                sites=sites,
            )
            q_to_location.update(assigned)

        return tuple(q_to_location[qid] for qid in qubits)

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
    ) -> tuple[layout.LocationAddress, ...]:
        qubits = tuple(sorted(all_qubits))
        cz_layers = _to_cz_layers(stages)
        return self._compute_layout_from_cz_layers(qubits, cz_layers)
