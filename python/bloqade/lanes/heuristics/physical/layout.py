from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import kahip

from bloqade.lanes import layout
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini.physical import (
    get_physical_layout_arch_spec,
)


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
class PhysicalLayoutHeuristicGraphPartitionCenterOut(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_physical_layout_arch_spec)
    max_words: int | None = None
    u_factor: int = 1
    partitioner_seed: int = 0

    KAHIP_MODE_ECO = 1

    def _validate_single_zone(self) -> None:
        if len(self.arch_spec.zones) != 1:
            raise ValueError(
                "PhysicalLayoutHeuristicGraphPartitionCenterOut expects exactly one "
                f"entangling zone, got {len(self.arch_spec.zones)}."
            )

    @property
    def home_word_ids(self) -> tuple[int, ...]:
        """Home words for one-zone physical layout.

        Uses the arch-level home-word computation from zone entangling pairs.
        """
        self._validate_single_zone()
        return tuple(sorted(self.arch_spec._home_words))

    @property
    def sites_per_partition(self) -> int:
        """Maximum number of qubits placed per home word."""
        return len(self.arch_spec.words[0].site_indices)

    def _effective_fill(self) -> int:
        return self.sites_per_partition

    def _word_count(self, qubit_count: int) -> int:
        max_words = (
            len(self.home_word_ids) if self.max_words is None else self.max_words
        )
        f_eff = self._effective_fill()
        return min(max_words, math.ceil(qubit_count / f_eff))

    def _target_block_sizes(self, qubit_count: int, k_words: int) -> tuple[int, ...]:
        sizes: list[int] = []
        remaining = qubit_count
        for _ in range(k_words):
            size = min(self.sites_per_partition, remaining)
            sizes.append(size)
            remaining -= size
            if remaining == 0:
                break
        if remaining > 0:
            raise RuntimeError(f"Too many qubits to fill {k_words}")
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
                assert c != t, "invalid CZ pair"
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
        # build weighted graph of CZs
        q_to_node, edge_weights = self._build_weighted_graph(qubits, cz_layers)
        qids = tuple(sorted(set(qubits)))
        n = len(qids)

        # if only 1 word, no partitioner needed
        if k_words == 1:
            return {qid: 0 for qid in qids}

        # Build a weighted undirected CSR graph expected by KaHIP.
        adjacency: list[list[int]] = [[] for _ in range(n)]
        adjacency_w: list[list[int]] = [[] for _ in range(n)]
        for (u, v), w in sorted(edge_weights.items()):
            adjacency[u].append(v)
            adjacency[v].append(u)
            adjacency_w[u].append(w)
            adjacency_w[v].append(w)
        xadj = [0]
        adjncy: list[int] = []
        adjcwgt: list[int] = []
        for node in range(n):
            adjncy.extend(adjacency[node])
            adjcwgt.extend(adjacency_w[node])
            xadj.append(len(adjncy))

        # KaHIP exposes a global imbalance tolerance (epsilon). We keep it strict.
        imbalance = max(float(self.u_factor) / 1000.0, 1e-6)
        vwgt = [1] * n
        _edge_cut, blocks = kahip.kaffpa(
            vwgt,
            xadj,
            adjcwgt,
            adjncy,
            k_words,
            imbalance,
            0,
            self.partitioner_seed,
            self.KAHIP_MODE_ECO,
        )
        assert len(blocks) == n, "KaHIP returned unexpected partition size."
        q_to_word = {qid: int(blocks[q_to_node[qid]]) for qid in qids}
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

    def _sites_center_out(self, word_id: int) -> list[layout.LocationAddress]:
        zone_id = self.arch_spec.word_zone_map[word_id]
        n = self.sites_per_partition
        sites = [
            layout.LocationAddress(word_id, site_id, zone_id) for site_id in range(n)
        ]
        center = (n - 1) / 2.0
        return sorted(
            sites,
            key=lambda addr: (abs(addr.site_id - center), addr.site_id),
        )

    def _sites_bottom_up(self, word_id: int) -> list[layout.LocationAddress]:
        zone_id = self.arch_spec.word_zone_map[word_id]
        return [
            layout.LocationAddress(word_id, site_id, zone_id)
            for site_id in range(self.sites_per_partition)
        ]

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

    def _site_distance_matrix(self) -> list[list[int]]:
        """Site distance weighted toward index similarity and bus proximity."""
        self._validate_single_zone()
        n_sites = self.sites_per_partition
        zone = self.arch_spec.zones[0]
        adjacency: list[set[int]] = [set() for _ in range(n_sites)]
        for bus in zone.site_buses:
            for src, dst in zip(bus.src, bus.dst, strict=True):
                if 0 <= src < n_sites and 0 <= dst < n_sites:
                    adjacency[src].add(dst)
                    adjacency[dst].add(src)

        # Fall back to linear site distance when no site-bus edges exist.
        if all(len(neighbors) == 0 for neighbors in adjacency):
            return [
                [abs(src - dst) for dst in range(n_sites)] for src in range(n_sites)
            ]

        distance: list[list[int]] = [
            [n_sites for _ in range(n_sites)] for _ in range(n_sites)
        ]
        for src in range(n_sites):
            distance[src][src] = 0
            frontier = [src]
            while frontier:
                next_frontier: list[int] = []
                for node in frontier:
                    d = distance[src][node]
                    for nbr in adjacency[node]:
                        if d + 1 < distance[src][nbr]:
                            distance[src][nbr] = d + 1
                            next_frontier.append(nbr)
                frontier = next_frontier

        # Blend a strong site-index term with site-bus shortest-path proximity.
        # This keeps "same site id across words" as the primary objective while
        # still preferring closer points in the site-bus graph as a tie-breaker.
        for src in range(n_sites):
            for dst in range(n_sites):
                bus_distance = distance[src][dst]
                if bus_distance >= n_sites:
                    bus_distance = abs(src - dst)
                distance[src][dst] = (100 * abs(src - dst)) + bus_distance
        return distance

    def _candidate_slots(
        self, k_words: int, target_sizes: tuple[int, ...]
    ) -> list[layout.LocationAddress]:
        home_words = self.home_word_ids
        word_zone = self.arch_spec.word_zone_map
        slots: list[layout.LocationAddress] = []
        for word_idx in range(k_words):
            word_id = home_words[word_idx]
            zone_id = word_zone[word_id]
            for site_id in range(target_sizes[word_idx]):
                slots.append(layout.LocationAddress(word_id, site_id, zone_id))
        return slots

    def _global_site_min_cost_assignment(
        self,
        qubits: tuple[int, ...],
        weighted_edges: dict[tuple[int, int], int],
        q_to_node: dict[int, int],
        slots: list[layout.LocationAddress],
    ) -> dict[int, layout.LocationAddress]:
        if len(qubits) != len(slots):
            raise RuntimeError("Qubit count and slot count must match for assignment.")

        site_distance = self._site_distance_matrix()
        qids = tuple(sorted(qubits))
        node_to_q = {node: qid for qid, node in q_to_node.items()}

        edge_weight_by_q: dict[int, dict[int, int]] = {qid: {} for qid in qids}
        for (u, v), weight in weighted_edges.items():
            q_u = node_to_q[u]
            q_v = node_to_q[v]
            edge_weight_by_q[q_u][q_v] = weight
            edge_weight_by_q[q_v][q_u] = weight

        weighted_degree = {qid: sum(edge_weight_by_q[qid].values()) for qid in qids}
        unweighted_degree = {qid: len(edge_weight_by_q[qid]) for qid in qids}
        qubit_order = sorted(
            qids,
            key=lambda qid: (
                weighted_degree[qid],
                unweighted_degree[qid],
                -qid,
            ),
            reverse=True,
        )

        slot_centrality = []
        for i, slot in enumerate(slots):
            centrality = sum(
                site_distance[slot.site_id][other.site_id] for other in slots
            )
            slot_centrality.append((centrality, slot.site_id, slot.word_id, i))
        slot_order = [idx for _, _, _, idx in sorted(slot_centrality)]

        q_to_slot_idx: dict[int, int] = {}
        used_slots: set[int] = set()
        first_q = qubit_order[0]
        first_slot = slot_order[0]
        q_to_slot_idx[first_q] = first_slot
        used_slots.add(first_slot)

        for qid in qubit_order[1:]:
            best_slot_idx: int | None = None
            best_key: tuple[float, int, int, int] | None = None
            for slot_idx in slot_order:
                if slot_idx in used_slots:
                    continue
                slot = slots[slot_idx]
                incremental = 0.0
                for other_qid, other_slot_idx in q_to_slot_idx.items():
                    w = edge_weight_by_q[qid].get(other_qid, 0)
                    if w == 0:
                        continue
                    other_slot = slots[other_slot_idx]
                    incremental += w * site_distance[slot.site_id][other_slot.site_id]
                key = (incremental, slot.site_id, slot.word_id, slot_idx)
                if best_key is None or key < best_key:
                    best_key = key
                    best_slot_idx = slot_idx
            assert best_slot_idx is not None
            q_to_slot_idx[qid] = best_slot_idx
            used_slots.add(best_slot_idx)

        def swap_delta(qid_a: int, qid_b: int) -> int:
            slot_a = slots[q_to_slot_idx[qid_a]]
            slot_b = slots[q_to_slot_idx[qid_b]]
            delta = 0
            for other_qid in qids:
                if other_qid == qid_a or other_qid == qid_b:
                    continue
                slot_other = slots[q_to_slot_idx[other_qid]]
                w_a = edge_weight_by_q[qid_a].get(other_qid, 0)
                w_b = edge_weight_by_q[qid_b].get(other_qid, 0)
                if w_a:
                    delta += w_a * (
                        site_distance[slot_b.site_id][slot_other.site_id]
                        - site_distance[slot_a.site_id][slot_other.site_id]
                    )
                if w_b:
                    delta += w_b * (
                        site_distance[slot_a.site_id][slot_other.site_id]
                        - site_distance[slot_b.site_id][slot_other.site_id]
                    )
            return delta

        # Deterministic hill-climbing over pairwise swaps.
        max_passes = max(1, len(qids))
        for _ in range(max_passes):
            improved = False
            for i, qid_a in enumerate(qids):
                for qid_b in qids[i + 1 :]:
                    delta = swap_delta(qid_a, qid_b)
                    if delta < 0:
                        q_to_slot_idx[qid_a], q_to_slot_idx[qid_b] = (
                            q_to_slot_idx[qid_b],
                            q_to_slot_idx[qid_a],
                        )
                        improved = True
            if not improved:
                break

        return {qid: slots[q_to_slot_idx[qid]] for qid in qids}

    def _compute_layout_from_cz_layers(
        self,
        qubits: tuple[int, ...],
        cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        pinned: dict[int, layout.LocationAddress] | None = None,
    ) -> tuple[layout.LocationAddress, ...]:
        if pinned is None:
            pinned = {}
        pinned_addresses = set(pinned.values())
        unpinned_qubits = tuple(q for q in qubits if q not in pinned)

        k_words = self._word_count(len(qubits))
        total_capacity = k_words * self.sites_per_partition

        # Compute how many candidate slots remain after reserving pinned addresses.
        # This check must happen before _target_block_sizes, which raises RuntimeError
        # when qubit_count exceeds arch capacity; we convert that to a user-facing
        # ValueError with the "no legal positions remain" message.
        unpinned_slot_count = max(0, total_capacity - len(pinned_addresses))
        if len(unpinned_qubits) > unpinned_slot_count:
            raise ValueError(
                f"layout heuristic cannot place {len(unpinned_qubits)} un-pinned qubits: "
                f"arch provides {total_capacity} total sites, "
                f"{len(pinned_addresses)} are pinned, leaving {unpinned_slot_count} available; "
                "no legal positions remain"
            )

        # _target_block_sizes expects qubit_count <= k_words * sites_per_partition.
        # Use total_capacity as the qubit_count when qubits exceed it (this can
        # only occur when all overflow qubits are covered by pinned addresses,
        # since the check above would have raised otherwise).
        target_qubit_count = min(len(qubits), total_capacity)
        target_sizes = self._target_block_sizes(target_qubit_count, k_words)
        q_to_node, weighted_edges = self._build_weighted_graph(qubits, cz_layers)
        all_slots = self._candidate_slots(k_words, target_sizes)

        unpinned_slots = [s for s in all_slots if s not in pinned_addresses]

        if unpinned_qubits:
            # Filter weighted_edges to only include edges between unpinned qubits.
            unpinned_set = set(unpinned_qubits)
            node_to_q = {node: qid for qid, node in q_to_node.items()}
            unpinned_edges: dict[tuple[int, int], int] = {
                (u, v): w
                for (u, v), w in weighted_edges.items()
                if node_to_q[u] in unpinned_set and node_to_q[v] in unpinned_set
            }
            # Build a q_to_node restricted to unpinned qubits so the assignment
            # sees a consistent mapping.
            unpinned_q_to_node = {q: q_to_node[q] for q in unpinned_qubits}
            q_to_location = self._global_site_min_cost_assignment(
                qubits=unpinned_qubits,
                weighted_edges=unpinned_edges,
                q_to_node=unpinned_q_to_node,
                slots=unpinned_slots[: len(unpinned_qubits)],
            )
        else:
            q_to_location = {}

        result = q_to_location | pinned
        return tuple(result[q] for q in qubits)

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, layout.LocationAddress] | None = None,
    ) -> tuple[layout.LocationAddress, ...]:
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
        cz_layers = _to_cz_layers(stages)
        return self._compute_layout_from_cz_layers(qubits, cz_layers, pinned)
