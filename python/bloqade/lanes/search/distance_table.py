"""Precomputed hop-count and time-weighted distance tables.

This is a direct Python mirror of Rust's ``DistanceTable`` in
``crates/bloqade-lanes-search/src/heuristic.rs``.

Build once via :meth:`DistanceTable.build`, then optionally attach
time distances via :meth:`DistanceTable.with_time_distances`.  Both
steps use the same reversed-graph BFS / Dijkstra pattern as the Rust
implementation so that Python and Rust produce identical values given
the same architecture and targets.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

from bloqade.lanes.layout import LocationAddress

if TYPE_CHECKING:
    from bloqade.lanes.search.tree import ConfigurationTree


@dataclass
class DistanceTable:
    """Precomputed minimum lane-hop and time distances to target locations.

    Built once via BFS on the reversed lane graph (hop distances) and
    optionally Dijkstra on the reversed weighted graph (time distances).

    Convention: unreachable / unknown → ``float('inf')``.
    """

    # encoded_target → { encoded_src → min hops }
    _distance_to: dict[int, dict[int, int]] = field(
        default_factory=dict, init=False, repr=False
    )
    # encoded_target → { encoded_src → min time (µs) }; None until with_time_distances
    _time_distance_to: dict[int, dict[int, float]] | None = field(
        default=None, init=False, repr=False
    )
    # Fastest positive lane duration; None until with_time_distances
    _fastest_lane_us: float | None = field(default=None, init=False, repr=False)

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        tree: ConfigurationTree,
        targets: Iterable[LocationAddress],
    ) -> "DistanceTable":
        """BFS on the reversed lane graph — mirror of ``DistanceTable::new`` in Rust.

        For each unique target, initialise dist[target]=0 and expand
        predecessors (locations that have a lane *into* the target) level by
        level.  This gives minimum hop counts from every reachable location to
        each target.
        """
        table = cls()

        # Deduplicate targets.
        unique_targets: list[LocationAddress] = list({t: None for t in targets})

        # Build reverse adjacency: dst_enc → [src_enc, …]
        # Mirrors Rust's reverse_adj construction which iterates all bus groups.
        reverse_adj: dict[int, list[int]] = {}
        for lanes in tree._lanes_by_triplet.values():
            for lane in lanes:
                src, dst = tree.arch_spec.get_endpoints(lane)
                src_enc = _encode(src)
                dst_enc = _encode(dst)
                reverse_adj.setdefault(dst_enc, []).append(src_enc)

        # BFS from each target on reversed edges.
        for target in unique_targets:
            target_enc = _encode(target)
            dist: dict[int, int] = {target_enc: 0}
            queue: deque[int] = deque([target_enc])
            while queue:
                current = queue.popleft()
                current_dist = dist[current]
                for pred in reverse_adj.get(current, []):
                    if pred not in dist:
                        dist[pred] = current_dist + 1
                        queue.append(pred)
            table._distance_to[target_enc] = dist

        return table

    def with_time_distances(self, tree: ConfigurationTree) -> "DistanceTable":
        """Dijkstra on the reversed weighted graph; sets fastest_lane_us.

        Mirror of ``DistanceTable::with_time_distances`` in Rust.

        Skips gracefully if no lanes have path-timing data (no paths in the
        arch spec), leaving ``_fastest_lane_us`` as ``None`` so that
        :meth:`distance` falls back to hop count.

        Mirrors Rust ``LaneIndex::fastest_lane_duration_us`` which only
        considers lanes with explicit ``arch_spec.paths`` entries (not
        geometric fallbacks).  When ``arch_spec.paths`` is empty the arch
        has no timing data and we skip time distances entirely.
        """
        # Use explicit arch-spec path lookup (same as Rust's arch_spec.paths
        # map) so that arches without transport-path data return early.
        # `arch_spec.paths` is empty when the JSON has no "paths" key.
        arch_paths = tree.arch_spec.paths  # MappingProxyType[LaneAddress, ...]

        if not arch_paths:
            # No path data in arch spec — fall back to hop-count only.
            # Matches Rust: `fastest_lane_duration_us() == None => return self`.
            return self

        # Compute fastest lane duration across lanes that have explicit paths.
        # Mirrors Rust's LaneIndex which populates lane_durations only from
        # arch_spec.paths, not from the geometric straight-line fallback.
        fastest = float("inf")
        for lanes in tree._lanes_by_triplet.values():
            for lane in lanes:
                if lane not in arch_paths:
                    continue
                dur = tree.path_finder.metrics.get_lane_duration_us(lane)
                if dur > 0.0:
                    fastest = min(fastest, dur)

        if fastest == float("inf"):
            # No usable path durations found — leave time distances unset.
            return self

        self._fastest_lane_us = fastest

        # Build reverse weighted adjacency: dst_enc → [(src_enc, duration)]
        # Only include lanes that have explicit path entries (mirrors Rust).
        reverse_adj: dict[int, list[tuple[int, float]]] = {}
        for lanes in tree._lanes_by_triplet.values():
            for lane in lanes:
                if lane not in arch_paths:
                    continue
                dur = tree.path_finder.metrics.get_lane_duration_us(lane)
                if dur <= 0.0:
                    # Skip lanes with no timing data (mirrors Rust's `let Some(dur) =` guard).
                    continue
                src, dst = tree.arch_spec.get_endpoints(lane)
                src_enc = _encode(src)
                dst_enc = _encode(dst)
                reverse_adj.setdefault(dst_enc, []).append((src_enc, dur))

        # Dijkstra from each target on reversed weighted edges.
        time_dist_to: dict[int, dict[int, float]] = {}
        for target_enc in self._distance_to:
            dist: dict[int, float] = {target_enc: 0.0}
            # Min-heap entries: (cost, node_enc)
            heap: list[tuple[float, int]] = [(0.0, target_enc)]
            while heap:
                cost, node = heapq.heappop(heap)
                if cost > dist.get(node, float("inf")):
                    continue
                for pred, dur in reverse_adj.get(node, []):
                    new_cost = cost + dur
                    if new_cost < dist.get(pred, float("inf")):
                        dist[pred] = new_cost
                        heapq.heappush(heap, (new_cost, pred))
            time_dist_to[target_enc] = dist

        self._time_distance_to = time_dist_to
        return self

    # ── queries ──────────────────────────────────────────────────────

    def hop_distance(self, src: LocationAddress, tgt: LocationAddress) -> float:
        """Minimum lane-hop count from *src* to *tgt*.

        Returns ``float('inf')`` if *tgt* was not registered as a target, or
        if *src* cannot reach *tgt* in the lane graph.
        """
        tgt_enc = _encode(tgt)
        per_target = self._distance_to.get(tgt_enc)
        if per_target is None:
            return float("inf")
        hops = per_target.get(_encode(src))
        if hops is None:
            return float("inf")
        return float(hops)

    def time_distance(self, src: LocationAddress, tgt: LocationAddress) -> float:
        """Minimum move time (µs) from *src* to *tgt*.

        Returns ``float('inf')`` if time distances were not computed or if
        *tgt* / *src* are unreachable.
        """
        if self._time_distance_to is None:
            return float("inf")
        tgt_enc = _encode(tgt)
        per_target = self._time_distance_to.get(tgt_enc)
        if per_target is None:
            return float("inf")
        td = per_target.get(_encode(src))
        if td is None:
            return float("inf")
        return td

    def fastest_lane_us(self) -> float | None:
        """Fastest lane duration seen during :meth:`with_time_distances`.

        Returns ``None`` if :meth:`with_time_distances` has not been called or
        found no usable timing data.
        """
        return self._fastest_lane_us

    def distance(self, src: LocationAddress, tgt: LocationAddress, w_t: float) -> float:
        """Blended distance: ``(1-w_t)*hop + w_t*(time/fastest)``.

        Exactly mirrors the Rust ``blended_distance`` formula:
        ``(1.0 - w_t) * hop_dist + w_t * normalized_time_d``
        where ``normalized_time_d = time_d / fastest``.

        Falls back to hop distance when ``w_t <= 0`` or time data is missing.
        Returns ``float('inf')`` if hop distance is infinite.
        """
        hop = self.hop_distance(src, tgt)
        if hop == float("inf"):
            return float("inf")

        if w_t <= 0.0 or self._fastest_lane_us is None:
            return hop

        time_us = self.time_distance(src, tgt)
        if time_us == float("inf"):
            # Time path unavailable — degrade to hop only.
            return hop

        normalized_time = time_us / self._fastest_lane_us
        return (1.0 - w_t) * hop + w_t * normalized_time


# ── helpers ──────────────────────────────────────────────────────────


def _encode(loc: LocationAddress) -> int:
    """Encode a LocationAddress to a stable integer key.

    Uses the Rust-side hash which is stable for the lifetime of the process
    and consistent with the encoded integers used elsewhere in this codebase.
    We use Python's built-in hash (delegated to Rust __hash__) as the key;
    this is consistent because we always look up within a single process run.
    """
    return hash(loc)
