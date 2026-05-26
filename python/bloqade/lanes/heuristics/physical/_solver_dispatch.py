"""Shared dispatch tables for Rust solver options.

Currently holds only ``_STRATEGY_MAP``, used by
:class:`PhysicalPlacementStrategy` to coerce the legacy string-based
``RustPlacementTraversal.strategy`` field into a :class:`SearchStrategy`
enum.

The no-return placement family (``NoReturnPlacementStrategy``,
``NoHomePlacementStrategy``, ``RecedingHorizonNoReturnPlacementStrategy``)
takes :class:`SearchStrategy` / :class:`DeadlockPolicy` enum values
directly and bypasses any string lookup.
"""

from __future__ import annotations

from bloqade.lanes.bytecode import _native

_STRATEGY_MAP: dict[str, _native.SearchStrategy] = {
    "astar": _native.SearchStrategy.ASTAR,
    "dfs": _native.SearchStrategy.DFS,
    "bfs": _native.SearchStrategy.BFS,
    "greedy": _native.SearchStrategy.GREEDY,
    "ids": _native.SearchStrategy.IDS,
    # "cascade" is an alias for "cascade-ids" (kept for backward compat).
    "cascade": _native.SearchStrategy.CASCADE_IDS,
    "cascade-ids": _native.SearchStrategy.CASCADE_IDS,
    "cascade-dfs": _native.SearchStrategy.CASCADE_DFS,
    "cascade-entropy": _native.SearchStrategy.CASCADE_ENTROPY,
    "entropy": _native.SearchStrategy.ENTROPY,
}
