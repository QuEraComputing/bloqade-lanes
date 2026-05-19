"""Shared dispatch tables for Rust solver options.

Both ``PhysicalPlacementStrategy`` (fixed-target) and the no-return strategy
family use the same string-to-enum mappings for the ``strategy`` and
``deadlock_policy`` knobs on ``_native.SolveOptions``. Defining them here
once avoids 4-way duplication across the placement modules.
"""

from __future__ import annotations

from bloqade.lanes.bytecode import _native

_STRATEGY_MAP: dict[str, _native.SearchStrategy] = {
    "astar": _native.SearchStrategy.ASTAR,
    "dfs": _native.SearchStrategy.DFS,
    "bfs": _native.SearchStrategy.BFS,
    "greedy": _native.SearchStrategy.GREEDY,
    "ids": _native.SearchStrategy.IDS,
    "cascade": _native.SearchStrategy.CASCADE_IDS,
    "cascade-ids": _native.SearchStrategy.CASCADE_IDS,
    "cascade-dfs": _native.SearchStrategy.CASCADE_DFS,
    "cascade-entropy": _native.SearchStrategy.CASCADE_ENTROPY,
    "entropy": _native.SearchStrategy.ENTROPY,
}

_DEADLOCK_MAP: dict[str, _native.DeadlockPolicy] = {
    "skip": _native.DeadlockPolicy.SKIP,
    "move_blockers": _native.DeadlockPolicy.MOVE_BLOCKERS,
    "all_moves": _native.DeadlockPolicy.ALL_MOVES,
}
