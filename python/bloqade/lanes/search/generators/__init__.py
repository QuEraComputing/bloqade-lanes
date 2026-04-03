"""Move generators for the configuration search tree.

A MoveGenerator produces candidate move sets from a configuration node.
Different implementations enable different search strategies — exhaustive
enumeration, goal-directed search, greedy grid growing, etc.

All generators yield candidate frozenset[LaneAddress]. Validation
(lane validity, collision checks, transposition table lookups) is
performed by ConfigurationTree.apply_move_set and higher-level helpers
such as ConfigurationTree.expand_node, so generators are free to
over-generate — invalid candidates are filtered out when moves are
applied to the tree.
"""

from bloqade.lanes.search.generators.aod_grouping import BusContext
from bloqade.lanes.search.generators.base import EntropyNode, MoveGenerator
from bloqade.lanes.search.generators.exhaustive import ExhaustiveMoveGenerator
from bloqade.lanes.search.generators.greedy import GreedyMoveGenerator
from bloqade.lanes.search.generators.heuristic import HeuristicMoveGenerator

__all__ = [
    "BusContext",
    "EntropyNode",
    "ExhaustiveMoveGenerator",
    "GreedyMoveGenerator",
    "HeuristicMoveGenerator",
    "MoveGenerator",
]
