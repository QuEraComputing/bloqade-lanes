"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 1) for the rationale. Removed in the final cleanup stage once all
in-flight branches have rebased onto ``bloqade.lanes.bytecode.encoding``.
"""

from bloqade.lanes.bytecode.encoding import (
    Direction as Direction,
    LaneAddress as LaneAddress,
    LocationAddress as LocationAddress,
    MoveType as MoveType,
    SiteLaneAddress as SiteLaneAddress,
    WordLaneAddress as WordLaneAddress,
    ZoneAddress as ZoneAddress,
)
