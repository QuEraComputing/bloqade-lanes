"""TRANSITIONAL SHIM — see ``.superpowers/plans/2026-04-27-archspec-package-merge.md``
(Stage 8) for the rationale. Removed in the final cleanup stage once all
in-flight branches have rebased onto ``bloqade.lanes.arch.build.topology``.
"""

from bloqade.lanes.arch.build.topology import (
    AllToAllSiteTopology as AllToAllSiteTopology,
    DiagonalWordTopology as DiagonalWordTopology,
    HypercubeSiteTopology as HypercubeSiteTopology,
    HypercubeWordTopology as HypercubeWordTopology,
    InterZoneTopology as InterZoneTopology,
    MatchingTopology as MatchingTopology,
    SiteTopology as SiteTopology,
    TransversalSiteTopology as TransversalSiteTopology,
    WordTopology as WordTopology,
)
