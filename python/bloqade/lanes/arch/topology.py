"""Backward-compatible re-exports of the topology strategy classes.

The canonical import path is ``bloqade.lanes.arch.build.topology``.
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
