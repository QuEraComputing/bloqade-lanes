"""Generic full architecture: 3-zone neutral atom processor.

Layout: storage_top → entangling → storage_bottom.
Each zone has 8 rows x 4 columns = 32 words, with 8 sites per word.
Total: 96 words, 768 sites, 16 entangling pairs.

All zones have full site (hypercube) and word (diagonal) connectivity.
"""

from __future__ import annotations

from bloqade.lanes.arch.builder import ArchResult, build_arch
from bloqade.lanes.arch.topology import (
    DiagonalWordTopology,
    HypercubeSiteTopology,
    MatchingTopology,
)
from bloqade.lanes.arch.zone import ArchBlueprint, DeviceLayout, ZoneSpec

_SITES_PER_WORD = 8
_NUM_ROWS = 8
_NUM_COLS = 4
_SITE_SPACING = 10.0


def get_arch() -> ArchResult:
    """Build the generic full architecture.

    Returns:
        ArchResult with the validated ArchSpec and zone metadata.
    """
    full_connectivity_zone = ZoneSpec(
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        entangling=False,
        measurement=False,
        site_topology=HypercubeSiteTopology(),
        word_topology=DiagonalWordTopology(),
    )

    entangling_zone = ZoneSpec(
        num_rows=_NUM_ROWS,
        num_cols=_NUM_COLS,
        entangling=True,
        measurement=True,
        site_topology=HypercubeSiteTopology(),
        word_topology=DiagonalWordTopology(),
    )

    blueprint = ArchBlueprint(
        zones={
            "storage_top": full_connectivity_zone,
            "entangling": entangling_zone,
            "storage_bottom": full_connectivity_zone,
        },
        layout=DeviceLayout(
            sites_per_word=_SITES_PER_WORD,
            site_spacing=_SITE_SPACING,
            pair_spacing=_SITE_SPACING,
            row_spacing=20.0,
            zone_gap=20.0,
        ),
        feed_forward=True,
        atom_reloading=True,
        blockade_radius=_SITE_SPACING,
    )

    return build_arch(
        blueprint,
        connections={
            ("storage_top", "entangling"): MatchingTopology(),
            ("entangling", "storage_bottom"): MatchingTopology(),
        },
    )
