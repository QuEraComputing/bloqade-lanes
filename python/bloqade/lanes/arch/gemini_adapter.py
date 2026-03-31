"""Adapter to express the current Gemini architecture via the zone-based API.

Maps the old Gemini parameters (hypercube_dims, word_size_y) to an
ArchBlueprint that, when built with build_arch(), produces a functionally
equivalent architecture.

Mapping:
    Old: 2^dims words, each 2 columns × word_size_y sites
    New: 1 row × 2^(dims+1) columns, each word_size_y sites (interleaved pairs)

    Old site buses (left↔right column within a word) become the CZ pair
    dimension (dim 0) of the HypercubeWordTopology.

    Old hypercube word buses (dims dimensions) become dims 1..dims of the
    HypercubeWordTopology on the column axis.

    Total: (dims+1)-dimensional column hypercube on 2^(dims+1) words.
"""

from __future__ import annotations

from .builder import ArchResult, build_arch
from .topology import AllToAllSiteTopology, HypercubeWordTopology
from .zone import ArchBlueprint, DeviceLayout, ZoneSpec


def gemini_blueprint(
    hypercube_dims: int = 4,
    word_size_y: int = 5,
    site_spacing: float = 10.0,
    pair_spacing: float = 10.0,
) -> ArchBlueprint:
    """Create an ArchBlueprint equivalent to the current Gemini architecture.

    Args:
        hypercube_dims: Number of hypercube dimensions (default 4 → 16 old words).
        word_size_y: Number of sites per word (default 5).
        site_spacing: Distance between adjacent atoms (micrometers).
        pair_spacing: Gap between CZ pairs (micrometers).

    Returns:
        An ArchBlueprint that, when built with build_arch(), produces a
        functionally equivalent architecture to generate_arch_hypercube().
    """
    num_cols = 2 * 2**hypercube_dims

    return ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=1,
                num_cols=num_cols,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=AllToAllSiteTopology(),
            ),
        },
        layout=DeviceLayout(
            sites_per_word=word_size_y,
            site_spacing=site_spacing,
            pair_spacing=pair_spacing,
        ),
    )


def build_gemini_arch(
    hypercube_dims: int = 4,
    word_size_y: int = 5,
    site_spacing: float = 10.0,
    pair_spacing: float = 10.0,
) -> ArchResult:
    """Build a Gemini-equivalent architecture using the zone-based API.

    Convenience function combining gemini_blueprint() + build_arch().
    """
    bp = gemini_blueprint(hypercube_dims, word_size_y, site_spacing, pair_spacing)
    return build_arch(bp)
