"""Synthetic-architecture fixtures for the heuristics test suite.

Several target-generator tests need architectures with a specific
topological property (two CZ pairs with distinct move costs, an
asymmetric blocker, a shared lane traversed in opposite directions, a
fully-blocked pair, ...). The real Gemini arch does not reliably provide
these, so this module exposes two factory fixtures that build tiny,
deterministic ``ArchSpec`` objects with known topology:

* ``gate_arch`` — a one-row entangling zone (hypercube word bus +
  all-to-all site bus). Home words are the even-indexed words; each
  word's CZ partner is its odd-indexed neighbour. Good for tests that
  just need a handful of feasible, non-partnered CZ pairs.

* ``arch_builder`` — a low-level single-zone builder driven by explicit
  word positions, word buses, and entangling pairs (one site per word).
  Costs follow physical distance, so callers control the exact move-cost
  landscape needed to exercise the congestion heuristics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pytest

from bloqade.lanes.arch import (
    AllToAllSiteTopology,
    ArchBlueprint,
    DeviceLayout,
    HypercubeWordTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.arch.build.imperative import ArchBuilder, ZoneBuilder
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import Grid as _RustGrid


def _make_gate_arch(num_cols: int = 4, sites_per_word: int = 1) -> ArchSpec:
    """Build a single one-row entangling zone.

    ``num_cols`` must be an even power of two (hypercube word topology).
    Words ``0, 2, 4, ...`` are home; word ``2k`` is CZ-partnered with
    word ``2k + 1``.
    """
    blueprint = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=1,
                num_cols=num_cols,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=AllToAllSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=sites_per_word),
    )
    return build_arch(blueprint).arch


def _build_custom_arch(
    positions: Sequence[tuple[float, float]],
    word_buses: Iterable[tuple[int, int]],
    entangling: Iterable[tuple[int, int]] = (),
    *,
    x_clearance: float = 3.0,
    y_clearance: float = 3.0,
    name: str = "synthetic",
) -> ArchSpec:
    """Build a single-zone, one-site-per-word arch with explicit topology.

    Args:
        positions: ``(x_um, y_um)`` for each word, indexed by ``word_id``
            (so ``positions[0]`` is word 0). Each word holds a single site.
        word_buses: ``(src_word, dst_word)`` single-element word buses.
            Each becomes one bidirectional edge in the movement graph;
            its cost tracks the physical distance between the two words.
        entangling: ``(word_a, word_b)`` CZ blockade pairs. ``word_a``'s
            CZ partner becomes ``word_b`` and vice versa.
        x_clearance / y_clearance: AOD path clearance (µm).
        name: zone name.

    Lay each bus out along a clear axis-aligned corridor (no third word
    strictly between its endpoints) so its cost stays proportional to the
    straight-line distance.
    """
    xs = sorted({x for x, _ in positions})
    ys = sorted({y for _, y in positions})
    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}

    grid = _RustGrid.from_positions([float(x) for x in xs], [float(y) for y in ys])
    zone = ZoneBuilder(
        name, grid, (1, 1), x_clearance=x_clearance, y_clearance=y_clearance
    )
    for x, y in positions:
        zone.add_word([x_index[x]], [y_index[y]])
    for src, dst in word_buses:
        zone.add_word_bus([src], [dst])
    pairs = list(entangling)
    if pairs:
        zone.add_entangling_pairs([a for a, _ in pairs], [b for _, b in pairs])

    builder = ArchBuilder()
    builder.add_zone(zone)
    builder.add_mode("all", [name])
    return builder.build()


@pytest.fixture
def gate_arch():
    """Factory fixture returning :func:`_make_gate_arch`."""
    return _make_gate_arch


@pytest.fixture
def arch_builder():
    """Factory fixture returning :func:`_build_custom_arch`."""
    return _build_custom_arch
