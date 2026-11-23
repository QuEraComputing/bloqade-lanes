"""Synthetic-architecture fixtures for the heuristics test suite.

Several target-generator tests need architectures with a specific
topological property (two CZ pairs with distinct move costs, an
asymmetric blocker, a shared lane traversed in opposite directions, a
fully-blocked pair, ...). The real Gemini arch does not reliably provide
these, so tests load tiny, deterministic ``ArchSpec`` snapshots with known
topology from ``tests/fixtures/arch``:

* ``gate_arch`` — a one-row entangling zone (hypercube word bus +
  all-to-all site bus). Home words are the even-indexed words; each
  word's CZ partner is its odd-indexed neighbour. Good for tests that
  just need a handful of feasible, non-partnered CZ pairs.

* ``congestion_hub_detour`` and ``congestion_bridge`` — single-zone,
  one-site-per-word layouts with explicit word positions, word buses and
  entangling pairs, loaded with ``load_arch``. Their move-cost landscapes
  are described in the tests that use them.
"""

from __future__ import annotations

import pytest
from tests._arch_fixtures import gate_zone_arch

from bloqade.lanes.arch.spec import ArchSpec


def _make_gate_arch(num_cols: int = 4, sites_per_word: int = 1) -> ArchSpec:
    """A single one-row entangling zone.

    ``num_cols`` must be an even power of two (hypercube word topology).
    Words ``0, 2, 4, ...`` are home; word ``2k`` is CZ-partnered with
    word ``2k + 1``.
    """
    return gate_zone_arch(
        num_rows=1,
        num_cols=num_cols,
        sites_per_word=sites_per_word,
        word_topology="hypercube",
        site_topology="all_to_all",
    )


@pytest.fixture
def gate_arch():
    """Factory fixture returning :func:`_make_gate_arch`."""
    return _make_gate_arch
