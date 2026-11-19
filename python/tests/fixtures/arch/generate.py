# pyright: reportMissingImports=false
"""Regenerate the architecture snapshots in this directory.

Lanes tests load these snapshots instead of building architectures, because the
architecture builder lives in ``bloqade.gemini.arch`` (the ``bloqade``
distribution), which lanes cannot depend on. This script is the record of how
each snapshot was built. It is not run by the test suite.

Run it from the repository root with Bloqade 0.38.0 installed::

    python python/tests/fixtures/arch/generate.py           # write every snapshot
    python python/tests/fixtures/arch/generate.py --check   # fail if any differs

To add a snapshot, add a parameter set to ``GATE_ZONE_FIXTURES`` in
``python/tests/_arch_fixtures.py``, or a recipe to ``CUSTOM_FIXTURES`` below.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from bloqade.gemini.arch import (
    AllToAllSiteTopology,
    ArchBuilder,
    SiteTopology,
    WordTopology,
    DiagonalWordTopology,
    HypercubeSiteTopology,
    HypercubeWordTopology,
    create_zone_words,
)

from bloqade.lanes.arch.spec import ArchSpec

FIXTURE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE_DIR.parents[1]))

from _arch_fixtures import (
    GATE_ZONE_FIXTURES,
    SiteTopologyName,
    WordTopologyName,
    gate_zone_name,
)

_WORD_TOPOLOGIES = {
    "hypercube": HypercubeWordTopology,
    "diagonal": DiagonalWordTopology,
}
_SITE_TOPOLOGIES = {
    "hypercube": HypercubeSiteTopology,
    "all_to_all": AllToAllSiteTopology,
}


def _fixture_arch(
    num_rows: int,
    num_cols: int,
    sites_per_word: int,
    zones: Sequence[tuple[str, bool, WordTopology | None, SiteTopology | None]],
    connections: Sequence[tuple[str, str]] = (),
) -> ArchSpec:
    """Build interleaved words on a 10 µm grid with zones stacked 20 µm apart."""
    grid = create_zone_words(
        num_rows=num_rows, num_cols=num_cols, sites_per_word=sites_per_word
    )
    columns = [10.0 * i for i in range(num_cols * sites_per_word)]
    builder = ArchBuilder(
        grid_shape=(num_rows, len(columns)), word_shape=(1, sites_per_word)
    )
    for word in grid.words:
        builder.add_word(
            rows=word.site_indices[0][1], columns=[x for x, _ in word.site_indices]
        )
    for i, (name, entangling, word_topology, site_topology) in enumerate(zones):
        builder.add_zone(
            name,
            rows=[20.0 * (i * num_rows + row) for row in range(num_rows)],
            columns=columns,
            x_clearance=3.0,
            y_clearance=3.0,
        )
        if site_topology is not None:
            for bus in site_topology.generate_site_buses(sites_per_word):
                builder.add_site_bus(name, bus.src, bus.dst)
        if word_topology is not None:
            for bus in word_topology.generate_word_buses(grid):
                builder.add_word_bus(name, bus.src, bus.dst)
        if entangling:
            left, right = zip(*grid.cz_pairs())
            builder.add_entangling_pairs(name, left, right)
    for source, destination in connections:
        builder.connect((source, grid.all_word_ids), (destination, grid.all_word_ids))
    names = [zone[0] for zone in zones]
    builder.add_mode("all", names)
    for name in names:
        builder.add_mode(name, [name])
    return builder.build()


def _gate_zone(
    num_rows: int,
    num_cols: int,
    sites_per_word: int,
    word_topology: WordTopologyName,
    site_topology: SiteTopologyName,
) -> ArchSpec:
    return _fixture_arch(
        num_rows,
        num_cols,
        sites_per_word,
        [
            (
                "gate",
                True,
                _WORD_TOPOLOGIES[word_topology](),
                _SITE_TOPOLOGIES[site_topology](),
            )
        ],
    )


def _custom_one_site_words(
    positions: Sequence[tuple[float, float]],
    word_buses: Iterable[tuple[int, int]],
    entangling: Iterable[tuple[int, int]] = (),
    *,
    x_clearance: float = 3.0,
    y_clearance: float = 3.0,
    name: str = "synthetic",
) -> ArchSpec:
    """A single-zone, one-site-per-word architecture with explicit topology.

    Args:
        positions: ``(x_um, y_um)`` of each word, indexed by ``word_id``.
        word_buses: ``(src_word, dst_word)`` single-element word buses. Each
            becomes one bidirectional edge whose cost tracks the distance
            between the two words.
        entangling: ``(word_a, word_b)`` CZ blockade pairs.
        x_clearance / y_clearance: AOD path clearance (µm).
        name: zone name.

    Each bus is laid out along a clear axis-aligned corridor (no third word
    strictly between its endpoints), so its cost stays proportional to the
    straight-line distance.
    """
    xs = sorted({x for x, _ in positions})
    ys = sorted({y for _, y in positions})
    x_index = {v: i for i, v in enumerate(xs)}
    y_index = {v: i for i, v in enumerate(ys)}

    builder = ArchBuilder(grid_shape=(len(ys), len(xs)), word_shape=(1, 1))
    for x, y in positions:
        builder.add_word(rows=y_index[y], columns=x_index[x])
    builder.add_zone(
        name, rows=ys, columns=xs, x_clearance=x_clearance, y_clearance=y_clearance
    )
    for src, dst in word_buses:
        builder.add_word_bus(name, [src], [dst])
    for a, b in entangling:
        builder.add_entangling_pairs(name, [a], [b])
    builder.add_mode("all", [name])
    return builder.build()


def _congestion_hub_detour() -> ArchSpec:
    """``test_target_direction_chosen_when_target_path_cheaper``.

    The control move (``w0 -> w3``) has a cheap two-hop route through hub
    ``w4`` and an expensive detour ``w0-w5-w6-w3``. The target move
    (``w2 -> w1``) is a fixed two-hop route through ``w7``::

        w0 --w4-- w3        (control short route via hub w4)
        |          |
        w5 ------- w6       (control detour)
        CZ pairs (w0,w1) and (w2,w3); target route w2 - w7 - w1
    """
    positions = [
        (0.0, 0.0),  # 0: w0  control start (X)
        (30.0, 200.0),  # 1: w1  partner(X)
        (0.0, 200.0),  # 2: w2  target start (Y)
        (20.0, 0.0),  # 3: w3  partner(Y)
        (10.0, 0.0),  # 4: w4  hub (blocker site)
        (0.0, 100.0),  # 5: w5  detour waypoint
        (20.0, 100.0),  # 6: w6  detour waypoint
        (15.0, 200.0),  # 7: w7  target-route midpoint
    ]
    buses = [(0, 4), (4, 3), (0, 5), (5, 6), (6, 3), (2, 7), (7, 1)]
    entangling = [(0, 1), (2, 3)]
    return _custom_one_site_words(positions, buses, entangling)


def _congestion_bridge() -> ArchSpec:
    """``test_multi_pair_avoids_opposite_direction_reuse``.

    Lane ``L = (w1, w2)`` is the only connection between the left cluster
    (``w0``, ``w5``, ...) and the right cluster (``w3``, ``w4``, ...).
    """
    positions = [
        (0.0, 70.0),  # 0  ctrl A (left)
        (0.0, 0.0),  # 1  L0  (bridge left end)
        (60.0, 0.0),  # 2  R0  (bridge right end); L = (w1, w2)
        (60.0, 70.0),  # 3  partner(tgtA); A-ctrl destination (right)
        (60.0, -60.0),  # 4  ctrl B (right)
        (0.0, -60.0),  # 5  partner(tgtB); B-ctrl destination (left)
        (0.0, 140.0),  # 6  tgt A (partner of w3)
        (60.0, 140.0),  # 7  partner(ctrlA); A-tgt dest (isolated)
        (0.0, -140.0),  # 8  tgt B (partner of w5)
        (195.0, -140.0),  # 9  partner(ctrlB); B-tgt destination
        (65.0, -140.0),  # 10 B-tgt route waypoint
        (130.0, -140.0),  # 11 B-tgt route waypoint
    ]
    buses = [(0, 1), (1, 2), (2, 3), (2, 4), (1, 5), (8, 10), (10, 11), (11, 9)]
    entangling = [(0, 7), (3, 6), (5, 8), (4, 9)]
    return _custom_one_site_words(positions, buses, entangling)


def _proc_mem_two_zone() -> ArchSpec:
    """An entangling ``proc`` zone matched to a bus-less ``mem`` zone."""
    return _fixture_arch(
        2,
        2,
        4,
        [
            ("proc", True, HypercubeWordTopology(), HypercubeSiteTopology()),
            ("mem", False, None, None),
        ],
        [("proc", "mem")],
    )


def _site_bus_scoping_two_zone() -> ArchSpec:
    """Zone ``a`` has site buses; zone ``b`` has none."""
    return _fixture_arch(
        1,
        2,
        4,
        [
            ("a", True, HypercubeWordTopology(), AllToAllSiteTopology()),
            ("b", True, HypercubeWordTopology(), None),
        ],
        [("a", "b")],
    )


def _two_entangling_zones() -> ArchSpec:
    """Two entangling zones, for heuristics that require exactly one."""
    return _fixture_arch(
        2,
        2,
        16,
        [
            ("zone_a", True, DiagonalWordTopology(), HypercubeSiteTopology()),
            ("zone_b", True, DiagonalWordTopology(), HypercubeSiteTopology()),
        ],
    )


CUSTOM_FIXTURES: dict[str, Callable[[], ArchSpec]] = {
    "congestion_hub_detour": _congestion_hub_detour,
    "congestion_bridge": _congestion_bridge,
    "proc_mem_two_zone": _proc_mem_two_zone,
    "site_bus_scoping_two_zone": _site_bus_scoping_two_zone,
    "two_entangling_zones": _two_entangling_zones,
}


def recipes() -> dict[str, Callable[[], ArchSpec]]:
    """Every snapshot name with the function that builds it."""
    gate_zones: dict[str, Callable[[], ArchSpec]] = {
        gate_zone_name(*parameters): (lambda p=parameters: _gate_zone(*p))
        for parameters in GATE_ZONE_FIXTURES
    }
    return {**gate_zones, **CUSTOM_FIXTURES}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the architecture snapshots for lanes tests."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare instead of writing; exit non-zero if any snapshot differs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIXTURE_DIR,
        help="directory to write snapshots to (default: this directory)",
    )
    args = parser.parse_args()

    stale: list[str] = []
    for name, build in recipes().items():
        text = build().to_json() + "\n"
        path = args.output / f"{name}.json"
        if args.check:
            current = path.read_text(encoding="utf-8") if path.is_file() else None
            if current != text:
                stale.append(path.name)
        else:
            path.write_text(text, encoding="utf-8")

    known = {f"{name}.json" for name in recipes()}
    unknown = sorted(p.name for p in args.output.glob("*.json") if p.name not in known)
    if stale or unknown:
        for name in stale:
            print(f"out of date: {name}")
        for name in unknown:
            print(f"no recipe: {name}")
        return 1 if args.check else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
