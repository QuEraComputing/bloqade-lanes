#!/usr/bin/env python3
"""Generate the conveyor-capable benchmark ArchSpec from the bundled Gemini one.

Chain assembly is dead code on both shipped Gemini specs: a site bus is an
elementwise hop map ``src[i] -> dst[i]``, and every shipped bus is
*endpoint-disjoint* (``set(src) & set(dst) == set()``). ``vacating_lane()`` looks
for an outgoing lane from a blocked destination on the same bus, which on a
disjoint bus is necessarily ``None`` — so every chain path is unreachable and CI
cannot see a chain regression (issue #939, part of #887).

This script converts each hypercube site bus of the bundled physical spec into
*conveyor* form. Dimension ``d`` has stride ``s = 2 ** d`` and becomes
``i -> i + s`` for every valid ``i``. On the 8-site Gemini word:

    dim 0 (s=1)  0->1, 2->3, 4->5, 6->7   ==>  0->1, 1->2, ... 6->7
    dim 1 (s=2)  0->2, 1->3, 4->6, 5->7   ==>  0->2, 1->3, 2->4, 3->5, 4->6, 5->7
    dim 2 (s=4)  0->4, 1->5, 2->6, 3->7   ==>  unchanged (already disjoint)

Two properties make this the right shape for a benchmark, and both are asserted
below rather than trusted:

1. **Each conveyor bus is a strict lane superset of the hypercube bus it
   replaces**, so the conveyor spec can do everything the builtin one can, plus
   chains. Note this makes *plan existence* monotone, which is weaker than it
   sounds: every strategy here is a bounded heuristic search, so a strictly
   larger search space can still make it find worse plans or exhaust its budget.
   Measured move counts do improve on every row, but that is an empirical result,
   not a consequence of the superset property. Traversing a longer distance along
   a strided conveyor also costs several hops where the hypercube did one bit
   flip; that is the accepted trade.
2. **Geometry is untouched** — words, grid, entangling pairs, word buses and
   modes are byte-identical — so the layout heuristic and every benchmark kernel
   work unchanged, and a conveyor row differs from its builtin row only by bus
   structure.

Transport paths are purely additive. A ``LaneAddr`` is keyed by *source site id*
rather than by an index into the bus ``src`` array, so converting a bus leaves
every existing path entry valid and byte-identical and only adds entries for the
new lanes. New lanes are synthesized in the bundled spec's own waypoint idiom.

Usage::

    python scripts/gen_conveyor_arch.py [--check]

Writes ``examples/arch/gemini-conveyor.json``. With ``--check`` it regenerates
in memory and exits non-zero if the committed file is stale, which is what CI
uses to stop the spec drifting from the bundled Gemini one.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "python/bloqade/lanes/arch/gemini/physical/_physical_spec.json"
DST = REPO_ROOT / "examples/arch/gemini-conveyor.json"

# The bundled spec routes every site-bus lane out to a clearance line, across,
# and back down. Matching it exactly keeps the synthesized lanes physically
# consistent with the hardware-derived ones (#753).
Y_CLEARANCE = 5.0

MOVE_TYPE_SITE_BUS = 0


def _absolute_grid(grid: dict) -> tuple[list[float], list[float]]:
    """Expand a grid's spacing lists into absolute x and y coordinates."""
    xs = [grid["x_start"]]
    for step in grid["x_spacing"]:
        xs.append(xs[-1] + step)
    ys = [grid["y_start"]]
    for step in grid["y_spacing"]:
        ys.append(ys[-1] + step)
    return xs, ys


def _encode_lane(
    *,
    direction: int,
    move_type: int,
    zone_id: int,
    word_id: int,
    site_id: int,
    bus_id: int,
) -> int:
    """Pack a ``LaneAddr``; mirrors ``LaneAddr::encode_u64`` in the core crate.

    Layout: ``data0 = [word_id:16][site_id:16]`` and
    ``data1 = [direction:1][move_type:2][pad][zone_id:8 @ 21][bus_id:16]``,
    packed as ``data0 | (data1 << 32)``.
    """
    data0 = ((word_id & 0xFFFF) << 16) | (site_id & 0xFFFF)
    data1 = (
        ((direction & 1) << 31)
        | ((move_type & 3) << 29)
        | ((zone_id & 0xFF) << 21)
        | (bus_id & 0xFFFF)
    )
    return data0 | (data1 << 32)


def build_conveyor_spec() -> dict:
    spec = json.loads(SRC.read_text(encoding="utf-8"))

    if len(spec["zones"]) != 1:
        raise SystemExit(
            f"expected a single-zone bundled spec, got {len(spec['zones'])} zones"
        )
    zone = spec["zones"][0]
    xs, ys = _absolute_grid(zone["grid"])
    n_sites = len(spec["words"][0]["sites"])

    # 1. Convert each hypercube dimension to conveyor form.
    old_buses = copy.deepcopy(zone["site_buses"])
    new_buses = []
    for dimension in range(len(old_buses)):
        stride = 1 << dimension
        src = list(range(n_sites - stride))
        new_buses.append({"src": src, "dst": [i + stride for i in src]})
    zone["site_buses"] = new_buses

    # The superset property is the whole reason this comparison is meaningful,
    # so assert it rather than documenting it.
    for dimension, (old, new) in enumerate(zip(old_buses, new_buses)):
        old_lanes = set(zip(old["src"], old["dst"]))
        new_lanes = set(zip(new["src"], new["dst"]))
        missing = old_lanes - new_lanes
        if missing:
            raise SystemExit(
                f"site bus {dimension} is not a superset of the hypercube bus "
                f"it replaces; missing lanes {sorted(missing)}"
            )

    # 2. Add transport paths for the newly introduced lanes only.
    existing = {int(path["lane"], 16) for path in spec["paths"]}
    preserved = len(spec["paths"])
    for word_id in zone["words_with_site_buses"]:
        coords = [(xs[i], ys[j]) for i, j in spec["words"][word_id]["sites"]]
        for bus_id, bus in enumerate(new_buses):
            for src_site, dst_site in zip(bus["src"], bus["dst"]):
                (x0, y0), (x1, y1) = coords[src_site], coords[dst_site]
                waypoints = [
                    [x0, y0],
                    [x0, y0 + Y_CLEARANCE],
                    [x1, y1 + Y_CLEARANCE],
                    [x1, y1],
                ]
                for direction, path in ((0, waypoints), (1, list(reversed(waypoints)))):
                    lane = _encode_lane(
                        direction=direction,
                        move_type=MOVE_TYPE_SITE_BUS,
                        zone_id=0,
                        word_id=word_id,
                        site_id=src_site,
                        bus_id=bus_id,
                    )
                    if lane not in existing:
                        spec["paths"].append(
                            {"lane": f"0x{lane:016x}", "waypoints": path}
                        )
                        existing.add(lane)

    added = len(spec["paths"]) - preserved
    print(
        f"site buses: {[len(b['src']) for b in old_buses]} -> "
        f"{[len(b['src']) for b in new_buses]} lanes; "
        f"paths: {preserved} preserved, {added} added"
    )
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed spec matches this generator instead of writing it",
    )
    args = parser.parse_args()

    spec = build_conveyor_spec()
    rendered = json.dumps(spec, indent=1) + "\n"

    if args.check:
        if not DST.exists():
            print(f"ERROR: {DST} does not exist; run without --check", file=sys.stderr)
            return 1
        if DST.read_text(encoding="utf-8") != rendered:
            print(
                f"ERROR: {DST} is stale. Regenerate with:\n"
                f"    python scripts/gen_conveyor_arch.py",
                file=sys.stderr,
            )
            return 1
        print(f"{DST.relative_to(REPO_ROOT)} is up to date")
        return 0

    DST.write_text(rendered, encoding="utf-8")
    print(f"wrote {DST.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
