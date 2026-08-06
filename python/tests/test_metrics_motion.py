"""Regression tests for the per-CZ motion replay in ``Metrics``.

The replay previously applied lanes sequentially with an unguarded dict
write, so a conveyor chain (overlapping-but-acyclic bus, legal since #874)
applied in ascending-source order overwrote the downstream atom's entry and
permanently lost it — silently corrupting per-qubit hop/distance metrics.
Lanes within one ``move`` execute simultaneously; the replay must resolve
every endpoint against the pre-move occupancy before committing.
"""

import json

import pytest
from bloqade.decoders.dialects import annotate

from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import ArchSpec as RustArchSpec
from bloqade.lanes.bytecode.encoding import (
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.bytecode.exceptions import (
    DestinationOccupiedError,
    MoveValidationError,
)
from bloqade.lanes.dialects import move
from bloqade.lanes.metrics import Metrics
from bloqade.lanes.prelude import kernel

kernel = kernel.add(annotate)

# Four words in one zone; word bus 0 is a conveyor chain 0→1, 1→2, 2→3
# (overlapping acyclic src/dst — legal per bus well-formedness).
CHAIN_ARCH_JSON = json.dumps(
    {
        "version": "2.0",
        "words": [{"sites": [[w, 0], [w, 1]]} for w in range(4)],
        "zones": [
            {
                "grid": {
                    "x_start": 0.0,
                    "y_start": 0.0,
                    "x_spacing": [5.0, 5.0, 5.0],
                    "y_spacing": [3.0],
                },
                "site_buses": [],
                "word_buses": [{"src": [0, 1, 2], "dst": [1, 2, 3]}],
                "words_with_site_buses": [],
                "sites_with_word_buses": [0],
                "entangling_pairs": [[1, 2]],
            }
        ],
        "zone_buses": [],
        "modes": [{"name": "full", "zones": [0], "bitstring_order": []}],
    }
)


def test_per_cz_motion_counts_chain_movers_once():
    arch_spec = ArchSpec(RustArchSpec.from_json_validated(CHAIN_ARCH_JSON))

    @kernel
    def main():
        state0 = move.load()
        state1 = move.fill(
            state0,
            location_addresses=(
                move.LocationAddress(0, 0),
                move.LocationAddress(1, 0),
            ),
        )
        state2 = move.logical_initialize(
            state1,
            thetas=(0.0, 0.0),
            phis=(0.0, 0.0),
            lams=(0.0, 0.0),
            location_addresses=(
                move.LocationAddress(0, 0),
                move.LocationAddress(1, 0),
            ),
        )
        # Conveyor chain 0→1, 1→2 in ascending-source order: the order the
        # old sequential replay corrupted (atom at word 1 was overwritten,
        # then double-moved as if it were the atom from word 0).
        state3 = move.move(
            state2,
            lanes=(WordLaneAddress(0, 0, 0), WordLaneAddress(1, 0, 0)),
        )
        state4 = move.cz(state3, zone_address=ZoneAddress(0))
        future = move.end_measure(state4, zone_addresses=(move.ZoneAddress(0),))
        return move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(1, 0),
        )

    metrics = Metrics(arch_spec=arch_spec)
    avg_hops, avg_distance_um = metrics.analyze_per_cz_motion(main)

    # Both atoms move exactly once (one 5 µm hop each). The sequential
    # replay reported a single atom with two hops (avg 2.0) and lost the
    # other atom entirely.
    assert avg_hops == 1.0
    assert avg_distance_um == 5.0


def test_per_cz_motion_chain_is_lane_order_independent():
    arch_spec = ArchSpec(RustArchSpec.from_json_validated(CHAIN_ARCH_JSON))

    def build(lanes: tuple[WordLaneAddress, ...]):
        @kernel
        def main():
            state0 = move.load()
            state1 = move.fill(
                state0,
                location_addresses=(
                    move.LocationAddress(0, 0),
                    move.LocationAddress(1, 0),
                ),
            )
            state2 = move.logical_initialize(
                state1,
                thetas=(0.0, 0.0),
                phis=(0.0, 0.0),
                lams=(0.0, 0.0),
                location_addresses=(
                    move.LocationAddress(0, 0),
                    move.LocationAddress(1, 0),
                ),
            )
            state3 = move.move(state2, lanes=lanes)
            state4 = move.cz(state3, zone_address=ZoneAddress(0))
            future = move.end_measure(state4, zone_addresses=(move.ZoneAddress(0),))
            return move.get_future_result(
                future,
                zone_address=move.ZoneAddress(0),
                location_address=move.LocationAddress(1, 0),
            )

        return main

    metrics = Metrics(arch_spec=arch_spec)
    forward = metrics.analyze_per_cz_motion(
        build((WordLaneAddress(0, 0, 0), WordLaneAddress(1, 0, 0)))
    )
    reverse = metrics.analyze_per_cz_motion(
        build((WordLaneAddress(1, 0, 0), WordLaneAddress(0, 0, 0)))
    )
    assert forward == reverse == (1.0, 5.0)


def test_per_cz_motion_rejects_inexecutable_move():
    """An occupied destination that is not vacated by the same group cannot
    execute. The replay delegates to the canonical execution model, so it
    fails fast instead of silently overwriting the occupant's entry."""
    arch_spec = ArchSpec(RustArchSpec.from_json_validated(CHAIN_ARCH_JSON))

    @kernel
    def main():
        state0 = move.load()
        state1 = move.fill(
            state0,
            location_addresses=(
                move.LocationAddress(0, 0),
                move.LocationAddress(1, 0),
            ),
        )
        state2 = move.logical_initialize(
            state1,
            thetas=(0.0, 0.0),
            phis=(0.0, 0.0),
            lams=(0.0, 0.0),
            location_addresses=(
                move.LocationAddress(0, 0),
                move.LocationAddress(1, 0),
            ),
        )
        # Only word 0 moves; its destination (word 1) holds a stationary atom.
        state3 = move.move(state2, lanes=(WordLaneAddress(0, 0, 0),))
        state4 = move.cz(state3, zone_address=ZoneAddress(0))
        future = move.end_measure(state4, zone_addresses=(move.ZoneAddress(0),))
        return move.get_future_result(
            future,
            zone_address=move.ZoneAddress(0),
            location_address=move.LocationAddress(1, 0),
        )

    metrics = Metrics(arch_spec=arch_spec)
    with pytest.raises(MoveValidationError) as excinfo:
        metrics.analyze_per_cz_motion(main)
    assert any(isinstance(e, DestinationOccupiedError) for e in excinfo.value.errors)
