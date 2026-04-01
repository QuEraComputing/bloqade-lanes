from bloqade.geometry.dialects import grid

from bloqade.lanes import layout
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout.encoding import (
    Direction,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
)
from bloqade.lanes.layout.word import Word


def test_get_blockaded_location_with_pair():
    """Test get_blockaded_location returns the correct paired location.

    In the new model, CZ pairs are between words (word 0 <-> word 1),
    mapping site i in one word to site i in the partner word.
    """
    arch_spec = logical.get_arch_spec()

    # location (0, 0) should pair with (1, 0) (word 0 <-> word 1, same site)
    location = layout.LocationAddress(0, 0)
    blockaded = arch_spec.get_blockaded_location(location)

    assert blockaded is not None
    assert blockaded == layout.LocationAddress(1, 0)

    # test reverse
    location2 = layout.LocationAddress(1, 0)
    blockaded2 = arch_spec.get_blockaded_location(location2)

    assert blockaded2 is not None
    assert blockaded2 == layout.LocationAddress(0, 0)


def test_get_blockaded_location_without_pair():
    """Test get_blockaded_location returns None for locations without pairs."""

    # archspec wno sites have CZ pairs
    word = Word(
        grid.Grid.from_positions([0.0, 5.0, 10.0, 15.0], [0.0]),
        ((0, 0), (1, 0), (2, 0), (3, 0)),
    )

    arch_spec = layout.ArchSpec.from_components(
        (word,),
        ((0,),),
        (0,),
        [],
        frozenset(),
        frozenset(),
        (),
        (),
    )

    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 0)) is None
    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 1)) is None
    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 2)) is None


def test_get_blockaded_location_multiple_words():
    """Test get_blockaded_location works across different words."""

    # Create ArchSpec with 4 words, each word having 4 sites
    # Entangling pairs: word 0 <-> word 1, word 2 <-> word 3
    words = tuple(
        Word(
            grid.Grid.from_positions([0.0, 2.0, 10.0, 12.0], [0.0]),
            ((0, 0), (1, 0), (2, 0), (3, 0)),
        )
        for ix in range(4)
    )

    arch_spec = layout.ArchSpec.from_components(
        words,
        (tuple(range(4)),),  # All 4 words in zone 0
        (0,),
        [[(0, 1), (2, 3)]],
        frozenset(),
        frozenset(),
        (),
        (),
    )

    # Test word 0, site 0 should pair with word 1, site 0
    blockaded = arch_spec.get_blockaded_location(layout.LocationAddress(0, 0))
    assert blockaded == layout.LocationAddress(1, 0)

    # Test word 1, site 0 should pair with word 0, site 0
    blockaded2 = arch_spec.get_blockaded_location(layout.LocationAddress(1, 0))
    assert blockaded2 == layout.LocationAddress(0, 0)

    # Test word 2, site 3 should pair with word 3, site 3
    blockaded3 = arch_spec.get_blockaded_location(layout.LocationAddress(2, 3))
    assert blockaded3 == layout.LocationAddress(3, 3)

    # Test word 3, site 2 should pair with word 2, site 2
    blockaded4 = arch_spec.get_blockaded_location(layout.LocationAddress(3, 2))
    assert blockaded4 == layout.LocationAddress(2, 2)


def test_blockaded_location_preserves_site_index():
    """Site-symmetric pairing: get_blockaded_location preserves site_id across all CZ pairs."""
    arch_spec = logical.get_arch_spec()
    for zone in arch_spec.entangling_zones:
        for w_a, w_b in zone:
            num_sites = len(arch_spec.words[w_a].site_indices)
            for s in range(num_sites):
                loc_a = layout.LocationAddress(w_a, s)
                loc_b = layout.LocationAddress(w_b, s)
                blockaded_a = arch_spec.get_blockaded_location(loc_a)
                blockaded_b = arch_spec.get_blockaded_location(loc_b)
                assert blockaded_a is not None
                assert (
                    blockaded_a.site_id == s
                ), f"Site mismatch: ({w_a},{s}) -> site {blockaded_a.site_id}"
                assert blockaded_a.word_id == w_b
                assert blockaded_b is not None
                assert blockaded_b.site_id == s
                assert blockaded_b.word_id == w_a


def test_get_lane_address_site_move_forward():
    """get_lane_address returns the correct lane for a site-bus move (forward)."""
    arch_spec = logical.get_arch_spec()
    src = layout.LocationAddress(0, 0)
    dst = layout.LocationAddress(0, 1)
    lane = arch_spec.get_lane_address(src, dst)
    assert lane is not None
    assert isinstance(lane, SiteLaneAddress)
    assert lane.move_type == MoveType.SITE
    assert lane.direction == Direction.FORWARD
    got_src, got_dst = arch_spec.get_endpoints(lane)
    assert (got_src, got_dst) == (src, dst)


def test_get_lane_address_site_move_backward():
    """get_lane_address returns the correct lane for a site-bus move (backward)."""
    arch_spec = logical.get_arch_spec()
    src = layout.LocationAddress(0, 0)
    dst = layout.LocationAddress(0, 1)
    forward_lane = arch_spec.get_lane_address(src, dst)
    assert forward_lane is not None
    backward_lane = arch_spec.get_lane_address(dst, src)
    assert backward_lane is not None
    assert backward_lane.direction == Direction.BACKWARD
    got_src, got_dst = arch_spec.get_endpoints(backward_lane)
    assert (got_src, got_dst) == (dst, src)


def test_get_lane_address_word_move():
    """get_lane_address returns the correct lane for a word-bus move."""
    arch_spec = logical.get_arch_spec()
    src = layout.LocationAddress(0, 0)
    dst = layout.LocationAddress(1, 0)
    lane = arch_spec.get_lane_address(src, dst)
    assert lane is not None
    assert isinstance(lane, WordLaneAddress)
    assert lane.move_type == MoveType.WORD
    assert lane.direction == Direction.FORWARD
    got_src, got_dst = arch_spec.get_endpoints(lane)
    assert (got_src, got_dst) == (src, dst)


def test_get_lane_address_returns_none_for_unconnected_pair():
    """get_lane_address returns None when no lane connects the two locations."""
    arch_spec = logical.get_arch_spec()
    loc = layout.LocationAddress(0, 0)
    assert arch_spec.get_lane_address(loc, loc) is None


def test_get_lane_address_roundtrip():
    """For every lane, get_lane_address(get_endpoints(lane)) returns the same lane."""
    arch_spec = logical.get_arch_spec()
    # Site lanes: one word, one bus, forward
    for word_id in arch_spec.has_site_buses:
        for bus_id, bus in enumerate(arch_spec.site_buses):
            for i in range(len(bus.src)):
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    lane = SiteLaneAddress(
                        word_id=word_id,
                        site_id=bus.src[i],
                        bus_id=bus_id,
                        direction=direction,
                    )
                    src, dst = arch_spec.get_endpoints(lane)
                    looked_up = arch_spec.get_lane_address(src, dst)
                    assert looked_up is not None
                    assert looked_up == lane
    # Word lanes
    for bus_id, bus in enumerate(arch_spec.word_buses):
        for site_id in arch_spec.has_word_buses:
            for word_id in bus.src:
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    lane = WordLaneAddress(
                        word_id=word_id,
                        site_id=site_id,
                        bus_id=bus_id,
                        direction=direction,
                    )
                    src, dst = arch_spec.get_endpoints(lane)
                    looked_up = arch_spec.get_lane_address(src, dst)
                    assert looked_up is not None
                    assert looked_up == lane
