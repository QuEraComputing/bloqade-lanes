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

    In the zone-centric model, CZ pairs are between zones (zone 0 <-> zone 1).
    The CZ partner of (zone=0, word=W, site=S) is (zone=1, word=W, site=S).
    """
    arch_spec = logical.get_arch_spec()

    # location (z=0, w=0, s=0) should pair with (z=0, w=1, s=0) (word 0 <-> word 1)
    location = layout.LocationAddress(0, 0, 0)
    blockaded = arch_spec.get_blockaded_location(location)

    assert blockaded is not None
    assert blockaded == layout.LocationAddress(0, 1, 0)

    # test reverse
    location2 = layout.LocationAddress(0, 1, 0)
    blockaded2 = arch_spec.get_blockaded_location(location2)

    assert blockaded2 is not None
    assert blockaded2 == layout.LocationAddress(0, 0, 0)


def test_get_blockaded_location_without_pair():
    """Test get_blockaded_location returns None for locations without pairs."""
    from bloqade.lanes.bytecode._native import (
        Grid as RustGrid,
        Mode as RustMode,
        Zone as RustZone,
    )
    from bloqade.lanes.bytecode._native import LocationAddress as RustLocAddr

    word = Word(sites=((0, 0), (1, 0), (2, 0), (3, 0)))
    rust_grid = RustGrid.from_positions([0.0, 5.0, 10.0, 15.0], [0.0])
    rust_zone = RustZone(
        grid=rust_grid,
        site_buses=[],
        word_buses=[],
        words_with_site_buses=[],
        sites_with_word_buses=[],
    )
    rust_mode = RustMode(
        name="all",
        zones=[0],
        bitstring_order=[RustLocAddr(0, 0, s) for s in range(4)],
    )

    arch_spec = layout.ArchSpec.from_components(
        (word,),
        (rust_zone,),
        [],
        [rust_mode],
    )

    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 0, 0)) is None
    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 0, 1)) is None
    assert arch_spec.get_blockaded_location(layout.LocationAddress(0, 0, 2)) is None


def test_blockaded_location_preserves_site_index():
    """Site-symmetric pairing: get_blockaded_location preserves site_id across all CZ pairs."""
    arch_spec = logical.get_arch_spec()
    # In the zone-centric model, entangling pairs are between zones
    for z_a, z_b in arch_spec.entangling_zone_pairs:
        num_sites = len(arch_spec.words[0].site_indices)
        for word_id in range(len(arch_spec.words)):
            for s in range(num_sites):
                loc_a = layout.LocationAddress(z_a, word_id, s)
                blockaded_a = arch_spec.get_blockaded_location(loc_a)
                if blockaded_a is not None:
                    assert (
                        blockaded_a.site_id == s
                    ), f"Site mismatch: (z={z_a},w={word_id},s={s}) -> site {blockaded_a.site_id}"
                    assert blockaded_a.zone_id == z_b


def test_get_lane_address_site_move_forward():
    """get_lane_address returns the correct lane for a site-bus move (forward)."""
    arch_spec = logical.get_arch_spec()
    src = layout.LocationAddress(0, 0, 0)
    dst = layout.LocationAddress(0, 0, 1)
    lane = arch_spec.get_lane_address(src, dst)
    assert lane is not None
    assert lane.move_type == MoveType.SITE
    assert lane.direction == Direction.FORWARD
    got_src, got_dst = arch_spec.get_endpoints(lane)
    assert (got_src, got_dst) == (src, dst)


def test_get_lane_address_site_move_backward():
    """get_lane_address returns the correct lane for a site-bus move (backward)."""
    arch_spec = logical.get_arch_spec()
    src = layout.LocationAddress(0, 0, 0)
    dst = layout.LocationAddress(0, 0, 1)
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
    src = layout.LocationAddress(0, 0, 0)
    dst = layout.LocationAddress(0, 1, 0)
    lane = arch_spec.get_lane_address(src, dst)
    assert lane is not None
    assert lane.move_type == MoveType.WORD
    assert lane.direction == Direction.FORWARD
    got_src, got_dst = arch_spec.get_endpoints(lane)
    assert (got_src, got_dst) == (src, dst)


def test_get_lane_address_returns_none_for_unconnected_pair():
    """get_lane_address returns None when no lane connects the two locations."""
    arch_spec = logical.get_arch_spec()
    loc = layout.LocationAddress(0, 0, 0)
    assert arch_spec.get_lane_address(loc, loc) is None


def test_get_lane_address_roundtrip():
    """For every lane, get_lane_address(get_endpoints(lane)) returns the same lane."""
    arch_spec = logical.get_arch_spec()
    for zone_id, zone in enumerate(arch_spec.zones):
        for bus_id, bus in enumerate(zone.site_buses):
            for word_id in zone.words_with_site_buses:
                for i in range(len(bus.src)):
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        lane = SiteLaneAddress(
                            zone_id=zone_id,
                            word_id=word_id,
                            site_id=bus.src[i],
                            bus_id=bus_id,
                            direction=direction,
                        )
                        src, dst = arch_spec.get_endpoints(lane)
                        looked_up = arch_spec.get_lane_address(src, dst)
                        assert looked_up is not None
                        assert looked_up == lane
        for bus_id, bus in enumerate(zone.word_buses):
            for site_id in zone.sites_with_word_buses:
                for word_id in bus.src:
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        lane = WordLaneAddress(
                            zone_id=zone_id,
                            word_id=word_id,
                            site_id=site_id,
                            bus_id=bus_id,
                            direction=direction,
                        )
                        src, dst = arch_spec.get_endpoints(lane)
                        looked_up = arch_spec.get_lane_address(src, dst)
                        assert looked_up is not None
                        assert looked_up == lane
