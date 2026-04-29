from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.bytecode.word import Word


def test_get_cz_partner_with_pair():
    """Test get_cz_partner returns the correct paired location.

    Entangling pairs are defined within each zone. The CZ partner of
    (zone=0, word=W, site=S) is (zone=0, partner_word, site=S).
    """
    arch_spec = logical.get_arch_spec()

    # get_cz_partner preserves zone_id and maps to partner word
    location = LocationAddress(0, 0)
    blockaded = arch_spec.get_cz_partner(location)

    assert blockaded is not None
    assert blockaded == LocationAddress(1, 0)

    # test reverse
    location2 = LocationAddress(1, 0)
    blockaded2 = arch_spec.get_cz_partner(location2)

    assert blockaded2 is not None
    assert blockaded2 == LocationAddress(0, 0)


def test_get_cz_partner_without_pair():
    """Test get_cz_partner returns None for locations without pairs."""
    from bloqade.lanes.bytecode._native import (
        Grid as RustGrid,
        LocationAddress as RustLocAddr,
        Mode as RustMode,
        Zone as RustZone,
    )

    word = Word(sites=((0, 0), (1, 0), (2, 0), (3, 0)))
    rust_grid = RustGrid.from_positions([0.0, 5.0, 10.0, 15.0], [0.0])
    rust_zone = RustZone(
        name="test",
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

    arch_spec = ArchSpec.from_components(
        words=(word,),
        zones=(rust_zone,),
        modes=[rust_mode],
    )

    assert arch_spec.get_cz_partner(LocationAddress(0, 0)) is None
    assert arch_spec.get_cz_partner(LocationAddress(0, 1)) is None
    assert arch_spec.get_cz_partner(LocationAddress(0, 2)) is None


def test_blockaded_location_preserves_site_index():
    """Site-symmetric pairing: get_cz_partner preserves site_id across all CZ pairs."""
    arch_spec = logical.get_arch_spec()
    # get_cz_partner maps to the partner word within the same zone
    num_sites = len(arch_spec.words[0].site_indices)
    for word_id in range(len(arch_spec.words)):
        for s in range(num_sites):
            loc = LocationAddress(word_id, s)
            blockaded = arch_spec.get_cz_partner(loc)
            if blockaded is not None:
                assert (
                    blockaded.site_id == s
                ), f"Site mismatch: (w={word_id},s={s}) -> site {blockaded.site_id}"


def test_get_lane_address_word_move_forward():
    """get_lane_address returns the correct lane for a word-bus move (forward)."""
    arch_spec = logical.get_arch_spec()
    # Word bus 0 maps word 0 -> word 1
    src = LocationAddress(0, 0)
    dst = LocationAddress(1, 0)
    lane = arch_spec.get_lane_address(src, dst)
    assert lane is not None
    assert lane.move_type == MoveType.WORD
    assert lane.direction == Direction.FORWARD
    got_src, got_dst = arch_spec.get_endpoints(lane)
    assert (got_src, got_dst) == (src, dst)


def test_get_lane_address_word_move_backward():
    """get_lane_address returns the correct lane for a word-bus move (backward)."""
    arch_spec = logical.get_arch_spec()
    src = LocationAddress(0, 0)
    dst = LocationAddress(1, 0)
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
    src = LocationAddress(0, 0)
    dst = LocationAddress(1, 0)
    lane = arch_spec.get_lane_address(src, dst)
    assert lane is not None
    assert lane.move_type == MoveType.WORD
    assert lane.direction == Direction.FORWARD
    got_src, got_dst = arch_spec.get_endpoints(lane)
    assert (got_src, got_dst) == (src, dst)


def test_get_lane_address_returns_none_for_unconnected_pair():
    """get_lane_address returns None when no lane connects the two locations."""
    arch_spec = logical.get_arch_spec()
    loc = LocationAddress(0, 0)
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


def test_arch_spec_value_equality_is_value_based():
    """Two independently constructed ArchSpecs from the same factory must
    compare equal. ArchSpec.__eq__ delegates to ``self.words == other.words``,
    which in turn relies on Word equality being value-based — a regression
    happened during the #466 wrapper refactor when Word lost its explicit
    ``__eq__``/``__hash__`` (the underlying Rust Word has no value-based
    equality yet).
    """
    a = logical.get_arch_spec()
    b = logical.get_arch_spec()
    assert a == b
    # Word value-equality is the underlying piece that broke.
    assert a.words == b.words
    assert all(wa == wb for wa, wb in zip(a.words, b.words))


# ── Tests for derived zone helpers (#421) ──


def _build_two_zone_spec():
    """Build a small 2-zone ArchSpec for testing zone-derivation helpers.

    Zone 0 has words 0,1 paired (CZ-capable) plus an unpaired word 2.
    Zone 1 has words 3,4 paired (CZ-capable). 5 words total.
    """
    from bloqade.lanes.bytecode._native import (
        Grid as RustGrid,
        LocationAddress as RustLocAddr,
        Mode as RustMode,
        WordBus,
        Zone as RustZone,
    )
    from bloqade.lanes.bytecode.word import Word

    words = (
        Word(sites=((0, 0),)),
        Word(sites=((1, 0),)),
        Word(sites=((0, 1),)),  # unpaired word in zone 0
        Word(sites=((0, 0),)),
        Word(sites=((1, 0),)),
    )
    zone0 = RustZone(
        name="zone0",
        grid=RustGrid.from_positions([0.0, 1.0], [0.0, 1.0]),
        site_buses=[],
        word_buses=[WordBus(src=[0], dst=[1])],
        words_with_site_buses=[],
        sites_with_word_buses=[0],
        entangling_pairs=[(0, 1)],
    )
    zone1 = RustZone(
        name="zone1",
        grid=RustGrid.from_positions([10.0, 11.0], [0.0, 1.0]),
        site_buses=[],
        word_buses=[WordBus(src=[3], dst=[4])],
        words_with_site_buses=[],
        sites_with_word_buses=[0],
        entangling_pairs=[(3, 4)],
    )
    mode = RustMode(
        name="all",
        zones=[0, 1],
        bitstring_order=[
            RustLocAddr(0, 0, 0),
            RustLocAddr(0, 1, 0),
            RustLocAddr(1, 3, 0),
            RustLocAddr(1, 4, 0),
        ],
    )
    return ArchSpec.from_components(
        words=words,
        zones=(zone0, zone1),
        modes=[mode],
    )


def test_word_zone_map_single_zone():
    """word_zone_map maps every word to zone 0 for the Gemini logical arch."""
    arch_spec = logical.get_arch_spec()
    mapping = arch_spec.word_zone_map
    assert set(mapping.keys()) == set(range(len(arch_spec.words)))
    assert all(zone_id == 0 for zone_id in mapping.values())


def test_word_zone_map_multi_zone():
    """word_zone_map assigns each word to its containing zone."""
    arch_spec = _build_two_zone_spec()
    mapping = arch_spec.word_zone_map
    # Words 0, 1 are in zone 0 via entangling_pairs; word 2 is unreferenced
    # (falls back to zone 0). Words 3, 4 are in zone 1 via entangling_pairs.
    assert mapping == {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}


def test_home_sites_single_zone_uses_zone_zero():
    """Every home site for the Gemini logical arch carries zone_id=0."""
    arch_spec = logical.get_arch_spec()
    sites = arch_spec.home_sites
    assert len(sites) > 0
    assert all(site.zone_id == 0 for site in sites)
    # Every home_site word_id should be a home word.
    assert all(site.word_id in arch_spec._home_words for site in sites)


def test_home_sites_multi_zone_uses_per_word_zone():
    """home_sites uses each word's actual zone_id, not a hardcoded 0."""
    arch_spec = _build_two_zone_spec()
    sites = arch_spec.home_sites
    # Home words are min of each entangling pair, plus unpaired words:
    # zone 0: words 0 (pair 0,1) and 2 (unpaired); zone 1: word 3 (pair 3,4).
    by_word = {site.word_id: site.zone_id for site in sites}
    assert by_word == {0: 0, 2: 0, 3: 1}


def test_cz_zone_addresses_single_zone():
    """cz_zone_addresses contains exactly the entangling zone (Gemini)."""
    arch_spec = logical.get_arch_spec()
    assert arch_spec.cz_zone_addresses == frozenset(
        [ZoneAddress(0)],
    )


def test_cz_zone_addresses_multi_zone():
    """cz_zone_addresses contains every zone with entangling_pairs."""
    arch_spec = _build_two_zone_spec()
    assert arch_spec.cz_zone_addresses == frozenset(
        [ZoneAddress(0), ZoneAddress(1)],
    )


def test_assert_single_cz_zone_rejects_multi_zone():
    """Strategies built on multi-zone arches fail loudly at construction."""
    import pytest

    from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone

    arch_spec = _build_two_zone_spec()
    with pytest.raises(ValueError, match="requires exactly one CZ-capable zone"):
        assert_single_cz_zone(arch_spec, "TestStrategy")
