import pytest

from bloqade.lanes.layout import encoding


def test_direction_repr():
    assert repr(encoding.Direction.FORWARD) == "Direction.FORWARD"
    assert repr(encoding.Direction.BACKWARD) == "Direction.BACKWARD"


def test_movetype_repr():
    assert repr(encoding.MoveType.SITE) == "MoveType.SITE"
    assert repr(encoding.MoveType.WORD) == "MoveType.WORD"


def test_zoneaddress_encode():
    za = encoding.ZoneAddress(zone_id=42)
    assert za.encode() == 42


def test_locationaddress_encode():
    la = encoding.LocationAddress(zone_id=0, word_id=1, site_id=2)
    # Rust encodes as [zone_id:8][word_id:16][site_id:16][pad:24]
    encoded = la.encode()
    assert encoded != 0
    from bloqade.lanes.bytecode._native import LocationAddress as RustLoc
    decoded = RustLoc.decode(encoded)
    assert decoded.zone_id == 0
    assert decoded.word_id == 1
    assert decoded.site_id == 2


def test_laneaddress_encode_and_reverse():
    la = encoding.LaneAddress(
        move_type=encoding.MoveType.SITE,
        zone_id=0,
        word_id=1,
        site_id=2,
        bus_id=3,
        direction=encoding.Direction.FORWARD,
    )
    encoded = la.encode()
    assert isinstance(encoded, int)
    rev = la.reverse()
    assert rev.direction == encoding.Direction.BACKWARD
    assert rev.word_id == la.word_id


def test_sitelaneaddress_and_wordlaneaddress():
    sla = encoding.SiteLaneAddress(
        zone_id=0, word_id=1, site_id=2, bus_id=3, direction=encoding.Direction.FORWARD
    )
    assert sla.move_type == encoding.MoveType.SITE
    wla = encoding.WordLaneAddress(
        zone_id=0, word_id=1, site_id=2, bus_id=3, direction=encoding.Direction.FORWARD
    )
    assert wla.move_type == encoding.MoveType.WORD


class TestRangeValidation:
    """IDs outside valid range are rejected at construction with ValueError."""

    def test_zone_address_overflow(self):
        with pytest.raises(ValueError, match="zone_id=65536 exceeds maximum"):
            encoding.ZoneAddress(zone_id=0x10000)

    def test_location_address_word_id_overflow(self):
        with pytest.raises(ValueError, match="word_id=65536 exceeds maximum"):
            encoding.LocationAddress(zone_id=0, word_id=0x10000, site_id=0)

    def test_location_address_site_id_overflow(self):
        with pytest.raises(ValueError, match="site_id=65536 exceeds maximum"):
            encoding.LocationAddress(zone_id=0, word_id=0, site_id=0x10000)

    def test_lane_address_word_id_overflow(self):
        with pytest.raises(ValueError, match="word_id=65536 exceeds maximum"):
            encoding.LaneAddress(encoding.MoveType.SITE, 0x10000, 0, 0)

    def test_lane_address_site_id_overflow(self):
        with pytest.raises(ValueError, match="site_id=65536 exceeds maximum"):
            encoding.LaneAddress(encoding.MoveType.SITE, 0, 0x10000, 0)

    def test_lane_address_bus_id_overflow(self):
        with pytest.raises(ValueError, match="bus_id=65536 exceeds maximum"):
            encoding.LaneAddress(encoding.MoveType.SITE, 0, 0, 0x10000)

    def test_max_valid_values_accepted(self):
        """Maximum valid values are accepted."""
        encoding.ZoneAddress(zone_id=0xFFFF)
        encoding.LocationAddress(zone_id=0xFF, word_id=0xFFFF, site_id=0xFFFF)
        encoding.LaneAddress(
            encoding.MoveType.SITE, 0xFFFF, 0xFFFF, 0xFFFF, zone_id=0xFF
        )

    def test_zone_address_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            encoding.ZoneAddress(zone_id=-1)

    def test_location_address_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            encoding.LocationAddress(zone_id=0, word_id=-1, site_id=0)
        with pytest.raises(ValueError, match="must be non-negative"):
            encoding.LocationAddress(zone_id=0, word_id=0, site_id=-1)

    def test_lane_address_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            encoding.LaneAddress(encoding.MoveType.SITE, -1, 0, 0)
        with pytest.raises(ValueError, match="must be non-negative"):
            encoding.LaneAddress(encoding.MoveType.SITE, 0, -1, 0)
        with pytest.raises(ValueError, match="must be non-negative"):
            encoding.LaneAddress(encoding.MoveType.SITE, 0, 0, -1)
