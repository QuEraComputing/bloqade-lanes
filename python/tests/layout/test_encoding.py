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
    la = encoding.LocationAddress(word_id=1, site_id=2)
    # Rust encodes as [word_id:16][site_id:16]
    assert la.encode() == (1 << 16) | 2


def test_laneaddress_encode_and_reverse():
    la = encoding.LaneAddress(
        move_type=encoding.MoveType.SITE,
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
        word_id=1, site_id=2, bus_id=3, direction=encoding.Direction.FORWARD
    )
    assert sla.move_type == encoding.MoveType.SITE
    wla = encoding.WordLaneAddress(
        word_id=1, site_id=2, bus_id=3, direction=encoding.Direction.FORWARD
    )
    assert wla.move_type == encoding.MoveType.WORD
