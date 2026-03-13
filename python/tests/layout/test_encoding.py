import pytest

from bloqade.lanes.layout import encoding
from bloqade.lanes.layout.arch import ArchSpec, Bus


def test_default_hex_repr():
    assert (
        encoding.USE_HEX_REPR is True
    ), "Expected USE_HEX_REPR to be True by default, please set it back to True"


def make_minimal_arch_spec():
    # Minimal Word mock
    Word = type("Word", (), {"site_indices": (0, 1)})
    words = (Word(), Word())
    zones = ((0, 1),)
    measurement_mode_zones = (0,)
    entangling_zones = frozenset([0])
    has_site_buses = frozenset([0, 1])
    has_word_buses = frozenset([0, 1])
    # Minimal Bus mock
    BusMock = Bus(src=(0, 1), dst=(0, 1))
    site_buses = (BusMock,)
    word_buses = (BusMock,)
    return ArchSpec(
        words=words,  # type: ignore
        zones=zones,
        measurement_mode_zones=measurement_mode_zones,
        entangling_zones=entangling_zones,
        has_site_buses=has_site_buses,
        has_word_buses=has_word_buses,
        site_buses=site_buses,
        word_buses=word_buses,
    )


def test_direction_repr():
    assert repr(encoding.Direction.FORWARD) == "Direction.FORWARD"
    assert repr(encoding.Direction.BACKWARD) == "Direction.BACKWARD"


def test_movetype_repr():
    assert repr(encoding.MoveType.SITE) == "MoveType.SITE"
    assert repr(encoding.MoveType.WORD) == "MoveType.WORD"


def test_encodingtype_repr_and_infer():
    spec = make_minimal_arch_spec()
    assert encoding.EncodingType.infer(spec) == encoding.EncodingType.BIT32
    assert repr(encoding.EncodingType.BIT32) == "EncodingType.BIT32"
    assert repr(encoding.EncodingType.BIT64) == "EncodingType.BIT64"


def test_zoneaddress_get_address():
    za = encoding.ZoneAddress(zone_id=42)
    assert za.get_address(encoding.EncodingType.BIT32) == 42
    assert za.get_address(encoding.EncodingType.BIT64) == 42
    with pytest.raises(ValueError):
        encoding.ZoneAddress(zone_id=0x1FFFF).get_address(encoding.EncodingType.BIT32)


def test_wordaddress_get_address():
    wa = encoding.WordAddress(word_id=42)
    assert wa.get_address(encoding.EncodingType.BIT32) == 42
    assert wa.get_address(encoding.EncodingType.BIT64) == 42
    with pytest.raises(ValueError):
        encoding.WordAddress(word_id=0x1FFFF).get_address(encoding.EncodingType.BIT32)


def test_siteaddress_get_address():
    sa = encoding.SiteAddress(site_id=42)
    assert sa.get_address(encoding.EncodingType.BIT32) == 42
    assert sa.get_address(encoding.EncodingType.BIT64) == 42
    with pytest.raises(ValueError):
        encoding.SiteAddress(site_id=0x1FFFF).get_address(encoding.EncodingType.BIT32)


def test_locationaddress_get_address():
    la = encoding.LocationAddress(word_id=1, site_id=2)
    assert la.get_address(encoding.EncodingType.BIT32) == (1 << 8) | 2
    assert la.get_address(encoding.EncodingType.BIT64) == (1 << 16) | 2
    with pytest.raises(ValueError):
        encoding.LocationAddress(word_id=0x1FFFF, site_id=2).get_address(
            encoding.EncodingType.BIT32
        )
    with pytest.raises(ValueError):
        encoding.LocationAddress(word_id=1, site_id=0x1FFFF).get_address(
            encoding.EncodingType.BIT32
        )


def test_laneaddress_get_address_and_reverse():
    la = encoding.LaneAddress(
        move_type=encoding.MoveType.SITE,
        word_id=1,
        site_id=2,
        bus_id=3,
        direction=encoding.Direction.FORWARD,
    )
    addr32 = la.get_address(encoding.EncodingType.BIT32)
    addr64 = la.get_address(encoding.EncodingType.BIT64)
    assert isinstance(addr32, int)
    assert isinstance(addr64, int)
    rev = la.reverse()
    assert rev.direction == encoding.Direction.BACKWARD
    assert rev.word_id == la.word_id
    with pytest.raises(ValueError):
        encoding.LaneAddress(
            move_type=encoding.MoveType.SITE,
            word_id=0x1FFFF,
            site_id=2,
            bus_id=3,
            direction=encoding.Direction.FORWARD,
        ).get_address(encoding.EncodingType.BIT32)


def test_sitelaneaddress_and_wordlaneaddress():
    sla = encoding.SiteLaneAddress(
        word_id=1, site_id=2, bus_id=3, direction=encoding.Direction.FORWARD
    )
    assert sla.move_type == encoding.MoveType.SITE
    wla = encoding.WordLaneAddress(
        word_id=1, site_id=2, bus_id=3, direction=encoding.Direction.FORWARD
    )
    assert wla.move_type == encoding.MoveType.WORD
