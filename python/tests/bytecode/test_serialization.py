"""Standard Kirin JSON round trips for Lanes address attributes and Move IR."""

import pytest
from kirin import ir
from kirin.prelude import structural_no_opt
from kirin.serialization import JSONSerializer
from kirin.serialization.base.deserializer import Deserializer
from kirin.serialization.base.serializer import Serializer
from kirin.serialization.core.serializationmodule import SerializationModule

from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)


def roundtrip(value):
    serializer = JSONSerializer()
    unit = Serializer().serialize(value)
    payload = serializer.encode(
        SerializationModule(symbol_table={}, body=unit, version="1.0.0")
    )
    module = serializer.decode(payload)
    assert module.body is not None
    return Deserializer(dialect_group=structural_no_opt).deserialize(module.body)


@pytest.mark.parametrize(
    "address",
    [
        ZoneAddress(0),
        ZoneAddress(255),
        LocationAddress(0, 0, 0),
        LocationAddress(37, 15, 2),
        LocationAddress(65535, 65535, 255),
    ]
    + [
        LaneAddress(move_type, 37, 5, 2, direction, 3)
        for move_type in (MoveType.SITE, MoveType.WORD, MoveType.ZONE)
        for direction in (Direction.FORWARD, Direction.BACKWARD)
    ]
    + [
        cls(37, 5, 2, direction, 3)
        for cls in (SiteLaneAddress, WordLaneAddress)
        for direction in (Direction.FORWARD, Direction.BACKWARD)
    ],
)
def test_address_roundtrip_preserves_type_value_hash_and_kirin_type(address):
    restored = roundtrip(address)
    assert type(restored) is type(address)
    assert restored == address
    assert restored.encode() == address.encode()
    assert hash(restored) == hash(address)
    assert restored.type == address.type
    assert restored.unwrap() is restored
    wrapped = roundtrip(ir.PyAttr(address))
    assert type(wrapped.unwrap()) is type(address)
    assert wrapped.unwrap() == address


def test_addresses_nested_in_tuples_and_collections():
    value = ir.PyAttr(
        {
            "locations": ((LocationAddress(1, 2),), (LocationAddress(4, 5, 1),)),
            "lanes": [SiteLaneAddress(2, 1, 0), WordLaneAddress(1, 2, 0)],
            "zones": (ZoneAddress(0), ZoneAddress(1)),
        }
    )
    assert roundtrip(value).unwrap() == value.unwrap()


@pytest.mark.parametrize(
    "field,value",
    [
        ("move_type", "INVALID"),
        ("direction", "INVALID"),
        ("word_id", True),
        ("zone_id", "0"),
    ],
)
def test_invalid_lane_payload_rejected(field, value):
    serializer = Serializer()
    unit = LaneAddress(MoveType.SITE, 0, 1, 2).serialize(serializer)
    unit.data[field] = serializer.serialize(value)
    with pytest.raises((TypeError, ValueError)):
        LaneAddress.deserialize(unit, Deserializer(dialect_group=structural_no_opt))


def test_fixed_lane_rejects_conflicting_move_type():
    serializer = Serializer()
    unit = SiteLaneAddress(0, 1, 2).serialize(serializer)
    unit.data["move_type"] = serializer.serialize("WORD")
    with pytest.raises(ValueError, match="does not match"):
        SiteLaneAddress.deserialize(unit, Deserializer(dialect_group=structural_no_opt))
