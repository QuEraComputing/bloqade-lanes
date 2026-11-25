from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from kirin.serialization.core.serializationunit import SerializationUnit

from bloqade.lanes.bytecode._native import (
    Direction as Direction,
    LaneAddress as _RustLaneAddress,
    LocationAddress as _RustLocationAddress,
    MoveType as MoveType,
    ZoneAddress as _RustZoneAddress,
)
from bloqade.lanes.bytecode._wrapper import KirinRustWrapper

if TYPE_CHECKING:
    from kirin.serialization.base.deserializer import Deserializer
    from kirin.serialization.base.serializer import Serializer
    from typing_extensions import Self


def _address_unit(
    address: ZoneAddress | LocationAddress | LaneAddress,
    serializer: Serializer,
    **fields: int | str,
) -> SerializationUnit:
    return SerializationUnit(
        kind="lanes-address",
        module_name=type(address).__module__,
        class_name=type(address).__name__,
        data={key: serializer.serialize(value) for key, value in fields.items()},
    )


def _address_int(unit: SerializationUnit, key: str, deserializer: Deserializer) -> int:
    value = deserializer.deserialize(unit.data[key])
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Address field {key!r} must be an integer")
    return value


class ZoneAddress(KirinRustWrapper[_RustZoneAddress]):
    """Address identifying a zone in the architecture."""

    def __init__(self, zone_id: int):
        self._inner = _RustZoneAddress(zone_id)
        self.__post_init__()

    @property
    def zone_id(self) -> int:
        return self._inner.zone_id

    def serialize(self, serializer: Serializer) -> SerializationUnit:
        """Encode the zone using the standard Kirin attribute protocol."""
        return _address_unit(self, serializer, zone_id=self.zone_id)

    @classmethod
    def deserialize(cls, unit: SerializationUnit, deserializer: Deserializer) -> Self:
        return cls(zone_id=_address_int(unit, "zone_id", deserializer))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ZoneAddress):
            return NotImplemented
        return self.zone_id < other.zone_id


class LocationAddress(KirinRustWrapper[_RustLocationAddress]):
    """Address identifying a physical atom location (zone + word + site)."""

    def __init__(self, word_id: int, site_id: int, zone_id: int = 0):
        self._inner = _RustLocationAddress(zone_id, word_id, site_id)
        self.__post_init__()

    @property
    def zone_id(self) -> int:
        return self._inner.zone_id

    @property
    def word_id(self) -> int:
        return self._inner.word_id

    @property
    def site_id(self) -> int:
        return self._inner.site_id

    def serialize(self, serializer: Serializer) -> SerializationUnit:
        """Encode address fields without serializing the native Rust object."""
        return _address_unit(
            self,
            serializer,
            zone_id=self.zone_id,
            word_id=self.word_id,
            site_id=self.site_id,
        )

    @classmethod
    def deserialize(cls, unit: SerializationUnit, deserializer: Deserializer) -> Self:
        return cls(
            zone_id=_address_int(unit, "zone_id", deserializer),
            word_id=_address_int(unit, "word_id", deserializer),
            site_id=_address_int(unit, "site_id", deserializer),
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LocationAddress):
            return NotImplemented
        return (self.zone_id, self.word_id, self.site_id) < (
            other.zone_id,
            other.word_id,
            other.site_id,
        )

    def replace(
        self,
        *,
        word_id: int | None = None,
        site_id: int | None = None,
        zone_id: int | None = None,
    ) -> Self:
        """Return a copy, optionally replacing fields."""
        return LocationAddress(  # type: ignore[return-value]
            word_id if word_id is not None else self.word_id,
            site_id if site_id is not None else self.site_id,
            zone_id if zone_id is not None else self.zone_id,
        )


class LaneAddress(KirinRustWrapper[_RustLaneAddress]):
    """Address identifying a transport lane."""

    _fixed_move_type: ClassVar[MoveType | None] = None

    def __init__(
        self,
        move_type: MoveType,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = Direction.FORWARD,
        zone_id: int = 0,
    ):
        self._inner = _RustLaneAddress(
            move_type,
            zone_id,
            word_id,
            site_id,
            bus_id,
            direction,
        )
        self.__post_init__()

    @property
    def move_type(self) -> MoveType:
        return self._inner.move_type

    @property
    def zone_id(self) -> int:
        return self._inner.zone_id

    @property
    def word_id(self) -> int:
        return self._inner.word_id

    @property
    def site_id(self) -> int:
        return self._inner.site_id

    @property
    def bus_id(self) -> int:
        return self._inner.bus_id

    @property
    def direction(self) -> Direction:
        return self._inner.direction

    def serialize(self, serializer: Serializer) -> SerializationUnit:
        """Encode enums by name, keeping Rust enum objects out of the payload."""
        return _address_unit(
            self,
            serializer,
            zone_id=self.zone_id,
            word_id=self.word_id,
            site_id=self.site_id,
            bus_id=self.bus_id,
            move_type=self.move_type.name,
            direction=self.direction.name,
        )

    @classmethod
    def deserialize(cls, unit: SerializationUnit, deserializer: Deserializer) -> Self:
        move_name = deserializer.deserialize(unit.data["move_type"])
        direction_name = deserializer.deserialize(unit.data["direction"])
        move_types = {
            "SITE": MoveType.SITE,
            "WORD": MoveType.WORD,
            "ZONE": MoveType.ZONE,
        }
        directions = {"FORWARD": Direction.FORWARD, "BACKWARD": Direction.BACKWARD}
        if not isinstance(move_name, str) or move_name not in move_types:
            raise ValueError(f"Unknown lane move type: {move_name!r}")
        if not isinstance(direction_name, str) or direction_name not in directions:
            raise ValueError(f"Unknown lane direction: {direction_name!r}")
        move_type = move_types[move_name]
        if cls._fixed_move_type is not None and move_type != cls._fixed_move_type:
            raise ValueError(f"Move type {move_name} does not match {cls.__name__}")
        # Reconstruct the native value directly so fixed-move-type subclass
        # constructors need not accept the base class's move_type argument.
        return cls.from_inner(
            _RustLaneAddress(
                move_type,
                _address_int(unit, "zone_id", deserializer),
                _address_int(unit, "word_id", deserializer),
                _address_int(unit, "site_id", deserializer),
                _address_int(unit, "bus_id", deserializer),
                directions[direction_name],
            )
        )

    def reverse(self) -> LaneAddress:
        new_direction = (
            Direction.BACKWARD
            if self.direction == Direction.FORWARD
            else Direction.FORWARD
        )
        return self.replace(direction=new_direction)

    def src_site(self) -> LocationAddress:
        """Get the source site as a LocationAddress."""
        return LocationAddress(self.word_id, self.site_id, self.zone_id)

    def replace(
        self,
        *,
        move_type: MoveType | None = None,
        word_id: int | None = None,
        site_id: int | None = None,
        bus_id: int | None = None,
        direction: Direction | None = None,
        zone_id: int | None = None,
    ) -> Self:
        """Return a copy, optionally replacing fields."""
        return LaneAddress(  # type: ignore[return-value]
            move_type if move_type is not None else self.move_type,
            word_id if word_id is not None else self.word_id,
            site_id if site_id is not None else self.site_id,
            bus_id if bus_id is not None else self.bus_id,
            direction if direction is not None else self.direction,
            zone_id if zone_id is not None else self.zone_id,
        )


class SiteLaneAddress(LaneAddress):
    """LaneAddress with move_type fixed to MoveType.SITE."""

    _fixed_move_type = MoveType.SITE

    def __init__(
        self,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = Direction.FORWARD,
        zone_id: int = 0,
    ):
        super().__init__(MoveType.SITE, word_id, site_id, bus_id, direction, zone_id)


class WordLaneAddress(LaneAddress):
    """LaneAddress with move_type fixed to MoveType.WORD."""

    _fixed_move_type = MoveType.WORD

    def __init__(
        self,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = Direction.FORWARD,
        zone_id: int = 0,
    ):
        super().__init__(MoveType.WORD, word_id, site_id, bus_id, direction, zone_id)
