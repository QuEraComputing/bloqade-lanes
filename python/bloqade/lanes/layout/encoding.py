import abc
from dataclasses import dataclass
from typing import Self

from kirin import ir, types
from kirin.print import Printer

from bloqade.lanes.bytecode._native import (
    Direction as Direction,
    LaneAddress as _RustLaneAddress,
    LocationAddress as _RustLocationAddress,
    MoveType as MoveType,
    ZoneAddress as _RustZoneAddress,
)


@dataclass()
class Encoder(ir.Data):
    """Base class of all encodable entities."""

    def __post_init__(self):
        self.type = types.PyClass(type(self))

    @abc.abstractmethod
    def encode(self) -> int:
        """Return the bit-packed encoded address as an integer."""
        ...

    def unwrap(self):
        return self

    def print_impl(self, printer: Printer):
        printer.plain_print(f"0x{self.encode():08x}")

    def __repr__(self) -> str:
        return f"0x{self.encode():08x}"


class ZoneAddress(Encoder):
    """Address identifying a zone in the architecture."""

    _inner: _RustZoneAddress

    def __init__(self, zone_id: int):
        self._inner = _RustZoneAddress(zone_id)
        self.__post_init__()

    @property
    def zone_id(self) -> int:
        return self._inner.zone_id

    def encode(self) -> int:
        return self._inner.encode()

    def __hash__(self) -> int:
        return self._inner.encode()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZoneAddress):
            return NotImplemented
        return self._inner == other._inner

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ZoneAddress):
            return NotImplemented
        return self.zone_id < other.zone_id


class LocationAddress(Encoder):
    """Address identifying a physical atom location (word + site)."""

    _inner: _RustLocationAddress

    def __init__(self, word_id: int, site_id: int):
        self._inner = _RustLocationAddress(word_id, site_id)
        self.__post_init__()

    @property
    def word_id(self) -> int:
        return self._inner.word_id

    @property
    def site_id(self) -> int:
        return self._inner.site_id

    def encode(self) -> int:
        return self._inner.encode()

    def __hash__(self) -> int:
        return self._inner.encode()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocationAddress):
            return NotImplemented
        return self._inner == other._inner

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LocationAddress):
            return NotImplemented
        return (self.word_id, self.site_id) < (other.word_id, other.site_id)

    def replace(
        self,
        *,
        word_id: int | None = None,
        site_id: int | None = None,
    ) -> Self:
        """Return a copy, optionally replacing fields."""
        return LocationAddress(  # type: ignore[return-value]
            word_id if word_id is not None else self.word_id,
            site_id if site_id is not None else self.site_id,
        )


class LaneAddress(Encoder):
    """Address identifying a transport lane."""

    _inner: _RustLaneAddress

    def __init__(
        self,
        move_type: MoveType,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = Direction.FORWARD,
    ):
        self._inner = _RustLaneAddress(
            move_type,
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

    def reverse(self) -> "LaneAddress":
        new_direction = (
            Direction.BACKWARD
            if self.direction == Direction.FORWARD
            else Direction.FORWARD
        )
        return self.replace(direction=new_direction)

    def encode(self) -> int:
        return self._inner.encode()

    def src_site(self) -> LocationAddress:
        """Get the source site as a LocationAddress."""
        return LocationAddress(self.word_id, self.site_id)

    def replace(
        self,
        *,
        move_type: MoveType | None = None,
        word_id: int | None = None,
        site_id: int | None = None,
        bus_id: int | None = None,
        direction: Direction | None = None,
    ) -> Self:
        """Return a copy, optionally replacing fields."""
        return LaneAddress(  # type: ignore[return-value]
            move_type if move_type is not None else self.move_type,
            word_id if word_id is not None else self.word_id,
            site_id if site_id is not None else self.site_id,
            bus_id if bus_id is not None else self.bus_id,
            direction if direction is not None else self.direction,
        )

    def __hash__(self) -> int:
        return self._inner.encode()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LaneAddress):
            return NotImplemented
        return self._inner == other._inner


class SiteLaneAddress(LaneAddress):
    """LaneAddress with move_type fixed to MoveType.SITE."""

    def __init__(
        self,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = Direction.FORWARD,
    ):
        super().__init__(MoveType.SITE, word_id, site_id, bus_id, direction)


class WordLaneAddress(LaneAddress):
    """LaneAddress with move_type fixed to MoveType.WORD."""

    def __init__(
        self,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = Direction.FORWARD,
    ):
        super().__init__(MoveType.WORD, word_id, site_id, bus_id, direction)
