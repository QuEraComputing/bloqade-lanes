import abc
import enum
from dataclasses import dataclass
from typing import Self

from kirin import ir, types
from kirin.print import Printer

from bloqade.lanes.bytecode._native import (
    Direction as _RustDirection,
    LaneAddress as _RustLaneAddress,
    LocationAddress as _RustLocationAddress,
    MoveType as _RustMoveType,
    ZoneAddress as _RustZoneAddress,
)

USE_HEX_REPR = True


class Direction(enum.IntEnum):
    FORWARD = 0
    BACKWARD = 1

    def __repr__(self):
        return f"Direction.{self.name}"

    def _to_rust(self) -> _RustDirection:
        if self == Direction.FORWARD:
            return _RustDirection.FORWARD
        return _RustDirection.BACKWARD

    @staticmethod
    def _from_rust(d: _RustDirection) -> "Direction":
        if d == _RustDirection.FORWARD:
            return Direction.FORWARD
        return Direction.BACKWARD


class MoveType(enum.IntEnum):
    SITE = 0
    WORD = 1

    def __repr__(self):
        return f"MoveType.{self.name}"

    def _to_rust(self) -> _RustMoveType:
        if self == MoveType.SITE:
            return _RustMoveType.SITE
        return _RustMoveType.WORD

    @staticmethod
    def _from_rust(m: _RustMoveType) -> "MoveType":
        if m == _RustMoveType.SITE:
            return MoveType.SITE
        return MoveType.WORD


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


@dataclass(repr=not USE_HEX_REPR, order=True)
class WordAddress(Encoder):
    """Data class representing a word address in the architecture."""

    word_id: int
    """The ID of the word."""

    def encode(self) -> int:
        return self.word_id

    def __hash__(self) -> int:
        return self.word_id


@dataclass(repr=not USE_HEX_REPR, order=True)
class SiteAddress(Encoder):
    """Data class representing a site address in the architecture."""

    site_id: int
    """The ID of the site."""

    def encode(self) -> int:
        return self.site_id

    def __hash__(self) -> int:
        return self.site_id


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

    def replace_word_id(self, word_id: int) -> Self:
        """Return a copy with a different word_id."""
        return LocationAddress(word_id, self.site_id)  # type: ignore[return-value]


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
            move_type._to_rust(),
            word_id,
            site_id,
            bus_id,
            direction._to_rust(),
        )
        self.__post_init__()

    @property
    def move_type(self) -> MoveType:
        return MoveType._from_rust(self._inner.move_type)

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
        return Direction._from_rust(self._inner.direction)

    def reverse(self) -> "LaneAddress":
        new_direction = (
            Direction.BACKWARD
            if self.direction == Direction.FORWARD
            else Direction.FORWARD
        )
        return LaneAddress(
            self.move_type, self.word_id, self.site_id, self.bus_id, new_direction
        )

    def encode(self) -> int:
        return self._inner.encode()

    def src_site(self) -> LocationAddress:
        """Get the source site as a LocationAddress."""
        return LocationAddress(self.word_id, self.site_id)

    def replace_word_id(self, word_id: int) -> Self:
        """Return a copy with a different word_id."""
        return LaneAddress(  # type: ignore[return-value]
            self.move_type, word_id, self.site_id, self.bus_id, self.direction
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
