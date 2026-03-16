import abc
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from kirin import ir, types
from kirin.print import Printer

from bloqade.lanes.bytecode._native import (
    Direction as _RustDirection,
    LaneAddress as _RustLaneAddress,
    LocationAddress as _RustLocationAddress,
    MoveType as _RustMoveType,
    ZoneAddress as _RustZoneAddress,
)

if TYPE_CHECKING:
    from bloqade.lanes.layout.arch import ArchSpec

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


class EncodingType(enum.IntEnum):
    BIT32 = 32
    BIT64 = 64

    @staticmethod
    def infer(spec: "ArchSpec") -> "EncodingType":
        num_words = len(spec.words)
        num_sites = len(spec.words[0].site_indices)
        num_site_buses = len(spec.site_buses)
        num_word_buses = len(spec.word_buses)

        max_id = max(
            num_words - 1,
            num_sites - 1,
            num_site_buses - 1,
            num_word_buses - 1,
        )

        if max_id < 256:
            return EncodingType.BIT32
        elif max_id < 65536:
            return EncodingType.BIT64
        else:
            raise ValueError("Architecture too large to encode with 64-bit addresses")

    def __repr__(self):
        return f"EncodingType.{self.name}"


@dataclass()
class Encoder(ir.Data):
    """Base class of all encodable entities."""

    def __post_init__(self):
        self.type = types.PyClass(type(self))

    @abc.abstractmethod
    def get_address(self, encoding: EncodingType) -> int:
        """Return the encoded physical address as an integer.

        Args:
            encoding: The encoding type to use (BIT32 or BIT64).

        Returns:
            The encoded physical address as an integer.

        Raises:
            ValueError: If the word_id or site_id is too large to encode.

        """
        pass

    def unwrap(self):
        return self

    def print_impl(self, printer: Printer):
        try:
            printer.plain_print(f"0x{self.get_address(EncodingType.BIT32):08x}")
        except ValueError:
            printer.plain_print(f"0x{self.get_address(EncodingType.BIT64):016x}")

    def __repr__(self) -> str:
        try:
            return f"0x{self.get_address(EncodingType.BIT32):08x}"
        except ValueError:
            return f"0x{self.get_address(EncodingType.BIT64):016x}"


class ZoneAddress(Encoder):
    """Address identifying a zone in the architecture."""

    _inner: _RustZoneAddress

    def __init__(self, zone_id: int):
        self._inner = _RustZoneAddress(zone_id)
        self.__post_init__()

    @property
    def zone_id(self) -> int:
        return self._inner.zone_id

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

    def get_address(self, encoding: EncodingType) -> int:
        if encoding == EncodingType.BIT32:
            mask = 0xFF
        elif encoding == EncodingType.BIT64:
            mask = 0xFFFF
        else:
            raise ValueError("Unsupported encoding type")

        zone_id_enc = mask & self.zone_id

        if zone_id_enc != self.zone_id:
            raise ValueError("Zone ID too large to encode")

        return zone_id_enc


@dataclass(repr=not USE_HEX_REPR, order=True)
class WordAddress(Encoder):
    """Data class representing a word address in the architecture."""

    word_id: int
    """The ID of the word."""

    def __hash__(self) -> int:
        return self.get_address(EncodingType.BIT64)

    def get_address(self, encoding: EncodingType) -> int:
        if encoding == EncodingType.BIT32:
            mask = 0xFF
        elif encoding == EncodingType.BIT64:
            mask = 0xFFFF
        else:
            raise ValueError("Unsupported encoding type")

        word_id_enc = mask & self.word_id

        if word_id_enc != self.word_id:
            raise ValueError("Word ID too large to encode")

        return word_id_enc


@dataclass(repr=not USE_HEX_REPR, order=True)
class SiteAddress(Encoder):
    """Data class representing a site address in the architecture."""

    site_id: int
    """The ID of the site."""

    def __hash__(self) -> int:
        return self.get_address(EncodingType.BIT64)

    def get_address(self, encoding: EncodingType) -> int:
        if encoding == EncodingType.BIT32:
            mask = 0xFF
        elif encoding == EncodingType.BIT64:
            mask = 0xFFFF
        else:
            raise ValueError("Unsupported encoding type")

        site_id_enc = mask & self.site_id

        if site_id_enc != self.site_id:
            raise ValueError("Site ID too large to encode")

        return site_id_enc


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

    def get_address(self, encoding: EncodingType) -> int:
        if encoding == EncodingType.BIT32:
            mask = 0xFF
            shift = 8
        elif encoding == EncodingType.BIT64:
            mask = 0xFFFF
            shift = 16
        else:
            raise ValueError("Unsupported encoding type")

        word_id_enc = mask & self.word_id
        site_id_enc = mask & self.site_id

        if word_id_enc != self.word_id:
            raise ValueError("Word ID too large to encode")

        if site_id_enc != self.site_id:
            raise ValueError("Site ID too large to encode")

        address = site_id_enc
        address |= word_id_enc << shift
        return address

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

    def get_address(self, encoding: EncodingType) -> int:
        if encoding == EncodingType.BIT32:
            mask = 0xFF
            shift = 8
            padding = 6
        elif encoding == EncodingType.BIT64:
            mask = 0xFFFF
            shift = 16
            padding = 14
        else:
            raise ValueError("Unsupported encoding type")

        word_id_enc = mask & self.word_id
        lane_id_enc = mask & self.bus_id
        site_id_enc = mask & self.site_id

        if lane_id_enc != self.bus_id:
            raise ValueError("Lane ID too large to encode")

        if site_id_enc != self.site_id:
            raise ValueError("Site ID too large to encode")

        if word_id_enc != self.word_id:
            raise ValueError("Word ID too large to encode")

        address = lane_id_enc
        address |= site_id_enc << shift
        address |= word_id_enc << (2 * shift)
        address |= self.move_type.value << (3 * shift + padding)
        address |= self.direction.value << (3 * shift + padding + 1)
        return address

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
