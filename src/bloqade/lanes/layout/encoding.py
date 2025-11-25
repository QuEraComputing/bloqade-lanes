import abc
import enum
from dataclasses import dataclass, field, replace


class Direction(enum.Enum):
    FORWARD = 0
    BACKWARD = 1


class MoveType(enum.Enum):
    SITE = 0
    WORD = 1


class EncodingType(enum.Enum):
    BIT32 = 32
    BIT64 = 64

    @staticmethod
    def infer(spec) -> "EncodingType":
        num_words = len(spec.words)
        num_sites = len(spec.words[0].sites)
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


class Encoder(abc.ABC):
    """Base class of all encodable entities."""

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

    def __repr__(self) -> str:
        return f"0x{self.get_address(EncodingType.BIT64):016x}"


@dataclass(frozen=True)
class ZoneAddress(Encoder):
    zone_id: int
    """The ID of the zone."""

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

    def __repr__(self) -> str:
        return super().__str__()


@dataclass(frozen=True)
class WordAddress(Encoder):
    """Data class representing a word address in the architecture."""

    word_id: int
    """The ID of the word."""

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

    def __repr__(self) -> str:
        return super().__str__()


@dataclass(frozen=True)
class SiteAddress(Encoder):
    """Data class representing a site address in the architecture."""

    site_id: int
    """The ID of the site."""

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

    def __repr__(self) -> str:
        return super().__str__()


@dataclass(frozen=True)
class LocationAddress(Encoder):
    """Data class representing a physical address in the architecture."""

    word_id: int
    """The ID of the word."""
    site_id: int
    """The ID of the site within the word."""

    def get_address(self, encoding: EncodingType):

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
        assert address.bit_length() <= (
            32 if encoding == EncodingType.BIT32 else 64
        ), "Bug in encoding"
        return address

    def __repr__(self) -> str:
        return super().__str__()


@dataclass(frozen=True, order=True)
class LaneAddress(Encoder):
    direction: Direction
    move_type: MoveType
    word_id: int
    site_id: int
    bus_id: int

    def reverse(self):
        new_direction = (
            Direction.BACKWARD
            if self.direction == Direction.FORWARD
            else Direction.FORWARD
        )
        return replace(self, direction=new_direction)

    def __str__(self) -> str:
        return "0x{:08x}".format(self.get_address(EncodingType.BIT32))

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
        assert address.bit_length() <= (
            32 if encoding == EncodingType.BIT32 else 64
        ), "Bug in encoding"
        return address

    def src_site(self) -> LocationAddress:
        """Get the source site as a PhysicalAddress."""
        return LocationAddress(self.word_id, self.site_id)

    def __repr__(self) -> str:
        return super().__str__()


@dataclass(frozen=True)
class SiteLaneAddress(LaneAddress):
    move_type: MoveType = field(default=MoveType.SITE, init=False)

    def __repr__(self) -> str:
        return super().__str__()


@dataclass(frozen=True)
class WordLaneAddress(LaneAddress):
    move_type: MoveType = field(default=MoveType.WORD, init=False)

    def __repr__(self) -> str:
        return super().__str__()
