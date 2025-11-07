import abc
import enum
from dataclasses import dataclass, replace


class Direction(enum.Enum):
    FORWARD = 0
    BACKWARD = 1


class MoveTypeEnum(enum.Enum):
    INTRA = 0
    INTER = 1


class EncodingType(enum.Enum):
    BIT32 = 0
    BIT64 = 1

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


@dataclass(frozen=True)
class MoveType(Encoder):
    direction: Direction

    @abc.abstractmethod
    def src_site(self) -> LocationAddress:
        """Get the source site as a PhysicalAddress."""
        pass

    def reverse(self):
        new_direction = (
            Direction.BACKWARD
            if self.direction == Direction.FORWARD
            else Direction.FORWARD
        )
        return replace(self, direction=new_direction)


@dataclass(frozen=True)
class IntraMove(MoveType):
    word_id: int
    site_id: int
    lane_id: int

    def src_site(self):
        return LocationAddress(self.word_id, self.site_id)

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
        lane_id_enc = mask & self.lane_id
        site_id_enc = mask & self.site_id

        if lane_id_enc != self.lane_id:
            raise ValueError("Lane ID too large to encode")

        if site_id_enc != self.site_id:
            raise ValueError("Site ID too large to encode")

        if word_id_enc != self.word_id:
            raise ValueError("Word ID too large to encode")

        address = lane_id_enc
        address |= site_id_enc << shift
        address |= word_id_enc << (2 * shift)
        address |= MoveTypeEnum.INTRA.value << (3 * shift + padding)
        address |= self.direction.value << (3 * shift + padding + 1)
        assert address.bit_length() <= (
            32 if encoding == EncodingType.BIT32 else 64
        ), "Bug in encoding"
        return address


@dataclass(frozen=True)
class InterMove(MoveType):
    start_word_id: int
    end_word_id: int
    lane_id: int

    def src_site(self):
        return LocationAddress(self.start_word_id, self.lane_id)

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

        lane_id_enc = mask & self.lane_id
        start_word_id = mask & self.start_word_id
        end_word_id = mask & self.end_word_id

        if lane_id_enc != self.lane_id:
            raise ValueError("Lane ID too large to encode")

        if start_word_id != self.start_word_id:
            raise ValueError("Start word ID too large to encode")

        if end_word_id != self.end_word_id:
            raise ValueError("End word ID too large to encode")

        address = lane_id_enc
        address |= end_word_id << shift
        address |= start_word_id << (2 * shift)
        address |= MoveTypeEnum.INTER.value << (3 * shift + padding)
        address |= self.direction.value << (3 * shift + padding + 1)
        assert address.bit_length() <= (
            32 if encoding == EncodingType.BIT32 else 64
        ), "Bug in encoding"
        return address
