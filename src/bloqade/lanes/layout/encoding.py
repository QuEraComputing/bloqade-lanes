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


@dataclass(frozen=True)
class MoveType(abc.ABC):
    direction: Direction

    @abc.abstractmethod
    def get_address(self, encoding: EncodingType) -> int:
        pass

    @abc.abstractmethod
    def src_site(self) -> tuple[int, int]:
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

    def src_site(self) -> tuple[int, int]:
        return self.word_id, self.site_id

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

        assert word_id_enc == self.word_id, "word ID too large to encode"
        assert lane_id_enc == self.lane_id, "Lane ID too large to encode"
        assert site_id_enc == self.site_id, "Site ID too large to encode"

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

    def src_site(self) -> tuple[int, int]:
        return self.start_word_id, self.lane_id

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

        assert lane_id_enc == self.lane_id, "Lane ID too large to encode"
        assert start_word_id == self.start_word_id, "Start word ID too large to encode"
        assert end_word_id == self.end_word_id, "End word ID too large to encode"

        address = lane_id_enc
        address |= end_word_id << shift
        address |= start_word_id << (2 * shift)
        address |= MoveTypeEnum.INTER.value << (3 * shift + padding)
        address |= self.direction.value << (3 * shift + padding + 1)
        assert address.bit_length() <= (
            32 if encoding == EncodingType.BIT32 else 64
        ), "Bug in encoding"
        return address
