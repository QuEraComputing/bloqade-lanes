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
    block_id: int
    site_id: int
    lane_id: int

    def src_site(self) -> tuple[int, int]:
        return self.block_id, self.site_id

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

        block_id_enc = mask & self.block_id
        lane_id_enc = mask & self.lane_id
        site_id_enc = mask & self.site_id

        assert block_id_enc == self.block_id, "Block ID too large to encode"
        assert lane_id_enc == self.lane_id, "Lane ID too large to encode"
        assert site_id_enc == self.site_id, "Site ID too large to encode"

        address = lane_id_enc
        address |= site_id_enc << shift
        address |= block_id_enc << (2 * shift)
        address |= MoveTypeEnum.INTRA.value << (3 * shift + padding)
        address |= self.direction.value << (3 * shift + padding + 1)
        assert address.bit_length() <= (
            32 if encoding == EncodingType.BIT32 else 64
        ), "Bug in encoding"
        return address


@dataclass(frozen=True)
class InterMove(MoveType):
    start_block_id: int
    end_block_id: int
    lane_id: int

    def src_site(self) -> tuple[int, int]:
        return self.start_block_id, self.lane_id

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
        start_block_id = mask & self.start_block_id
        end_block_id = mask & self.end_block_id

        assert lane_id_enc == self.lane_id, "Lane ID too large to encode"
        assert (
            start_block_id == self.start_block_id
        ), "Start Block ID too large to encode"
        assert end_block_id == self.end_block_id, "End Block ID too large to encode"

        address = lane_id_enc
        address |= end_block_id << shift
        address |= start_block_id << (2 * shift)
        address |= MoveTypeEnum.INTER.value << (3 * shift + padding)
        address |= self.direction.value << (3 * shift + padding + 1)
        assert address.bit_length() <= (
            32 if encoding == EncodingType.BIT32 else 64
        ), "Bug in encoding"
        return address
