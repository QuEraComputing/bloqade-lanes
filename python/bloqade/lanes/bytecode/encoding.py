from __future__ import annotations

from typing import TYPE_CHECKING

from bloqade.lanes.bytecode._native import (
    Direction as Direction,
    LaneAddress as _RustLaneAddress,
    LocationAddress as _RustLocationAddress,
    MoveType as MoveType,
    ZoneAddress as _RustZoneAddress,
)
from bloqade.lanes.bytecode._wrapper import KirinRustWrapper

if TYPE_CHECKING:
    from typing_extensions import Self


class ZoneAddress(KirinRustWrapper[_RustZoneAddress]):
    """Address identifying a zone in the architecture."""

    def __init__(self, zone_id: int):
        self._inner = _RustZoneAddress(zone_id)
        self.__post_init__()

    @property
    def zone_id(self) -> int:
        return self._inner.zone_id

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

    def reverse(self) -> "LaneAddress":
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

    def __init__(
        self,
        word_id: int,
        site_id: int,
        bus_id: int,
        direction: Direction = Direction.FORWARD,
        zone_id: int = 0,
    ):
        super().__init__(MoveType.WORD, word_id, site_id, bus_id, direction, zone_id)
