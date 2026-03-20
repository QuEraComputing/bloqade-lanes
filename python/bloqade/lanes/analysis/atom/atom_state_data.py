from __future__ import annotations

from dataclasses import dataclass, field

from kirin.interp import InterpreterError

from bloqade.lanes.bytecode import AtomStateData as _RustAtomStateData
from bloqade.lanes.bytecode._native import (
    LaneAddress as _RustLaneAddress,
    LocationAddress as _RustLocationAddress,
)
from bloqade.lanes.layout import LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.path import PathFinder


def _from_rust_loc(rust_loc: _RustLocationAddress) -> LocationAddress:
    """Convert a Rust LocationAddress to a Python LocationAddress."""
    return LocationAddress(rust_loc.word_id, rust_loc.site_id)


def _from_rust_lane(rust_lane: _RustLaneAddress) -> LaneAddress:
    """Convert a Rust LaneAddress to a Python LaneAddress."""
    return LaneAddress(
        rust_lane.move_type,
        rust_lane.word_id,
        rust_lane.site_id,
        rust_lane.bus_id,
        rust_lane.direction,
    )


def _to_rust_state(state: AtomStateData) -> _RustAtomStateData:
    """Convert a Python AtomStateData to a Rust AtomStateData."""
    return state._inner


def _from_rust_state(rust_state: _RustAtomStateData) -> AtomStateData:
    """Convert a Rust AtomStateData to a Python AtomStateData."""
    return AtomStateData(_inner=rust_state)


@dataclass(frozen=True)
class AtomStateData:
    _inner: _RustAtomStateData = field(default_factory=_RustAtomStateData, repr=False)

    @classmethod
    def from_fields(
        cls,
        locations_to_qubit: dict[LocationAddress, int] | None = None,
        qubit_to_locations: dict[int, LocationAddress] | None = None,
        collision: dict[int, int] | None = None,
        prev_lanes: dict[int, LaneAddress] | None = None,
        move_count: dict[int, int] | None = None,
    ) -> AtomStateData:
        """Construct from explicit field values."""
        rust_state = _RustAtomStateData(
            locations_to_qubit=(
                {loc._inner: qid for loc, qid in locations_to_qubit.items()}  # type: ignore[attr-defined]
                if locations_to_qubit
                else None
            ),
            qubit_to_locations=(
                {qid: loc._inner for qid, loc in qubit_to_locations.items()}  # type: ignore[attr-defined]
                if qubit_to_locations
                else None
            ),
            collision=collision,
            prev_lanes=(
                {qid: lane._inner for qid, lane in prev_lanes.items()}  # type: ignore[attr-defined]
                if prev_lanes
                else None
            ),
            move_count=move_count,
        )
        return cls(_inner=rust_state)

    @classmethod
    def new(cls, locations: dict[int, LocationAddress] | list[LocationAddress]):
        if isinstance(locations, list):
            locations = {i: loc for i, loc in enumerate(locations)}

        rust_state = _RustAtomStateData.from_qubit_locations(
            {qid: loc._inner for qid, loc in locations.items()}  # type: ignore[attr-defined]
        )
        return cls(_inner=rust_state)

    @property
    def locations_to_qubit(self) -> dict[LocationAddress, int]:
        """Mapping from location to qubit id."""
        return {
            _from_rust_loc(loc): qid
            for loc, qid in self._inner.locations_to_qubit.items()
        }

    @property
    def qubit_to_locations(self) -> dict[int, LocationAddress]:
        """Mapping from qubit id to its current location."""
        return {
            qid: _from_rust_loc(loc)
            for qid, loc in self._inner.qubit_to_locations.items()
        }

    @property
    def collision(self) -> dict[int, int]:
        """Mapping from qubit id to another qubit id that it has collided with."""
        return dict(self._inner.collision)

    @property
    def prev_lanes(self) -> dict[int, LaneAddress]:
        """Mapping from qubit id to the lane it took to reach this state."""
        return {
            qid: _from_rust_lane(lane) for qid, lane in self._inner.prev_lanes.items()
        }

    @property
    def move_count(self) -> dict[int, int]:
        """Mapping from qubit id to number of moves it has had."""
        return dict(self._inner.move_count)

    def __hash__(self):
        return self._inner.__hash__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomStateData):
            return NotImplemented
        return self._inner == other._inner

    def add_atoms(self, locations: dict[int, LocationAddress]):
        rust_locs = {
            qid: loc._inner for qid, loc in locations.items()  # type: ignore[attr-defined]
        }
        try:
            result = self._inner.add_atoms(rust_locs)
        except RuntimeError as e:
            raise InterpreterError(str(e)) from e
        return _from_rust_state(result)

    def apply_moves(
        self,
        lanes: tuple[LaneAddress, ...],
        path_finder: PathFinder,
    ):
        rust_lanes = [lane._inner for lane in lanes]  # type: ignore[attr-defined]
        result = self._inner.apply_moves(
            rust_lanes, path_finder.spec._inner  # type: ignore[attr-defined]
        )
        if result is None:
            return None
        return _from_rust_state(result)

    def get_qubit(self, location: LocationAddress):
        return self._inner.get_qubit(location._inner)  # type: ignore[attr-defined]

    def get_qubit_pairing(self, zone_address: ZoneAddress, arch_spec: ArchSpec):
        result = self._inner.get_qubit_pairing(
            zone_address._inner, arch_spec._inner  # type: ignore[attr-defined]
        )
        if result is None:
            return [], [], []
        return result

    def copy(self):
        return _from_rust_state(self._inner.copy())
