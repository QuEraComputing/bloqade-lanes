import abc
from collections import defaultdict
from dataclasses import dataclass, field

from bloqade.analysis.address.lattice import Address, AddressQubit
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from kirin.interp.exceptions import InterpreterError
from typing_extensions import Self

from bloqade.lanes.layout import LaneAddress, LocationAddress, ZoneAddress

from .lattice import AtomState, ConcreteState, ExecuteCZ, ExecuteMeasure


class PlacementStrategyABC(abc.ABC):

    @abc.abstractmethod
    def validate_initial_layout(
        self,
        initial_layout: tuple[LocationAddress, ...],
    ) -> None: ...

    @abc.abstractmethod
    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> AtomState: ...

    @abc.abstractmethod
    def sq_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState: ...

    @abc.abstractmethod
    def measure_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState: ...


class SingleZonePlacementStrategyABC(PlacementStrategyABC):

    @abc.abstractmethod
    def compute_moves(
        self,
        state_before: ConcreteState,
        state_after: ConcreteState,
    ) -> tuple[tuple[LaneAddress, ...], ...]: ...

    @abc.abstractmethod
    def desired_cz_layout(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> ConcreteState: ...

    def cz_placements(
        self, state: AtomState, controls: tuple[int, ...], targets: tuple[int, ...]
    ) -> AtomState:
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()

        if not isinstance(state, ConcreteState):
            return AtomState.top()

        desired_state = self.desired_cz_layout(state, controls, targets)
        move_layers = self.compute_moves(state, desired_state)

        return ExecuteCZ(
            occupied=state.occupied,
            layout=desired_state.layout,
            move_count=tuple(
                mc + int(src != dst)
                for mc, src, dst in zip(
                    state.move_count, state.layout, desired_state.layout
                )
            ),
            active_cz_zones=frozenset(
                [ZoneAddress(0)]
            ),  # Assuming single zone with address 0
            move_layers=move_layers,
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        return state  # No movement needed for single zone

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state

        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(
                ZoneAddress(0) for _ in qubits
            ),  # Assuming single zone with address 0
        )


@dataclass
class PlacementAnalysis(Forward[AtomState]):
    keys = ("runtime.placement",)

    initial_layout: tuple[LocationAddress, ...]
    address_analysis: dict[ir.SSAValue, Address]
    move_count: defaultdict[ir.SSAValue, int] = field(init=False)

    placement_strategy: PlacementStrategyABC
    """The strategy function to use for calculating placements."""
    lattice = AtomState

    def __post_init__(self):
        self.placement_strategy.validate_initial_layout(self.initial_layout)
        super().__post_init__()

    def initialize(self) -> Self:
        self.move_count = defaultdict(int)
        return super().initialize()

    def get_inintial_state(self, qubits: tuple[ir.SSAValue, ...]):
        occupied = set(self.initial_layout)
        layout = []
        move_count = []
        for q in qubits:
            if not isinstance(addr := self.address_analysis.get(q), AddressQubit):
                raise InterpreterError(f"Qubit {q} does not have a qubit address.")

            loc_addr = self.initial_layout[addr.data]
            occupied.discard(loc_addr)
            layout.append(loc_addr)
            move_count.append(self.move_count[q])

        return ConcreteState(
            layout=tuple(layout),
            occupied=frozenset(occupied),
            move_count=tuple(move_count),
        )

    def method_self(self, method: ir.Method) -> AtomState:
        return AtomState.bottom()

    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        return tuple(AtomState.bottom() for _ in node.results)
