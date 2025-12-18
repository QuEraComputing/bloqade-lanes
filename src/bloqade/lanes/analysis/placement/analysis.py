import abc
from collections import defaultdict
from dataclasses import dataclass, field

from bloqade.analysis.address.lattice import Address, AddressQubit
from kirin import ir
from kirin.analysis import Forward
from kirin.analysis.forward import ForwardFrame
from kirin.interp.exceptions import InterpreterError
from typing_extensions import Self

from bloqade.lanes.layout import LocationAddress

from .lattice import AtomState, ConcreteState


class PlacementStrategyABC(abc.ABC):

    @abc.abstractmethod
    def validate_initial_layout(
        self,
        initial_layout: tuple[LocationAddress, ...],
    ) -> None:
        pass

    @abc.abstractmethod
    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> AtomState:
        pass

    @abc.abstractmethod
    def sq_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState:
        pass

    @abc.abstractmethod
    def measure_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState:
        pass


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
