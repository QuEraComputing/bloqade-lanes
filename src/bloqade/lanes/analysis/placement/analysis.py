import abc
from typing import Callable
from dataclasses import field, dataclass

from bloqade.analysis.address.lattice import Address, AddressQubit
from kirin.analysis import Forward
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import LocationAddress
from kirin.interp.exceptions import InterpreterError
from .lattice import AtomsState, ConcreteState, LocalID

from kirin import ir


class MoveStrategy(abc.ABC):
    @abc.abstractmethod
    def calculate_cz_placements(
        self,
        state: ConcreteState,
        controls: tuple[LocalID, ...],
        targets: tuple[LocalID, ...],
    ) -> ConcreteState:
        pass

    @abc.abstractmethod
    def calculate_sq_placements(
        self,
        state: ConcreteState,
        qubits: tuple[LocalID, ...],
    ) -> ConcreteState:
        pass





@dataclass
class PlacementAnalysis(Forward[AtomsState]):
    keys = ["runtime.placement"]

    initial_layout: dict[int, LocationAddress]
    address_analysis: dict[ir.SSAValue, Address]

    move_strategy: MoveStrategy
    """The strategy function to use for calculating placements."""
    lattice = AtomsState

    def get_inintial_state(self, qubits: tuple[ir.SSAValue, ...]):
        occupied = set(self.initial_layout.values())
        layout = {}
        for i, q in enumerate(qubits):
            addr = self.address_analysis[q]
            if not isinstance(addr, AddressQubit):
                raise InterpreterError(f"Qubit {q} does not have a qubit address.")
            
            loc_addr = self.initial_layout[addr.data]
            occupied.discard(loc_addr)
            layout[LocalID(i)] = loc_addr

        return ConcreteState(
            layout=layout,
            moves={LocalID(i): [] for i in range(len(qubits))},
            occupied=frozenset(occupied),
        )

    def get_placement_cz(self, state: ConcreteState, controls: tuple[LocalID, ...], targets: tuple[LocalID, ...]) -> ConcreteState:
        return self.move_strategy.calculate_cz_placements(
            state,
            controls,
            targets,
        )
    
    def get_placement_sq(self, state: ConcreteState, qubits: tuple[LocalID, ...]) -> ConcreteState:
        return self.move_strategy.calculate_sq_placements(
            state,
            qubits,
        )
    
    

    def method_self(self, method: ir.Method) -> AtomsState:
        return AtomsState.bottom()
    


            

    
