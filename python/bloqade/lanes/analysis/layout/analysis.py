import abc
from dataclasses import dataclass, field

from bloqade.analysis import address
from kirin import ir
from kirin.analysis.forward import Forward, ForwardFrame
from kirin.lattice import EmptyLattice

from bloqade.lanes.layout.encoding import LocationAddress


@dataclass
class LayoutHeuristicABC(abc.ABC):

    @abc.abstractmethod
    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
        pinned: dict[int, LocationAddress] | None = None,
    ) -> tuple[LocationAddress, ...]:
        """
        Compute the initial qubit layout from circuit stages.

        Args:
            all_qubits: Tuple of logical qubit indices to be mapped.
            stages: List of circuit stages, where each stage is a tuple of
                (control, target) qubit pairs representing two-qubit gates.
            pinned: Map from logical qubit ID to pre-pinned LocationAddress.
                Implementations MUST place each pinned qubit at its requested
                address and MUST NOT use any address in pinned.values() for
                un-pinned qubits. None or empty preserves previous behavior.
                Values are assumed to be valid addresses for the architecture;
                the caller is responsible for validating addresses against the
                ArchSpec before invoking. Implementations MAY (but need not)
                re-check duplicates and extra qubit-id keys.

        Returns:
            A tuple of LocationAddress objects mapping logical qubit indices
            to physical locations. Pinned IDs return their pinned address;
            un-pinned IDs return the heuristic's choice. Raises if no legal
            layout exists.
        """
        ...  # pragma: no cover


@dataclass
class LayoutAnalysis(Forward):
    keys = ("place.layout",)
    lattice = EmptyLattice

    heuristic: LayoutHeuristicABC
    address_entries: dict[ir.SSAValue, address.Address]
    all_qubits: tuple[int, ...]
    stages: list[tuple[tuple[int, int], ...]] = field(default_factory=list, init=False)
    global_address_stack: list[int] = field(default_factory=list, init=False)
    location_addresses: dict[int, LocationAddress] = field(
        default_factory=dict, init=False
    )

    def initialize(self):
        self.stages.clear()
        self.global_address_stack.clear()
        self.location_addresses.clear()
        return super().initialize()

    def eval_stmt_fallback(self, frame, stmt):
        return (self.lattice.bottom(),)

    def add_stage(self, control: tuple[int, ...], target: tuple[int, ...]):
        global_controls = tuple(self.global_address_stack[c] for c in control)
        global_targets = tuple(self.global_address_stack[t] for t in target)
        self.stages.append(tuple(zip(global_controls, global_targets)))

    def method_self(self, method: ir.Method):
        return EmptyLattice.bottom()

    def process_results(self):
        layout = self.heuristic.compute_layout(
            self.all_qubits, self.stages, pinned=self.location_addresses
        )
        return layout

    def get_layout_no_raise(self, method: ir.Method):
        """Get the layout for a given method."""
        self.run_no_raise(method)
        return self.process_results()

    def get_layout(self, method: ir.Method):
        """Get the layout for a given method."""
        self.run(method)
        return self.process_results()

    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        return tuple(EmptyLattice.bottom() for _ in node.results)
