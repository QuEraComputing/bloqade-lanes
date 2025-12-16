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
    ) -> tuple[LocationAddress, ...]:
        """
        Compute the initial qubit layout from circuit stages.

        Args:
            stages: List of circuit stages, where each stage is a tuple of (control, target)
                qubit pairs representing two-qubit gates. For example:
                [
                    ((control1, target1), (control2, target2)),  # stage 1
                    ((control3, target3),),                      # stage 2
                    ...
                ]

        Returns:
            A tuple of LocationAddress objects mapping logical qubit indices to physical locations.
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

    def initialize(self):
        self.stages.clear()
        self.global_address_stack.clear()
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
        layout = self.heuristic.compute_layout(self.all_qubits, self.stages)
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
