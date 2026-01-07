import abc
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Sequence, TypeGuard, TypeVar

from kirin import ir
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import func, ilist
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import LaneAddress, LocationAddress, MoveType, ZoneAddress

from .base import AtomStateRewriter

Len = TypeVar("Len")


class NoiseModelABC(abc.ABC):
    def get_cz_paired_noise(
        self, zone_address: ZoneAddress
    ) -> (
        ir.Method[[ilist.IList[qubit.Qubit, Len], ilist.IList[qubit.Qubit, Len]], None]
        | None
    ):
        return None

    def get_global_rz_noise(
        self,
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float], None] | None:
        return None

    def get_local_rz_noise(
        self, locations: Sequence[LocationAddress]
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float], None] | None:
        return None

    def get_global_r_noise(
        self,
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float, float], None] | None:
        return None

    def get_local_r_noise(
        self, locations: Sequence[LocationAddress]
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float, float], None] | None:
        return None

    @abc.abstractmethod
    def get_lane_noise(self, lane: LaneAddress) -> ir.Method[[qubit.Qubit], None]: ...

    @abc.abstractmethod
    def get_bus_idle_noise(
        self, move_type: MoveType, bus_id: int
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any]], None]: ...

    @abc.abstractmethod
    def get_cz_unpaired_noise(
        self, zone_address: ZoneAddress
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any]], None]: ...


@dataclass
class SimpleNoiseModel(NoiseModelABC):
    lane_noise: ir.Method[[qubit.Qubit], None]
    idle_noise: ir.Method[[ilist.IList[qubit.Qubit, Any]], None]
    cz_unpaired_noise: ir.Method[[ilist.IList[qubit.Qubit, Any]], None]
    cz_paired_noise: ir.Method[
        [ilist.IList[qubit.Qubit, Any], ilist.IList[qubit.Qubit, Any]], None
    ]
    global_rz_noise: ir.Method[[ilist.IList[qubit.Qubit, Any], float], None]
    local_rz_noise: ir.Method[[ilist.IList[qubit.Qubit, Any], float], None]
    global_r_noise: ir.Method[[ilist.IList[qubit.Qubit, Any], float, float], None]
    local_r_noise: ir.Method[[ilist.IList[qubit.Qubit, Any], float, float], None]

    def get_lane_noise(self, lane: LaneAddress):
        return self.lane_noise

    def get_bus_idle_noise(self, move_type: MoveType, bus_id: int):
        return self.idle_noise

    def get_cz_unpaired_noise(self, zone_address: ZoneAddress):
        return self.cz_unpaired_noise

    def get_cz_paired_noise(self, zone_address: ZoneAddress):
        return self.cz_paired_noise

    def get_global_r_noise(self):
        return self.global_r_noise

    def get_local_r_noise(self, locations: Sequence[LocationAddress]):
        return self.local_r_noise

    def get_global_rz_noise(self):
        return self.global_rz_noise

    def get_local_rz_noise(self, locations: Sequence[LocationAddress]):
        return self.local_rz_noise


@dataclass
class InsertNoise(AtomStateRewriter):
    atom_state_map: ForwardFrame[atom.MoveExecution]
    noise_model: NoiseModelABC

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if (trait := node.get_trait(move.EmitsState)) is None:
            return rewrite_abc.RewriteResult()

        if not isinstance(
            atom_state := self.atom_state_map.get(trait.get_state_result(node)),
            atom.AtomState,
        ):
            return rewrite_abc.RewriteResult()

        return self.rewriter(node, atom_state)

    @singledispatchmethod
    def rewriter(self, node: ir.Statement, atom_state: atom.AtomState):
        return rewrite_abc.RewriteResult()

    @rewriter.register(move.Move)
    def _(
        self, node: move.Move, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        if len(node.lanes) == 0:
            return rewrite_abc.RewriteResult()

        move_noise_methods = tuple(map(self.noise_model.get_lane_noise, node.lanes))
        qubit_ssas = self.get_qubit_ssa_from_locations(
            atom_state,
            tuple(self.arch_spec.get_endpoints(lane)[1] for lane in node.lanes),
        )
        qubit_ssas = list(filter(None, qubit_ssas))

        def is_stationary(qubit_ssa: ir.SSAValue) -> bool:
            return qubit_ssa not in qubit_ssas

        stationary_qubits = tuple(
            filter(is_stationary, self.physical_ssa_values.values())
        )

        def filter_no_qubit(
            pair: tuple[ir.Method[[qubit.Qubit], None], ir.SSAValue | None],
        ) -> TypeGuard[tuple[ir.Method[[qubit.Qubit], None], ir.SSAValue]]:
            return pair[1] is not None

        all_pairs = filter(filter_no_qubit, zip(move_noise_methods, qubit_ssas))

        for noise_method, qubit_ssa in all_pairs:
            func.Invoke((qubit_ssa,), callee=noise_method).insert_before(node)

        first_lane, *_ = node.lanes
        if len(stationary_qubits) > 0:
            bus_idle_method = self.noise_model.get_bus_idle_noise(
                first_lane.move_type, first_lane.bus_id
            )
            (idle_reg := ilist.New(tuple(stationary_qubits))).insert_before(node)
            func.Invoke((idle_reg.result,), callee=bus_idle_method).insert_before(node)

        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.CZ)
    def _(self, node: move.CZ, atom_state: atom.AtomState) -> rewrite_abc.RewriteResult:
        cz_unpaired_noise = self.noise_model.get_cz_unpaired_noise(node.zone_address)
        cz_paired_noise = self.noise_model.get_cz_paired_noise(node.zone_address)

        controls, targets, unpaired = atom_state.data.get_qubit_pairing(
            node.zone_address, self.arch_spec
        )

        unpaired_qubits: tuple[ir.SSAValue, ...] = tuple(
            self.physical_ssa_values[i] for i in unpaired
        )
        controls_ssa: tuple[ir.SSAValue, ...] = tuple(
            self.physical_ssa_values[i] for i in controls
        )
        targets_ssa: tuple[ir.SSAValue, ...] = tuple(
            self.physical_ssa_values[i] for i in targets
        )

        has_done_something = False
        if len(unpaired_qubits) > 0 and cz_unpaired_noise is not None:
            unpaired_reg = ilist.New(unpaired_qubits)
            func.Invoke((unpaired_reg.result,), callee=cz_unpaired_noise).insert_after(
                node
            )
            unpaired_reg.insert_after(node)

            has_done_something = True

        if len(controls_ssa) > 0 and cz_paired_noise is not None:
            assert len(targets_ssa) == len(controls_ssa), "Mismatched CZ pairing."
            controls_reg = ilist.New(controls_ssa)
            targets_reg = ilist.New(targets_ssa)
            func.Invoke(
                (controls_reg.result, targets_reg.result),
                callee=cz_paired_noise,
            ).insert_after(node)
            targets_reg.insert_after(node)
            controls_reg.insert_after(node)

            has_done_something = True

        return rewrite_abc.RewriteResult(has_done_something=has_done_something)

    def insert_gate_noise(
        self,
        node: ir.Statement,
        method: ir.Method,
        qubit_ssas: tuple[ir.SSAValue | None, ...],
    ):
        reg = ilist.New(tuple(filter(None, qubit_ssas)))
        inputs = (reg.result, *node.args[1:])
        func.Invoke(inputs, callee=method).insert_after(node)
        reg.insert_after(node)

    @singledispatchmethod
    def get_noise_method(self, node: ir.Statement) -> ir.Method | None:
        return None

    @get_noise_method.register(move.LocalR)
    def _(self, node: move.LocalR) -> ir.Method | None:
        return self.noise_model.get_local_r_noise(node.location_addresses)

    @get_noise_method.register(move.LocalRz)
    def _(self, node: move.LocalRz) -> ir.Method | None:
        return self.noise_model.get_local_rz_noise(node.location_addresses)

    @get_noise_method.register(move.GlobalRz)
    def _(self, node: move.GlobalRz) -> ir.Method | None:
        return self.noise_model.get_global_rz_noise()

    @get_noise_method.register(move.GlobalR)
    def _(self, node: move.GlobalR) -> ir.Method | None:
        return self.noise_model.get_global_r_noise()

    @rewriter.register(move.LocalRz)
    @rewriter.register(move.LocalR)
    def _(
        self, node: move.LocalRz | move.LocalR, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        if (noise_method := self.get_noise_method(node)) is None:
            return rewrite_abc.RewriteResult()

        qubit_ssas = self.get_qubit_ssa_from_locations(
            atom_state, node.location_addresses
        )

        self.insert_gate_noise(node, noise_method, qubit_ssas)

        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.GlobalRz)
    @rewriter.register(move.GlobalR)
    def rewrite_global(
        self, node: move.GlobalRz | move.GlobalR, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        if (noise_method := self.get_noise_method(node)) is None:
            return rewrite_abc.RewriteResult()

        all_qubit_ids = tuple(sorted(self.physical_ssa_values.keys()))
        qubit_ssas = tuple(
            self.physical_ssa_values.get(qubit_id) for qubit_id in all_qubit_ids
        )
        self.insert_gate_noise(node, noise_method, qubit_ssas)

        return rewrite_abc.RewriteResult(has_done_something=True)
