import abc
from dataclasses import dataclass
from typing import Any, TypeGuard

from kirin import ir
from kirin.dialects import func, ilist
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LaneAddress, MoveType, ZoneAddress

from .base import AtomStateRewriter


class NoiseModelABC(abc.ABC):

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

    def get_lane_noise(self, lane: LaneAddress):
        return self.lane_noise

    def get_bus_idle_noise(self, move_type: MoveType, bus_id: int):
        return self.idle_noise

    def get_cz_unpaired_noise(self, zone_address: ZoneAddress):
        return self.cz_unpaired_noise


@dataclass
class InsertNoise(AtomStateRewriter):
    atom_state_map: dict[ir.Statement, atom.AtomStateType]
    noise_model: NoiseModelABC

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not (
            isinstance(node, (move.Move, move.CZ))
            and isinstance(atom_state := self.atom_state_map.get(node), atom.AtomState)
        ):
            return rewrite_abc.RewriteResult()

        rewriter = getattr(self, f"rewrite_{type(node).__name__}")
        return rewriter(atom_state, node)

    def rewrite_Move(
        self, atom_state: atom.AtomState, node: move.Move
    ) -> rewrite_abc.RewriteResult:
        if len(node.lanes) == 0:
            return rewrite_abc.RewriteResult()

        first_lane = node.lanes[0]

        stationary_qubits: set[ir.SSAValue] = set(self.physical_ssa_values)
        move_noise_methods = tuple(map(self.noise_model.get_lane_noise, node.lanes))

        qubit_ssas = self.get_qubit_ssa_from_locations(
            atom_state,
            tuple(self.arch_spec.get_endpoints(lane)[1] for lane in node.lanes),
        )
        stationary_qubits.difference_update(filter(None, qubit_ssas))

        def filter_no_qubit(
            pair: tuple[ir.Method[[qubit.Qubit], None], ir.SSAValue | None],
        ) -> TypeGuard[tuple[ir.Method[[qubit.Qubit], None], ir.SSAValue]]:
            return pair[1] is not None

        all_pairs = filter(filter_no_qubit, zip(move_noise_methods, qubit_ssas))

        for noise_method, qubit_ssa in all_pairs:
            func.Invoke((qubit_ssa,), callee=noise_method).insert_before(node)

        if len(stationary_qubits) > 0:
            bus_idle_method = self.noise_model.get_bus_idle_noise(
                first_lane.move_type, first_lane.bus_id
            )
            (idle_reg := ilist.New(tuple(stationary_qubits))).insert_before(node)
            func.Invoke((idle_reg.result,), callee=bus_idle_method).insert_before(node)

        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_CZ(
        self, atom_state: atom.AtomState, node: move.CZ
    ) -> rewrite_abc.RewriteResult:
        if (
            cz_unpaired_noise := self.noise_model.get_cz_unpaired_noise(
                node.zone_address
            )
        ) is None:
            return rewrite_abc.RewriteResult()

        assert len(atom_state.locations) == len(
            self.physical_ssa_values
        ), "Mismatch between atom state and physical SSA values"

        _, _, unpaired = atom_state.get_qubit_pairing(node.zone_address, self.arch_spec)

        unpaired_qubits: tuple[ir.SSAValue, ...] = tuple(
            self.physical_ssa_values[i] for i in unpaired
        )

        if len(unpaired_qubits) > 0:
            unpaired_reg = ilist.New(unpaired_qubits)
            func.Invoke((unpaired_reg.result,), callee=cz_unpaired_noise).insert_after(
                node
            )
            unpaired_reg.insert_after(node)

            return rewrite_abc.RewriteResult(has_done_something=True)

        return rewrite_abc.RewriteResult()
