import abc
import math
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Sequence, TypeGuard, TypeVar

from kirin import ir
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import func, ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import LaneAddress, LocationAddress, MoveType, ZoneAddress

from ... import utils
from .base import AtomStateRewriter

Len = TypeVar("Len")


class NoiseModelABC(abc.ABC):
    """Abstract base class for noise models used during move-to-squin compilation.

    Subclass this to define custom noise kernels for each physical operation
    (lane moves, CZ gates, idle periods, local/global rotations). Methods that
    return ``None`` indicate no noise is inserted for that operation.
    """

    def get_cz_paired_noise(
        self, zone_address: ZoneAddress
    ) -> (
        ir.Method[[ilist.IList[qubit.Qubit, Len], ilist.IList[qubit.Qubit, Len]], None]
        | None
    ):
        """Return the noise kernel for paired qubits during a CZ gate, or ``None``."""
        return None

    def get_global_rz_noise(
        self,
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float], None] | None:
        """Return the noise kernel for global Rz rotations, or ``None``."""
        return None

    def get_local_rz_noise(
        self, locations: Sequence[LocationAddress]
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float], None] | None:
        """Return the noise kernel for local Rz rotations, or ``None``."""
        return None

    def get_global_r_noise(
        self,
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float, float], None] | None:
        """Return the noise kernel for global R rotations, or ``None``."""
        return None

    def get_local_r_noise(
        self, locations: Sequence[LocationAddress]
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any], float, float], None] | None:
        """Return the noise kernel for local R rotations, or ``None``."""
        return None

    def get_logical_initialize(
        self,
    ) -> tuple[
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Any]], None] | None,
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Any]], None] | None,
    ]:
        """Return (clean_kernel, noisy_kernel) for logical initialization.

        The clean kernel is used by InsertGates for the noiseless initialization.
        The noisy kernel, if not None, is invoked by InsertNoise after initialization
        to apply initialization noise.

        Returns ``(None, None)`` by default, meaning no initialization is provided
        by the noise model.
        """
        return None, None

    @abc.abstractmethod
    def get_lane_noise(self, lane: LaneAddress) -> ir.Method[[qubit.Qubit], None]:
        """Return the noise kernel applied to a qubit after a lane move."""
        ...

    @abc.abstractmethod
    def get_bus_idle_noise(
        self, move_type: MoveType, bus_id: int
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any]], None]:
        """Return the noise kernel applied to stationary qubits during a move."""
        ...

    @abc.abstractmethod
    def get_cz_unpaired_noise(
        self, zone_address: ZoneAddress
    ) -> ir.Method[[ilist.IList[qubit.Qubit, Any]], None]:
        """Return the noise kernel for unpaired qubits during a CZ gate."""
        ...


@dataclass
class SimpleNoiseModel(NoiseModelABC):
    """A concrete noise model that applies the same noise kernel for each operation type.

    Unlike :class:`NoiseModelABC`, which allows different noise per zone or location,
    this model uses a single kernel per operation category (e.g. one kernel for all
    lane moves, one for all CZ gates). Created by :func:`~bloqade.lanes.noise_model.generate_simple_noise_model`.
    """

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
    logical_initialize_clean: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Any]], None] | None
    ) = None
    logical_initialize_noisy: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Any]], None] | None
    ) = None

    def get_logical_initialize(self):
        return self.logical_initialize_clean, self.logical_initialize_noisy

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
    initialize_noise_kernel: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Any]], None] | None
    ) = None

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if (trait := node.get_trait(move.EmitsState)) is None:
            return rewrite_abc.RewriteResult()

        if not isinstance(
            state_after := self.atom_state_map.get(trait.get_state_result(node)),
            atom.AtomState,
        ):
            return rewrite_abc.RewriteResult()

        return self.rewriter(node, state_after)

    @singledispatchmethod
    def rewriter(self, node: ir.Statement, state_after: atom.AtomState):
        return rewrite_abc.RewriteResult()

    @rewriter.register(move.Move)
    def _(
        self, node: move.Move, state_after: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        if len(node.lanes) == 0:
            return rewrite_abc.RewriteResult()

        move_noise_methods = tuple(map(self.noise_model.get_lane_noise, node.lanes))
        qubit_ssas = self.get_qubit_ssa_from_locations(
            state_after,
            tuple(self.arch_spec.get_endpoints(lane)[1] for lane in node.lanes),
        )
        qubit_ssas = tuple(filter(None, qubit_ssas))

        qubit_ssa_set = set(qubit_ssas)

        def is_stationary(qubit_ssa: ir.SSAValue) -> bool:
            return qubit_ssa not in qubit_ssa_set

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

    @rewriter.register(move.PhysicalInitialize)
    def _(
        self, node: move.PhysicalInitialize, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        if self.initialize_noise_kernel is None:
            return rewrite_abc.RewriteResult()

        nodes_to_insert: list[ir.Statement] = [tau := py.Constant(math.tau)]
        for theta, phi, lam, locations in zip(
            node.thetas, node.phis, node.lams, node.location_addresses
        ):
            qubit_ssa = self.get_qubit_ssa_from_locations(atom_state, locations)
            if not utils.no_none_elements_tuple(qubit_ssa):
                return rewrite_abc.RewriteResult()

            nodes_to_insert.append(theta_rad := py.Mult(tau.result, theta))
            nodes_to_insert.append(phi_rad := py.Mult(tau.result, phi))
            nodes_to_insert.append(lam_rad := py.Mult(tau.result, lam))
            nodes_to_insert.append(reg_stmt := ilist.New(qubit_ssa))
            inputs = (
                theta_rad.result,
                phi_rad.result,
                lam_rad.result,
                reg_stmt.result,
            )
            nodes_to_insert.append(
                func.Invoke(inputs, callee=self.initialize_noise_kernel)
            )

        for n in reversed(nodes_to_insert):
            n.insert_after(node)

        return rewrite_abc.RewriteResult(has_done_something=True)

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
