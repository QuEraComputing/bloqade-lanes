from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, TypeVar

from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from bloqade.rewrite.passes import AggressiveUnroll
from kirin import ir, passes, types
from kirin.dialects import func, ilist, py

from bloqade import qubit, types as bloqade_types

from .abstract_simulator import (
    AbstractSimulator,
    CliffTSimulatorTask,
    Result,
    TsimSimulatorTask,
)

if TYPE_CHECKING:
    from bloqade.lanes.analysis import atom
    from bloqade.lanes.analysis.placement import PlacementStrategyABC
    from bloqade.lanes.arch.spec import ArchSpec

RetType = TypeVar("RetType")
PhysicalResult = Result
_S = TypeVar("_S", bound=ir.Statement)


def _default_arch_spec() -> "ArchSpec":
    from bloqade.lanes.arch.gemini.physical import get_arch_spec

    return get_arch_spec()


def _find_qubit_ssas(mt: ir.Method) -> list[ir.SSAValue]:
    """Walk the IR and collect SSA values for physical qubit allocations."""
    qubits: list[ir.SSAValue] = []
    for stmt in mt.callable_region.walk():
        for result in stmt.results:
            if result.type.is_subseteq(
                bloqade_types.QubitType
            ) and not result.type.is_subseteq(types.Bottom):
                qubits.append(result)
    return qubits


def _find_return_stmt(mt: ir.Method) -> func.Return:
    """Find the final return statement in a single-block kernel."""
    block = mt.callable_region.blocks[0]
    last = block.last_stmt
    assert isinstance(last, func.Return), f"Expected func.Return, got {type(last)}"
    return last


def _insert_before(stmt: _S, anchor: ir.Statement) -> _S:
    """Insert stmt before anchor and return stmt for chaining."""
    stmt.insert_before(anchor)
    return stmt


def _validate_m2_matrix(matrix: list[list[int]], name: str) -> int:
    if len(matrix) == 0:
        raise ValueError(f"{name} must have at least one row")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError(f"{name} rows must all have the same length")
    return len(matrix)


def append_measurements_and_annotations_physical(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
) -> None:
    """Append physical detector/observable annotations from per-block matrices.

    The method is mutated in-place. The supplied matrices are interpreted as
    measurement-to-detector/observable maps for one logical block laid out in a
    flat physical measurement list. If the kernel measures multiple such blocks,
    the matrices are repeated block-by-block. The original return value is
    preserved.
    """
    if m2dets is None and m2obs is None:
        raise ValueError("At least one of m2dets or m2obs must be provided")

    AggressiveUnroll(mt.dialects, no_raise=True).fixpoint(mt)

    physical_qubits_per_logical_qubit: int | None = None
    if m2dets is not None:
        physical_qubits_per_logical_qubit = _validate_m2_matrix(m2dets, "m2dets")
    if m2obs is not None:
        m2obs_rows = _validate_m2_matrix(m2obs, "m2obs")
        if (
            physical_qubits_per_logical_qubit is not None
            and physical_qubits_per_logical_qubit != m2obs_rows
        ):
            raise ValueError("m2dets and m2obs must have the same number of rows")
        physical_qubits_per_logical_qubit = m2obs_rows
    assert physical_qubits_per_logical_qubit is not None

    num_physical_qubits = len(_find_qubit_ssas(mt))
    if num_physical_qubits == 0:
        raise ValueError("No physical qubit allocations found in the kernel")

    num_logical_qubits, remainder = divmod(
        num_physical_qubits, physical_qubits_per_logical_qubit
    )
    if remainder != 0:
        raise ValueError(
            "The number of physical qubits must be divisible by the number of "
            "physical qubits per logical qubit"
        )

    measure_stmts = [
        stmt
        for stmt in mt.callable_region.walk()
        if isinstance(stmt, qubit.stmts.Measure)
    ]
    if len(measure_stmts) != 1:
        raise ValueError("Expected exactly one physical qubit.Measure statement")
    measure_stmt = measure_stmts[0]
    return_stmt = _find_return_stmt(mt)

    @cache
    def _get_physical_measurement(q_idx: int, m_idx: int) -> ir.SSAValue:
        flat_idx = q_idx * physical_qubits_per_logical_qubit + m_idx
        (idx := py.Constant(flat_idx)).insert_before(return_stmt)
        (getitem := py.GetItem(measure_stmt.result, idx.result)).insert_before(
            return_stmt
        )
        return getitem.result

    if m2dets is not None:
        for q_idx in range(num_logical_qubits):
            for j in range(len(m2dets[0])):
                meas_ssas = [
                    _get_physical_measurement(q_idx, m_idx)
                    for m_idx, row in enumerate(m2dets)
                    if row[j]
                ]
                meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)
                coord_0 = _insert_before(py.Constant(float(q_idx)), return_stmt)
                coord_1 = _insert_before(py.Constant(float(j)), return_stmt)
                coords = _insert_before(
                    ilist.New([coord_0.result, coord_1.result]), return_stmt
                )
                _insert_before(
                    SetDetector(meas_list.result, coords.result), return_stmt
                )

    if m2obs is not None:
        for q_idx in range(num_logical_qubits):
            for j in range(len(m2obs[0])):
                meas_ssas = [
                    _get_physical_measurement(q_idx, m_idx)
                    for m_idx, row in enumerate(m2obs)
                    if row[j]
                ]
                meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)
                _insert_before(SetObservable(meas_list.result), return_stmt)


@dataclass(frozen=True)
class PhysicalSimulatorTask(TsimSimulatorTask[RetType]):
    """A compiled simulation task for physical SQuIn programs."""

    @property
    def source_squin_kernel(self) -> ir.Method[[], RetType]:
        """The input physical SQuIn kernel."""
        return self.logical_squin_kernel


@dataclass(frozen=True)
class PhysicalCliffTSimulatorTask(CliffTSimulatorTask[RetType]):
    """A CliffT-backed compiled simulation task for physical SQuIn programs."""

    @property
    def source_squin_kernel(self) -> ir.Method[[], RetType]:
        """The input physical SQuIn kernel."""
        return self.logical_squin_kernel


@dataclass
class PhysicalSimulator(AbstractSimulator):
    """Simulator for programs written directly at the physical SQuIn level."""

    arch_spec: ArchSpec = field(default_factory=_default_arch_spec)
    """The physical architecture specification used for compilation."""

    def task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
        m2dets: list[list[int]] | None = None,
        m2obs: list[list[int]] | None = None,
    ) -> PhysicalSimulatorTask[RetType]:
        """Compile a physical SQuIn kernel into a reusable simulation task.

        Args:
            physical_kernel (ir.Method[[], RetType]): The physical SQuIn kernel to compile.
            place_opt_type (type[passes.Pass] | None): Optional placement pass class.
            placement_strategy (PlacementStrategyABC | None): Optional placement strategy.

        Returns:
            PhysicalSimulatorTask[RetType]: The compiled simulation task.

        """
        physical_move_kernel, post_processing = self._compile_physical_task(
            physical_kernel, place_opt_type, placement_strategy, m2dets, m2obs
        )
        return PhysicalSimulatorTask(
            physical_kernel,
            self.noise_model,
            self.arch_spec,
            physical_move_kernel,
            post_processing,
            seed=self.seed,
        )

    def _compile_physical_task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
        m2dets: list[list[int]] | None = None,
        m2obs: list[list[int]] | None = None,
    ) -> tuple[ir.Method[[], RetType], "atom.PostProcessing[RetType]"]:
        """Compile a physical SQuIn kernel and extract post-processing."""
        from bloqade.lanes.analysis import atom
        from bloqade.lanes.passes import SequentialPlacePass
        from bloqade.lanes.pipeline import PhysicalPipeline

        if m2dets is not None or m2obs is not None:
            append_measurements_and_annotations_physical(physical_kernel, m2dets, m2obs)

        if place_opt_type is None:
            place_opt_type = SequentialPlacePass

        physical_pipeline = PhysicalPipeline(
            arch_spec=self.arch_spec,
            place_opt_type=place_opt_type,
            placement_strategy=placement_strategy,
        )
        physical_move_kernel = physical_pipeline.emit(physical_kernel, no_raise=False)
        post_processing = atom.AtomInterpreter(
            physical_move_kernel.dialects, arch_spec=self.arch_spec
        ).get_post_processing(physical_move_kernel)
        return physical_move_kernel, post_processing


@dataclass
class PhysicalCliffTSimulator(AbstractSimulator):
    """CliffT-backed simulator for physical SQuIn programs."""

    arch_spec: ArchSpec = field(default_factory=_default_arch_spec)
    """The physical architecture specification used for compilation."""

    def task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
        # TODO: support m2dets, m2obs
        m2dets: list[list[int]] | None = None,
        m2obs: list[list[int]] | None = None,
    ) -> PhysicalCliffTSimulatorTask[RetType]:
        """Compile a physical SQuIn kernel into a reusable CliffT-backed task.

        Args:
            physical_kernel (ir.Method[[], RetType]): The physical SQuIn kernel to compile.
            place_opt_type (type[passes.Pass] | None): Optional placement pass class.
            placement_strategy (PlacementStrategyABC | None): Optional placement strategy.

        Returns:
            PhysicalCliffTSimulatorTask[RetType]: The compiled simulation task.

        """
        physical_move_kernel, post_processing = self._compile_physical_task(
            physical_kernel, place_opt_type, placement_strategy, m2dets, m2obs
        )
        return PhysicalCliffTSimulatorTask(
            physical_kernel,
            self.noise_model,
            self.arch_spec,
            physical_move_kernel,
            post_processing,
            seed=self.seed,
        )

    def _compile_physical_task(
        self,
        physical_kernel: ir.Method[[], RetType],
        place_opt_type: type[passes.Pass] | None = None,
        placement_strategy: "PlacementStrategyABC | None" = None,
        m2dets: list[list[int]] | None = None,
        m2obs: list[list[int]] | None = None,
    ) -> tuple[ir.Method[[], RetType], "atom.PostProcessing[RetType]"]:
        """Compile a physical kernel using the shared physical simulator helper."""
        return PhysicalSimulator(
            noise_model=self.noise_model, arch_spec=self.arch_spec
        )._compile_physical_task(
            physical_kernel, place_opt_type, placement_strategy, m2dets, m2obs
        )
