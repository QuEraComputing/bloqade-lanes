from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import cache, cached_property
from typing import (
    TYPE_CHECKING,
    Generic,
    Literal,
    TypeVar,
    overload,
)

from bloqade.analysis.address import AddressAnalysis, AddressQubit
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from bloqade.rewrite.passes import AggressiveUnroll
from kirin import ir, passes
from kirin.dialects import func, ilist, py

from bloqade import qubit

from ._task_runtime import DetectorResult, Result, _SimulatorTaskBase
from .simulator_backend import AbstractSimulatorBackend, TsimSimulatorBackend

if TYPE_CHECKING:
    import tsim as tsim_backend  # type: ignore[reportMissingImports]

    from bloqade.lanes.analysis import atom
    from bloqade.lanes.analysis.placement import PlacementStrategyABC
    from bloqade.lanes.arch.spec import ArchSpec
    from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC

RetType = TypeVar("RetType")
PhysicalResult = Result
_S = TypeVar("_S", bound=ir.Statement)


def _default_noise_model() -> "NoiseModelABC":
    from bloqade.lanes.noise_model import generate_simple_noise_model

    return generate_simple_noise_model()


def _default_arch_spec() -> "ArchSpec":
    from bloqade.lanes.arch.gemini.physical import get_arch_spec

    return get_arch_spec()


def _find_qubit_ssas(mt: ir.Method) -> list[ir.SSAValue]:
    """Return one physical-qubit SSA value per concrete qubit address."""
    address_analysis = AddressAnalysis(mt.dialects)
    frame, _ = address_analysis.run(mt)
    qubits_by_address: dict[int, ir.SSAValue] = {}

    for stmt in mt.callable_region.walk():
        for result in stmt.results:
            address = frame.get(result)
            if isinstance(address, AddressQubit):
                qubits_by_address.setdefault(address.data, result)

    return [
        qubits_by_address[address] for address in range(address_analysis.qubit_count)
    ]


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
class PhysicalSimulatorTask(_SimulatorTaskBase[RetType], Generic[RetType]):
    """A compiled simulation task for physical SQuIn programs."""

    source_squin_kernel: ir.Method[[], RetType]
    """The input physical SQuIn kernel."""
    noise_model: NoiseModelABC
    """The physical noise model to insert into the SQuIn kernel."""
    physical_arch_spec: ArchSpec = field(repr=False)
    """The physical architecture specification."""
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    """The physical move kernel compiled from the source SQuIn kernel."""
    _post_processing: atom.PostProcessing[RetType] = field(repr=False)
    """The post-processing object for extracting detectors, observables, and return values."""
    simulator: GeminiPhysicalSimulator = field(repr=False)
    """The simulator configuration that created this task."""

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel with noise channels."""
        from bloqade.lanes.transform import MoveToSquinPhysical

        return MoveToSquinPhysical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
        ).emit(self.physical_move_kernel)

    @cached_property
    def noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel without noise channels."""
        from bloqade.lanes.transform import MoveToSquinPhysical

        return MoveToSquinPhysical(
            arch_spec=self.physical_arch_spec,
        ).emit(self.physical_move_kernel)


@dataclass
class GeminiPhysicalSimulator:
    """Simulator for programs written directly at the physical SQuIn level."""

    noise_model: NoiseModelABC = field(default_factory=_default_noise_model)
    """The physical noise model used for simulation."""
    arch_spec: ArchSpec = field(default_factory=_default_arch_spec)
    """The physical architecture specification used for compilation."""
    backend: AbstractSimulatorBackend = field(default_factory=TsimSimulatorBackend)
    """Sampling and detector-model backend used by created tasks."""
    place_opt_type: type[passes.Pass] | None = None
    """Optional placement pass used for compilation."""
    placement_strategy: PlacementStrategyABC | None = None
    """Optional physical placement strategy."""
    m2dets: list[list[int]] | None = None
    """Optional measurement-to-detector matrix."""
    m2obs: list[list[int]] | None = None
    """Optional measurement-to-observable matrix."""

    def task(
        self,
        physical_kernel: ir.Method[[], RetType],
    ) -> PhysicalSimulatorTask[RetType]:
        """Compile a physical SQuIn kernel into a reusable simulation task."""
        from bloqade.lanes.analysis import atom
        from bloqade.lanes.passes import SequentialPlacePass
        from bloqade.lanes.pipeline import PhysicalPipeline

        # Annotation, unrolling, and physical compilation all mutate their
        # input. Keep the method supplied by the caller reusable.
        source_squin_kernel = physical_kernel.similar()

        if self.m2dets is not None or self.m2obs is not None:
            append_measurements_and_annotations_physical(
                source_squin_kernel, self.m2dets, self.m2obs
            )

        place_opt_type = self.place_opt_type or SequentialPlacePass

        physical_pipeline = PhysicalPipeline(
            arch_spec=self.arch_spec,
            place_opt_type=place_opt_type,
            placement_strategy=self.placement_strategy,
        )
        physical_move_kernel = physical_pipeline.emit(
            source_squin_kernel, no_raise=False
        )
        post_processing = atom.AtomInterpreter(
            physical_move_kernel.dialects, arch_spec=self.arch_spec
        ).get_post_processing(physical_move_kernel)

        return PhysicalSimulatorTask(
            source_squin_kernel,
            self.noise_model,
            self.arch_spec,
            physical_move_kernel,
            post_processing,
            self,
        )

    @overload
    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
        seed: int | None = None,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
        seed: int | None = None,
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult:
        """Compile and run a physical SQuIn kernel."""
        return self.task(physical_kernel).run(
            shots, with_noise, run_detectors=run_detectors, seed=seed
        )

    @overload
    def run_async(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
        seed: int | None = None,
    ) -> Future[Result[RetType]]: ...

    @overload
    def run_async(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
        seed: int | None = None,
    ) -> Future[DetectorResult]: ...

    @overload
    def run_async(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
        seed: int | None = None,
    ) -> Future[Result[RetType]] | Future[DetectorResult]: ...

    def run_async(
        self,
        physical_kernel: ir.Method[[], RetType],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        """Compile and run a physical SQuIn kernel asynchronously."""
        task = self.task(physical_kernel)
        if run_detectors:
            return task.run_async(shots, with_noise, run_detectors=True, seed=seed)
        return task.run_async(shots, with_noise, run_detectors=False, seed=seed)

    def visualize(
        self,
        physical_kernel: ir.Method[[], RetType],
        animated: bool = False,
        interactive: bool = True,
    ):
        """Visualize the compiled physical move kernel."""
        self.task(physical_kernel).visualize(animated=animated, interactive=interactive)

    def physical_squin_kernel(
        self, physical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the source physical SQuIn kernel to a noisy physical SQuIn kernel."""
        return self.task(physical_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, physical_kernel: ir.Method[[], RetType]
    ) -> ir.Method[[], RetType]:
        """Compile the source physical SQuIn kernel to the physical move dialect."""
        return self.task(physical_kernel).physical_move_kernel

    def tsim_circuit(
        self, physical_kernel: ir.Method[[], RetType], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        """Compile the physical SQuIn kernel to a tsim circuit."""
        task = self.task(physical_kernel)
        if with_noise:
            return task.tsim_circuit
        return task.noiseless_tsim_circuit

    def fidelity_bounds(
        self, physical_kernel: ir.Method[[], RetType]
    ) -> tuple[float, float]:
        """Get the fidelity bounds for the physical SQuIn kernel."""
        return self.task(physical_kernel).fidelity_bounds()


PhysicalSimulator = GeminiPhysicalSimulator
