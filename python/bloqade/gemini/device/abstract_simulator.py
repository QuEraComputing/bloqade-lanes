from __future__ import annotations

import abc
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

import numpy as np
from bloqade.analysis.fidelity import FidelityAnalysis
from kirin import ir, rewrite
from stim import DetectorErrorModel

if TYPE_CHECKING:
    import stim as stim_backend
    import tsim as tsim_backend  # type: ignore[reportMissingImports]
    from clifft import Program, SampleResult

    from bloqade.lanes.analysis import atom
    from bloqade.lanes.arch.spec import ArchSpec
    from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

RetType = TypeVar("RetType")
TaskRet = TypeVar("TaskRet")


def _tsim():
    try:
        from bloqade import tsim
    except ImportError as exc:
        raise ImportError(
            "Gemini simulation requires the optional `tsim` extra. "
            "Install it with `bloqade-lanes[tsim]` or `uv sync --extra tsim`."
        ) from exc

    return tsim


def _default_noise_model() -> "LogicalNoiseModelABC":
    from bloqade.lanes.noise_model import generate_logical_noise_model

    return generate_logical_noise_model()


def _clifft_compatible_stim_text(circuit: Any) -> str:
    """Return Stim text with instruction tags stripped for CliffT parsing."""
    # CliffT currently rejects Stim instruction tags like I_ERROR[loss](0).
    # The tags are metadata, so stripping them preserves the sampled semantics.
    return "\n".join(
        re.sub(r"^([A-Z][A-Z0-9_]*)(\[[^\]\n]+\])", r"\1", line)
        for line in str(circuit).splitlines()
    )


def _kernel_to_tsim_circuit(kernel: ir.Method) -> tsim_backend.Circuit:
    """Compile a SQuIn kernel copy to a tsim circuit."""
    from bloqade.lanes.rewrite.squin2stim import RemoveReturn

    mt = kernel.similar()
    rewrite.Walk(RemoveReturn()).rewrite(mt.code)
    return _tsim().Circuit(mt)


def _kernel_to_stim_circuit(kernel: ir.Method) -> stim_backend.Circuit:
    """Compile a SQuIn kernel copy to a Stim circuit."""
    import stim
    from bloqade.stim.circuit import _codegen

    from bloqade.lanes.rewrite.squin2stim import RemoveReturn

    mt = kernel.similar()
    rewrite.Walk(RemoveReturn()).rewrite(mt.code)
    stim_str = _codegen(mt)
    return stim.Circuit(stim_program_text=stim_str)


@dataclass(frozen=True)
class DetectorResult:
    """Result from the detector sampler containing only detector and observable outcomes."""

    _detector_error_model: DetectorErrorModel
    _fidelity_min: float
    _fidelity_max: float
    _detectors: list[list[bool]]
    _observables: list[list[bool]]

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the upper and lower fidelity bounds.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit.

        Returns:
            DetectorErrorModel: The STIM detector error model.

        """
        return self._detector_error_model

    @property
    def detectors(self) -> tuple[tuple[bool, ...], ...]:
        """The detector outcomes from the simulation.

        Returns:
            tuple[tuple[bool, ...], ...]: The detector outcomes, one tuple per shot.

        """
        return tuple(tuple(shot) for shot in self._detectors)

    @property
    def observables(self) -> tuple[tuple[bool, ...], ...]:
        """The observable outcomes from the simulation.

        Returns:
            tuple[tuple[bool, ...], ...]: The observable outcomes, one tuple per shot.

        """
        return tuple(tuple(shot) for shot in self._observables)


@dataclass(frozen=True)
class Result(Generic[RetType]):
    """Simulation result including measurement outcomes, detector error model, post-processing, and fidelity bounds."""

    _raw_measurements: list[list[bool]]
    _detector_error_model: DetectorErrorModel
    _post_processing: atom.PostProcessing[RetType]
    _fidelity_min: float
    _fidelity_max: float

    def fidelity_bounds(self) -> tuple[float, float]:
        """Return the upper and lower fidelity bounds.

        Note: The upper and lower bounds are related to branching logic in the kernel.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return (self._fidelity_min, self._fidelity_max)

    @property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the physical noise circuit.

        Returns:
            DetectorErrorModel: The STIM detector error model.

        """
        return self._detector_error_model

    @cached_property
    def return_values(self) -> list[RetType]:
        """The return values of the logical kernel.

        Returns:
            list[RetType]: The return values, one per shot.

        """
        return list(self._post_processing.emit_return(self._raw_measurements))

    @cached_property
    def detectors(self) -> list[list[bool]]:
        """The detector outcomes from the simulation.

        Returns:
            list[list[bool]]: The detector outcomes, one list per shot.

        """
        return list(self._post_processing.emit_detectors(self._raw_measurements))

    @cached_property
    def measurements(self) -> list[list[bool]]:
        """The raw measurement outcomes used to compute detectors and observables.

        Returns:
            list[list[bool]]: The raw measurement outcomes, one list per shot.

        """
        return list(map(list, self._raw_measurements))

    @cached_property
    def observables(self) -> list[list[bool]]:
        """The observable outcomes from the simulation.

        Returns:
            list[list[bool]]: The observable outcomes, one list per shot.

        """
        return list(self._post_processing.emit_observables(self._raw_measurements))


@dataclass(frozen=True)
class AbstractSimulatorTask(abc.ABC, Generic[RetType]):
    """A compiled simulation task.

    The squin-to-move compilation and post-processing extraction are performed
    eagerly at construction time. Simulation artifacts (physical squin kernel,
    stim circuits, samplers, detector error model) are computed lazily on first
    access since they depend on the noise model.
    """

    logical_squin_kernel: ir.Method[[], RetType]
    """The input SQuIn kernel compiled by the simulator frontend."""
    noise_model: LogicalNoiseModelABC
    """The noise model to be inserted into the physical SQuIn kernel."""
    physical_arch_spec: ArchSpec = field(repr=False)
    """The physical architecture specification."""
    physical_move_kernel: ir.Method[[], RetType] = field(repr=False)
    """The physical move kernel that executes the kernel on the physical architecture."""
    _post_processing: atom.PostProcessing[RetType] = field(repr=False)
    """The post-processing object for detectors, observables, and return values."""
    _thread_pool_executor: ThreadPoolExecutor = field(
        default_factory=ThreadPoolExecutor, init=False, repr=False
    )
    seed: int | None = None
    """Optional backend seed for task sampling."""

    @cached_property
    def _physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel with noise channels.

        The shared implementation intentionally uses ``MoveToSquinLogical``.
        Logical move programs need this pass to lower logical initialize
        statements, while physical move programs do not contain those statements
        and therefore pass through the same rewrite path without triggering the
        logical-initialize rewrite.
        """
        from bloqade.lanes.transform import MoveToSquinLogical

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=True,
        ).emit(self.physical_move_kernel)

    @property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical squin kernel with noise channels."""
        return self._physical_squin_kernel

    @cached_property
    def _noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical SQuIn kernel without noise channels.

        This follows the same shared ``MoveToSquinLogical`` path as
        :attr:`physical_squin_kernel`, but asks the rewrite to omit noise
        channel insertion.
        """
        from bloqade.lanes.transform import MoveToSquinLogical

        return MoveToSquinLogical(
            arch_spec=self.physical_arch_spec,
            noise_model=self.noise_model,
            add_noise=False,
        ).emit(self.physical_move_kernel)

    @property
    def noiseless_physical_squin_kernel(self) -> ir.Method[[], RetType]:
        """The physical squin kernel without noise channels."""
        return self._noiseless_physical_squin_kernel

    @cached_property
    def tsim_circuit(self) -> tsim_backend.Circuit:
        """The tsim circuit corresponding to the physical squin kernel."""
        return _kernel_to_tsim_circuit(self.physical_squin_kernel)

    @cached_property
    def noiseless_tsim_circuit(self) -> tsim_backend.Circuit:
        """The noiseless tsim circuit compiled without noise channels."""
        return _kernel_to_tsim_circuit(self.noiseless_physical_squin_kernel)

    @cached_property
    def stim_circuit(self) -> stim_backend.Circuit:
        """The Stim circuit corresponding to the noisy physical SQuIn kernel."""
        return _kernel_to_stim_circuit(self.physical_squin_kernel)

    @cached_property
    def noiseless_stim_circuit(self) -> stim_backend.Circuit:
        """The Stim circuit corresponding to the noiseless physical SQuIn kernel."""
        return _kernel_to_stim_circuit(self.noiseless_physical_squin_kernel)

    def visualize(self, animated: bool = False, interactive: bool = True):
        """Visualize the physical move kernel using the built-in debugger.

        Args:
            animated (bool): Whether to use the animated debugger. Defaults to False.
            interactive (bool): Whether to enable interactive mode. Defaults to True.

        """
        from bloqade.lanes.visualize import animated_debugger, debugger

        if animated:
            animated_debugger(
                self.physical_move_kernel,
                self.physical_arch_spec,
                interactive=interactive,
            )
        else:
            debugger(
                self.physical_move_kernel,
                self.physical_arch_spec,
                interactive=interactive,
            )

    def fidelity_bounds(self) -> tuple[float, float]:
        """Compute the fidelity bounds for the physical squin kernel.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        analysis = FidelityAnalysis(self.physical_squin_kernel.dialects)
        analysis.run(self.physical_squin_kernel)

        max_fidelity = 1.0
        min_fidelity = 1.0

        for gate_fid in analysis.gate_fidelities:
            min_fidelity *= gate_fid.min
            max_fidelity *= gate_fid.max

        return min_fidelity, max_fidelity

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Result[RetType] | DetectorResult: ...

    @abc.abstractmethod
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Result[RetType] | DetectorResult:
        """Run the kernel and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler for faster detector/observable sampling.
                Defaults to False.

        Returns:
            Result[RetType]: When ``run_detectors=False``, the full simulation result
                including measurement outcomes, return values, detectors, and observables.
            DetectorResult: When ``run_detectors=True``, the result containing only
                detector and observable outcomes.

        """
        ...

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[Result[RetType]]: ...

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    @abc.abstractmethod
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler. Defaults to False.

        Returns:
            Future[Result[RetType]]: When ``run_detectors=False``, a future resolving
                to the full simulation result.
            Future[DetectorResult]: When ``run_detectors=True``, a future resolving
                to the detector result.

        """
        ...


class TsimSimulatorTask(AbstractSimulatorTask[RetType]):
    """Shared tsim-backed simulator task implementation."""

    @cached_property
    def measurement_sampler(self):
        """The noisy tsim measurement sampler."""
        return self.tsim_circuit.compile_sampler()

    @cached_property
    def noiseless_measurement_sampler(self):
        """The noiseless tsim measurement sampler."""
        return self.noiseless_tsim_circuit.compile_sampler()

    @cached_property
    def detector_sampler(self):
        """The noisy tsim detector sampler."""
        return self.tsim_circuit.compile_detector_sampler()

    @cached_property
    def noiseless_detector_sampler(self):
        """The noiseless tsim detector sampler."""
        return self.noiseless_tsim_circuit.compile_detector_sampler()

    @cached_property
    def detector_error_model(self):
        """The STIM detector error model corresponding to the noisy tsim circuit."""
        return self.tsim_circuit.detector_error_model(approximate_disjoint_errors=True)

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Result[RetType] | DetectorResult:
        """Run the task and return measurement or detector samples."""
        if run_detectors:
            return self._run_detectors(shots, with_noise)

        circuit = self.tsim_circuit if with_noise else self.noiseless_tsim_circuit
        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler()
        else:
            sampler = (
                self.measurement_sampler
                if with_noise
                else self.noiseless_measurement_sampler
            )

        raw_results = sampler.sample(shots=shots).tolist()
        fidelity_min, fidelity_max = self.fidelity_bounds()
        return Result(
            raw_results,
            self.detector_error_model,
            self._post_processing,
            fidelity_min,
            fidelity_max,
        )

    def _run_detectors(self, shots: int = 1, with_noise: bool = True) -> DetectorResult:
        """Run the detector sampler for faster detector/observable sampling.

        This skips the full measurement sampler and directly samples detector
        and observable outcomes, which is significantly faster when only
        detectors and observables are needed.

        Args:
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.

        Returns:
            DetectorResult: The result containing detector and observable outcomes.

        """
        circuit = self.tsim_circuit if with_noise else self.noiseless_tsim_circuit
        if circuit.is_clifford:
            # Use Stim for the Clifford case. Since .detector_sampler uses a reference
            # sample which flips outcomes, we use the measurement sampler and convert to
            # detectors and observables while explicitly skipping the reference sample.
            sampler = circuit.stim_circuit.compile_sampler()
            m2det = circuit.compile_m2d_converter(skip_reference_sample=True)
            samples = sampler.sample(shots=shots)
            det_obs = m2det.convert(measurements=samples, separate_observables=True)
        else:
            sampler = (
                self.detector_sampler if with_noise else self.noiseless_detector_sampler
            )
            det_obs: tuple[np.ndarray, np.ndarray] = sampler.sample(
                shots=shots, separate_observables=True
            )

        fidelity_min, fidelity_max = self.fidelity_bounds()
        return DetectorResult(
            _detector_error_model=self.detector_error_model,
            _fidelity_min=fidelity_min,
            _fidelity_max=fidelity_max,
            _detectors=det_obs[0].tolist(),
            _observables=det_obs[1].tolist(),
        )

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[Result[RetType]]: ...

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        """Run the task asynchronously."""
        if run_detectors:
            return self._thread_pool_executor.submit(
                self._run_detectors, shots, with_noise
            )
        return self._thread_pool_executor.submit(self.run, shots, with_noise)


class CliffTSimulatorTask(AbstractSimulatorTask[RetType]):
    """Shared CliffT-backed simulator task implementation."""

    @cached_property
    def _stim_text(self) -> str:
        """Stim program text for the noisy circuit."""
        return str(self.stim_circuit)

    @cached_property
    def _noiseless_stim_text(self) -> str:
        """Stim program text for the noiseless circuit."""
        return str(self.noiseless_stim_circuit)

    @cached_property
    def detector_error_model(self) -> DetectorErrorModel:
        """The STIM detector error model corresponding to the noisy circuit."""
        import stim

        # TODO: redefine this function somewhere else so CliffT sim path doesn't have to import tsim?
        from tsim.circuit import (
            get_detector_error_model,  # type: ignore[reportMissingImports]
        )

        stim_circuit = stim.Circuit(self._stim_text)
        return get_detector_error_model(
            stim_circuit,
            allow_non_deterministic_observables=True,
            approximate_disjoint_errors=True,
        )

    @cached_property
    def clifft_tsim_program(self) -> Program:
        """The noisy CliffT program.

        CliffT consumes tsim shorthand for non-Clifford instructions such as
        ``U3``. The cached Stim text remains Stim-compatible for DEM
        construction, and this property converts back to tsim shorthand only for
        CliffT compilation.
        """
        import clifft

        # TODO: redefine this function somewhere else so CliffT sim path doesn't have to import tsim?
        from tsim.utils.program_text import (
            stim_to_shorthand,  # type: ignore[reportMissingImports]
        )

        return clifft.compile(
            _clifft_compatible_stim_text(stim_to_shorthand(self._stim_text))
        )

    @cached_property
    def clifft_noiseless_tsim_program(self) -> Program:
        """The noiseless CliffT program."""
        import clifft

        # TODO: redefine this function somewhere else so CliffT sim path doesn't have to import tsim?
        from tsim.utils.program_text import (
            stim_to_shorthand,  # type: ignore[reportMissingImports]
        )

        return clifft.compile(
            _clifft_compatible_stim_text(stim_to_shorthand(self._noiseless_stim_text))
        )

    def _sample_clifft(
        self,
        shots: int,
        *,
        with_noise: bool = True,
    ) -> SampleResult:
        import clifft

        # TODO: check if _run_clifft() is ever called with run()... because I think we might be calling sample_clifft_det_obs always?
        program = (
            self.clifft_tsim_program
            if with_noise
            else self.clifft_noiseless_tsim_program
        )
        sample_kwargs: dict[str, int] = {"shots": int(shots)}
        if self.seed is not None:
            sample_kwargs["seed"] = int(self.seed)
        return cast("SampleResult", clifft.sample(program, **sample_kwargs))

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Result[RetType] | DetectorResult:
        """Sample detector and observable arrays using CliffT."""
        sample_result = self._sample_clifft(shots, with_noise=with_noise)
        # NOTE: this should be a no-op if detectors/observables are already of the right type
        detectors = np.asarray(sample_result.detectors, dtype=np.uint8)
        observables = np.asarray(sample_result.observables, dtype=np.uint8)
        fidelity_min, fidelity_max = self.fidelity_bounds()
        if run_detectors:
            return DetectorResult(
                _detector_error_model=self.detector_error_model,
                _fidelity_min=fidelity_min,
                _fidelity_max=fidelity_max,
                _detectors=detectors.astype(bool).tolist(),
                _observables=observables.astype(bool).tolist(),
            )

        # TODO: should GeminiLogicalSimulatorTask.run expose NumPy arrays instead
        # of list-backed Result/DetectorResult objects? CliffT natively returns
        # measurement, detector, and observable arrays.
        return Result(
            _raw_measurements=np.asarray(sample_result.measurements, dtype=np.uint8)
            .astype(bool)
            .tolist(),
            _detector_error_model=self.detector_error_model,
            _post_processing=self._post_processing,
            _fidelity_min=fidelity_min,
            _fidelity_max=fidelity_max,
        )

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[Result[RetType]]: ...

    @overload
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    # NOTE: defining run_async because AbstractSimulator exposes this in public API.
    # overriding because run_async in GeminiLogicalSimulatorTask depends on a private method which we don't need to add to _CliffTSimulatorTask.
    def run_async(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[Result[RetType]] | Future[DetectorResult]:
        """Run the CliffT sampler asynchronously."""

        # TODO: should GeminiLogicalSimulatorTask.run_async preserve NumPy-array
        # results for detector-heavy workflows instead of wrapping list-backed
        # Result/DetectorResult containers?
        # ^ We can't return numpy arrays natively because they aren't a Kirin type, I believe
        if run_detectors:
            return cast(
                Future[DetectorResult],
                self._thread_pool_executor.submit(
                    self.run,
                    shots,
                    with_noise,
                    run_detectors=True,
                ),
            )
        return cast(
            Future[Result[RetType]],
            self._thread_pool_executor.submit(
                self.run,
                shots,
                with_noise,
                run_detectors=False,
            ),
        )

    # NOTE: this function is unused, but can be used if we want to get the detectors/observables directly without applying the self._post_processing
    # function; based on the annotated detectors/observables in the CliffT circuit.
    # QUESTION: I don't know what the point of _post_processing is for the simulator if we already annotated our circuit with detectors/observables..
    # why can't we just get that directly
    def _sample_detector_observables(
        self,
        shots: int,
        *,
        with_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample detector and observable arrays."""
        sample_result = self._sample_clifft(shots, with_noise=with_noise)
        return (
            np.asarray(sample_result.detectors, dtype=np.uint8),
            np.asarray(sample_result.observables, dtype=np.uint8),
        )


@dataclass
class AbstractSimulator(abc.ABC):
    """Shared convenience API for simulator frontends."""

    noise_model: LogicalNoiseModelABC = field(default_factory=_default_noise_model)
    """The noise model used for simulation."""
    seed: int | None = None
    """Optional backend seed for task sampling."""

    # TODO: rename "kernel" arg in "task" function to something that both logical and physical simulators can agree on? for arg names
    # ^ ideally, in a "nonbreaking" fashion
    @abc.abstractmethod
    def task(
        self,
        kernel: ir.Method[[], TaskRet],
        *args: Any,
        **kwargs: Any,
    ) -> AbstractSimulatorTask[TaskRet]:
        """Compile a kernel into a reusable simulator task."""
        ...

    @overload
    def run(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Result[TaskRet]: ...

    @overload
    def run(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Result[TaskRet] | DetectorResult: ...

    def run(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Result[TaskRet] | DetectorResult:
        """Run the kernel and get simulation results.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to run.
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler for faster detector/observable sampling.
                Defaults to False.

        Returns:
            Result[RetType]: When ``run_detectors=False``, the full simulation result.
            DetectorResult: When ``run_detectors=True``, the result containing only
                detector and observable outcomes.

        """
        return self.task(logical_squin_kernel).run(
            shots, with_noise, run_detectors=run_detectors
        )

    @overload
    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
    ) -> Future[Result[TaskRet]]: ...

    @overload
    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
    ) -> Future[DetectorResult]: ...

    @overload
    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool,
    ) -> Future[Result[TaskRet]] | Future[DetectorResult]: ...

    def run_async(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
    ) -> Future[Result[TaskRet]] | Future[DetectorResult]:
        """Run the kernel asynchronously and get simulation results.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to run.
            shots (int): Number of shots to run. Defaults to 1.
            with_noise (bool): Whether to include noise in the simulation. Defaults to True.
            run_detectors (bool): When ``True``, use the detector sampler instead of
                the measurement sampler. Defaults to False.

        Returns:
            Future[Result[RetType]]: When ``run_detectors=False``, a future resolving
                to the full simulation result.
            Future[DetectorResult]: When ``run_detectors=True``, a future resolving
                to the detector result.

        """
        task = self.task(logical_squin_kernel)
        if run_detectors:
            return task.run_async(shots, with_noise, run_detectors=True)
        return task.run_async(shots, with_noise)

    def tsim_circuit(
        self, logical_squin_kernel: ir.Method[[], TaskRet], with_noise: bool = True
    ) -> tsim_backend.Circuit:
        """Compile the logical squin kernel to the tsim circuit.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.
            with_noise (bool): Whether to include noise in the tsim circuit. Defaults to True.

        Returns:
            tsim.Circuit: The compiled tsim circuit.

        """
        task = self.task(logical_squin_kernel)
        if with_noise:
            return task.tsim_circuit
        return task.noiseless_tsim_circuit

    def visualize(
        self,
        logical_squin_kernel: ir.Method[[], TaskRet],
        animated: bool = False,
        interactive: bool = True,
    ):
        """Visualize the physical move kernel using the built-in debugger.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to visualize.
            animated (bool): Whether to use the animated debugger. Defaults to False.
            interactive (bool): Whether to enable interactive mode. Defaults to True.

        """
        self.task(logical_squin_kernel).visualize(
            animated=animated, interactive=interactive
        )

    def physical_squin_kernel(
        self, logical_squin_kernel: ir.Method[[], TaskRet]
    ) -> ir.Method[[], TaskRet]:
        """Compile the logical squin kernel to the physical squin kernel.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.

        Returns:
            ir.Method[[], RetType]: The physical squin kernel.

        """
        return self.task(logical_squin_kernel).physical_squin_kernel

    def physical_move_kernel(
        self, logical_squin_kernel: ir.Method[[], TaskRet]
    ) -> ir.Method[[], TaskRet]:
        """Compile the logical squin kernel to the physical move kernel.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to compile.

        Returns:
            ir.Method[[], RetType]: The physical move kernel.

        """
        return self.task(logical_squin_kernel).physical_move_kernel

    def fidelity_bounds(
        self, logical_squin_kernel: ir.Method[[], TaskRet]
    ) -> tuple[float, float]:
        """Get the fidelity bounds for the logical squin kernel.

        Args:
            logical_squin_kernel (ir.Method[[], RetType]): The logical squin kernel to analyze.

        Returns:
            tuple[float, float]: The (min, max) fidelity bounds.

        """
        return self.task(logical_squin_kernel).fidelity_bounds()
