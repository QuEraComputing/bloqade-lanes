from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable
from weakref import WeakKeyDictionary

import numpy as np
from kirin import ir, rewrite
from kirin.rewrite.abc import RewriteResult, RewriteRule

if TYPE_CHECKING:
    from stim import DetectorErrorModel
    from tsim import Circuit as TsimCircuit  # type: ignore[reportMissingImports]


@dataclass(frozen=True)
class BackendSample:
    """Raw samples returned by a simulator backend.

    A backend may return measurements, or it may return detector and observable
    arrays directly. Task execution validates and normalizes the selected form.
    """

    measurements: np.ndarray | None = None
    detectors: np.ndarray | None = None
    observables: np.ndarray | None = None


class AbstractSimulatorBackend(abc.ABC):
    """Sampling and detector-model contract for Gemini simulators."""

    @abc.abstractmethod
    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> BackendSample:
        """Sample a compiled physical SQuIn kernel."""

    @abc.abstractmethod
    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        """Build the detector error model for a physical SQuIn kernel."""


@runtime_checkable
class _TsimCircuitCapability(Protocol):
    """Structural capability for backends that can provide a Tsim circuit."""

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit: ...


def _get_tsim_circuit(
    backend: AbstractSimulatorBackend, physical_squin_kernel: ir.Method
) -> TsimCircuit:
    if not isinstance(backend, _TsimCircuitCapability):
        raise TypeError(
            f"{type(backend).__name__} does not provide Tsim circuit compatibility"
        )
    return backend._tsim_circuit(physical_squin_kernel)


def _tsim() -> Any:
    try:
        from bloqade import tsim
    except ImportError as exc:
        raise ImportError(
            "Tsim simulation requires the optional `sim` extra. "
            "Install it with `bloqade-lanes[sim]` or `uv sync --extra sim`."
        ) from exc

    return tsim


@dataclass
class TsimSimulatorBackend(AbstractSimulatorBackend):
    """Backend that uses Tsim for both sampling and DEM generation."""

    _circuits: WeakKeyDictionary[ir.Method, TsimCircuit] = field(
        default_factory=WeakKeyDictionary, init=False, repr=False
    )

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        """Convert a physical SQuIn kernel to a cached Tsim circuit."""
        from bloqade.lanes.rewrite.squin2stim import RemoveReturn

        try:
            return self._circuits[physical_squin_kernel]
        except KeyError:
            pass

        owned_kernel = physical_squin_kernel.similar()
        rewrite.Walk(RemoveReturn()).rewrite(owned_kernel.code)
        circuit = _tsim().Circuit(owned_kernel)
        self._circuits[physical_squin_kernel] = circuit
        return circuit

    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> BackendSample:
        circuit = self._tsim_circuit(physical_squin_kernel)
        if run_detectors:
            return self._sample_detectors(circuit, shots, seed=seed)

        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler(seed=seed)
        else:
            sampler = circuit.compile_sampler(seed=seed)

        return BackendSample(
            measurements=np.asarray(sampler.sample(shots=shots), dtype=bool)
        )

    @staticmethod
    def _sample_detectors(
        circuit: TsimCircuit, shots: int, seed: int | None = None
    ) -> BackendSample:
        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler(seed=seed)
            converter = circuit.compile_m2d_converter(skip_reference_sample=True)
            measurements = sampler.sample(shots=shots)
            detectors, observables = converter.convert(
                measurements=measurements, separate_observables=True
            )
        else:
            sampler = circuit.compile_detector_sampler(seed=seed)
            detectors, observables = sampler.sample(
                shots=shots, separate_observables=True
            )

        return BackendSample(
            detectors=np.asarray(detectors, dtype=bool),
            observables=np.asarray(observables, dtype=bool),
        )

    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        return self._tsim_circuit(physical_squin_kernel).detector_error_model(
            approximate_disjoint_errors=True
        )


def _clifft_compatible_stim_text(circuit: Any) -> str:
    """Strip Stim instruction tags that CliffT does not currently parse."""

    return "\n".join(
        re.sub(r"^([A-Z][A-Z0-9_]*)(\[[^\]\n]+\])", r"\1", line)
        for line in str(circuit).splitlines()
    )


def _clifft() -> Any:
    try:
        import clifft  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "CliffT simulation requires the optional `clifft` dependency. "
            "Install it with `bloqade-lanes[msd-reprod]` or include `clifft` "
            "in your environment."
        ) from exc

    return clifft


def _clifft_tsim_import_error(exc: ImportError) -> ImportError:
    return ImportError(
        "CliffT simulation also requires `bloqade-lanes[sim]`: Tsim performs "
        "physical SQuIn conversion and detector error model generation even "
        "when CliffT is selected for sampling."
    )


@dataclass
class CliffTSimulatorBackend(AbstractSimulatorBackend):
    """Backend using Tsim for conversion/DEM generation and CliffT for sampling."""

    seed: int | None = None
    _tsim_backend: TsimSimulatorBackend = field(
        default_factory=TsimSimulatorBackend, repr=False
    )
    _programs: WeakKeyDictionary[ir.Method, Any] = field(
        default_factory=WeakKeyDictionary, init=False, repr=False
    )

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        try:
            return self._tsim_backend._tsim_circuit(physical_squin_kernel)
        except ImportError as exc:
            raise _clifft_tsim_import_error(exc) from exc

    def _clifft_program(self, physical_squin_kernel: ir.Method) -> Any:
        try:
            return self._programs[physical_squin_kernel]
        except KeyError:
            pass

        program = _clifft().compile(
            _clifft_compatible_stim_text(self._tsim_circuit(physical_squin_kernel))
        )
        self._programs[physical_squin_kernel] = program
        return program

    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> BackendSample:
        sample_kwargs: dict[str, int] = {"shots": int(shots)}
        effective_seed = self.seed if seed is None else seed
        if effective_seed is not None:
            sample_kwargs["seed"] = effective_seed

        sample_result = _clifft().sample(
            self._clifft_program(physical_squin_kernel), **sample_kwargs
        )
        if run_detectors:
            return BackendSample(
                detectors=np.asarray(sample_result.detectors, dtype=bool),
                observables=np.asarray(sample_result.observables, dtype=bool),
            )
        return BackendSample(
            measurements=np.asarray(sample_result.measurements, dtype=bool)
        )

    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        try:
            return self._tsim_backend._detector_error_model(physical_squin_kernel)
        except ImportError as exc:
            raise _clifft_tsim_import_error(exc) from exc


def _pyqrack_tsim_import_error(exc: ImportError) -> ImportError:
    return ImportError(
        "PyQrack simulation also requires `bloqade-lanes[sim]`: Tsim provides "
        "the guaranteed detector error model even when PyQrack is selected "
        "for sampling."
    )


class _RemovePyQrackAnnotations(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        from bloqade.decoders.dialects.annotate.stmts import (
            SetDetector,
            SetObservable,
        )

        if isinstance(node, (SetDetector, SetObservable)):
            node.delete()
            return RewriteResult(has_done_something=True)
        return RewriteResult()


@dataclass
class PyQrackSimulatorBackend(AbstractSimulatorBackend):
    """Backend using PyQrack for sampling and Tsim for guaranteed DEM generation."""

    seed: int | None = None
    options: dict[str, Any] | None = None
    min_qubits: int = 0
    _tsim_backend: TsimSimulatorBackend = field(
        default_factory=TsimSimulatorBackend, repr=False
    )
    _rng_state: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng_state = np.random.default_rng(self.seed)

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        try:
            return self._tsim_backend._tsim_circuit(physical_squin_kernel)
        except ImportError as exc:
            raise _pyqrack_tsim_import_error(exc) from exc

    def sample(
        self,
        physical_squin_kernel: ir.Method,
        *,
        shots: int,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> BackendSample:
        from bloqade.pyqrack.base import PyQrackInterpreter
        from bloqade.pyqrack.device import StackMemorySimulator
        from bloqade.pyqrack.reg import MeasurementResultValue
        from bloqade.pyqrack.task import PyQrackSimulatorTask

        rng_state = np.random.default_rng(seed) if seed is not None else self._rng_state

        class _RecordingPyQrackInterpreter(PyQrackInterpreter):
            measurements: list[MeasurementResultValue]

            def initialize(self):
                self.measurements = []
                # This resets memory and constructs the fresh QrackSimulator.
                initialized = super().initialize()

                # Give each one-shot Qrack simulator a distinct deterministic seed.
                qrack_seed = int(initialized.rng_state.integers(0, 2**63))
                initialized.memory.sim_reg.seed(qrack_seed)

                return initialized

            def set_global_measurement_id(self, m):
                super().set_global_measurement_id(m)
                self.measurements.append(m)

        class _RecordingStackMemorySimulator(StackMemorySimulator):
            def new_task(self, mt, args, kwargs, memory):
                interpreter = _RecordingPyQrackInterpreter(
                    mt.dialects,
                    memory=memory,
                    rng_state=self.rng_state,
                    loss_m_result=self.loss_m_result,
                )
                return PyQrackSimulatorTask(
                    kernel=mt,
                    args=args,
                    kwargs=kwargs,
                    pyqrack_interp=interpreter,
                )

        # Annotation removal mutates its input, so prepare one owned kernel for
        # this sampling request and reuse it only to construct fresh shot tasks.
        owned_kernel = physical_squin_kernel.similar()
        rewrite.Walk(_RemovePyQrackAnnotations()).rewrite(owned_kernel.code)
        simulator = _RecordingStackMemorySimulator(
            options=cast(Any, self.options or {}),
            rng_state=rng_state,
            min_qubits=self.min_qubits,
        )
        recorded_measurements: list[list[bool]] = []
        for _ in range(shots):
            task = simulator.task(owned_kernel)
            task.run()
            interpreter = cast(_RecordingPyQrackInterpreter, task.pyqrack_interp)
            recorded_measurements.append(
                [
                    self._measurement_to_bool(measurement, MeasurementResultValue)
                    for measurement in interpreter.measurements
                ]
            )

        return BackendSample(measurements=np.asarray(recorded_measurements, dtype=bool))

    def _detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        try:
            return self._tsim_backend._detector_error_model(physical_squin_kernel)
        except ImportError as exc:
            raise _pyqrack_tsim_import_error(exc) from exc

    @staticmethod
    def _measurement_to_bool(measurement: Any, measurement_result_value: Any) -> bool:
        if measurement == measurement_result_value.Zero:
            return False
        if measurement == measurement_result_value.One:
            return True
        raise ValueError(f"Unsupported PyQrack measurement result: {measurement!r}")
