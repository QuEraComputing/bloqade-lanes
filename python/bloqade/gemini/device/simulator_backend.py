from __future__ import annotations

import abc
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

import numpy as np
from kirin import ir, rewrite

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
    ) -> BackendSample:
        """Sample a compiled physical SQuIn kernel."""

    @abc.abstractmethod
    def detector_error_model(
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
    ) -> BackendSample:
        circuit = self._tsim_circuit(physical_squin_kernel)
        if run_detectors:
            return self._sample_detectors(circuit, shots)

        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler()
        else:
            sampler = circuit.compile_sampler()

        return BackendSample(
            measurements=np.asarray(sampler.sample(shots=shots), dtype=bool)
        )

    @staticmethod
    def _sample_detectors(circuit: TsimCircuit, shots: int) -> BackendSample:
        if circuit.is_clifford:
            sampler = circuit.stim_circuit.compile_sampler()
            converter = circuit.compile_m2d_converter(skip_reference_sample=True)
            measurements = sampler.sample(shots=shots)
            detectors, observables = converter.convert(
                measurements=measurements, separate_observables=True
            )
        else:
            sampler = circuit.compile_detector_sampler()
            detectors, observables = sampler.sample(
                shots=shots, separate_observables=True
            )

        return BackendSample(
            detectors=np.asarray(detectors, dtype=bool),
            observables=np.asarray(observables, dtype=bool),
        )

    def detector_error_model(
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
    tsim_backend: TsimSimulatorBackend = field(
        default_factory=TsimSimulatorBackend, repr=False
    )
    _programs: WeakKeyDictionary[ir.Method, Any] = field(
        default_factory=WeakKeyDictionary, init=False, repr=False
    )

    def _tsim_circuit(self, physical_squin_kernel: ir.Method) -> TsimCircuit:
        try:
            return self.tsim_backend._tsim_circuit(physical_squin_kernel)
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
    ) -> BackendSample:
        sample_kwargs: dict[str, int] = {"shots": int(shots)}
        if self.seed is not None:
            sample_kwargs["seed"] = int(self.seed)

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

    def detector_error_model(
        self, physical_squin_kernel: ir.Method
    ) -> DetectorErrorModel:
        try:
            return self.tsim_backend.detector_error_model(physical_squin_kernel)
        except ImportError as exc:
            raise _clifft_tsim_import_error(exc) from exc
