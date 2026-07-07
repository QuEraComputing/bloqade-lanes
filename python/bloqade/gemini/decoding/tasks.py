from __future__ import annotations

import re
from concurrent.futures import Future
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import numpy as np

from bloqade.gemini.device import DetectorResult, GeminiLogicalSimulatorTask, Result

if TYPE_CHECKING:
    from clifft import Program, SampleResult  # type: ignore[reportMissingImports]

RetType = TypeVar("RetType")


def _clifft_compatible_stim_text(circuit: Any) -> str:
    """Return Stim text with instruction tags stripped for CliffT parsing."""

    # CliffT currently rejects Stim instruction tags like I_ERROR[loss](0).
    # The tags are metadata, so stripping them preserves the sampled semantics.
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


# TODO: inherits from GeminiLogicalSimulatorTask for now because a lot of fields used in the experiment are basically in GeminiLogicalSimulatorTask.
# ^ these fields are primarily used in _apply_special_tsim_circuit_strategy where we need some way to modify the tsim_circuit associated w/ a task.
# ^ can think of a better design later.
@dataclass(frozen=True)
class _CliffTSimulatorTask(GeminiLogicalSimulatorTask[RetType]):
    """Gemini logical simulator task that samples through CliffT."""

    seed: int | None = None

    @cached_property
    def clifft_tsim_program(self) -> Program:
        clifft = _clifft()

        return cast(
            "Program",
            clifft.compile(_clifft_compatible_stim_text(self.tsim_circuit)),
        )

    @cached_property
    def clifft_noiseless_tsim_program(self) -> Program:
        clifft = _clifft()

        return cast(
            "Program",
            clifft.compile(_clifft_compatible_stim_text(self.noiseless_tsim_circuit)),
        )

    def _sample_clifft(
        self,
        shots: int,
        *,
        with_noise: bool = True,
    ) -> SampleResult:
        clifft = _clifft()

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

    # NOTE: defining run_async because GeminiLogicalSimulatorTask exposes this in public API.
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

        sample_result = self._sample_clifft(
            shots,
            with_noise=with_noise,
        )
        return (
            np.asarray(sample_result.detectors, dtype=np.uint8),
            np.asarray(sample_result.observables, dtype=np.uint8),
        )
