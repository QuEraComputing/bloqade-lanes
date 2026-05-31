from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Generic, Literal, TypeVar, cast, overload

import numpy as np
import stim
import tsim as tsim_backend

from bloqade.gemini.device import DetectorResult, GeminiLogicalSimulatorTask, Result

RetType = TypeVar("RetType")

if TYPE_CHECKING:
    from clifft import Program, SampleResult


# TODO: ideally, not sure if we even want this ObservableFrame class.
class _ObservableFrame(str, Enum):
    """Observable-frame normalization modes for demo tasks."""

    RAW = "raw"
    NOISELESS_REFERENCE_FLIPS = "noiseless_reference_flips"


def _clifft_compatible_stim_text(circuit: tsim_backend.Circuit) -> str:
    """Return Stim text with instruction tags stripped for CliffT parsing."""

    # CliffT currently rejects Stim instruction tags like I_ERROR[loss](0).
    # The tags are metadata, so stripping them preserves the sampled semantics.
    return "\n".join(
        re.sub(r"^([A-Z][A-Z0-9_]*)(\[[^\]\n]+\])", r"\1", line)
        for line in str(circuit).splitlines()
    )


@dataclass
class DemoTask(Generic[RetType]):
    """Task wrapper adding observable-frame metadata and CliffT sampling.

    Args:
        task: Underlying Gemini logical simulator task.
        observable_frame: Observable normalization mode applied by sampling
            helpers.
        observable_reference: Optional cached noiseless observable reference
            used for rebasing special tasks.
        metadata: Implementation metadata used by special task construction.
    """

    task: GeminiLogicalSimulatorTask[RetType]
    observable_frame: _ObservableFrame = _ObservableFrame.RAW
    observable_reference: np.ndarray | None = None
    # TODO: this is SOLELY to pass down the "special" logical kernel for the "prefix_prepare" path.
    metadata: dict[str, object] = field(default_factory=dict)

    def __getattr__(self, name: str) -> object:
        return getattr(self.task, name)

    @property
    def detector_error_model(self) -> stim.DetectorErrorModel:
        return self.task.detector_error_model

    @cached_property
    def clifft_tsim_program(self) -> Program:
        import clifft

        return clifft.compile(_clifft_compatible_stim_text(self.task.tsim_circuit))

    @cached_property
    def clifft_noiseless_tsim_program(self) -> Program:
        import clifft

        return clifft.compile(
            _clifft_compatible_stim_text(self.task.noiseless_tsim_circuit)
        )

    @overload
    def _run_clifft(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
        seed: int | None = None,
    ) -> Result[RetType]: ...

    @overload
    def _run_clifft(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
        seed: int | None = None,
    ) -> DetectorResult: ...

    @overload
    def _run_clifft(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult: ...

    # TODO: check if _run_clifft() is ever called with run()... because I think we might be calling sample_clifft_det_obs always?
    def _run_clifft(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult:
        sample_result = self._sample_clifft(shots, with_noise=with_noise, seed=seed)

        fidelity_min, fidelity_max = self.task.fidelity_bounds()
        if run_detectors:
            return DetectorResult(
                _detector_error_model=self.task.detector_error_model,
                _fidelity_min=fidelity_min,
                _fidelity_max=fidelity_max,
                _detectors=np.asarray(sample_result.detectors, dtype=bool).tolist(),
                _observables=np.asarray(sample_result.observables, dtype=bool).tolist(),
            )

        return Result(
            np.asarray(sample_result.measurements, dtype=bool).tolist(),
            self.task.detector_error_model,
            self.task._post_processing,
            fidelity_min,
            fidelity_max,
        )

    # TODO: not sure if I like this overload logic, but it mirrors GeminiLogicalSimulatorTask; do we need it/can we get rid of it?
    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[False] = ...,
        sim_type: str = "tsim",
        seed: int | None = None,
    ) -> Result[RetType]: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: Literal[True],
        sim_type: str = "tsim",
        seed: int | None = None,
    ) -> DetectorResult: ...

    @overload
    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        sim_type: str = "tsim",
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult: ...

    def run(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        sim_type: str = "tsim",
        seed: int | None = None,
    ) -> Result[RetType] | DetectorResult:
        if sim_type == "tsim":
            return self.task.run(
                shots,
                with_noise=with_noise,
                run_detectors=run_detectors,
            )
        if sim_type != "clifft":
            raise ValueError(
                f"sim_type is {sim_type}; currently, the only supported simulator "
                "backends are 'tsim' and 'clifft'"
            )
        return self._run_clifft(
            shots,
            with_noise,
            run_detectors=run_detectors,
            seed=seed,
        )

    def _sample_clifft(
        self,
        shots: int,
        *,
        with_noise: bool = True,
        seed: int | None = None,
    ) -> SampleResult:
        import clifft

        program = (
            self.clifft_tsim_program
            if with_noise
            else self.clifft_noiseless_tsim_program
        )
        sample_kwargs: dict[str, int] = {"shots": int(shots)}
        if seed is not None:
            sample_kwargs["seed"] = int(seed)
        return cast("SampleResult", clifft.sample(program, **sample_kwargs))

    def sample_clifft_det_obs(
        self,
        shots: int,
        *,
        with_noise: bool = True,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_result = self._sample_clifft(shots, with_noise=with_noise, seed=seed)
        return (
            np.asarray(sample_result.detectors, dtype=np.uint8),
            np.asarray(sample_result.observables, dtype=np.uint8),
        )
