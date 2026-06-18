from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, cast

import numpy as np
import stim
import tsim as tsim_backend

from bloqade.gemini.device import GeminiLogicalSimulatorTask

if TYPE_CHECKING:
    from clifft import Program, SampleResult


def _clifft_compatible_stim_text(circuit: tsim_backend.Circuit) -> str:
    """Return Stim text with instruction tags stripped for CliffT parsing."""

    # CliffT currently rejects Stim instruction tags like I_ERROR[loss](0).
    # The tags are metadata, so stripping them preserves the sampled semantics.
    return "\n".join(
        re.sub(r"^([A-Z][A-Z0-9_]*)(\[[^\]\n]+\])", r"\1", line)
        for line in str(circuit).splitlines()
    )


@dataclass
class DemoTask:
    """Small wrapper around a Gemini logical simulator task."""

    task: GeminiLogicalSimulatorTask

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

    def _sample_clifft(
        self,
        shots: int,
        *,
        with_noise: bool = True,
        seed: int | None = None,
    ) -> SampleResult:
        import clifft

        # TODO: check if _run_clifft() is ever called with run()... because I think we might be calling sample_clifft_det_obs always?
        program = (
            self.clifft_tsim_program
            if with_noise
            else self.clifft_noiseless_tsim_program
        )
        sample_kwargs: dict[str, int] = {"shots": int(shots)}
        if seed is not None:
            sample_kwargs["seed"] = int(seed)
        return cast("SampleResult", clifft.sample(program, **sample_kwargs))

    def sample_detector_observables(
        self,
        shots: int,
        *,
        with_noise: bool = True,
        sim_type: str = "tsim",
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample detector and observable arrays."""

        # TODO: has separate clifft explicit check to avoid converting from
        # np.array -> list -> np.array. I think the problem is that DetectorResult
        # contains lists, so we have to do conversions to numpy arrays. So that's why
        # we are currently calling a clifft method that returns np.array's directly.
        # To change this, we'd need to change the task.run() interface/the DetectorResult
        # return type.
        # TODO: not sure if I like this overload logic, but it mirrors GeminiLogicalSimulatorTask; do we need it/can we get rid of it?
        if sim_type == "clifft":
            sample_result = self._sample_clifft(
                shots,
                with_noise=with_noise,
                seed=seed,
            )
            return (
                np.asarray(sample_result.detectors, dtype=np.uint8),
                np.asarray(sample_result.observables, dtype=np.uint8),
            )
        if sim_type != "tsim":
            raise ValueError("sim_type must be either 'tsim' or 'clifft'.")

        result = self.task.run(shots, with_noise=with_noise, run_detectors=True)
        return (
            np.asarray(result.detectors, dtype=np.uint8),
            np.asarray(result.observables, dtype=np.uint8),
        )


GeminiDecoderTask = DemoTask
