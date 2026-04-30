from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any

import numpy as np


class ObservableFrame(str, Enum):
    RAW = "raw"
    NOISELESS_REFERENCE_FLIPS = "noiseless_reference_flips"


@dataclass(frozen=True)
class LogicalKernelSpec:
    kernel: Any
    special_tsim_circuit_strategy: str | None = None
    # TODO: for observable_frame, we need to NOT have this auto-correction
    observable_frame: ObservableFrame = ObservableFrame.RAW


@dataclass(frozen=True)
class SyndromeLayout:
    output_detector_count: int = 3
    output_observable_count: int = 1


DEFAULT_SYNDROME_LAYOUT = SyndromeLayout()


def _clifft_compatible_stim_text(circuit: Any) -> str:
    # CliffT currently rejects Stim instruction tags like I_ERROR[loss](0).
    # The tags are metadata, so stripping them preserves the sampled semantics.
    return "\n".join(
        re.sub(r"^([A-Z][A-Z0-9_]*)(\[[^\]\n]+\])", r"\1", line)
        for line in str(circuit).splitlines()
    )


@dataclass
class DemoTask:
    task: Any
    observable_frame: ObservableFrame = ObservableFrame.RAW
    observable_reference: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.task, name)

    @cached_property
    def clifft_tsim_program(self) -> Any:
        import clifft

        return clifft.compile(_clifft_compatible_stim_text(self.task.tsim_circuit))

    @cached_property
    def clifft_noiseless_tsim_program(self) -> Any:
        import clifft

        return clifft.compile(
            _clifft_compatible_stim_text(self.task.noiseless_tsim_circuit)
        )

    def _run_clifft(
        self,
        shots: int = 1,
        with_noise: bool = True,
        *,
        run_detectors: bool = False,
        seed: int | None = None,
    ) -> Any:
        import clifft

        from bloqade.gemini.device import DetectorResult, Result

        program = (
            self.clifft_tsim_program
            if with_noise
            else self.clifft_noiseless_tsim_program
        )
        sample_kwargs: dict[str, Any] = {"shots": int(shots)}
        if seed is not None:
            sample_kwargs["seed"] = int(seed)
        sample_result = clifft.sample(program, **sample_kwargs)

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

    def run(self, *args: Any, sim_type: str = "tsim", **kwargs: Any) -> Any:
        if sim_type == "tsim":
            return self.task.run(*args, **kwargs)
        if sim_type != "clifft":
            raise ValueError(
                f"sim_type is {sim_type}; currently, the only supported simulator "
                "backends are 'tsim' and 'clifft'"
            )

        if len(args) > 2:
            raise TypeError(
                "DemoTask.run accepts at most two positional arguments: "
                "shots and with_noise."
            )
        shots = args[0] if len(args) >= 1 else kwargs.pop("shots", 1)
        with_noise = args[1] if len(args) >= 2 else kwargs.pop("with_noise", True)
        run_detectors = kwargs.pop("run_detectors", False)
        seed = kwargs.pop("seed", None)
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"DemoTask.run got an unexpected keyword argument {unexpected!r}"
            )
        return self._run_clifft(
            shots,
            with_noise,
            run_detectors=run_detectors,
            seed=seed,
        )
