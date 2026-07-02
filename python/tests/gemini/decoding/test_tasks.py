from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

from bloqade.gemini.decoding.tasks import _CliffTSimulatorTask
from bloqade.gemini.device.abstract_simulator import _clifft_compatible_stim_text


@dataclass(frozen=True)
class _FakeSampleResult:
    detectors: list[list[bool]]
    observables: list[list[bool]]
    measurements: list[list[bool]]


def _task_with_cached_programs(
    *,
    seed: int | None,
) -> _CliffTSimulatorTask[Any]:
    task = object.__new__(_CliffTSimulatorTask)
    object.__setattr__(task, "seed", seed)
    object.__setattr__(task, "clifft_tsim_program", "noisy-program")
    object.__setattr__(task, "clifft_noiseless_tsim_program", "noiseless-program")
    return task


def test_sample_clifft_uses_noisy_program_and_omits_missing_seed(monkeypatch):
    calls: list[tuple[object, dict[str, int]]] = []
    fake_clifft = ModuleType("clifft")

    def sample(program: object, **kwargs: int) -> _FakeSampleResult:
        calls.append((program, kwargs))
        return _FakeSampleResult(
            detectors=[[False]],
            observables=[[True]],
            measurements=[[False]],
        )

    cast(Any, fake_clifft).sample = sample
    monkeypatch.setitem(sys.modules, "clifft", fake_clifft)

    result = _task_with_cached_programs(seed=None)._sample_clifft(
        7,
        with_noise=True,
    )

    assert result.detectors == [[False]]
    assert calls == [("noisy-program", {"shots": 7})]


def test_sample_clifft_uses_noiseless_program_and_seed(monkeypatch):
    calls: list[tuple[object, dict[str, int]]] = []
    fake_clifft = ModuleType("clifft")

    def sample(program: object, **kwargs: int) -> _FakeSampleResult:
        calls.append((program, kwargs))
        return _FakeSampleResult(
            detectors=[[True]],
            observables=[[False]],
            measurements=[[True]],
        )

    cast(Any, fake_clifft).sample = sample
    monkeypatch.setitem(sys.modules, "clifft", fake_clifft)

    result = _task_with_cached_programs(seed=123)._sample_clifft(
        11,
        with_noise=False,
    )

    assert result.observables == [[False]]
    assert calls == [("noiseless-program", {"shots": 11, "seed": 123})]


def test_clifft_detector_error_model_accepts_stim_compatible_u3_tag():
    task = object.__new__(_CliffTSimulatorTask)
    object.__setattr__(
        task,
        "_stim_text",
        """
I[U3(theta=0.0*pi, phi=-0.27542338*pi, lambda=0.0*pi)] 0
M 0
DETECTOR rec[-1]
""",
    )

    dem = task.detector_error_model

    assert "detector D0" in str(dem)


def test_clifft_program_converts_stim_tags_back_to_tsim_shorthand(monkeypatch):
    calls: list[str] = []
    fake_clifft = ModuleType("clifft")

    def compile(program_text: str) -> str:
        calls.append(program_text)
        return "compiled-program"

    cast(Any, fake_clifft).compile = compile
    monkeypatch.setitem(sys.modules, "clifft", fake_clifft)

    task = object.__new__(_CliffTSimulatorTask)
    object.__setattr__(
        task,
        "_stim_text",
        """
I[U3(theta=0.39957968*pi, phi=0.25*pi, lambda=0.0*pi)] 6
I_ERROR[loss](0) 6
""",
    )

    assert task.clifft_tsim_program == "compiled-program"
    assert [call.strip() for call in calls] == ["""
U3(0.39957968, 0.25, 0.0) 6
I_ERROR(0) 6
""".strip()]


def test_clifft_compatible_text_preserves_raw_tsim_u3_shorthand():
    text = """
U3(0.39957968, 0.25, 0.0) 6
I_ERROR[loss](0) 6
"""

    compatible = _clifft_compatible_stim_text(text)

    assert "U3(0.39957968, 0.25, 0.0) 6" in compatible
    assert "I_ERROR(0) 6" in compatible
