from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

from bloqade.gemini.decoding.tasks import _CliffTSimulatorTask


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
