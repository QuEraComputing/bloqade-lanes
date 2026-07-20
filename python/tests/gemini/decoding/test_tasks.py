from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

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


def test_sample_clifft_uses_noiseless_program_and_task_seed(monkeypatch):
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


def test_sample_clifft_per_call_seed_overrides_task_seed(monkeypatch):
    calls: list[tuple[object, dict[str, int]]] = []
    fake_clifft = ModuleType("clifft")

    def sample(program: object, **kwargs: int) -> _FakeSampleResult:
        calls.append((program, kwargs))
        return _FakeSampleResult([], [], [])

    cast(Any, fake_clifft).sample = sample
    monkeypatch.setitem(sys.modules, "clifft", fake_clifft)

    _task_with_cached_programs(seed=123)._sample_clifft(
        2,
        with_noise=True,
        seed=0,
    )

    assert calls == [("noisy-program", {"shots": 2, "seed": 0})]


def test_sample_clifft_none_seed_falls_back_to_task_seed(monkeypatch):
    calls: list[tuple[object, dict[str, int]]] = []
    fake_clifft = ModuleType("clifft")

    def sample(program: object, **kwargs: int) -> _FakeSampleResult:
        calls.append((program, kwargs))
        return _FakeSampleResult([], [], [])

    cast(Any, fake_clifft).sample = sample
    monkeypatch.setitem(sys.modules, "clifft", fake_clifft)

    _task_with_cached_programs(seed=123)._sample_clifft(
        2,
        with_noise=True,
        seed=None,
    )

    assert calls == [("noisy-program", {"shots": 2, "seed": 123})]


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "1"])
def test_sample_clifft_rejects_invalid_per_call_seed_before_sampling(
    monkeypatch, seed: object
):
    calls: list[tuple[object, dict[str, int]]] = []
    fake_clifft = ModuleType("clifft")

    def sample(program: object, **kwargs: int) -> _FakeSampleResult:
        calls.append((program, kwargs))
        return _FakeSampleResult([], [], [])

    cast(Any, fake_clifft).sample = sample
    monkeypatch.setitem(sys.modules, "clifft", fake_clifft)

    with pytest.raises((TypeError, ValueError), match="seed must be"):
        _task_with_cached_programs(seed=123)._sample_clifft(
            2,
            with_noise=True,
            seed=cast(int | None, seed),
        )

    assert calls == []


def test_clifft_run_async_forwards_per_call_seed_and_returns_result(monkeypatch):
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

    task = _task_with_cached_programs(seed=123)
    object.__setattr__(task, "_thread_pool_executor", ThreadPoolExecutor(max_workers=1))
    object.__setattr__(task, "fidelity_bounds", lambda: (0.5, 0.9))
    object.__setattr__(task, "detector_error_model", object())
    object.__setattr__(task, "_post_processing", MagicMock())

    try:
        result = task.run_async(shots=2, with_noise=False, seed=0).result()
    finally:
        task._thread_pool_executor.shutdown()

    assert result.measurements == [[False]]
    assert calls == [("noiseless-program", {"shots": 2, "seed": 0})]
