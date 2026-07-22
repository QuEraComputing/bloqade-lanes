import math
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from bloqade.decoders.dialects import annotate
from kirin.dialects import ilist

from bloqade import qubit, squin, types
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.device import (
    DetectorResult,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask,
    Result,
)
from bloqade.gemini.steane_defaults import steane7_m2dets, steane7_m2obs
from bloqade.lanes.noise_model import generate_logical_noise_model


@gemini_logical.kernel(verify=False)
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_detector([meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0])
    annotate.set_detector([meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1])
    annotate.set_detector([meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2])


@gemini_logical.kernel(verify=False)
def set_observable(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_observable([meas[0], meas[1], meas[5]])


@gemini_logical.kernel(aggressive_unroll=True)
def main():
    # see arXiv: 2412.15165v1, Figure 3a
    reg = qubit.qalloc(5)
    squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

    squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
    squin.sqrt_x_adj(reg[0])
    squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
    squin.broadcast.sqrt_y_adj(reg)

    measurements = gemini_logical.terminal_measure(reg)

    for i in range(len(reg)):
        set_detector(measurements[i])
        set_observable(measurements[i])


@pytest.mark.slow
@pytest.mark.parametrize("size", [2, 6])
def test_physical_compilation(size: int):
    @gemini_logical.kernel(aggressive_unroll=True)
    def main():
        reg = qubit.qalloc(1)
        squin.h(reg[0])
        for _ in range(size):
            current = len(reg)
            missing = size - current
            if missing > current:
                num_alloc = current
            else:
                num_alloc = missing

            if num_alloc > 0:
                new_qubits = qubit.qalloc(num_alloc)
                squin.broadcast.cx(reg[-num_alloc:], new_qubits)
                reg = reg + new_qubits

        meas = gemini_logical.terminal_measure(reg)

        def set_observable(qubit_index: int):
            return squin.set_observable(
                [meas[qubit_index][0], meas[qubit_index][1], meas[qubit_index][5]]
            )

        return ilist.map(set_observable, ilist.range(len(reg)))

    result = GeminiLogicalSimulator().run(main, 1000, with_noise=False)
    # checks to make sure logical GHZ state is created.
    assert all(len(set(rv)) == 1 for rv in result.observables)


@pytest.mark.slow
def test_run_default():
    """Test that run() without run_detectors returns a Result."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=5, with_noise=False)

    assert isinstance(result, Result)
    assert result.fidelity_bounds() is not None
    assert result.detector_error_model is not None


@pytest.mark.slow
def test_run_with_run_detectors_flag():
    """Test that run(run_detectors=True) returns a DetectorResult."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=False, run_detectors=True)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None
    assert result.detector_error_model is not None


@pytest.mark.slow
def test_run_detectors_with_noise():
    """Test run(run_detectors=True) with noise enabled uses the noisy detector sampler."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=True, run_detectors=True)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None


@pytest.mark.slow
def test_run_async_with_run_detectors_flag():
    """Test run_async(run_detectors=True) returns a Future[DetectorResult]."""
    sim = GeminiLogicalSimulator()
    future = sim.run_async(main, shots=5, with_noise=False, run_detectors=True)
    result = future.result()

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


@pytest.mark.slow
def test_run_detectors_via_task():
    """Test calling run(run_detectors=True) on a task directly."""
    sim = GeminiLogicalSimulator()
    task = sim.task(main)
    result = task.run(shots=5, with_noise=False, run_detectors=True)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


@pytest.mark.slow
def test_run_detectors_task_directly():
    """Test creating a GeminiLogicalSimulatorTask and calling run(run_detectors=True)."""
    sim = GeminiLogicalSimulator(noise_model=generate_logical_noise_model())
    task = sim.task(main)
    result = task.run(shots=5, with_noise=False, run_detectors=True)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


@pytest.mark.slow
def test_run_detectors_task_async():
    """Test run_async(run_detectors=True) directly on GeminiLogicalSimulatorTask."""
    sim = GeminiLogicalSimulator(noise_model=generate_logical_noise_model())
    task = sim.task(main)
    future = task.run_async(shots=5, with_noise=False, run_detectors=True)
    result = future.result()
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


def test_result_property_caching():
    """Test that Result properties return cached values on subsequent access."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def returning_kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        meas = gemini_logical.terminal_measure(reg)

        def set_observable(qubit_index: int):
            return squin.set_observable(
                [meas[qubit_index][0], meas[qubit_index][1], meas[qubit_index][5]]
            )

        return ilist.map(set_observable, ilist.range(len(reg)))

    sim = GeminiLogicalSimulator()
    result = sim.run(returning_kernel, shots=5, with_noise=False)

    # Access each property twice to exercise the caching path
    detectors_first = result.detectors
    detectors_second = result.detectors
    assert detectors_first is detectors_second

    observables_first = result.observables
    observables_second = result.observables
    assert observables_first is observables_second

    measurements_first = result.measurements
    measurements_second = result.measurements
    assert measurements_first is measurements_second

    return_values_first = result.return_values
    return_values_second = result.return_values
    assert return_values_first is return_values_second


def test_noiseless_tsim_circuit_compiles_samplers():
    sim = GeminiLogicalSimulator()
    task = sim.task(main)

    noiseless = task.noiseless_tsim_circuit

    assert noiseless.compile_sampler() is not None
    assert noiseless.compile_detector_sampler() is not None


def _steane_matrices(num_qubits: int):
    return steane7_m2dets(num_qubits), steane7_m2obs(num_qubits)


@pytest.mark.slow
@pytest.mark.parametrize("run_detectors", [False, True])
def test_logical_x_observable_is_one(run_detectors: bool):
    m2dets, m2obs = _steane_matrices(1)

    @gemini_logical.kernel(aggressive_unroll=True)
    def logical_x():
        reg = qubit.qalloc(1)
        squin.x(reg[0])
        gemini_logical.terminal_measure(reg)

    task = GeminiLogicalSimulator().task(logical_x, m2dets=m2dets, m2obs=m2obs)
    result = task.run(10, with_noise=False, run_detectors=run_detectors)

    assert all(all(obs) for obs in result.observables)


@pytest.mark.slow
@pytest.mark.parametrize(
    "use_dets, use_obs",
    [(True, True), (True, False), (False, True)],
    ids=["both", "dets_only", "obs_only"],
)
def test_append_annotations_to_kernel_with_terminal_measure(
    use_dets: bool, use_obs: bool
):
    """Append detectors/observables via the task() API to a squin kernel
    that already has a terminal_measure."""
    num_qubits = 2
    m2dets, m2obs = _steane_matrices(num_qubits)

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel_with_measure():
        reg = qubit.qalloc(num_qubits)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    task = GeminiLogicalSimulator().task(
        kernel_with_measure,
        m2dets=m2dets if use_dets else None,
        m2obs=m2obs if use_obs else None,
    )
    result = task.run(10, with_noise=False)

    if use_dets:
        assert len(result.detectors) == 10
        assert all(len(det) == len(m2dets[0]) for det in result.detectors)
        assert all(isinstance(b, bool) for det in result.detectors for b in det)

    if use_obs:
        assert len(result.observables) == 10
        assert all(len(obs) == len(m2obs[0]) for obs in result.observables)
        assert all(isinstance(b, bool) for obs in result.observables for b in obs)


@pytest.mark.parametrize(
    "use_dets, use_obs",
    [(True, True), (True, False), (False, True)],
    ids=["both", "dets_only", "obs_only"],
)
def test_cudaq_kernel_integration(use_dets: bool, use_obs: bool):
    cudaq = pytest.importorskip("cudaq")

    num_qubits = 2
    m2dets, m2obs = _steane_matrices(num_qubits)

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(num_qubits)
        h(q[0])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
        cx(q[0], q[1])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]

    task = GeminiLogicalSimulator().task(
        bell_pair,
        m2dets=m2dets if use_dets else None,
        m2obs=m2obs if use_obs else None,
    )
    result = task.run(10, with_noise=False)

    if use_dets:
        assert len(result.detectors) == 10
        assert all(len(det) == len(m2dets[0]) for det in result.detectors)
        assert all(isinstance(b, bool) for det in result.detectors for b in det)

    if use_obs:
        assert len(result.observables) == 10
        assert all(len(obs) == len(m2obs[0]) for obs in result.observables)
        assert all(isinstance(b, bool) for obs in result.observables for b in obs)


def _mock_task(*, is_clifford: bool) -> MagicMock:
    task = MagicMock()
    task.tsim_circuit.is_clifford = is_clifford
    task.noiseless_tsim_circuit.is_clifford = is_clifford
    task.fidelity_bounds.return_value = (0.5, 0.9)
    return task


def test_run_clifford_uses_stim_sampler():
    task = _mock_task(is_clifford=True)
    samples = np.array([[True, False]])
    task.tsim_circuit.stim_circuit.compile_sampler.return_value.sample.return_value = (
        samples
    )

    result = GeminiLogicalSimulatorTask.run(task, shots=1, with_noise=True)

    task.tsim_circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=None)
    task.measurement_sampler.sample.assert_not_called()
    assert isinstance(result, Result)
    assert result._raw_measurements == samples.tolist()


def test_run_non_clifford_uses_measurement_sampler():
    task = _mock_task(is_clifford=False)
    task.measurement_sampler.sample.return_value = np.array([[True]])

    GeminiLogicalSimulatorTask.run(task, shots=1, with_noise=True)

    task.measurement_sampler.sample.assert_called_once_with(shots=1)
    task.tsim_circuit.stim_circuit.compile_sampler.assert_not_called()


@pytest.mark.parametrize(
    ("with_noise", "circuit_attribute"),
    [(True, "tsim_circuit"), (False, "noiseless_tsim_circuit")],
)
def test_seed_zero_is_forwarded_to_stim_sampler(
    with_noise: bool, circuit_attribute: str
):
    task = _mock_task(is_clifford=True)
    circuit = getattr(task, circuit_attribute)
    samples = np.array([[True]])
    circuit.stim_circuit.compile_sampler.return_value.sample.return_value = samples

    result = GeminiLogicalSimulatorTask.run(
        task, shots=1, with_noise=with_noise, seed=0
    )

    circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=0)
    assert isinstance(result, Result)
    assert result._raw_measurements == samples.tolist()


def test_seeded_stim_sampler_reproduces_fixed_batch():
    import stim

    task = _mock_task(is_clifford=True)
    task.tsim_circuit.stim_circuit = stim.Circuit("H 0\nM 0")

    first = GeminiLogicalSimulatorTask.run(task, shots=64, seed=17)
    second = GeminiLogicalSimulatorTask.run(task, shots=64, seed=17)

    assert first.measurements == second.measurements


@pytest.mark.parametrize(
    ("with_noise", "circuit_attribute"),
    [(True, "tsim_circuit"), (False, "noiseless_tsim_circuit")],
)
@pytest.mark.parametrize(
    ("run_detectors", "compile_method", "cached_sampler_attribute"),
    [
        (False, "compile_sampler", "measurement_sampler"),
        (True, "compile_detector_sampler", "detector_sampler"),
    ],
)
def test_seeded_non_clifford_run_compiles_fresh_sampler(
    with_noise: bool,
    circuit_attribute: str,
    run_detectors: bool,
    compile_method: str,
    cached_sampler_attribute: str,
):
    task = _mock_task(is_clifford=False)
    circuit = getattr(task, circuit_attribute)
    sampler = getattr(circuit, compile_method).return_value
    if run_detectors:
        sampler.sample.return_value = (
            np.array([[True]]),
            np.array([[False]]),
        )
    else:
        sampler.sample.return_value = np.array([[True]])

    if run_detectors:
        GeminiLogicalSimulatorTask._run_detectors(
            task, shots=1, with_noise=with_noise, seed=17
        )
    else:
        GeminiLogicalSimulatorTask.run(
            task,
            shots=1,
            with_noise=with_noise,
            run_detectors=False,
            seed=17,
        )

    getattr(circuit, compile_method).assert_called_once_with(seed=17)
    selected_cached_sampler_attribute = (
        cached_sampler_attribute
        if with_noise
        else f"noiseless_{cached_sampler_attribute}"
    )
    getattr(task, selected_cached_sampler_attribute).sample.assert_not_called()


def test_seeded_non_clifford_run_does_not_leak_into_unseeded_cache():
    task = _mock_task(is_clifford=False)
    seeded_sampler = task.tsim_circuit.compile_sampler.return_value
    seeded_sampler.sample.return_value = np.array([[True]])
    task.measurement_sampler.sample.return_value = np.array([[False]])

    GeminiLogicalSimulatorTask.run(task, shots=1, seed=17)
    result = GeminiLogicalSimulatorTask.run(task, shots=1, seed=None)

    task.tsim_circuit.compile_sampler.assert_called_once_with(seed=17)
    seeded_sampler.sample.assert_called_once_with(shots=1)
    task.measurement_sampler.sample.assert_called_once_with(shots=1)
    assert result.measurements == [[False]]


@pytest.mark.parametrize(
    ("with_noise", "cached_sampler_attribute"),
    [(True, "measurement_sampler"), (False, "noiseless_measurement_sampler")],
)
@pytest.mark.parametrize("run_detectors", [False, True])
def test_unseeded_non_clifford_run_uses_cached_sampler(
    with_noise: bool, cached_sampler_attribute: str, run_detectors: bool
):
    task = _mock_task(is_clifford=False)
    sampler_attribute = (
        cached_sampler_attribute
        if not run_detectors
        else cached_sampler_attribute.replace("measurement", "detector")
    )
    sampler = getattr(task, sampler_attribute)
    if run_detectors:
        sampler.sample.return_value = (
            np.array([[True]]),
            np.array([[False]]),
        )
    else:
        sampler.sample.return_value = np.array([[True]])

    if run_detectors:
        GeminiLogicalSimulatorTask._run_detectors(task, shots=1, with_noise=with_noise)
    else:
        GeminiLogicalSimulatorTask.run(
            task, shots=1, with_noise=with_noise, run_detectors=False
        )

    sampler.sample.assert_called_once()
    task.tsim_circuit.compile_sampler.assert_not_called()
    task.noiseless_tsim_circuit.compile_sampler.assert_not_called()
    task.tsim_circuit.compile_detector_sampler.assert_not_called()
    task.noiseless_tsim_circuit.compile_detector_sampler.assert_not_called()


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_task_run_accepts_valid_seed_values(seed: int):
    task = _mock_task(is_clifford=True)
    task.tsim_circuit.stim_circuit.compile_sampler.return_value.sample.return_value = (
        np.array([[True]])
    )

    GeminiLogicalSimulatorTask.run(task, shots=1, seed=seed)

    task.tsim_circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=seed)


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "1"])
def test_task_run_rejects_invalid_seed_before_sampling(seed: object):
    task = _mock_task(is_clifford=True)

    with pytest.raises((TypeError, ValueError), match="seed must be"):
        GeminiLogicalSimulatorTask.run(task, shots=1, seed=cast(int | None, seed))

    task.tsim_circuit.stim_circuit.compile_sampler.assert_not_called()


def test_task_run_async_forwards_seed_and_returns_result():
    task = _mock_task(is_clifford=True)
    executor = ThreadPoolExecutor(max_workers=1)
    task._thread_pool_executor = executor
    expected_result = object()
    task.run.return_value = expected_result

    try:
        result = GeminiLogicalSimulatorTask.run_async(
            task, shots=3, with_noise=False, seed=0
        ).result()
    finally:
        executor.shutdown()

    assert result is expected_result
    task.run.assert_called_once_with(3, False, seed=0)


def test_task_run_async_forwards_seed_to_detector_path():
    task = _mock_task(is_clifford=False)
    executor = ThreadPoolExecutor(max_workers=1)
    task._thread_pool_executor = executor
    expected_result = object()
    task._run_detectors.return_value = expected_result

    try:
        result = GeminiLogicalSimulatorTask.run_async(
            task, shots=3, with_noise=False, run_detectors=True, seed=0
        ).result()
    finally:
        executor.shutdown()

    assert result is expected_result
    task._run_detectors.assert_called_once_with(3, False, seed=0)


def test_task_run_forwards_seed_to_detector_path():
    task = _mock_task(is_clifford=False)
    expected_result = object()
    task._run_detectors.return_value = expected_result

    result = GeminiLogicalSimulatorTask.run(
        task, shots=3, with_noise=False, run_detectors=True, seed=0
    )

    assert result is expected_result
    task._run_detectors.assert_called_once_with(3, False, seed=0)


@pytest.mark.parametrize(
    ("with_noise", "circuit_attribute"),
    [(True, "tsim_circuit"), (False, "noiseless_tsim_circuit")],
)
def test_seeded_clifford_detector_sampling_uses_stim_seed_and_m2d(
    with_noise: bool, circuit_attribute: str
):
    task = _mock_task(is_clifford=True)
    samples = np.array([[True, False]])
    detectors, observables = np.array([[True]]), np.array([[False]])
    circuit = getattr(task, circuit_attribute)
    sampler = circuit.stim_circuit.compile_sampler.return_value
    sampler.sample.return_value = samples
    m2d = circuit.compile_m2d_converter.return_value
    m2d.convert.return_value = (detectors, observables)

    result = GeminiLogicalSimulatorTask._run_detectors(
        task, shots=1, with_noise=with_noise, seed=17
    )

    circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=17)
    sampler.sample.assert_called_once_with(shots=1)
    circuit.compile_m2d_converter.assert_called_once_with(skip_reference_sample=True)
    m2d.convert.assert_called_once_with(measurements=samples, separate_observables=True)
    assert isinstance(result, DetectorResult)


def test_task_run_accepts_explicit_none_seed():
    task = _mock_task(is_clifford=True)
    task.tsim_circuit.stim_circuit.compile_sampler.return_value.sample.return_value = (
        np.array([[True]])
    )

    GeminiLogicalSimulatorTask.run(task, shots=1, seed=None)

    task.tsim_circuit.stim_circuit.compile_sampler.assert_called_once_with(seed=None)


@pytest.mark.parametrize("seed", [0, None])
@pytest.mark.parametrize("run_detectors", [False, True])
def test_simulator_run_forwards_seed_to_task(
    monkeypatch, seed: int | None, run_detectors: bool
):
    sim = GeminiLogicalSimulator()
    task = MagicMock()
    expected_result = object()
    task.run.return_value = expected_result
    monkeypatch.setattr(sim, "task", MagicMock(return_value=task))

    result = sim.run(
        main,
        shots=3,
        with_noise=False,
        run_detectors=run_detectors,
        seed=seed,
    )

    assert result is expected_result
    task.run.assert_called_once_with(3, False, run_detectors=run_detectors, seed=seed)


@pytest.mark.parametrize("run_detectors", [False, True])
def test_simulator_run_async_forwards_seed_and_returns_result(
    monkeypatch, run_detectors: bool
):
    sim = GeminiLogicalSimulator()
    task = MagicMock()
    expected_result = object()
    future = Future()
    future.set_result(expected_result)
    task.run_async.return_value = future
    monkeypatch.setattr(sim, "task", MagicMock(return_value=task))

    result = sim.run_async(
        main,
        shots=3,
        with_noise=False,
        run_detectors=run_detectors,
        seed=0,
    ).result()

    assert result is expected_result
    if run_detectors:
        task.run_async.assert_called_once_with(3, False, run_detectors=True, seed=0)
    else:
        task.run_async.assert_called_once_with(3, False, seed=0)


def test_simulator_clifft_task_seed_rejected_before_compilation(monkeypatch):
    sim = GeminiLogicalSimulator(backend="clifft", seed=-1)
    compile_task = MagicMock()
    monkeypatch.setattr("bloqade.lanes.logical_mvp.compile_task", compile_task)

    with pytest.raises(ValueError, match="seed must be"):
        sim.run(main, seed=None)

    compile_task.assert_not_called()


def test_run_detectors_clifford_converts_via_m2d():
    task = _mock_task(is_clifford=True)
    samples = np.array([[True, False]])
    detectors, observables = np.array([[True]]), np.array([[False]])
    task.tsim_circuit.stim_circuit.compile_sampler.return_value.sample.return_value = (
        samples
    )
    m2d = task.tsim_circuit.compile_m2d_converter.return_value
    m2d.convert.return_value = (detectors, observables)

    result = GeminiLogicalSimulatorTask._run_detectors(task, shots=1, with_noise=True)

    task.tsim_circuit.compile_m2d_converter.assert_called_once_with(
        skip_reference_sample=True
    )
    task.detector_sampler.sample.assert_not_called()
    assert isinstance(result, DetectorResult)
    assert result._detectors == detectors.tolist()
    assert result._observables == observables.tolist()


def test_run_detectors_non_clifford_uses_detector_sampler():
    task = _mock_task(is_clifford=False)
    task.detector_sampler.sample.return_value = (
        np.array([[True]]),
        np.array([[False]]),
    )

    GeminiLogicalSimulatorTask._run_detectors(task, shots=1, with_noise=True)

    task.detector_sampler.sample.assert_called_once_with(
        shots=1, separate_observables=True
    )
    task.tsim_circuit.compile_m2d_converter.assert_not_called()
