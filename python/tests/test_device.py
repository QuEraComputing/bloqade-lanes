import math
from concurrent.futures import Future
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, assert_type, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from bloqade.decoders.dialects import annotate
from kirin.dialects import ilist

from bloqade import qubit, squin, types
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.device import (
    BackendSample,
    CliffTSimulatorBackend,
    DetectorResult,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask,
    PyQrackSimulatorBackend,
    Result,
    TsimSimulatorBackend,
)
from bloqade.lanes.cudaq_integration import cudaq_to_squin
from bloqade.lanes.logical_mvp import append_measurements_and_annotations
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs


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


@gemini_logical.kernel(aggressive_unroll=True)
def small_backend_kernel():
    reg = qubit.qalloc(1)
    squin.x(reg[0])
    measurements = gemini_logical.terminal_measure(reg)
    set_detector(measurements[0])
    set_observable(measurements[0])


def _plain_callable():
    return None


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


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TsimSimulatorBackend(), id="tsim"),
        pytest.param(CliffTSimulatorBackend(seed=17), id="clifft"),
        pytest.param(PyQrackSimulatorBackend(seed=17), id="pyqrack"),
    ],
)
def test_builtin_backends_run_logical_workflow_with_guaranteed_dem(backend):
    result = GeminiLogicalSimulator(backend=backend).run(
        small_backend_kernel, shots=2, with_noise=False
    )

    assert isinstance(result, Result)
    assert len(result.measurements) == 2
    assert result.detector_error_model is not None
    if isinstance(backend, CliffTSimulatorBackend):
        backend._programs.clear()


def test_pyqrack_logical_detector_run_uses_measurement_fallback():
    result = GeminiLogicalSimulator(backend=PyQrackSimulatorBackend(seed=18)).run(
        small_backend_kernel, shots=2, with_noise=False, run_detectors=True
    )

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 2
    assert len(result.observables) == 2
    assert result.detector_error_model is not None


@pytest.mark.parametrize("seed", [None, 0])
@pytest.mark.parametrize("run_detectors", [False, True])
def test_logical_simulator_run_forwards_seed(
    monkeypatch, seed: int | None, run_detectors: bool
):
    simulator = GeminiLogicalSimulator()
    task = MagicMock()
    expected = object()
    task.run.return_value = expected
    monkeypatch.setattr(simulator, "task", MagicMock(return_value=task))

    result = simulator.run(
        main,
        shots=3,
        with_noise=False,
        run_detectors=run_detectors,
        seed=seed,
    )

    assert result is expected
    task.run.assert_called_once_with(3, False, run_detectors=run_detectors, seed=seed)


@pytest.mark.parametrize("run_detectors", [False, True])
def test_logical_simulator_run_async_forwards_seed(monkeypatch, run_detectors: bool):
    simulator = GeminiLogicalSimulator()
    task = MagicMock()
    expected = object()
    future = Future()
    future.set_result(expected)
    task.run_async.return_value = future
    monkeypatch.setattr(simulator, "task", MagicMock(return_value=task))

    result = simulator.run_async(
        main,
        shots=3,
        with_noise=False,
        run_detectors=run_detectors,
        seed=0,
    ).result()

    assert result is expected
    task.run_async.assert_called_once_with(
        3, False, run_detectors=run_detectors, seed=0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"m2dets": [[1]]}, id="m2dets"),
        pytest.param({"m2obs": [[1]]}, id="m2obs"),
    ],
)
def test_logical_simulator_constructor_rejects_measurement_matrices(kwargs):
    simulator_type: Any = GeminiLogicalSimulator

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        simulator_type(**kwargs)


def test_logical_simulator_task_rejects_non_squin_before_compilation(monkeypatch):
    import bloqade.lanes.logical_mvp as logical_mvp

    compile_task = MagicMock()
    monkeypatch.setattr(logical_mvp, "compile_task", compile_task)
    invalid_kernel: Any = _plain_callable

    with pytest.raises(TypeError, match="Squin ir.Method"):
        GeminiLogicalSimulator().task(invalid_kernel)

    compile_task.assert_not_called()


@pytest.mark.parametrize("entrypoint", ["run", "run_async", "visualize"])
def test_logical_simulator_rejects_non_squin_through_public_entrypoints(
    monkeypatch, entrypoint
):
    simulator = GeminiLogicalSimulator()
    task_spy = MagicMock(wraps=simulator.task)
    monkeypatch.setattr(simulator, "task", task_spy)
    invalid_kernel: Any = _plain_callable

    with pytest.raises(TypeError, match="Squin ir.Method"):
        getattr(simulator, entrypoint)(invalid_kernel)

    task_spy.assert_called_once_with(invalid_kernel)


def test_logical_simulator_task_preserves_source_kernel():
    source_ir = small_backend_kernel.print_str()

    task = GeminiLogicalSimulator().task(small_backend_kernel)

    assert small_backend_kernel.print_str() == source_ir
    assert task.logical_squin_kernel is not small_backend_kernel


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

    append_measurements_and_annotations(logical_x, m2dets, m2obs)
    task = GeminiLogicalSimulator().task(logical_x)
    result = task.run(10, with_noise=False, run_detectors=run_detectors)

    assert all(all(obs) for obs in result.observables)


@pytest.mark.slow
@pytest.mark.parametrize(
    "use_dets, use_obs",
    [(True, True), (True, False), (False, True)],
    ids=["both", "dets_only", "obs_only"],
)
def test_explicit_annotations_on_kernel_with_terminal_measure(
    use_dets: bool, use_obs: bool
):
    """Explicitly append annotations to a kernel with a terminal measure."""
    num_qubits = 2
    m2dets, m2obs = _steane_matrices(num_qubits)

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel_with_measure():
        reg = qubit.qalloc(num_qubits)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    selected_m2dets = m2dets if use_dets else None
    selected_m2obs = m2obs if use_obs else None
    append_measurements_and_annotations(
        kernel_with_measure, selected_m2dets, selected_m2obs
    )
    task = GeminiLogicalSimulator().task(kernel_with_measure)
    result = task.run(10, with_noise=False)

    expected_detector_width = len(m2dets[0]) if use_dets else 0
    expected_observable_width = len(m2obs[0]) if use_obs else 0
    assert len(result.detectors) == 10
    assert all(len(det) == expected_detector_width for det in result.detectors)
    assert len(result.observables) == 10
    assert all(len(obs) == expected_observable_width for obs in result.observables)

    if use_dets:
        assert all(isinstance(b, bool) for det in result.detectors for b in det)

    if use_obs:
        assert all(isinstance(b, bool) for obs in result.observables for b in obs)


@pytest.mark.parametrize(
    "use_dets, use_obs",
    [(True, True), (True, False), (False, True)],
    ids=["both", "dets_only", "obs_only"],
)
def test_cudaq_kernel_explicit_conversion_and_annotation(use_dets: bool, use_obs: bool):
    cudaq = pytest.importorskip("cudaq")

    num_qubits = 2
    m2dets, m2obs = _steane_matrices(num_qubits)

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(num_qubits)
        h(q[0])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
        cx(q[0], q[1])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]

    squin_kernel = cudaq_to_squin(bell_pair)
    selected_m2dets = m2dets if use_dets else None
    selected_m2obs = m2obs if use_obs else None
    append_measurements_and_annotations(squin_kernel, selected_m2dets, selected_m2obs)
    task = GeminiLogicalSimulator().task(squin_kernel)
    result = task.run(10, with_noise=False)

    expected_detector_width = len(m2dets[0]) if use_dets else 0
    expected_observable_width = len(m2obs[0]) if use_obs else 0
    assert len(result.detectors) == 10
    assert all(len(det) == expected_detector_width for det in result.detectors)
    assert len(result.observables) == 10
    assert all(len(obs) == expected_observable_width for obs in result.observables)

    if use_dets:
        assert all(isinstance(b, bool) for det in result.detectors for b in det)

    if use_obs:
        assert all(isinstance(b, bool) for obs in result.observables for b in obs)


def _mock_task() -> Any:
    task = object.__new__(GeminiLogicalSimulatorTask)
    backend = MagicMock()
    backend.detector_error_model.return_value = "dem"
    object.__setattr__(task, "physical_squin_kernel", "noisy-kernel")
    object.__setattr__(task, "noiseless_physical_squin_kernel", "noiseless-kernel")
    object.__setattr__(task, "simulator", SimpleNamespace(backend=backend))
    object.__setattr__(task, "fidelity_bounds", MagicMock(return_value=(0.5, 0.9)))
    object.__setattr__(task, "_post_processing", MagicMock())
    return task


def test_run_samples_noisy_kernel_through_backend_after_dem_generation():
    task = _mock_task()
    samples = np.array([[True, False]])
    task.simulator.backend.sample.return_value = BackendSample(measurements=samples)

    result = GeminiLogicalSimulatorTask.run(task, shots=1, with_noise=True)

    task.simulator.backend.detector_error_model.assert_called_once_with("noisy-kernel")
    task.simulator.backend.sample.assert_called_once_with(
        "noisy-kernel", shots=1, run_detectors=False, seed=None
    )
    assert isinstance(result, Result)
    assert result._raw_measurements == samples.tolist()
    assert result.detector_error_model == "dem"


def test_run_samples_noiseless_kernel_through_backend():
    task = _mock_task()
    task.simulator.backend.sample.return_value = BackendSample(
        measurements=np.array([[True]])
    )

    GeminiLogicalSimulatorTask.run(task, shots=1, with_noise=False)

    task.simulator.backend.sample.assert_called_once_with(
        "noiseless-kernel", shots=1, run_detectors=False, seed=None
    )


def test_run_detectors_uses_native_backend_detector_samples():
    task = _mock_task()
    detectors, observables = np.array([[True]]), np.array([[False]])
    task.simulator.backend.sample.return_value = BackendSample(
        detectors=detectors, observables=observables
    )

    result = GeminiLogicalSimulatorTask.run(
        task, shots=1, with_noise=True, run_detectors=True
    )

    task.simulator.backend.sample.assert_called_once_with(
        "noisy-kernel", shots=1, run_detectors=True, seed=None
    )
    assert isinstance(result, DetectorResult)
    assert result._detectors == detectors.tolist()
    assert result._observables == observables.tolist()


def test_run_detectors_falls_back_to_postprocessing_measurements():
    task = _mock_task()
    task.simulator.backend.sample.return_value = BackendSample(
        measurements=np.array([[True, False]])
    )
    task._post_processing.emit_detectors.return_value = [[True]]
    task._post_processing.emit_observables.return_value = [[False]]

    result = GeminiLogicalSimulatorTask.run(
        task, shots=1, with_noise=True, run_detectors=True
    )

    task._post_processing.emit_detectors.assert_called_once_with([[True, False]])
    task._post_processing.emit_observables.assert_called_once_with([[True, False]])
    assert result._detectors == [[True]]
    assert result._observables == [[False]]


@pytest.mark.parametrize(
    "sample, message",
    [
        (BackendSample(), "did not return measurement"),
        (BackendSample(measurements=np.array([True])), "two-dimensional"),
        (BackendSample(measurements=np.array([[True], [False]])), "rows"),
    ],
)
def test_run_rejects_invalid_backend_measurement_payloads(sample, message):
    task = _mock_task()
    task.simulator.backend.sample.return_value = sample

    with pytest.raises(ValueError, match=message):
        GeminiLogicalSimulatorTask.run(task, shots=1)


def test_run_detectors_rejects_partial_native_payload():
    task = _mock_task()
    task.simulator.backend.sample.return_value = BackendSample(
        detectors=np.array([[True]])
    )

    with pytest.raises(ValueError, match="detector and observable.*together"):
        GeminiLogicalSimulatorTask.run(task, shots=1, run_detectors=True)


def test_run_fails_on_dem_before_sampling():
    task = _mock_task()
    task.simulator.backend.detector_error_model.side_effect = ImportError("no tsim")

    with pytest.raises(ImportError, match="no tsim"):
        GeminiLogicalSimulatorTask.run(task, shots=1)

    task.simulator.backend.sample.assert_not_called()


@pytest.mark.parametrize("seed", [0, 2**63 - 1])
def test_task_run_accepts_valid_seed_values(seed: int):
    task = _mock_task()
    task.simulator.backend.sample.return_value = BackendSample(
        measurements=np.array([[True]])
    )

    GeminiLogicalSimulatorTask.run(task, shots=1, seed=seed)

    task.simulator.backend.sample.assert_called_once_with(
        "noisy-kernel", shots=1, run_detectors=False, seed=seed
    )


def test_task_run_accepts_explicit_none_seed():
    task = _mock_task()
    task.simulator.backend.sample.return_value = BackendSample(
        measurements=np.array([[True]])
    )

    GeminiLogicalSimulatorTask.run(task, shots=1, seed=None)

    task.simulator.backend.sample.assert_called_once_with(
        "noisy-kernel", shots=1, run_detectors=False, seed=None
    )


def test_task_run_forwards_seed_to_detector_backend():
    task = _mock_task()
    task.simulator.backend.sample.return_value = BackendSample(
        detectors=np.array([[True]]),
        observables=np.array([[False]]),
    )

    result = GeminiLogicalSimulatorTask.run(task, shots=1, run_detectors=True, seed=0)

    task.simulator.backend.sample.assert_called_once_with(
        "noisy-kernel", shots=1, run_detectors=True, seed=0
    )
    assert isinstance(result, DetectorResult)


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "1"])
def test_task_run_rejects_invalid_seed_before_dem_or_sampling(seed: object):
    task = _mock_task()

    with pytest.raises(ValueError, match="seed must be"):
        GeminiLogicalSimulatorTask.run(task, seed=cast(int | None, seed))

    task.simulator.backend.detector_error_model.assert_not_called()
    task.simulator.backend.sample.assert_not_called()


@pytest.mark.parametrize("seed", [True, -1, 2**63, 1.5, "1"])
def test_task_run_async_rejects_invalid_seed_before_submission(seed: object):
    task = _mock_task()
    executor = MagicMock()
    object.__setattr__(task, "_thread_pool_executor", executor)

    with pytest.raises(ValueError, match="seed must be"):
        GeminiLogicalSimulatorTask.run_async(task, seed=cast(int | None, seed))

    executor.submit.assert_not_called()


@pytest.mark.parametrize("run_detectors", [False, True])
def test_task_run_async_forwards_seed(run_detectors: bool):
    task = _mock_task()
    executor = MagicMock()
    future = Future()
    future.set_result(object())
    executor.submit.return_value = future
    run = MagicMock()
    object.__setattr__(task, "_thread_pool_executor", executor)
    object.__setattr__(task, "run", run)

    result = GeminiLogicalSimulatorTask.run_async(
        task,
        shots=3,
        with_noise=False,
        run_detectors=run_detectors,
        seed=0,
    )

    assert result is future
    executor.submit.assert_called_once_with(
        run,
        3,
        False,
        run_detectors=run_detectors,
        seed=0,
    )


if TYPE_CHECKING:

    def _check_task_run_overloads(
        task: GeminiLogicalSimulatorTask[Any], dynamic: bool
    ) -> None:
        assert_type(task.run(run_detectors=False), Result[Any])
        assert_type(task.run(run_detectors=True), DetectorResult)
        assert_type(task.run(run_detectors=dynamic), Result[Any] | DetectorResult)
        assert_type(task.run_async(run_detectors=False), Future[Result[Any]])
        assert_type(task.run_async(run_detectors=True), Future[DetectorResult])
        assert_type(
            task.run_async(run_detectors=dynamic),
            Future[Result[Any]] | Future[DetectorResult],
        )

    def _check_simulator_run_overloads(
        simulator: GeminiLogicalSimulator, dynamic: bool
    ) -> None:
        assert_type(simulator.run(main, run_detectors=False), Result[None])
        assert_type(simulator.run(main, run_detectors=True), DetectorResult)
        assert_type(
            simulator.run(main, run_detectors=dynamic),
            Result[None] | DetectorResult,
        )
        assert_type(
            simulator.run_async(main, run_detectors=False), Future[Result[None]]
        )
        assert_type(
            simulator.run_async(main, run_detectors=True), Future[DetectorResult]
        )
        assert_type(
            simulator.run_async(main, run_detectors=dynamic),
            Future[Result[None]] | Future[DetectorResult],
        )
