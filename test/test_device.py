import math
from typing import Any

import pytest
from bloqade.decoders.dialects import annotate
from bloqade.gemini import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import qubit, squin, types
from bloqade.lanes.device import (
    DetectorResult,
    GeminiLogicalSimulator,
    Result,
)
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs


@gemini_logical.kernel(verify=False)
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_detector([meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0])
    annotate.set_detector([meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1])
    annotate.set_detector([meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2])


@gemini_logical.kernel(verify=False)
def set_observable(meas: ilist.IList[types.MeasurementResult, Any], index: int):
    annotate.set_observable([meas[0], meas[1], meas[5]], index)


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
        set_observable(measurements[i], i)


@pytest.mark.parametrize("size", [2, 3, 4, 5, 6])
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
                [meas[qubit_index][0], meas[qubit_index][1], meas[qubit_index][5]],
                qubit_index,
            )

        return ilist.map(set_observable, ilist.range(len(reg)))

    result = GeminiLogicalSimulator().run(main, 1000, with_noise=False)
    # checks to make sure logical GHZ state is created.
    assert all(len(set(rv)) == 1 for rv in result.observables)


def test_run_default():
    """Test that run() without run_detectors returns a Result."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=5, with_noise=False)

    assert isinstance(result, Result)
    assert result.fidelity_bounds() is not None
    assert result.detector_error_model is not None


def test_run_with_run_detectors_flag():
    """Test that run(run_detectors=True) returns a DetectorResult."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=False, run_detectors=True)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None
    assert result.detector_error_model is not None


def test_run_detectors_with_noise():
    """Test run(run_detectors=True) with noise enabled uses the noisy detector sampler."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=True, run_detectors=True)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None


def test_run_async_with_run_detectors_flag():
    """Test run_async(run_detectors=True) returns a Future[DetectorResult]."""
    sim = GeminiLogicalSimulator()
    future = sim.run_async(main, shots=5, with_noise=False, run_detectors=True)
    result = future.result()

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


def test_run_detectors_via_task():
    """Test calling run(run_detectors=True) on a task directly."""
    sim = GeminiLogicalSimulator()
    task = sim.task(main)
    result = task.run(shots=5, with_noise=False, run_detectors=True)

    assert isinstance(result, DetectorResult)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


def test_run_detectors_task_directly():
    """Test creating a GeminiLogicalSimulatorTask and calling run(run_detectors=True)."""
    sim = GeminiLogicalSimulator(noise_model=generate_simple_noise_model())
    task = sim.task(main)
    result = task.run(shots=5, with_noise=False, run_detectors=True)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


def test_run_detectors_task_async():
    """Test run_async(run_detectors=True) directly on GeminiLogicalSimulatorTask."""
    sim = GeminiLogicalSimulator(noise_model=generate_simple_noise_model())
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
                [meas[qubit_index][0], meas[qubit_index][1], meas[qubit_index][5]],
                qubit_index,
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


def _steane_matrices(num_qubits: int):
    return steane7_m2dets(num_qubits), steane7_m2obs(num_qubits)


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


def test_cudaq_kernel_requires_annotation_matrices():
    cudaq = pytest.importorskip("cudaq")

    @cudaq.kernel
    def bell_pair():
        q = cudaq.qvector(2)
        h(q[0])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
        cx(q[0], q[1])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]

    with pytest.raises(
        ValueError,
        match="At least one of m2dets or m2obs must be provided for CUDA-Q kernels",
    ):
        GeminiLogicalSimulator().task(bell_pair)


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
