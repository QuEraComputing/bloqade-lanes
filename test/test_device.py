import math
from typing import Any

import pytest
from bloqade.decoders.dialects import annotate
from bloqade.gemini import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import qubit, squin, types
from bloqade.lanes.device import GeminiLogicalSimulator, GeminiLogicalSimulatorTask
from bloqade.lanes.noise_model import generate_simple_noise_model


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


def test_no_measurements_run():
    """Test that no_measurements mode uses detector sampler and produces detectors/observables."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=False, no_measurements=True)

    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None
    assert result.detector_error_model is not None


def test_no_measurements_blocks_measurements_access():
    """Test that accessing measurements raises ValueError in no_measurements mode."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=False, no_measurements=True)

    with pytest.raises(ValueError, match="measurements not accessible"):
        result.measurements


def test_no_measurements_blocks_return_values_access():
    """Test that accessing return_values raises ValueError in no_measurements mode."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=False, no_measurements=True)

    with pytest.raises(ValueError, match="return values not accessible"):
        result.return_values


def test_no_measurements_rejects_non_none_return_type():
    """Test that no_measurements mode rejects kernels with non-None return type."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def returning_kernel():
        reg = qubit.qalloc(1)
        squin.h(reg[0])
        meas = gemini_logical.terminal_measure(reg)
        return squin.set_observable([meas[0][0], meas[0][1], meas[0][5]], 0)

    sim = GeminiLogicalSimulator()
    with pytest.raises(ValueError, match="None return type"):
        sim.task(returning_kernel, no_measurements=True)


def test_no_measurements_task_directly():
    """Test creating a GeminiLogicalSimulatorTask with no_measurements=True."""
    noise_model = generate_simple_noise_model()
    task = GeminiLogicalSimulatorTask(main, noise_model, no_measurements=True)
    result = task.run(shots=5, with_noise=False)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5


def test_no_measurements_via_task():
    """Test passing no_measurements to task() on GeminiLogicalSimulator."""
    sim = GeminiLogicalSimulator()
    task = sim.task(main, no_measurements=True)
    result = task.run(shots=5, with_noise=False)
    assert len(result.detectors) == 5
    assert len(result.observables) == 5
    with pytest.raises(ValueError, match="measurements not accessible"):
        result.measurements


def test_no_measurements_with_noise():
    """Test no_measurements mode with noise enabled uses the noisy detector sampler."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=10, with_noise=True, no_measurements=True)

    assert len(result.detectors) == 10
    assert len(result.observables) == 10
    assert result.fidelity_bounds() is not None


def test_no_measurements_run_async():
    """Test run_async with no_measurements=True returns a Future with valid Result."""
    sim = GeminiLogicalSimulator()
    future = sim.run_async(main, shots=5, with_noise=False, no_measurements=True)
    result = future.result()

    assert len(result.detectors) == 5
    assert len(result.observables) == 5
    with pytest.raises(ValueError, match="measurements not accessible"):
        result.measurements


def test_no_measurements_blocks_detectors_observables_error_messages():
    """Test that detectors/observables have distinct error messages in no_measurements mode.

    When Result is constructed via the no_measurements path, detectors and
    observables are populated directly, so they should not raise. But
    measurements and return_values should raise with their specific messages.
    """
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=5, with_noise=False, no_measurements=True)

    # detectors and observables should be accessible
    assert len(result.detectors) == 5
    assert len(result.observables) == 5

    # measurements and return_values should raise with distinct messages
    with pytest.raises(ValueError, match="measurements not accessible"):
        result.measurements
    with pytest.raises(ValueError, match="return values not accessible"):
        result.return_values


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


def test_no_measurements_result_property_caching():
    """Test that Result properties cache correctly in no_measurements mode."""
    sim = GeminiLogicalSimulator()
    result = sim.run(main, shots=5, with_noise=False, no_measurements=True)

    detectors_first = result.detectors
    detectors_second = result.detectors
    assert detectors_first is detectors_second

    observables_first = result.observables
    observables_second = result.observables
    assert observables_first is observables_second


def test_no_measurements_task_run_async():
    """Test run_async directly on GeminiLogicalSimulatorTask with no_measurements."""
    noise_model = generate_simple_noise_model()
    task = GeminiLogicalSimulatorTask(main, noise_model, no_measurements=True)
    future = task.run_async(shots=5, with_noise=False)
    result = future.result()
    assert len(result.detectors) == 5
    assert len(result.observables) == 5
