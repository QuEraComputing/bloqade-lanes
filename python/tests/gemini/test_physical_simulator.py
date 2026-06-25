from unittest.mock import MagicMock

import numpy as np

from bloqade.gemini.device.physical_simulator import (
    PhysicalResult,
    PhysicalSimulatorTask,
)


def test_physical_result_converts_measurements_to_detectors_and_observables():
    raw_measurements = [[True, False, True]]
    circuit = MagicMock()
    detectors = np.array([[True, False]])
    observables = np.array([[False]])
    circuit.compile_m2d_converter.return_value.convert.return_value = (
        detectors,
        observables,
    )

    result = PhysicalResult(
        _raw_measurements=raw_measurements,
        _tsim_circuit=circuit,
        _detector_error_model=MagicMock(),
        _fidelity_min=0.5,
        _fidelity_max=0.9,
    )

    assert result.measurements == raw_measurements
    assert result.detectors == detectors.tolist()
    assert result.observables == observables.tolist()
    circuit.compile_m2d_converter.assert_called_once_with(skip_reference_sample=True)
    call_kwargs = circuit.compile_m2d_converter.return_value.convert.call_args.kwargs
    np.testing.assert_array_equal(
        call_kwargs["measurements"], np.asarray(raw_measurements, dtype=bool)
    )
    assert call_kwargs["separate_observables"] is True


def test_physical_task_run_clifford_returns_physical_result():
    task = MagicMock()
    task.tsim_circuit.is_clifford = True
    task.tsim_circuit.stim_circuit.compile_sampler.return_value.sample.return_value = (
        np.array([[True, False]])
    )
    task.fidelity_bounds.return_value = (0.5, 0.9)
    task.detector_error_model = MagicMock()

    result = PhysicalSimulatorTask.run(task, shots=1, with_noise=True)

    task.tsim_circuit.stim_circuit.compile_sampler.assert_called_once_with()
    task.measurement_sampler.sample.assert_not_called()
    assert isinstance(result, PhysicalResult)
    assert result.measurements == [[True, False]]


def test_physical_simulator_exports_from_gemini_namespace():
    from bloqade.gemini import PhysicalSimulator
    from bloqade.gemini.device import PhysicalSimulator as DevicePhysicalSimulator

    assert PhysicalSimulator is DevicePhysicalSimulator
