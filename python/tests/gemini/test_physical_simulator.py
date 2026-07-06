from unittest.mock import MagicMock

import numpy as np

from bloqade.gemini.device.physical_simulator import (
    PhysicalResult,
    PhysicalSimulator,
    PhysicalSimulatorTask,
)


def test_physical_result_uses_post_processing():
    raw_measurements = [[True, False, True]]
    post_processing = MagicMock()
    post_processing.emit_return.return_value = ["return-value"]
    post_processing.emit_detectors.return_value = [[True, False]]
    post_processing.emit_observables.return_value = [[False]]

    result = PhysicalResult(
        _raw_measurements=raw_measurements,
        _detector_error_model=MagicMock(),
        _post_processing=post_processing,
        _fidelity_min=0.5,
        _fidelity_max=0.9,
    )

    assert result.measurements == raw_measurements
    assert result.return_values == ["return-value"]
    assert result.detectors == [[True, False]]
    assert result.observables == [[False]]
    post_processing.emit_return.assert_called_once_with(raw_measurements)
    post_processing.emit_detectors.assert_called_once_with(raw_measurements)
    post_processing.emit_observables.assert_called_once_with(raw_measurements)


def test_physical_task_run_clifford_returns_physical_result():
    task = MagicMock()
    task.tsim_circuit.is_clifford = True
    task.tsim_circuit.stim_circuit.compile_sampler.return_value.sample.return_value = (
        np.array([[True, False]])
    )
    task.fidelity_bounds.return_value = (0.5, 0.9)
    task.detector_error_model = MagicMock()
    task._post_processing = MagicMock()

    result = PhysicalSimulatorTask.run(task, shots=1, with_noise=True)

    task.tsim_circuit.stim_circuit.compile_sampler.assert_called_once_with()
    task.measurement_sampler.sample.assert_not_called()
    assert isinstance(result, PhysicalResult)
    assert result.measurements == [[True, False]]


def test_physical_simulator_exports_from_gemini_namespace():
    from bloqade.gemini import PhysicalSimulator
    from bloqade.gemini.device import PhysicalSimulator as DevicePhysicalSimulator

    assert PhysicalSimulator is DevicePhysicalSimulator


def test_physical_simulator_task_passes_placement_strategy(monkeypatch):
    import bloqade.lanes.pipeline as pipeline_module
    from bloqade.lanes.analysis import atom

    captured = {}
    move_kernel = MagicMock()
    move_kernel.dialects = "dialects"

    class FakePhysicalPipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def emit(self, kernel, no_raise=True):
            captured["kernel"] = kernel
            captured["no_raise"] = no_raise
            return move_kernel

    class FakeAtomInterpreter:
        def __init__(self, dialects, arch_spec):
            captured["atom_dialects"] = dialects
            captured["atom_arch_spec"] = arch_spec

        def get_post_processing(self, kernel):
            captured["post_processing_kernel"] = kernel
            return "post_processing"

    monkeypatch.setattr(pipeline_module, "PhysicalPipeline", FakePhysicalPipeline)
    monkeypatch.setattr(atom, "AtomInterpreter", FakeAtomInterpreter)

    kernel = MagicMock()
    placement_strategy = MagicMock()

    task = PhysicalSimulator().task(kernel, placement_strategy=placement_strategy)

    assert captured["placement_strategy"] is placement_strategy
    assert captured["kernel"] is kernel
    assert captured["no_raise"] is False
    assert captured["atom_dialects"] == "dialects"
    assert captured["post_processing_kernel"] is move_kernel
    assert task.physical_move_kernel is move_kernel
    assert task._post_processing == "post_processing"
