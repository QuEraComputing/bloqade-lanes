from concurrent.futures import Future
from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, assert_type
from unittest.mock import MagicMock

import numpy as np
import pytest
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import func, ilist, ssacfg

import bloqade.gemini.device.physical_simulator as physical_simulator_module
from bloqade import squin, types as bloqade_types
from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.device import (
    BackendSample,
    CliffTSimulatorBackend,
    DetectorResult,
    GeminiLogicalSimulatorTask,
    GeminiPhysicalSimulator,
    PyQrackSimulatorBackend,
    Result,
    TsimSimulatorBackend,
)
from bloqade.gemini.device._task_runtime import _SimulatorTaskBase
from bloqade.gemini.device.physical_simulator import (
    PhysicalResult,
    PhysicalSimulator,
    PhysicalSimulatorTask,
    append_measurements_and_annotations_physical,
)
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.logical_mvp import _find_qubit_ssas
from bloqade.lanes.pipeline import PhysicalPipeline


@squin.kernel
def small_physical_kernel():
    reg = squin.qalloc(1)
    squin.x(reg[0])
    return squin.broadcast.measure(reg)


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


def _mock_physical_task() -> Any:
    task = object.__new__(PhysicalSimulatorTask)
    backend = MagicMock()
    backend.detector_error_model.return_value = "dem"
    object.__setattr__(task, "physical_squin_kernel", "noisy-kernel")
    object.__setattr__(task, "noiseless_physical_squin_kernel", "noiseless-kernel")
    object.__setattr__(task, "simulator", SimpleNamespace(backend=backend))
    object.__setattr__(task, "fidelity_bounds", MagicMock(return_value=(0.5, 0.9)))
    object.__setattr__(task, "_post_processing", MagicMock())
    return task


def test_physical_task_run_routes_through_backend():
    task = _mock_physical_task()
    task.simulator.backend.sample.return_value = BackendSample(
        measurements=np.array([[True, False]])
    )

    result = PhysicalSimulatorTask.run(task, shots=1, with_noise=True)

    task.simulator.backend.detector_error_model.assert_called_once_with("noisy-kernel")
    task.simulator.backend.sample.assert_called_once_with(
        "noisy-kernel", shots=1, run_detectors=False
    )
    assert isinstance(result, PhysicalResult)
    assert result.measurements == [[True, False]]
    assert result.detector_error_model == "dem"


def test_physical_simulator_exports_from_gemini_namespace():
    from bloqade.gemini import GeminiPhysicalSimulator, PhysicalSimulator
    from bloqade.gemini.device import PhysicalSimulator as DevicePhysicalSimulator

    assert PhysicalSimulator is DevicePhysicalSimulator
    assert PhysicalSimulator is GeminiPhysicalSimulator


def test_logical_and_physical_tasks_share_non_dataclass_runtime():
    assert not is_dataclass(_SimulatorTaskBase)
    assert issubclass(GeminiLogicalSimulatorTask, _SimulatorTaskBase)
    assert issubclass(PhysicalSimulatorTask, _SimulatorTaskBase)


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

    def fake_append_measurements_and_annotations_physical(kernel, m2dets, m2obs):
        captured["annotated_kernel"] = kernel
        captured["m2dets"] = m2dets
        captured["m2obs"] = m2obs

    monkeypatch.setattr(pipeline_module, "PhysicalPipeline", FakePhysicalPipeline)
    monkeypatch.setattr(atom, "AtomInterpreter", FakeAtomInterpreter)
    monkeypatch.setattr(
        physical_simulator_module,
        "append_measurements_and_annotations_physical",
        fake_append_measurements_and_annotations_physical,
    )

    kernel = MagicMock()
    owned_kernel = MagicMock()
    kernel.similar.return_value = owned_kernel
    placement_strategy = MagicMock()
    place_opt_type: Any = MagicMock()
    m2dets = [[1]]
    m2obs = [[1]]

    simulator = PhysicalSimulator(
        place_opt_type=place_opt_type,
        placement_strategy=placement_strategy,
        m2dets=m2dets,
        m2obs=m2obs,
    )
    task = simulator.task(kernel)

    assert captured["place_opt_type"] is place_opt_type
    assert captured["placement_strategy"] is placement_strategy
    kernel.similar.assert_called_once_with()
    assert captured["annotated_kernel"] is owned_kernel
    assert captured["m2dets"] is m2dets
    assert captured["m2obs"] is m2obs
    assert captured["kernel"] is owned_kernel
    assert captured["no_raise"] is False
    assert captured["atom_dialects"] == "dialects"
    assert captured["post_processing_kernel"] is move_kernel
    assert task.physical_move_kernel is move_kernel
    assert task._post_processing == "post_processing"
    assert task.source_squin_kernel is owned_kernel
    assert task.simulator is simulator


def test_physical_task_preserves_source_kernel_across_repeated_compilation():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        return squin.broadcast.measure(reg)

    source_ir = kernel.print_str()
    simulator = PhysicalSimulator(m2dets=[[1], [1]], m2obs=[[1], [0]])
    first = simulator.task(kernel)
    second = simulator.task(kernel)

    assert kernel.print_str() == source_ir
    assert first.source_squin_kernel is not kernel
    assert second.source_squin_kernel is not kernel
    assert first.source_squin_kernel is not second.source_squin_kernel
    for task in (first, second):
        assert (
            sum(
                isinstance(stmt, SetDetector)
                for stmt in task.source_squin_kernel.callable_region.walk()
            )
            == 1
        )
        assert (
            sum(
                isinstance(stmt, SetObservable)
                for stmt in task.source_squin_kernel.callable_region.walk()
            )
            == 1
        )


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TsimSimulatorBackend(), id="tsim"),
        pytest.param(CliffTSimulatorBackend(seed=29), id="clifft"),
        pytest.param(PyQrackSimulatorBackend(seed=29), id="pyqrack"),
    ],
)
@pytest.mark.parametrize("with_noise", [False, True], ids=["noiseless", "noisy"])
def test_builtin_backends_run_physical_workflow_with_guaranteed_dem(
    backend, with_noise
):
    result = PhysicalSimulator(backend=backend).run(
        small_physical_kernel, shots=2, with_noise=with_noise
    )

    assert isinstance(result, Result)
    assert len(result.measurements) == 2
    assert all(len(measurements) == 1 for measurements in result.measurements)
    assert result.detector_error_model is not None
    if isinstance(backend, CliffTSimulatorBackend):
        backend._programs.clear()


def test_pyqrack_physical_detector_run_uses_measurement_fallback():
    simulator = PhysicalSimulator(
        backend=PyQrackSimulatorBackend(seed=30),
        m2dets=[[1]],
        m2obs=[[1]],
    )

    result = simulator.run(
        small_physical_kernel, shots=2, with_noise=False, run_detectors=True
    )

    assert isinstance(result, DetectorResult)
    assert result.detectors == ((True,), (True,))
    assert result.observables == ((True,), (True,))
    assert result.detector_error_model is not None


def test_append_measurements_and_annotations_physical_preserves_kernel_return():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(4)
        measurements = squin.broadcast.measure(reg)
        return measurements

    append_measurements_and_annotations_physical(
        kernel,
        m2dets=[[1], [1]],
        m2obs=[[1], [0]],
    )

    assert sum(isinstance(s, SetDetector) for s in kernel.callable_region.walk()) == 2
    assert sum(isinstance(s, SetObservable) for s in kernel.callable_region.walk()) == 2

    arch_spec = get_arch_spec()
    physical_move_kernel = PhysicalPipeline(arch_spec=arch_spec).emit(
        kernel, no_raise=False
    )
    post_processing = atom.AtomInterpreter(
        physical_move_kernel.dialects, arch_spec=arch_spec
    ).get_post_processing(physical_move_kernel)

    raw_shots = [[True, False, False, True]]

    return_values = list(post_processing.emit_return(raw_shots))
    assert len(return_values) == 1
    assert list(return_values[0]) == raw_shots[0]
    assert list(post_processing.emit_detectors(raw_shots)) == [[True, True]]
    assert list(post_processing.emit_observables(raw_shots)) == [[True, False]]


def test_append_measurements_and_annotations_physical_uses_logical_mvp_helpers(
    monkeypatch,
):
    import bloqade.lanes.logical_mvp as logical_mvp

    find_qubit_ssas = MagicMock(wraps=logical_mvp._find_qubit_ssas)
    find_return_stmt = MagicMock(wraps=logical_mvp._find_return_stmt)
    insert_before = MagicMock(wraps=logical_mvp._insert_before)
    monkeypatch.setattr(logical_mvp, "_find_qubit_ssas", find_qubit_ssas)
    monkeypatch.setattr(logical_mvp, "_find_return_stmt", find_return_stmt)
    monkeypatch.setattr(logical_mvp, "_insert_before", insert_before)

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.broadcast.measure(reg)

    append_measurements_and_annotations_physical(
        kernel,
        m2dets=[[1], [1]],
        m2obs=None,
    )

    find_qubit_ssas.assert_called_once_with(kernel)
    find_return_stmt.assert_called_once_with(kernel)
    assert insert_before.call_count > 0


def test_append_measurements_and_annotations_physical_validates_block_size():
    @squin.kernel
    def kernel():
        reg = squin.qalloc(3)
        squin.broadcast.measure(reg)

    with pytest.raises(
        ValueError,
        match="physical qubits must be divisible",
    ):
        append_measurements_and_annotations_physical(
            kernel,
            m2dets=[[1], [1]],
            m2obs=None,
        )


def test_append_measurements_and_annotations_physical_accepts_new_at_allocations():
    kernel = squin.kernel.add(gemini_qubit)
    kernel.run_pass = squin.kernel.run_pass

    @kernel(typeinfer=True)
    def pinned_kernel():
        q0 = gemini_qubit.new_at(0, 0, 0)
        q1 = gemini_qubit.new_at(0, 1, 0)
        squin.broadcast.measure(ilist.IList([q0, q1]))

    append_measurements_and_annotations_physical(
        pinned_kernel,
        m2dets=[[1], [1]],
        m2obs=[[1], [0]],
    )

    assert (
        sum(isinstance(s, SetDetector) for s in pinned_kernel.callable_region.walk())
        == 1
    )
    assert (
        sum(isinstance(s, SetObservable) for s in pinned_kernel.callable_region.walk())
        == 1
    )


def test_find_qubit_ssas_ignores_qubit_typed_results_without_addresses():
    test_dialect = ir.Dialect("test.qubit_alloc")

    @statement(dialect=test_dialect)
    class CustomQubitAlloc(ir.Statement):
        result: ir.ResultValue = info.result(bloqade_types.QubitType)

    block = ir.Block(argtypes=(types.MethodType,))
    alloc = CustomQubitAlloc()
    none_stmt = func.ConstantNone()
    for stmt in (alloc, none_stmt, func.Return(none_stmt.result)):
        block.stmts.append(stmt)

    function = func.Function(
        sym_name="custom_alloc",
        signature=func.Signature((), types.NoneType),
        slots=(),
        body=ir.Region(blocks=block),
    )
    method = ir.Method(
        dialects=ir.DialectGroup([func.dialect, ssacfg.dialect, test_dialect]),
        code=function,
        sym_name="custom_alloc",
        arg_names=[],
    )

    assert _find_qubit_ssas(method) == []


if TYPE_CHECKING:

    def _check_physical_run_overloads(
        simulator: GeminiPhysicalSimulator,
        task: PhysicalSimulatorTask[Any],
        kernel: ir.Method[[], Any],
        dynamic: bool,
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
        assert_type(simulator.run(kernel, run_detectors=False), Result[Any])
        assert_type(simulator.run(kernel, run_detectors=True), DetectorResult)
        assert_type(
            simulator.run(kernel, run_detectors=dynamic),
            Result[Any] | DetectorResult,
        )
        assert_type(
            simulator.run_async(kernel, run_detectors=False), Future[Result[Any]]
        )
        assert_type(
            simulator.run_async(kernel, run_detectors=True), Future[DetectorResult]
        )
        assert_type(
            simulator.run_async(kernel, run_detectors=dynamic),
            Future[Result[Any]] | Future[DetectorResult],
        )
