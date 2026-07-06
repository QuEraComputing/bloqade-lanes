from unittest.mock import MagicMock

import bloqade.squin as squin
from kirin.dialects import ilist

from bloqade.gemini.device.abstract_simulator import (
    AbstractSimulator,
    AbstractSimulatorTask,
    CliffTSimulatorTask,
    TsimSimulatorTask,
)
from bloqade.gemini.device.physical_simulator import (
    PhysicalCliffTSimulator,
    PhysicalCliffTSimulatorTask,
    PhysicalSimulator,
    PhysicalSimulatorTask,
)
from bloqade.gemini.device.simulator import (
    GeminiLogicalCliffTSimulator,
    GeminiLogicalCliffTSimulatorTask,
    GeminiLogicalSimulator,
    GeminiLogicalSimulatorTask,
)


def test_simulator_tasks_use_shared_backend_bases():
    assert issubclass(GeminiLogicalSimulatorTask, TsimSimulatorTask)
    assert issubclass(PhysicalSimulatorTask, TsimSimulatorTask)
    assert issubclass(GeminiLogicalCliffTSimulatorTask, CliffTSimulatorTask)
    assert issubclass(PhysicalCliffTSimulatorTask, CliffTSimulatorTask)


def test_simulators_use_shared_abstract_base():
    assert issubclass(GeminiLogicalSimulator, AbstractSimulator)
    assert issubclass(GeminiLogicalCliffTSimulator, AbstractSimulator)
    assert issubclass(PhysicalSimulator, AbstractSimulator)
    assert issubclass(PhysicalCliffTSimulator, AbstractSimulator)


def test_logical_clifft_simulator_task_uses_compile_task(monkeypatch):
    import bloqade.lanes.logical_mvp as logical_mvp

    compiled = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    monkeypatch.setattr(logical_mvp, "compile_task", MagicMock(return_value=compiled))

    task = GeminiLogicalCliffTSimulator(seed=123).task(MagicMock())

    assert isinstance(task, GeminiLogicalCliffTSimulatorTask)
    assert task.seed == 123


def test_logical_backend_clifft_compatibility_returns_legacy_task(monkeypatch):
    import bloqade.lanes.logical_mvp as logical_mvp
    from bloqade.gemini.decoding.tasks import _CliffTSimulatorTask

    compiled = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    monkeypatch.setattr(logical_mvp, "compile_task", MagicMock(return_value=compiled))

    task = GeminiLogicalSimulator(backend="clifft", seed=456).task(MagicMock())

    assert isinstance(task, _CliffTSimulatorTask)
    assert task.seed == 456


def test_task_subclasses_only_declare_source_kernel_fields():
    assert GeminiLogicalSimulatorTask.__annotations__ == {}
    assert GeminiLogicalCliffTSimulatorTask.__annotations__ == {}
    assert PhysicalSimulatorTask.__annotations__ == {}
    assert PhysicalCliffTSimulatorTask.__annotations__ == {}


def test_task_source_kernel_lives_on_abstract_task():
    assert AbstractSimulatorTask.__annotations__["logical_squin_kernel"] == (
        "ir.Method[[], RetType]"
    )


def test_simulator_task_entrypoints_expect_zero_arg_kernels():
    assert AbstractSimulator.task.__annotations__["kernel"] == "ir.Method[[], TaskRet]"
    assert AbstractSimulator.run.__annotations__["logical_squin_kernel"] == (
        "ir.Method[[], TaskRet]"
    )
    assert AbstractSimulator.run_async.__annotations__["logical_squin_kernel"] == (
        "ir.Method[[], TaskRet]"
    )
    assert AbstractSimulator.tsim_circuit.__annotations__["logical_squin_kernel"] == (
        "ir.Method[[], TaskRet]"
    )
    assert AbstractSimulator.visualize.__annotations__["logical_squin_kernel"] == (
        "ir.Method[[], TaskRet]"
    )
    assert (
        AbstractSimulator.physical_squin_kernel.__annotations__["logical_squin_kernel"]
        == "ir.Method[[], TaskRet]"
    )
    assert (
        AbstractSimulator.physical_move_kernel.__annotations__["logical_squin_kernel"]
        == "ir.Method[[], TaskRet]"
    )
    assert AbstractSimulator.fidelity_bounds.__annotations__[
        "logical_squin_kernel"
    ] == ("ir.Method[[], TaskRet]")

    assert GeminiLogicalSimulator.task.__annotations__["logical_kernel"] == (
        "Union[ir.Method[[], RetType], Callable[..., Any]]"
    )
    assert GeminiLogicalCliffTSimulator.task.__annotations__["logical_kernel"] == (
        "Union[ir.Method[[], RetType], Callable[..., Any]]"
    )
    assert PhysicalSimulator.task.__annotations__["physical_kernel"] == (
        "ir.Method[[], RetType]"
    )
    assert PhysicalCliffTSimulator.task.__annotations__["physical_kernel"] == (
        "ir.Method[[], RetType]"
    )


def test_physical_task_source_kernel_aliases_logical_kernel_field():
    source_kernel = MagicMock()
    task = PhysicalSimulatorTask(
        source_kernel,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        seed=789,
    )

    assert task.logical_squin_kernel is source_kernel
    assert task.source_squin_kernel is source_kernel
    assert task.seed == 789


def test_physical_simulator_uses_shared_logical_noise_model_default():
    from bloqade.lanes.rewrite.move2squin.noise import LogicalNoiseModelABC

    assert isinstance(PhysicalSimulator().noise_model, LogicalNoiseModelABC)
    assert isinstance(PhysicalCliffTSimulator().noise_model, LogicalNoiseModelABC)


def test_task_physical_squin_kernel_uses_logical_move_to_squin(monkeypatch):
    import bloqade.lanes.transform as transform

    emitted_kernel = MagicMock()
    source_kernel = MagicMock()
    noise_model = MagicMock()
    arch_spec = MagicMock()
    move_kernel = MagicMock()
    captured: dict[str, object] = {}

    class FakeMoveToSquinLogical:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def emit(self, kernel):
            captured["kernel"] = kernel
            return emitted_kernel

    monkeypatch.setattr(transform, "MoveToSquinLogical", FakeMoveToSquinLogical)

    task = PhysicalSimulatorTask(
        source_kernel,
        noise_model,
        arch_spec,
        move_kernel,
        MagicMock(),
    )

    assert task.physical_squin_kernel is emitted_kernel
    assert captured == {
        "arch_spec": arch_spec,
        "noise_model": noise_model,
        "add_noise": True,
        "kernel": move_kernel,
    }


def test_logical_move_to_squin_matches_physical_for_physical_move_kernel():
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.noise_model import generate_logical_noise_model
    from bloqade.lanes.pipeline import PhysicalPipeline
    from bloqade.lanes.transform import MoveToSquinLogical, MoveToSquinPhysical

    @squin.kernel
    def physical_kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        squin.broadcast.measure(ilist.IList([reg[0], reg[1]]))

    arch_spec = get_arch_spec()
    physical_move_kernel = PhysicalPipeline(arch_spec=arch_spec).emit(
        physical_kernel, no_raise=False
    )
    noise_model = generate_logical_noise_model()

    logical_squin = MoveToSquinLogical(
        arch_spec=arch_spec,
        noise_model=noise_model,
        add_noise=False,
    ).emit(physical_move_kernel, no_raise=False)
    physical_squin = MoveToSquinPhysical(
        arch_spec=arch_spec,
    ).emit(physical_move_kernel, no_raise=False)

    assert logical_squin.print_str() == physical_squin.print_str()


def test_logical_move_to_squin_matches_physical_for_physical_move_kernel_noisy():
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.noise_model import (
        generate_logical_noise_model,
        generate_simple_noise_model,
    )
    from bloqade.lanes.pipeline import PhysicalPipeline
    from bloqade.lanes.transform import MoveToSquinLogical, MoveToSquinPhysical

    @squin.kernel
    def physical_kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        squin.broadcast.measure(ilist.IList([reg[0], reg[1]]))

    arch_spec = get_arch_spec()
    physical_move_kernel = PhysicalPipeline(arch_spec=arch_spec).emit(
        physical_kernel, no_raise=False
    )
    noise_model = generate_logical_noise_model()
    noise_model_phys = generate_simple_noise_model()

    logical_squin = MoveToSquinLogical(
        arch_spec=arch_spec,
        noise_model=noise_model,
        add_noise=True,
    ).emit(physical_move_kernel, no_raise=False)
    physical_squin = MoveToSquinPhysical(
        arch_spec=arch_spec, noise_model=noise_model_phys
    ).emit(physical_move_kernel, no_raise=False)

    assert logical_squin.print_str() == physical_squin.print_str()


def test_concrete_simulators_inherit_shared_tsim_circuit_method():
    method_names = (
        "run",
        "run_async",
        "visualize",
        "physical_squin_kernel",
        "physical_move_kernel",
        "fidelity_bounds",
        "tsim_circuit",
    )
    simulator_classes = (
        GeminiLogicalSimulator,
        GeminiLogicalCliffTSimulator,
        PhysicalSimulator,
        PhysicalCliffTSimulator,
    )
    for simulator_cls in simulator_classes:
        for method_name in method_names:
            assert getattr(simulator_cls, method_name) is getattr(
                AbstractSimulator, method_name
            )


def test_backend_tasks_inherit_shared_tsim_circuit_properties():
    assert TsimSimulatorTask.tsim_circuit is AbstractSimulatorTask.tsim_circuit
    assert CliffTSimulatorTask.tsim_circuit is AbstractSimulatorTask.tsim_circuit
    assert (
        TsimSimulatorTask.noiseless_tsim_circuit
        is AbstractSimulatorTask.noiseless_tsim_circuit
    )
    assert (
        CliffTSimulatorTask.noiseless_tsim_circuit
        is AbstractSimulatorTask.noiseless_tsim_circuit
    )
