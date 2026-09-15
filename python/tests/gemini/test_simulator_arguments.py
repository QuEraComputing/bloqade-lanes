"""Classical arguments are specialized before Gemini task compilation."""

# Expected type errors below must remain errors as the public signature evolves.
# pyright: reportUnnecessaryTypeIgnoreComment=true

from typing import TYPE_CHECKING

import pytest
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.gemini import GeminiLogicalSimulator, logical
from bloqade.gemini.device._arguments import bind_task_arguments


@logical.kernel(verify=False)
def parameterized(n: int, flip: bool = False):
    q = squin.qalloc(n)
    if flip:
        squin.x(q[0])
    return logical.terminal_measure(q)


@logical.kernel(aggressive_unroll=True)
def fixed():
    q = squin.qalloc(1)
    return logical.terminal_measure(q)


@pytest.mark.parametrize(
    "args, kwargs", [((1,), None), ((), {"n": 1}), ((1,), {"flip": True})]
)
def test_classical_arguments_compile(args, kwargs):
    original = parameterized.similar()
    task = GeminiLogicalSimulator().task(parameterized, *args, **(kwargs or {}))
    assert task.logical_squin_kernel.args == ()
    assert task.physical_move_kernel.args == ()
    assert parameterized.code.is_structurally_equal(original.code)
    assert parameterized.arg_names == original.arg_names
    assert parameterized.nargs == original.nargs


def test_zero_argument_api_unchanged():
    task = GeminiLogicalSimulator().task(fixed)
    assert task.logical_squin_kernel.args == ()


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        ((), None, "missing"),
        ((1, False, 3), None, "too many"),
        ((1,), {"n": 2}, "multiple values"),
        ((1,), {"typo": True}, "unexpected keyword"),
        (("one",), None, "has type"),
        (([],), None, "immutable classical"),
    ],
)
def test_argument_errors(args, kwargs, message):
    with pytest.raises(TypeError, match=message):
        GeminiLogicalSimulator().task(parameterized, *args, **(kwargs or {}))


def test_bound_program_is_still_validated():
    with pytest.raises(ValidationErrorGroup, match="exceeding the maximum"):
        GeminiLogicalSimulator().task(parameterized, 11)


def test_ir_only_kernel_can_bind_named_arguments():
    kernel = parameterized.similar()
    kernel.py_func = None
    bound = bind_task_arguments(kernel, n=1, flip=False)
    assert bound.args == ()
    bound.verify()


def test_distinct_arguments_produce_independent_tasks():
    sim = GeminiLogicalSimulator()
    first = sim.task(parameterized, n=1, flip=False)
    second = sim.task(parameterized, n=2, flip=False)
    assert not first.logical_squin_kernel.code.is_structurally_equal(
        second.logical_squin_kernel.code
    )
    assert first.logical_squin_kernel is not second.logical_squin_kernel


@logical.kernel(verify=False)
def rotated(theta: float):
    q = squin.qalloc(1)
    squin.ry(theta, q[0])
    return logical.terminal_measure(q)


@logical.kernel(verify=False)
def tuple_argument(config: tuple[int, bool]):
    q = squin.qalloc(config[0])
    if config[1]:
        squin.x(q[0])
    return logical.terminal_measure(q)


def test_float_and_tuple_arguments_compile():
    sim = GeminiLogicalSimulator()
    assert sim.task(rotated, theta=0.0).logical_squin_kernel.args == ()
    assert sim.task(tuple_argument, (1, True)).logical_squin_kernel.args == ()
    with pytest.raises(TypeError, match="has type"):
        sim.task(tuple_argument, ("bad", True))  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("flip", [False, True])
def test_noiseless_run_obeys_bound_argument(flip):
    task = GeminiLogicalSimulator().task(parameterized, 1, flip=flip)
    result = task.run(shots=4, with_noise=False)
    assert len(result.return_values) == 4
    # The logical Z result is the parity of the seven Steane data-qubit bits.
    for shot in result.return_values:
        assert bool(sum(bool(bit) for bit in shot[0]) % 2) is flip


def test_static_arguments_unroll_loops():
    @logical.kernel(verify=False)
    def repeated(count: int):
        q = squin.qalloc(1)
        for _ in range(count):
            squin.x(q[0])
        return logical.terminal_measure(q)

    task = GeminiLogicalSimulator().task(repeated, 3)
    result = task.run(shots=2, with_noise=False)
    assert all(
        sum(bool(bit) for bit in shot[0]) % 2 == 1 for shot in result.return_values
    )


@squin.kernel
def echo(value):
    return value


@pytest.mark.parametrize("value", [None, "label", 0.25, True, 3, (1, (False, "x"))])
def test_supported_constants_preserve_values(value):
    bound = bind_task_arguments(echo, value)
    assert bound() == value


@pytest.mark.parametrize("value", [object(), {"x": 1}, (1, []), [1, 2]])
def test_mutable_or_opaque_values_are_rejected(value):
    with pytest.raises(TypeError, match="immutable classical"):
        bind_task_arguments(echo, value)


def test_zero_argument_kernel_rejects_extra_arguments():
    with pytest.raises(TypeError, match="too many"):
        GeminiLogicalSimulator().task(fixed, 1)  # pyright: ignore[reportCallIssue]


def test_named_kernel_parameter_and_forwarded_keyword_arguments():
    @logical.kernel(verify=False)
    def named(args: int, kwargs: float):
        q = squin.qalloc(args)
        squin.broadcast.rx(kwargs, q)
        return logical.terminal_measure(q)

    task = GeminiLogicalSimulator().task(kernel=named, args=1, kwargs=0.0)
    assert task.logical_squin_kernel.args == ()


if TYPE_CHECKING:
    from kirin import ir
    from typing_extensions import assert_type

    from bloqade.gemini.device import GeminiLogicalSimulatorTask

    def check_task_signature(kernel: ir.Method[[float, float], int]) -> None:
        simulator = GeminiLogicalSimulator()
        assert_type(simulator.task(kernel, 0.1, 0.2), GeminiLogicalSimulatorTask[int])
        assert_type(bind_task_arguments(kernel, 0.1, 0.2), ir.Method[[], int])
        bind_task_arguments(kernel, "wrong", 1.0)  # pyright: ignore[reportArgumentType]
        bind_task_arguments(kernel, 0.1)  # pyright: ignore[reportCallIssue]
        bind_task_arguments(kernel, 0.1, 0.2, 0.3)  # pyright: ignore[reportCallIssue]
        bind_task_arguments(rotated, angle=0.1)  # pyright: ignore[reportCallIssue]
        simulator.task(kernel, "wrong", 1.0)  # pyright: ignore[reportArgumentType]
        simulator.task(kernel, 0.1)  # pyright: ignore[reportCallIssue]
        simulator.task(kernel, 0.1, 0.2, 0.3)  # pyright: ignore[reportCallIssue]
        simulator.task(rotated, angle=0.1)  # pyright: ignore[reportCallIssue]
