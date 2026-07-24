from collections.abc import Sequence
from typing import TypeGuard, TypeVar

import numpy as np
from kirin import ir

T = TypeVar("T")


def no_none_elements(xs: Sequence[T | None]) -> TypeGuard[Sequence[T]]:
    """Check that there are no None elements in the sequence.

    Args:
        xs: A sequence that may contain None elements.

    Returns:
        A TypeGuard indicating that all elements are not None.

    """
    return all(x is not None for x in xs)


def no_none_elements_tuple(xs: tuple[T | None, ...]) -> TypeGuard[tuple[T, ...]]:
    """Check that there are no None elements in the tuple.

    Args:
        xs: A tuple that may contain None elements.

    Returns:
        A TypeGuard indicating that all elements are not None.

    """
    return all(x is not None for x in xs)


def statements_outside_dialect_group(method: ir.Method) -> list[ir.Statement]:
    """Return every statement in *method* whose dialect is not in its group.

    Kirin's ``Method.verify()`` checks per-statement structure but never
    validates that each statement's dialect is a member of the method's
    declared ``DialectGroup``. An out-of-group statement therefore passes
    ``verify()`` cleanly and only surfaces lazily — as an ``InterpreterError``
    at interpretation/emit time (see ``kirin.interp.abc``), far from its cause.

    Use this after a lowering pass that is expected to leave no statements from
    the source dialect behind (e.g. ``move`` after ``RewriteMoveToStackMove``)
    to fail fast with a precise, local error instead.

    Statements with no dialect (``stmt.dialect is None``, e.g. dialect-agnostic
    base statements) are ignored.

    Args:
        method: The kernel to scan.

    Returns:
        The offending statements, in walk order. Empty if every statement's
        dialect is a member of ``method.dialects``.
    """
    dialects = method.dialects
    return [
        stmt
        for stmt in method.code.walk()
        if stmt.dialect is not None and stmt.dialect not in dialects
    ]


def check_circuit(
    squin_method: ir.Method[[], None],
    other_squin_method: ir.Method[[], None],
    atol: float = 1.0e-5,
    rtol: float = 1.0e-8,
) -> bool:
    """Check if two squin methods are equivalent.

    Args:
        squin_method (ir.Method[[], None]): The first squin method. This method should not take
            any arguments and return None.
        other_squin_method (ir.Method[[], None]): The second squin method. This method should not take
            any arguments and return None.
        atol (float, optional): Absolute tolerance for state vector comparison. Defaults to 1.0e-5.
        rtol (float, optional): Relative tolerance for state vector comparison. Defaults to 1

    Returns:
        bool: True if the methods are equivalent, False otherwise.

    Note:
        The methods should not perform any measurements. Otherwise, the state vectors may not be comparable.
    """
    from bloqade.pyqrack.device import StackMemorySimulator

    simulator = StackMemorySimulator()
    state_vector = np.asarray(simulator.state_vector(squin_method))
    other_state_vector = np.asarray(simulator.state_vector(other_squin_method))

    i = np.argmax(np.abs(state_vector))
    j = np.argmax(np.abs(other_state_vector))
    state_vector *= np.exp(-1j * np.angle(state_vector[i]))
    other_state_vector *= np.exp(-1j * np.angle(other_state_vector[j]))

    if state_vector.shape != other_state_vector.shape:
        return False

    return np.allclose(state_vector, other_state_vector, atol=atol, rtol=rtol)
