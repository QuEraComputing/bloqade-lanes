from typing import Sequence, TypeGuard, TypeVar

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
    state_vector *= np.exp(-1j * np.angle(state_vector[i]))
    other_state_vector *= np.exp(-1j * np.angle(other_state_vector[i]))

    print(state_vector)
    print(other_state_vector)
    if state_vector.shape != other_state_vector.shape:
        return False

    return np.allclose(state_vector, other_state_vector, atol=atol, rtol=rtol)
