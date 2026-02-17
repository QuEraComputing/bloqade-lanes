import numpy as np
from bloqade.pyqrack.device import StackMemorySimulator
from kirin import ir


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

    Returns:
        bool: True if the methods are equivalent, False otherwise.

    Note:
        The methods should not perform any measurements. Otherwise, the state vectors may not be comparable.
    """

    simulator = StackMemorySimulator()
    state_vector = np.asarray(simulator.state_vector(squin_method))
    other_state_vector = np.asarray(simulator.state_vector(other_squin_method))

    state_vector *= np.exp(-1j * np.angle(state_vector[0]))
    other_state_vector *= np.exp(-1j * np.angle(other_state_vector[0]))

    return np.allclose(state_vector, other_state_vector, atol=atol, rtol=rtol)
