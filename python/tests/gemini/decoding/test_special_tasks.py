from typing import Any, cast

import stim

from bloqade.gemini.decoding.special_tasks import (
    _apply_noiseless_unitary_prefix,
    _apply_special_tsim_circuit_strategy,
)


def test_apply_noiseless_unitary_prefix_returns_stim_circuits_without_mutating_inputs():
    circuit = stim.Circuit("""
        H 0
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
        """)
    original_text = str(circuit)

    transformed = _apply_noiseless_unitary_prefix({"X": circuit})

    assert transformed["X"] is not circuit
    assert isinstance(transformed["X"], stim.Circuit)
    assert str(circuit) == original_text
    assert str(transformed["X"]).startswith("H 0 0")


def test_apply_special_tsim_circuit_strategy_updates_noisy_and_noiseless_stim_cache():
    class Task:
        pass

    task = cast(Any, Task())
    task.stim_circuit = stim.Circuit("""
        H 0
        X_ERROR(0.1) 0
        M 0
        DETECTOR rec[-1]
        """)
    task.noiseless_stim_circuit = stim.Circuit("""
        H 0
        M 0
        DETECTOR rec[-1]
        """)

    transformed = _apply_special_tsim_circuit_strategy(cast(Any, {"X": task}))

    assert transformed["X"] is task
    assert str(task.stim_circuit).startswith("H 0 0")
    assert str(task.noiseless_stim_circuit).startswith("H 0 0")
