import os
import subprocess
import sys
import textwrap

import cirq
import pytest
from bloqade.cirq_utils import emit_circuit, load_circuit
from kirin import lowering

from bloqade.gemini import physical
from bloqade.gemini.common.dialects.qubit.stmts import NewAt


def test_physical_pinned_qubit_round_trip():
    from bloqade.gemini.common.cirq_conversion import GeminiQubit

    pinned = GeminiQubit(0, pin=(0, 0, 0))
    circuit = cirq.Circuit(cirq.H(pinned), cirq.measure(pinned))

    method = load_circuit(circuit, dialects=physical.kernel)

    assert sum(isinstance(stmt, NewAt) for stmt in method.callable_region.walk()) == 1
    assert emit_circuit(method) == circuit


def test_physical_pinned_loader_rejects_register_as_argument():
    from bloqade.gemini.common.cirq_conversion import GeminiQubit

    pinned = GeminiQubit(0, pin=(0, 0, 0))
    circuit = cirq.Circuit(cirq.measure(pinned))

    with pytest.raises(lowering.BuildError, match="pinned"):
        load_circuit(circuit, dialects=physical.kernel, register_as_argument=True)


def test_physical_new_at_emits_without_importing_logical_converter(tmp_path):
    script = tmp_path / "physical_cirq_emission.py"
    script.write_text(textwrap.dedent("""
            import sys

            from kirin.dialects import ilist
            from bloqade import squin
            from bloqade.cirq_utils import emit_circuit
            from bloqade.gemini import physical
            from bloqade.gemini.common.dialects.qubit import new_at

            @physical.kernel()
            def program():
                qubit = new_at(0, 0, 0)
                squin.h(qubit)
                return squin.broadcast.measure(ilist.IList([qubit]))

            assert "bloqade.gemini.logical.cirq_conversion" not in sys.modules
            circuit = emit_circuit(program, ignore_returns=True)
            assert "bloqade.gemini.logical.cirq_conversion" not in sys.modules
            print(repr(next(iter(circuit.all_qubits()))))
            """))

    completed = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path)},
    )

    assert completed.returncode == 0, completed.stderr
    assert "GeminiQubit(index=0, pin=(0, 0, 0))" in completed.stdout
