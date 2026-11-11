"""Public APIs remain available while compiler definitions belong to lanes."""

import subprocess
import sys

import pytest
from kirin.dialects import ilist
from kirin.serialization import JSONSerializer

from bloqade.gemini import logical
from bloqade.lanes.dialects import arch, steane713
from bloqade.lanes.dialects.steane713 import extensions


def test_gemini_frontend_reexports_lanes_definitions():
    assert logical.loc is arch.loc
    assert logical.terminal_measure is steane713.terminal_measure
    assert extensions.dialect is not steane713.dialect


def test_new_dialect_serialization_roundtrip():
    @logical.kernel(verify=False)
    def program():
        q = logical.qalloc_at(ilist.IList([2]))
        logical.extensions.star_rz(0.1, q[0])
        return logical.terminal_measure(q)

    serializer = JSONSerializer()
    payload = serializer.encode(program.dialects.encode(program, version="1.0.0"))
    restored = program.dialects.decode(serializer.decode(payload))
    # Deserialization may intern equal type attributes, changing JSON reference
    # IDs without changing the IR. Compare the decoded program structurally.
    assert program.code.is_structurally_equal(restored.code)


@pytest.mark.parametrize("first", ["bloqade.lanes.transform", "bloqade.gemini"])
def test_import_order(first):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module({first!r}); import bloqade.gemini; import bloqade.lanes.transform",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lanes_registers_compiler_method_tables():
    # A fresh process distinguishes registration at lanes import from test
    # import order. Every table a lanes pipeline needs must come from lanes.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from bloqade.lanes.dialects import arch, steane713, arrange
from bloqade.lanes.dialects.steane713 import extensions
from bloqade.lanes.validation import (
    address, const_address, duplicate_address, steane, terminal_measure,
    terminal_measurement,
)
from bloqade import qubit
from bloqade.squin import gate
from kirin.dialects import func, scf
requirements = [
    (arch.dialect, 'qubit.address'),
    (arch.dialect, 'lanes.arch.const_address'),
    (arch.dialect, 'lanes.arch.duplicates'),
    (arch.dialect, 'move.address.validation'),
    (arrange.dialect, 'move.address.validation'),
    (qubit.dialect, 'lanes.validate.physical.terminal_measure'),
    (steane713.dialect, 'typeinfer'),
    (steane713.dialect, 'measure_id'),
]
requirements += [(d, 'lanes.validate.logical') for d in
    (qubit.dialect, gate.dialect, scf.dialect, extensions.dialect)]
requirements += [(d, 'lanes.validate.terminal_measurement') for d in
    (qubit.dialect, gate.dialect, func.dialect, steane713.dialect, extensions.dialect)]
for dialect, key in requirements:
    assert key in dialect.interps, (dialect.name, key)
    assert type(dialect.interps[key]).__module__.startswith('bloqade.lanes.')
assert not any(n.startswith('bloqade.gemini') for n in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
