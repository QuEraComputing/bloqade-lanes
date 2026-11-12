"""Compiler imports are independent; Gemini supplies its own method tables."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "bloqade.lanes.dialects.move",
        "bloqade.lanes.dialects.arch",
        "bloqade.lanes.dialects.steane713",
        "bloqade.lanes.prelude",
        "bloqade.lanes.validation.address",
        "bloqade.lanes.transform",
    ],
)
def test_lanes_module_imports_without_gemini(module: str):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import importlib
import importlib.abc
import sys
class NoGemini(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'bloqade.gemini' or fullname.startswith('bloqade.gemini.'):
            raise AssertionError('Compiler imported the Gemini product: ' + fullname)
sys.meta_path.insert(0, NoGemini())
importlib.import_module({module!r})
assert not any(n.startswith('bloqade.gemini') for n in sys.modules)
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_lanes_to_gemini_imports():
    root = Path(__file__).parents[1] / "bloqade" / "lanes"
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = ["bloqade", "lanes", *path.relative_to(root).parts[:-1]]
                    module = ".".join(package[: len(package) - node.level + 1]) + (
                        "." + module if module else ""
                    )
                names = [module, *(module + "." + alias.name for alias in node.names)]
            else:
                continue
            if any(
                name == "bloqade.gemini" or name.startswith("bloqade.gemini.")
                for name in names
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not violations, "Lanes must not depend on Gemini: " + ", ".join(violations)


def test_lanes_compiles_without_gemini():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_independent_compilation.py"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lanes_leaves_gemini_method_tables_unregistered():
    # A fresh process ensures test import order cannot hide duplicate frontend
    # registrations. These tables are supplied by the separate Gemini package.
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
    assert key not in dialect.interps, (dialect.name, key)
assert not any(n.startswith('bloqade.gemini') for n in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
