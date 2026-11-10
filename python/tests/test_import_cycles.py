"""Compiler imports are independent; frontend tables are explicitly required."""

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


def test_missing_frontend_tables_fail_clearly():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_independent_compilation.py"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
