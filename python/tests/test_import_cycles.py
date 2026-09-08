import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "bloqade.lanes.dialects.move",
        "bloqade.lanes.prelude",
        "bloqade.lanes.validation.address",
        "bloqade.gemini.device.logical.utils",
    ],
)
def test_lanes_module_imports_in_fresh_interpreter(module: str):
    """Public Lanes modules import without relying on prior import order."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import importlib; importlib.import_module({module!r})",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_logical_utils_defers_lanes_compilation_imports():
    """Importing result helpers does not initialize the Lanes compiler stack.

    ``bloqade.gemini.device.logical.utils`` is imported while the public Gemini
    package is initialized.  Eagerly importing these modules reintroduces the
    Gemini--Lanes import cycle through the move dialect.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bloqade.gemini.device.logical.utils; "
                "assert 'bloqade.lanes.analysis.atom' not in sys.modules; "
                "assert 'bloqade.lanes.analysis.atom._shot_remapping' not in sys.modules; "
                "assert 'bloqade.lanes.arch.gemini.physical' not in sys.modules; "
                "assert 'bloqade.lanes.transform' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
