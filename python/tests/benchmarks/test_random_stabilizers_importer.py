from __future__ import annotations

import importlib.util
from pathlib import Path

import stim
from benchmarks.utils.import_random_stabilizers import (
    generate_random_stabilizers,
    render_squin_kernel_module,
)
from kirin import ir


def _load_methods_from_file(module_path):
    spec = importlib.util.spec_from_file_location(
        "test_random_stabilizer_module", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [value for value in vars(module).values() if isinstance(value, ir.Method)]


def test_rendered_module_exports_exactly_one_kernel_method(tmp_path):
    circuit = stim.Circuit("X 0\nY 0")
    kernel_name = "multi_qubit_rb_5_0_X"
    source = render_squin_kernel_module(kernel_name=kernel_name, circuit=circuit)

    module_path = tmp_path / f"{kernel_name}.py"
    module_path.write_text(source, encoding="utf-8")
    kernels = _load_methods_from_file(module_path)
    assert len(kernels) == 1
    assert "@squin.kernel(typeinfer=True, fold=True)" in source
    assert "import stim" not in source


def test_generate_random_stabilizers_writes_python_modules(tmp_path):
    output_dir = tmp_path / "random_stabilizers"
    count = generate_random_stabilizers(
        gemini_repo=Path("/Users/jasonludmir/Documents/gemini_benchmarking"),
        output_dir=output_dir,
    )

    assert count == 60
    assert not (output_dir / "_provenance").exists()
    generated_files = [
        path for path in output_dir.glob("*.py") if path.name != "__init__.py"
    ]
    assert len(generated_files) == 60
