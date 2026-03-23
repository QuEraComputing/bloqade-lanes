from __future__ import annotations

from typing import Any, Callable

from kirin import ir

from bloqade.gemini import logical


def is_cudaq_kernel(obj: Any) -> bool:
    """Check whether *obj* is a CUDA-Q kernel (``PyKernelDecorator``)."""
    try:
        from cudaq import PyKernelDecorator  # type: ignore[reportMissingImports]
    except ImportError:
        return False
    return isinstance(obj, PyKernelDecorator)


def cudaq_to_squin(kernel: Callable[..., Any]) -> ir.Method:
    """Convert a CUDA-Q kernel to a squin ``ir.Method``.

    The conversion pipeline is::

        CUDA-Q kernel  →  QIR (base profile)  →  squin ir.Method

    Args:
        kernel: A CUDA-Q ``PyKernelDecorator`` instance.

    Returns:
        The squin ``ir.Method`` corresponding to *kernel*.
    """
    import cudaq as cudaq_module  # type: ignore[reportMissingImports]
    from qbraid_qir.squin import load  # type: ignore[reportMissingImports]

    qir_str = cudaq_module.translate(kernel, format="qir-base")
    mt: ir.Method = load(qir_str, dialects=logical.kernel)

    assert (run_pass := logical.kernel.run_pass) is not None
    run_pass(mt, aggressive_unroll=True, verify=False)

    return mt
