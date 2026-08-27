# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Converting a CUDA-Q kernel
#
# Bloqade Lanes compiles SQuIN/Kirin methods. A CUDA-Q kernel must therefore be
# converted explicitly through QIR before it is passed to a Gemini device.
# Install the optional dependencies with `pip install "bloqade-lanes[cudaq]"`.

# %%
import importlib.util

from bloqade.gemini.cudaq import cudaq_to_squin

# %%
if importlib.util.find_spec("cudaq") is None:
    print("CUDA-Q is not installed; install bloqade-lanes[cudaq] to run this cell.")
else:
    import cudaq  # type: ignore[reportMissingImports]

    @cudaq.kernel
    def cudaq_bell_pair():
        qubits = cudaq.qvector(2)
        h(qubits[0])  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
        cx(  # noqa: F821  # pyright: ignore[reportUndefinedVariable]
            qubits[0], qubits[1]
        )

    squin_kernel = cudaq_to_squin(cudaq_bell_pair)
    squin_kernel.print()

# %% [markdown]
# `cudaq_to_squin` performs the following conversion:
#
# ```text
# CUDA-Q kernel -> base-profile QIR -> SQuIN ir.Method
# ```
#
# The converted method contains the circuit but not application-specific
# terminal measurement, detector, or observable annotations. Add those before
# passing it to `GeminiLogicalSimulator.task(...)`. This keeps conversion
# separate from Gemini-specific post-processing policy.
