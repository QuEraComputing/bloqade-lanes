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
# # Compilation pipeline and dialects
#
# Bloqade Lanes progressively lowers a circuit through multiple intermediate
# representations. A logical program follows this simplified path:
#
# ```text
# logical SQuIN -> native gates -> place dialect -> move dialect
#               -> physical SQuIN / hardware lowering
# ```
#
# A *dialect* is a collection of IR statements with shared semantics. Keeping
# placement and movement in separate dialects lets analyses reason about the
# circuit before committing to concrete atom paths.

# %%
from bloqade import squin
from bloqade.gemini import logical
from bloqade.lanes.dialects import move, place
from bloqade.lanes.transform import LogicalNativeToPlace, LogicalPipeline


# %%
@logical.kernel(aggressive_unroll=True)
def bell_program():
    qubits = squin.qalloc(2)
    squin.h(qubits[0])
    squin.cx(qubits[0], qubits[1])
    return logical.terminal_measure(qubits)


# %% [markdown]
# The source kernel contains circuit-level operations.

# %%
bell_program.print()

# %% [markdown]
# `LogicalNativeToPlace` lowers gates and allocations into static placement
# regions. The result contains statements from the `place` dialect.

# %%
logical_pipeline = LogicalPipeline()
place_program = LogicalNativeToPlace(arch_spec=logical_pipeline.arch_spec).emit(
    bell_program, no_raise=False
)

place_statements = [
    statement
    for statement in place_program.callable_region.walk()
    if type(statement).__module__.startswith(place.__name__)
]
print(f"place statements: {len(place_statements)}")
place_program.print()

# %% [markdown]
# The complete `LogicalPipeline` additionally schedules placement regions,
# computes an initial layout, chooses routes, and rewrites the result into the
# move dialect.

# %%
move_program = logical_pipeline.emit(bell_program, no_raise=False)
move_statements = [
    statement
    for statement in move_program.callable_region.walk()
    if type(statement).__module__.startswith(move.__name__)
]
print(f"move statements: {len(move_statements)}")
move_program.print()

# %% [markdown]
# The public simulator/device APIs run these stages for you. Calling the passes
# directly is useful for compiler development, inspecting intermediate IR, and
# experimenting with layout or placement strategies.
