# %% [markdown]
# # Example: Gemini Logical Lanes Compilation Pipeline
# ---
# In this example, we demonstrate the compilation pipeline for a simple quantum kernel
# using the Gemini logical lanes architecture. We will walk through the steps of compiling
# a basic squin kernel, performing layout and placement analyses, rewriting to move dialect,
# and finally inverting back to the squin dialect with physical qubits and noise models inserted.

# to start we create a simple squin kernel that uses the gemini logical dialect to define a
# terminal measurement on the circuit.

from bloqade.decoders.dialects import annotate
from bloqade.gemini import logical as gemini_logical

# %%
from bloqade import squin

kernel = squin.kernel.add(gemini_logical.dialect).add(annotate)
kernel.run_pass = squin.kernel.run_pass


@kernel
def example_kernel():
    reg = squin.qalloc(2)

    squin.cx(reg[0], reg[1])

    gemini_logical.terminal_measure(reg)


example_kernel.print()

# %% [markdown]
# ## Compilation Pipeline
# The first step is to compile the logical squin kernel to the native gate set.
# this can be done by using the existing SquinToNative transformer inside bloqade-circuit.

# %%
from bloqade.native.upstream import SquinToNative  # noqa: E402

example_kernel = SquinToNative().emit(example_kernel)
example_kernel.print()

# %% [markdown]
# Note that the kernel looks identical to before, because the rewrite pass rewrites the call
# graph (after copying) but does not change the call graph structure itself.

# %% [markdown]
# Next, we lowering from the native dialect to the place dialect. The place dialect attempts
# to encapsulate the circuit into code blocks stored in regions which can be used later on for
# placement and routing analyses. Another important aspect of the insert special qubit allocation
# statements that are used to indicate the logical state prepration of the qubits. Given the nature
# of how the current move synthesis works the call graph is flattened out into a single kernel.

# %%
from bloqade.lanes.upstream import NativeToPlace  # noqa: E402

example_kernel = NativeToPlace().emit(example_kernel)
example_kernel.print()

# %% [markdown]
# now that we have the place dialect inside the kernel we can run the initial layout analysis
# to determine the initial placement of logical qubits onto the gemini logical lanes architecture.

# part of the input to the layout analysis is the layout heuristic which takes data collected from
# the analysis interpreter to generate the initial layout. In this example we use a heuristic
# that priorities placing qubits that interact frequently within the same word. This heuristic
# is implemented in the `fixed` module inside bloqade-lanes.

# %%
from bloqade.analysis import address  # noqa: E402

from bloqade.lanes.analysis import layout  # noqa: E402
from bloqade.lanes.heuristics import fixed  # noqa: E402

address_frame, _ = address.AddressAnalysis(example_kernel.dialects).run(example_kernel)


layout_analysis = layout.LayoutAnalysis(
    example_kernel.dialects,
    fixed.LogicalLayoutHeuristic(),
    address_frame.entries,
    tuple(range(2)),
)

initial_layout = layout_analysis.get_layout(example_kernel)
print(initial_layout)

# %% [markdown]
# After running the layout analysis we can use it to plot the initial placement of logical qubits

# %%
from bloqade.lanes.arch.gemini.logical import get_arch_spec  # noqa: E402

logical_arch = get_arch_spec()

ax = logical_arch.plot(show_words=(0, 1))

(pos_0,) = logical_arch.get_positions(initial_layout[0])
(pos_1,) = logical_arch.get_positions(initial_layout[1])
ax.scatter(
    [pos_0[0], pos_1[0]],
    [pos_0[1], pos_1[1]],
    s=200,
    label="Initial Placement",
)

# %% [markdown]
# Next, we can run the placement analysis to determine the intermediate  placement of logical qubits
# during the circuit execution. The difference here being that the placement analysis will be
# mapped onto the linear logic within the bodies of the StaticPlacement statements.

# %%
from bloqade.lanes.analysis import placement  # noqa: E402

placement_analysis = placement.PlacementAnalysis(
    example_kernel.dialects,
    initial_layout,
    address_frame.entries,
    fixed.LogicalPlacementStrategy(),
)

placement_frame, _ = placement_analysis.run(example_kernel)
example_kernel.print(analysis=placement_frame.entries)

# %% [markdown]
# With the placement analysis complete we can now proceed to rewrite the place dialect to the
# move dialect. This involves using the MoveScheduler to insert move operations before gates
# in the place dialect to based on the current placement of qubits and the placement of the qubits
# required for the gate operation. There are also options to insert palindrome moves after gates
# to move the qubits back to their original locations at the end of the StaticPlacement body.
# Here we never have more than one CZ operation within a StaticPlacement body because we want
# the logical qubits to always return back to their original locations after each gate. Hence
# we set `insert_palindrome_moves=True` in the MoveToSquin transformer.


# %%
from bloqade.lanes.upstream import PlaceToMove  # noqa: E402

example_kernel = PlaceToMove(
    layout_heristic=fixed.LogicalLayoutHeuristic(),
    placement_strategy=fixed.LogicalPlacementStrategy(),
    insert_palindrome_moves=True,
).emit(example_kernel)

example_kernel.print()

# %% [markdown]
# Now at this point we can either continue to hardware by specializing the move dialect
# to a gemini specific statements which can be further lowered to pulse level control. We
# will not cover that here, instead we will demonstrate to integrate with the tsim but rewriting
# the move program back to squin dialect on the physical qubits along with inserting noise.

# Before we rewrite to physical squin we can also run the atom state analysis on the logical program
# just to demonstrate how the analysis works on the move dialect.

# To start we simply construct the atom interpreter with the logical architecture spec, then
# run the analysis on the move dialect kernel.

# %%
from bloqade.lanes.analysis import atom  # noqa: E402
from bloqade.lanes.arch.gemini.logical import get_arch_spec  # noqa: E402

frame, _ = atom.AtomInterpreter(example_kernel.dialects, arch_spec=logical_arch).run(
    example_kernel
)
example_kernel.print(analysis=frame.entries)

# %% [markdown]

# Now if we want the physical noise model of the program we first have to rewrite from logical moves
# on the logical architecture to physical squin on the physical architecture. Because all the gates
# are clifford gates and can be implemented transversally we simply rewrite all logical addresses
# to groups of physical addresses. If you are interested in how this is done please see the
# `transversal_rewrites` function inside the `bloqade.lanes.logical_mvp` module.

# %%
from bloqade.lanes.logical_mvp import transversal_rewrites  # noqa: E402

example_kernel = transversal_rewrites(example_kernel)
example_kernel.print()

# %% [markdown]
# Now that we have the physical move kernel we can run the atom state analysis again on the physical
# architecture to get the atom states on physical qubits.

# %%
from bloqade.lanes.arch.gemini.impls import generate_arch  # noqa: E402

physical_arch = generate_arch(4)

frame, _ = atom.AtomInterpreter(example_kernel.dialects, arch_spec=physical_arch).run(
    example_kernel
)
example_kernel.print(analysis=frame.entries)

# %% [markdown]
# from here we can rewrite the physical move program to physical squin with noise models inserted.
# This is done using the `MoveToSquin` transformer inside the `bloqade.lanes.transform` module.
# this transformation requires the physical architecture spec, a logical initialization function
# to prepare logical qubit from the physical qubits, and a noise model to insert noise channels during
# the compilation.

# %%

from bloqade.lanes.arch.gemini.logical import steane7_initialize  # noqa: E402
from bloqade.lanes.noise_model import generate_simple_noise_model  # noqa: E402
from bloqade.lanes.transform import MoveToSquin  # noqa: E402

noise_model = generate_simple_noise_model()
example_kernel = MoveToSquin(
    arch_spec=physical_arch,
    logical_initialization=steane7_initialize,
    noise_model=noise_model,
).emit(example_kernel)

example_kernel.print()

# %% [markdown]
# From here we can use the existing tools within bloqade-circuit to simulate the physical squin
# program with noise, or compile to stim programs for further analysis.
