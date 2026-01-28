# %%
# %matplotlib inline
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.layout import ArchSpec


def show_lanes(arch: ArchSpec):
    from matplotlib import pyplot as plt

    f, axs = plt.subplots(1, 2, figsize=(12, 5))
    arch.plot(show_words=range(len(arch.words)), show_site_bus=(0,), ax=axs[0])
    arch.plot(show_words=range(len(arch.words)), show_word_bus=(0,), ax=axs[1])
    axs[0].set_title("site bus 0")
    axs[1].set_title("word bus 0")
    plt.show()


logical_arch = generate_arch_hypercube(1, 5)
show_lanes(logical_arch)

# %%
from bloqade.lanes import kernel  # noqa F402
from bloqade.lanes.dialects import move  # noqa F402
from bloqade.lanes.layout import (  # noqa F402
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)


@kernel
def main():
    # linear type State
    empty_state = move.load()

    state = move.fill(
        empty_state, location_addresses=(LocationAddress(0, 0), LocationAddress(1, 0))
    )
    state = move.local_r(
        state, 0.25, -0.25, location_addresses=(LocationAddress(0, 0),)
    )
    state = move.move(state, lanes=(SiteLaneAddress(0, 0, 0),))
    state = move.move(state, lanes=(WordLaneAddress(0, 5, 0),))
    state = move.cz(state, zone_address=ZoneAddress(0))

    state = move.move(state, lanes=(WordLaneAddress(0, 5, 0, Direction.BACKWARD),))
    state = move.move(state, lanes=(SiteLaneAddress(0, 0, 0, Direction.BACKWARD),))
    state = move.local_r(state, 0.25, 0.25, location_addresses=(LocationAddress(0, 0),))


main.print()

# %%
# %matplotlib qt
from bloqade.lanes.visualize import debugger  # noqa F402

debugger(main, arch_spec=logical_arch, atom_marker="s")

# %%
# %matplotlib inline
physical_arch = generate_arch_hypercube(4, 5)
show_lanes(physical_arch)

# %% [markdown]
# Snippet of map used to rewrite addresses
#
# ```python
# def steane7_transversal_map(address: AddressType) -> Iterator[AddressType] | None:
#     """This function is used to map logical addresses to physical addresses.
#
#     The Steane [[7,1,3]] code encodes one logical qubit into seven physical qubits.
#     The mapping is as follows:
#
#     Logical Word ID 0 -> Physical Word IDs 0 to 6
#     Logical Word ID 1 -> Physical Word IDs 8 to 14
#
#     All other Word IDs remain unchanged.
#
#     """
#     if address.word_id == 0:
#         return (replace(address, word_id=word_id) for word_id in range(7))
#     elif address.word_id == 1:
#         return (replace(address, word_id=word_id) for word_id in range(8, 15, 1))
#     else:
#         return None
# ```

# %%
from bloqade.lanes.logical_mvp import transversal_rewrites  # noqa F402

# rewrites to transversal moves on steane code
main.print()
transversal_rewrites(transversal_main := main.similar())
transversal_main.print()

# %%
# %matplotlib qt

debugger(transversal_main, arch_spec=physical_arch)

# %%
from bloqade.lanes.noise_model import generate_simple_noise_model  # noqa F402
from bloqade.lanes.transform import MoveToSquin  # noqa F402

squin_kernel = MoveToSquin(
    physical_arch, noise_model=generate_simple_noise_model()
).emit(transversal_main)
squin_kernel.print()


# %%
from bloqade.cirq_utils import emit_circuit  # noqa F402
from cirq.contrib.svg import SVGCircuit  # noqa F402

circ = emit_circuit(squin_kernel)
SVGCircuit(circ)
