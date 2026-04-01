# %%
# %matplotlib inline
from bloqade.lanes.arch.gemini import logical, physical
from bloqade.lanes.layout import ArchSpec


def show_lanes(arch: ArchSpec):
    from matplotlib import pyplot as plt

    f, axs = plt.subplots(1, 2, figsize=(12, 5))
    arch.plot(show_words=range(min(4, len(arch.words))), show_site_bus=(0,), ax=axs[0])
    arch.plot(show_words=range(min(4, len(arch.words))), show_word_bus=(0,), ax=axs[1])
    axs[0].set_title("site bus 0")
    axs[1].set_title("word bus 0")
    plt.show()


logical_arch = logical.get_arch_spec()
show_lanes(logical_arch)

# %%
from bloqade.lanes._prelude import kernel  # noqa F402
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

    # Fill two qubits in the same word (word 0, sites 0 and 1)
    state = move.fill(
        empty_state, location_addresses=(LocationAddress(0, 0), LocationAddress(0, 1))
    )
    state = move.local_r(
        state, 0.25, -0.25, location_addresses=(LocationAddress(0, 0),)
    )
    # Word bus: move atom at site 1 from word 0 → word 1
    state = move.move(state, lanes=(WordLaneAddress(0, 1, 0),))
    # Site bus: move atom in word 1 from site 1 → site 0 (aligns for CZ)
    state = move.move(state, lanes=(SiteLaneAddress(1, 0, 0, Direction.BACKWARD),))
    # CZ between words 0 and 1 at matching site 0
    state = move.cz(state, zone_address=ZoneAddress(0))

    # Reverse: site bus forward (site 0 → site 1 in word 1)
    state = move.move(state, lanes=(SiteLaneAddress(1, 0, 0),))
    # Reverse: word bus backward (word 1 → word 0 at site 1)
    state = move.move(state, lanes=(WordLaneAddress(0, 1, 0, Direction.BACKWARD),))
    state = move.local_r(state, 0.25, 0.25, location_addresses=(LocationAddress(0, 0),))


main.print()

# %%
# %matplotlib qt
from bloqade.lanes.visualize import debugger  # noqa F402

debugger(main, arch_spec=logical_arch, atom_marker="s")

# %%
# %matplotlib inline
physical_arch = physical.get_arch_spec()
show_lanes(physical_arch)

# %% [markdown]
# Snippet of map used to rewrite addresses
#
# ```python
# def steane7_transversal_map(address: AddressType) -> Iterator[AddressType] | None:
#     """Map logical addresses to physical addresses via site expansion.
#
#     The Steane [[7,1,3]] code encodes one logical qubit into seven physical qubits.
#     Each logical site expands to 7 physical sites within the same word:
#
#         Logical site s → Physical sites s*7, s*7+1, ..., s*7+6
#
#     Word ID is preserved. Only expands logical site IDs (0 and 1).
#     Returns None for site IDs >= 2 (already physical / not a logical site).
#     """
#     if address.site_id >= 2:
#         return None
#     base = address.site_id * 7
#     return (address.replace(site_id=base + i) for i in range(7))
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
from bloqade.lanes.transform import MoveToSquinPhysical  # noqa F402

squin_kernel = MoveToSquinPhysical(
    physical_arch, noise_model=generate_simple_noise_model()
).emit(transversal_main)
squin_kernel.print()


# %%
from bloqade.cirq_utils import emit_circuit  # noqa F402
from cirq.contrib.svg import SVGCircuit  # noqa F402

circ = emit_circuit(squin_kernel)
SVGCircuit(circ)
