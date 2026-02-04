from matplotlib import pyplot as plt
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.arch.gemini.impls import (
    generate_arch_hypercube,
    generate_arch_linear,
)


hypercube_dim = 4
arch_spec = generate_arch_hypercube(hypercube_dims=hypercube_dim, word_size_y=1)
# arch_spec = generate_arch_linear(num_words=16, word_size_y=1)
# arch_spec = get_arch_spec()

ax = plt.gca()

print("Sitebuses:", arch_spec.site_buses)
print("Wordbuses:", arch_spec.word_buses)
print("Words:", arch_spec.words)

arch_spec.plot(
    ax=ax,
    show_words=[i for i in range(len(arch_spec.words))],
    show_site_bus=[],
    # show_word_bus=[i for i in range(len(arch_spec.word_buses))],
    show_word_bus=[3]
)

plt.show()

