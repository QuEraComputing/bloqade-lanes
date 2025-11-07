from matplotlib import pyplot as plt

from bloqade.lanes.gemini import generate_arch

arch_physical = generate_arch()

ax = arch_physical.plot(show_blocks=(0,), show_intra=tuple(range(4)))


ax.set_aspect(0.25)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_xlim(xmin - 4, xmax + 4)
ax.set_ylim(ymin - 4, ymax + 4)

plt.show()

f, axs = plt.subplots(2, 2, figsize=(10, 8))

arch_physical.plot(show_blocks=tuple(range(16)), show_inter=(0,), ax=axs[0, 0])
arch_physical.plot(show_blocks=tuple(range(16)), show_inter=(1,), ax=axs[0, 1])
arch_physical.plot(show_blocks=tuple(range(16)), show_inter=(2,), ax=axs[1, 0])
arch_physical.plot(show_blocks=tuple(range(16)), show_inter=(3,), ax=axs[1, 1])


plt.show()
