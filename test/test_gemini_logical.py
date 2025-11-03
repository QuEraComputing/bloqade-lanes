from matplotlib import pyplot as plt

from bloqade.lanes.gemini.logical import logical_arch

arch = logical_arch(physical=True)

ax = arch.plot(show_blocks=(0, 1), show_inter=range(len(arch.allowed_inter_lanes)))
ax.set_aspect("equal")
xlim = ax.get_xlim()
ylim = ax.get_ylim()

width = xlim[1] - xlim[0]
height = ylim[1] - ylim[0]

padding = 0.1
ax.set_xlim(xlim[0] - width * padding, xlim[1] + width * padding)
ax.set_ylim(ylim[0] - height * padding, ylim[1] + height * padding)


plt.show()
