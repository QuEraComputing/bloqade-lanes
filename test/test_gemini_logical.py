from matplotlib import pyplot as plt

from bloqade.lanes.gemini.logical import logical_arch

arch = logical_arch(physical=False)
arch_physical = logical_arch(physical=True)


f, (ax_l, *rest) = plt.subplots(1, 8)


arch.plot(ax_l, show_blocks=(0, 1), show_inter=(0,))

gs = ax_l.get_gridspec()
for ax in rest:
    ax.remove()

ax_p = f.add_subplot(gs[0, 1:])

arch_physical.plot(ax_p, show_blocks=(0, 1), show_inter=(0,))
for ax in (ax_l, ax_p):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    padding = 0.1
    ax.set_xlim(xlim[0] - width * padding, xlim[1] + width * padding)
    ax.set_ylim(ylim[0] - height * padding, ylim[1] + height * padding)

plt.subplots_adjust(wspace=1)
plt.show()
