from bloqade import gemini, squin


@gemini.logical.kernel(aggressive_unroll=True, no_raise=False)
def invalid():
    q = squin.qalloc(2)
    squin.swap(q[0], q[1])
    # squin.u3(0.1, 0.2, 0.3, q[1])
    return gemini.logical.terminal_measure(q)


sim_device = gemini.device.GeminiLogicalSimulator()

invalid_task = sim_device.task(invalid)

from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.visualize.plotly_debug import plotly_debugger

plotly_debugger(invalid_task.physical_move_kernel, get_physical_arch_spec())

from bloqade.lanes.visualize.arch import ArchVisualizer

figure = ArchVisualizer(get_arch_spec()).plot_interactive()

figure.show(renderer="browser")
