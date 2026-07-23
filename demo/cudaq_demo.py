import cudaq  # type: ignore[reportMissingImports]

from bloqade.gemini import GeminiLogicalSimulator
from bloqade.gemini.compile import append_measurements_and_annotations
from bloqade.gemini.cudaq import cudaq_to_squin
from bloqade.gemini.steane_defaults import steane7_m2dets, steane7_m2obs

m2dets, m2obs = steane7_m2dets(5), steane7_m2obs(5)


@cudaq.kernel
def main_cuda():
    q = cudaq.qvector(5)
    h(q[0])  # noqa: F821  # pyright: ignore
    cx(q[0], q[1])  # noqa: F821  # pyright: ignore
    h(q[2])  # noqa: F821  # pyright: ignore
    h(q[3])  # noqa: F821  # pyright: ignore
    h(q[4])  # noqa: F821  # pyright: ignore


prepared_squin = cudaq_to_squin(main_cuda)
append_measurements_and_annotations(prepared_squin, m2dets, m2obs)
task = GeminiLogicalSimulator().task(prepared_squin)

result = task.run(10)
print(result.detectors)
