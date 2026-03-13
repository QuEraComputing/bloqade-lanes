import cudaq  # type: ignore[reportMissingImports]

from bloqade.lanes import GeminiLogicalSimulator, steane7_m2dets, steane7_m2obs

m2dets, m2obs = steane7_m2dets(5), steane7_m2obs(5)


@cudaq.kernel
def main_cuda():
    q = cudaq.qvector(5)
    h(q[0])  # noqa: F821  # pyright: ignore
    cx(q[0], q[1])  # noqa: F821  # pyright: ignore
    h(q[2])  # noqa: F821  # pyright: ignore
    h(q[3])  # noqa: F821  # pyright: ignore
    h(q[4])  # noqa: F821  # pyright: ignore


task = GeminiLogicalSimulator().task(main_cuda, m2dets=m2dets, m2obs=m2obs)

result = task.run(10)
print(result.detectors)
