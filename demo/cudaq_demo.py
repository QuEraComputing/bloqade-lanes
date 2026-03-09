import cudaq
import numpy as np
from scipy.linalg import block_diag

from bloqade.lanes.device import GeminiLogicalSimulator


def get_measurement_to_detector_matrices(
    n: int,
) -> tuple[list[list[int]], list[list[int]]]:
    d = np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1]])
    o = np.array([[1, 1, 0, 0, 0, 1, 0]])
    m2obs = block_diag(*[o.T] * n)  # shape (n * 7, n) - n * 7 meas, n observables
    m2dets = block_diag(*[d.T] * n)  # shape (n * 7, n * 3) - n * 7 meas, n * 3 dets
    return m2dets.tolist(), m2obs.tolist()


m2dets, m2obs = get_measurement_to_detector_matrices(5)


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
result.return_values
