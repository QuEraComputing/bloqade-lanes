from typing import Literal
import cudaq
import numpy as np
from scipy.linalg import block_diag
from bloqade.lanes.device import GeminiLogicalSimulator


# manually specify
def measurement_to_detector_matrices(
    num_qubits: int, code: Literal["steane", "surface_code"]
):
    d = np.array([[1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 1, 0, 1]])
    o = np.array([[1, 1, 0, 0, 0, 1, 0]])
    m2obs = block_diag(*[o.T] * 5)  # shape (35, 5) - 35 measurements, 5 observables
    m2dets = block_diag(*[d.T] * 5)  # shape (35, 15) - 35 measurements, 15 detectors
    return m2dets, m2obs


@cudaq.kernel
def main_cuda():
    q = cudaq.qvector(5)
    h(q[0])
    cx(q[0], q[1])
    h(q[2])
    h(q[3])
    h(q[4])


m2dets, m2obs = measurement_to_detector_matrices(5, "steane")

task = GeminiLogicalSimulator().task(main_cuda, m2dets=m2dets, m2obs=m2obs)
task.tsim_circuit.diagram(height=400)
