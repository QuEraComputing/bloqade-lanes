import math

import numpy as np
from demo.msd_utils.circuits import (
    build_naive_kernel_bundle,
    build_task_map,
    make_noisy_steane7_initializer,
)

from bloqade.lanes import GeminiLogicalSimulator
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs

THETA = 0.3041 * math.pi
PHI = 0.25 * math.pi
LAM = 0.0
TARGET = np.array([0, 0, 0, 0], dtype=np.uint8)

sim = GeminiLogicalSimulator()
kb = build_naive_kernel_bundle(THETA, PHI, LAM, output_qubit=0)

custom_tasks = build_task_map(
    sim,
    kb.distilled,
    m2dets=steane7_m2dets(5),
    m2obs=steane7_m2obs(5),
    noisy_initializer=make_noisy_steane7_initializer(sim),
)

plain_tasks = {
    basis: sim.task(
        kb.distilled[basis],
        m2dets=steane7_m2dets(5),
        m2obs=steane7_m2obs(5),
    )
    for basis in ["X", "Y", "Z"]
}


def summarize(tasks, name):
    print(f"\nPATH {name}")
    for basis in ["X", "Y", "Z"]:
        res = tasks[basis].run(3000, with_noise=False, run_detectors=False)
        obs = np.asarray(res.observables, dtype=np.uint8)
        det = np.asarray(res.detectors, dtype=np.uint8)

        mask = np.all(obs[:, 1:] == TARGET, axis=1) & np.all(det == 0, axis=1)
        bits = obs[mask, 0].astype(np.float64)
        exp = np.mean(1.0 - 2.0 * bits) if len(bits) else float("nan")

        print(basis, "rate", mask.mean(), "exp", exp)


summarize(custom_tasks, "custom")
summarize(plain_tasks, "plain")
