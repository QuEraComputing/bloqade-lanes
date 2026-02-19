from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar

import tsim as tsim_backend
from kirin import ir

from bloqade import tsim
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.arch.gemini.logical import steane7_initialize
from bloqade.lanes.logical_mvp import compile_squin_to_move
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.transform import MoveToSquin

RetType = TypeVar("RetType")


@dataclass(frozen=True)
class GeminiLogicalSimulatorTask(Generic[RetType]):
    logical_squin_kernel: ir.Method[[], RetType]
    noise_model: NoiseModelABC

    def __post_init__(self):
        if self._post_processing is None:
            raise ValueError(
                "The provided logical_squin_kernel has invalid return values for measurement processing."
            )

    @cached_property
    def physical_arch_spec(self):
        return generate_arch_hypercube(4)

    @cached_property
    def physical_squin_kernel(self) -> ir.Method[[], RetType]:

        return MoveToSquin(
            generate_arch_hypercube(4),
            steane7_initialize,
            self.noise_model,
        ).emit(self.physical_move_kernel)

    @cached_property
    def physical_move_kernel(self) -> ir.Method[[], RetType]:
        return compile_squin_to_move(
            self.logical_squin_kernel, transversal_rewrite=True
        )

    @cached_property
    def _post_processing(self):
        return atom.AtomInterpreter(
            self.physical_move_kernel.dialects, arch_spec=self.physical_arch_spec
        ).get_post_processing(self.physical_move_kernel)

    @cached_property
    def tsim_circuit(self) -> tsim_backend.Circuit:
        physical_squin_kernel = self.physical_squin_kernel.similar()
        RemoveReturn().rewrite(physical_squin_kernel.code)
        return tsim.Circuit(physical_squin_kernel)

    @cached_property
    def measurement_sampler(self):
        return self.tsim_circuit.compile_sampler()

    @cached_property
    def detector_error_model(self):
        return self.tsim_circuit.detector_error_model(approximate_disjoint_errors=True)

    @property
    def post_processing(self):
        assert (
            self._post_processing is not None
        ), "validation failed during initialization"
        return self._post_processing

    def visualize(self, animated: bool = False, interactive: bool = True):
        from bloqade.lanes.visualize import animated_debugger, debugger

        if animated:
            animated_debugger(
                self.physical_move_kernel,
                self.physical_arch_spec,
                interactive=interactive,
            )
        else:
            debugger(
                self.physical_move_kernel,
                self.physical_arch_spec,
                interactive=interactive,
            )

    def run(self, shots: int = 1):
        raw_results = self.measurement_sampler.sample(shots=shots).tolist()
        return list(self.post_processing(raw_results))


@dataclass
class GeminiLogicalSimulator:
    noise_model: NoiseModelABC

    def task(
        self, logical_squin_kernel: ir.Method[[], RetType]
    ) -> GeminiLogicalSimulatorTask[RetType]:
        return GeminiLogicalSimulatorTask(
            logical_squin_kernel=logical_squin_kernel,
            noise_model=self.noise_model,
        )

    def run(self, logical_squin_kernel: ir.Method[[], RetType], shots: int = 1):
        return self.task(logical_squin_kernel).run(shots=shots)

    def visualize(
        self, logical_squin_kernel: ir.Method[[], RetType], animated: bool = False
    ):
        self.task(logical_squin_kernel).visualize(animated=animated)
