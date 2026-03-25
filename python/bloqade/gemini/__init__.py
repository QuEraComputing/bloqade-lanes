from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import logical as logical

if TYPE_CHECKING:
    from . import compile as compile, simulator as simulator
    from .compile import (
        append_measurements_and_annotations as append_measurements_and_annotations,
        compile_squin_to_move as compile_squin_to_move,
        compile_squin_to_move_and_visualize as compile_squin_to_move_and_visualize,
        compile_to_physical_squin_noise_model as compile_to_physical_squin_noise_model,
        compile_to_stim_program as compile_to_stim_program,
        run_squin_kernel_validation as run_squin_kernel_validation,
        transversal_rewrites as transversal_rewrites,
    )
    from .simulator import (
        DetectorResult as DetectorResult,
        GeminiLogicalSimulator as GeminiLogicalSimulator,
        GeminiLogicalSimulatorTask as GeminiLogicalSimulatorTask,
        NoiseModelABC as NoiseModelABC,
        Result as Result,
        generate_simple_noise_model as generate_simple_noise_model,
        steane7_m2dets as steane7_m2dets,
        steane7_m2obs as steane7_m2obs,
    )

__all__ = [
    "logical",
    "compile",
    "simulator",
    "DetectorResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Result",
    "NoiseModelABC",
    "generate_simple_noise_model",
    "steane7_m2dets",
    "steane7_m2obs",
    "run_squin_kernel_validation",
    "compile_squin_to_move",
    "compile_squin_to_move_and_visualize",
    "compile_to_physical_squin_noise_model",
    "compile_to_stim_program",
    "transversal_rewrites",
    "append_measurements_and_annotations",
]

_SIMULATOR_EXPORTS = {
    "DetectorResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Result",
    "NoiseModelABC",
    "generate_simple_noise_model",
    "steane7_m2dets",
    "steane7_m2obs",
}

_COMPILE_EXPORTS = {
    "run_squin_kernel_validation",
    "compile_squin_to_move",
    "compile_squin_to_move_and_visualize",
    "compile_to_physical_squin_noise_model",
    "compile_to_stim_program",
    "transversal_rewrites",
    "append_measurements_and_annotations",
}


def __getattr__(name: str) -> Any:
    if name == "compile":
        return import_module(".compile", __name__)
    if name == "simulator":
        return import_module(".simulator", __name__)
    if name in _SIMULATOR_EXPORTS:
        module = import_module(".simulator", __name__)
        return getattr(module, name)
    if name in _COMPILE_EXPORTS:
        module = import_module(".compile", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
