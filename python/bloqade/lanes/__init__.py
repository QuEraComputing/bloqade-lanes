from importlib import import_module

from .metrics import Metrics as Metrics
from .noise_model import generate_simple_noise_model as generate_simple_noise_model
from .rewrite.move2squin.noise import NoiseModelABC as NoiseModelABC
from .steane_defaults import (
    steane7_m2dets as steane7_m2dets,
    steane7_m2obs as steane7_m2obs,
)

__all__ = [
    "DetectorResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Metrics",
    "NoiseModelABC",
    "Result",
    "generate_simple_noise_model",
    "steane7_m2dets",
    "steane7_m2obs",
]

_DEVICE_EXPORTS = {
    "DetectorResult",
    "GeminiLogicalSimulator",
    "GeminiLogicalSimulatorTask",
    "Result",
}


def __getattr__(name: str):
    if name in _DEVICE_EXPORTS:
        return getattr(import_module("bloqade.gemini.device"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
