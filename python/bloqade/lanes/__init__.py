from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports are for type-checkers only. At runtime they are deferred via
    # __getattr__ below to break a circular import:
    #   bloqade.gemini.common.dialects.movement.stmts imports
    #   bloqade.lanes.bytecode.encoding, which would otherwise trigger the eager
    #   imports here and transitively load bloqade.gemini.logical before
    #   bloqade.gemini.common has finished initializing.
    from bloqade.gemini.device import (
        DetectorResult as DetectorResult,
        GeminiLogicalSimulator as GeminiLogicalSimulator,
        GeminiLogicalSimulatorTask as GeminiLogicalSimulatorTask,
        Result as Result,
    )

    from .metrics import Metrics as Metrics
    from .noise_model import (
        generate_logical_noise_model as generate_logical_noise_model,
        generate_simple_noise_model as generate_simple_noise_model,
    )
    from .rewrite.move2squin.noise import NoiseModelABC as NoiseModelABC
    from .steane_defaults import (
        steane7_m2dets as steane7_m2dets,
        steane7_m2obs as steane7_m2obs,
    )


def __getattr__(name: str):
    if name in (
        "DetectorResult",
        "GeminiLogicalSimulator",
        "GeminiLogicalSimulatorTask",
        "Result",
    ):
        from bloqade.gemini.device import (
            DetectorResult,
            GeminiLogicalSimulator,
            GeminiLogicalSimulatorTask,
            Result,
        )

        g = globals()
        g.update(
            {
                "DetectorResult": DetectorResult,
                "GeminiLogicalSimulator": GeminiLogicalSimulator,
                "GeminiLogicalSimulatorTask": GeminiLogicalSimulatorTask,
                "Result": Result,
            }
        )
        return g[name]
    if name == "Metrics":
        from .metrics import Metrics

        globals()["Metrics"] = Metrics
        return Metrics
    if name in ("generate_logical_noise_model", "generate_simple_noise_model"):
        from .noise_model import (
            generate_logical_noise_model,
            generate_simple_noise_model,
        )

        g = globals()
        g.update(
            {
                "generate_logical_noise_model": generate_logical_noise_model,
                "generate_simple_noise_model": generate_simple_noise_model,
            }
        )
        return g[name]
    if name == "NoiseModelABC":
        from .rewrite.move2squin.noise import NoiseModelABC

        globals()["NoiseModelABC"] = NoiseModelABC
        return NoiseModelABC
    if name in ("steane7_m2dets", "steane7_m2obs"):
        from .steane_defaults import steane7_m2dets, steane7_m2obs

        g = globals()
        g.update({"steane7_m2dets": steane7_m2dets, "steane7_m2obs": steane7_m2obs})
        return g[name]
    raise AttributeError(f"module 'bloqade.lanes' has no attribute {name!r}")
