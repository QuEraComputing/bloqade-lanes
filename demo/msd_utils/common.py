from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObservableFrame(str, Enum):
    RAW = "raw"
    NOISELESS_REFERENCE_FLIPS = "noiseless_reference_flips"


@dataclass(frozen=True)
class LogicalKernelSpec:
    kernel: Any
    # TODO: these "special" fields should ONLY be added for "special" kernels. To be honest, not sure if they should be added at all.
    special_prepare_kernel: Any | None = None
    special_tsim_circuit_strategy: str | None = None
    # TODO: for observable_frame, we need to NOT have this auto-correction
    observable_frame: ObservableFrame = ObservableFrame.RAW


@dataclass(frozen=True)
class SyndromeLayout:
    output_detector_count: int = 3
    output_observable_count: int = 1


DEFAULT_SYNDROME_LAYOUT = SyndromeLayout()


@dataclass
class DemoTask:
    task: Any
    observable_frame: ObservableFrame = ObservableFrame.RAW
    observable_reference: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.task, name)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.task.run(*args, **kwargs)
