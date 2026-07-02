from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from bloqade.gemini.device.simulator import GeminiLogicalCliffTSimulatorTask

RetType = TypeVar("RetType")


@dataclass(frozen=True)
class _CliffTSimulatorTask(GeminiLogicalCliffTSimulatorTask[RetType]):
    """Backward-compatible name for the Gemini logical CliffT simulator task."""
