from bloqade.cirq_registry import register_cirq_loader as _register_cirq_loader

from bloqade.gemini.common.dialects.arrange import move_to as move_to
from bloqade.lanes.dialects.arch import loc as loc

from . import dialects as dialects, impl as impl, validation as validation
from .dialects.operations import terminal_measure as terminal_measure
from .group import kernel as kernel
from .stdlib import (
    broadcast as broadcast,
    default_post_processing as default_post_processing,
    qalloc_at as qalloc_at,
    star_rz as star_rz,
)


def _get_cirq_loader():
    from .cirq_conversion import GeminiLogicalCirqLowerer

    return GeminiLogicalCirqLowerer


# Keep Cirq optional: load its Gemini integration only when conversion is used.
_register_cirq_loader(dialects.operations.dialect, _get_cirq_loader)
