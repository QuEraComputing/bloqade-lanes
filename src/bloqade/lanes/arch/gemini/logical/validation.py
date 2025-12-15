from dataclasses import dataclass, field

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.validation.address import Validation

from .spec import get_arch_spec


@dataclass
class AddressValidation(Validation):
    arch_spec: "ArchSpec" = field(default_factory=get_arch_spec)

    def name(self) -> str:
        return "Gemini Logical Address Validation"
