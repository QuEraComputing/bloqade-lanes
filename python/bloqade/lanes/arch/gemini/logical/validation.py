from bloqade.lanes.validation.address import get_validation

from .spec import get_arch_spec

# A no-arg-constructible address-validation pass pre-bound to the Gemini
# logical arch spec, suitable for use in a kirin ``ValidationSuite`` (which
# instantiates passes via ``pass_cls()``).
AddressValidation = get_validation(get_arch_spec())
