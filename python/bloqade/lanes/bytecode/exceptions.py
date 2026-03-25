"""Structured exception classes for bloqade-lanes-bytecode.

Each Rust error enum maps to a base exception class. Category-based enums
(like ArchSpecError) map to subclasses that carry a descriptive message
string. Fine-grained enums (like LaneGroupError, ValidationError) map to
subclasses with the variant's fields as attributes.
"""

# ── ArchSpec validation errors ──


class ArchSpecError(Exception):
    """Base class for architecture specification validation errors.

    When multiple errors are collected, this is raised with an ``errors``
    attribute containing the individual subclass instances.
    """

    def __init__(self, message: str, errors: "list[ArchSpecError] | None" = None):
        super().__init__(message)
        self.errors: list[ArchSpecError] = errors or []


class ArchSpecZoneError(ArchSpecError):
    """Zone configuration error (zone 0 coverage, measurement/entangling zone IDs)."""

    def __init__(self, message: str):
        super().__init__(message)


class ArchSpecGeometryError(ArchSpecError):
    """Word geometry error (site counts, grid indices, grid shape, non-finite values)."""

    def __init__(self, message: str):
        super().__init__(message)


class ArchSpecBusError(ArchSpecError):
    """Bus topology error (site/word bus structure, membership lists)."""

    def __init__(self, message: str):
        super().__init__(message)


class ArchSpecPathError(ArchSpecError):
    """Transport path error (invalid lanes, waypoint counts, endpoint mismatches)."""

    def __init__(self, message: str):
        super().__init__(message)


# ── Bytecode validation errors ──


class ValidationError(Exception):
    """Base class for bytecode validation errors.

    When multiple errors are collected, this is raised with an ``errors``
    attribute containing the individual subclass instances.
    """

    def __init__(self, message: str, errors: "list[ValidationError] | None" = None):
        super().__init__(message)
        self.errors: list[ValidationError] = errors or []


class NewArrayZeroDim0Error(ValidationError):
    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(f"pc {pc}: new_array dim0 must be > 0")


class NewArrayInvalidTypeTagError(ValidationError):
    def __init__(self, pc: int, type_tag: int):
        self.pc = pc
        self.type_tag = type_tag
        super().__init__(f"pc {pc}: invalid type tag 0x{type_tag:x}")


class InitialFillNotFirstError(ValidationError):
    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(
            f"pc {pc}: initial_fill must be the first non-constant instruction"
        )


class StackUnderflowError(ValidationError):
    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(f"pc {pc}: stack underflow")


class TypeMismatchError(ValidationError):
    def __init__(self, pc: int, expected: int, got: int):
        self.pc = pc
        self.expected = expected
        self.got = got
        super().__init__(
            f"pc {pc}: type mismatch: expected tag 0x{expected:x}, got 0x{got:x}"
        )


class InvalidZoneError(ValidationError):
    def __init__(self, pc: int, zone_id: int):
        self.pc = pc
        self.zone_id = zone_id
        super().__init__(f"pc {pc}: invalid zone_id={zone_id}")


class LocationValidationError(ValidationError):
    """Wraps a LocationGroupError with a program counter for bytecode context."""

    def __init__(self, pc: int, error: "LocationGroupError"):
        self.pc = pc
        self.error = error
        super().__init__(f"pc {pc}: {error}")


class LaneValidationError(ValidationError):
    """Wraps a LaneGroupError with a program counter for bytecode context."""

    def __init__(self, pc: int, error: "LaneGroupError"):
        self.pc = pc
        self.error = error
        super().__init__(f"pc {pc}: {error}")


# ── Location group errors (from ArchSpec.check_locations) ──


class LocationGroupError(Exception):
    """Base class for location group validation errors.

    When multiple errors are collected, this is raised with an ``errors``
    attribute containing the individual subclass instances.
    """

    def __init__(self, message: str, errors: "list[LocationGroupError] | None" = None):
        super().__init__(message)
        self.errors: list[LocationGroupError] = errors or []


class DuplicateLocationAddressError(LocationGroupError):
    def __init__(self, address: int):
        self.address = address
        super().__init__(f"duplicate location address 0x{address:04x}")


class InvalidLocationAddressError(LocationGroupError):
    def __init__(self, word_id: int, site_id: int):
        self.word_id = word_id
        self.site_id = site_id
        super().__init__(f"invalid location word_id={word_id}, site_id={site_id}")


# ── Lane group errors (from ArchSpec.check_lanes) ──


class LaneGroupError(Exception):
    """Base class for lane group validation errors.

    When multiple errors are collected, this is raised with an ``errors``
    attribute containing the individual subclass instances.
    """

    def __init__(self, message: str, errors: "list[LaneGroupError] | None" = None):
        super().__init__(message)
        self.errors: list[LaneGroupError] = errors or []


class DuplicateLaneAddressError(LaneGroupError):
    def __init__(self, address: int):
        self.address = address
        super().__init__(f"duplicate lane address 0x{address:016x}")


class InvalidLaneAddressError(LaneGroupError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"invalid lane: {message}")


class LaneGroupInconsistentError(LaneGroupError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"lane group inconsistent: {message}")


class LaneWordNotInSiteBusListError(LaneGroupError):
    def __init__(self, word_id: int):
        self.word_id = word_id
        super().__init__(f"word_id {word_id} not in words_with_site_buses")


class LaneSiteNotInWordBusListError(LaneGroupError):
    def __init__(self, site_id: int):
        self.site_id = site_id
        super().__init__(f"site_id {site_id} not in sites_with_word_buses")


class LaneGroupAODConstraintViolationError(LaneGroupError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"AOD constraint violation: {message}")


# ── Parse errors ──


class ParseError(Exception):
    """Base class for SST text format parse errors."""


class MissingVersionError(ParseError):
    def __init__(self):
        super().__init__("missing .version directive")


class InvalidVersionError(ParseError):
    def __init__(self, message: str):
        self.message = message
        super().__init__(f"invalid version: {message}")


class UnknownMnemonicError(ParseError):
    def __init__(self, line: int, mnemonic: str):
        self.line = line
        self.mnemonic = mnemonic
        super().__init__(f"line {line}: unknown mnemonic '{mnemonic}'")


class MissingOperandError(ParseError):
    def __init__(self, line: int, mnemonic: str):
        self.line = line
        self.mnemonic = mnemonic
        super().__init__(f"line {line}: missing operand for '{mnemonic}'")


class InvalidOperandError(ParseError):
    def __init__(self, line: int, message: str):
        self.line = line
        self.message = message
        super().__init__(f"line {line}: {message}")


# ── Program binary format errors ──


class ProgramError(Exception):
    """Base class for BLQD binary format errors."""


class BadMagicError(ProgramError):
    def __init__(self):
        super().__init__("bad magic bytes (expected BLQD)")


class TruncatedError(ProgramError):
    def __init__(self, expected: int, got: int):
        self.expected = expected
        self.got = got
        super().__init__(f"truncated: expected {expected} bytes, got {got}")


class UnknownSectionTypeError(ProgramError):
    def __init__(self, section_type: int):
        self.section_type = section_type
        super().__init__(f"unknown section type: {section_type}")


class InvalidCodeSectionLengthError(ProgramError):
    def __init__(self, length: int):
        self.length = length
        super().__init__(f"code section length {length} is not a multiple of 8")


class MissingMetadataSectionError(ProgramError):
    def __init__(self):
        super().__init__("missing metadata section")


class MissingCodeSectionError(ProgramError):
    def __init__(self):
        super().__init__("missing code section")


# ── Decode errors ──


class DecodeError(Exception):
    """Base class for instruction decode errors."""


class UnknownOpcodeError(DecodeError):
    def __init__(self, opcode: int):
        self.opcode = opcode
        super().__init__(f"unknown opcode: 0x{opcode:02x}")


class InvalidOperandDecodeError(DecodeError):
    def __init__(self, opcode: int, message: str):
        self.opcode = opcode
        self.message = message
        super().__init__(f"invalid operand for opcode 0x{opcode:02x}: {message}")


class DecodeErrorInProgram(ProgramError):
    """A decode error encountered while parsing a binary program."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(f"decode error: {message}")
