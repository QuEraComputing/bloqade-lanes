"""Structured exception classes for bloqade-lanes-bytecode.

Each Rust error enum maps to a base exception class. Each variant maps
to a subclass with the variant's fields as attributes.
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


class Zone0MissingWordsError(ArchSpecError):
    def __init__(self, missing: list[int]):
        self.missing = missing
        super().__init__(f"zone 0 must include all words: missing word IDs {missing}")


class MeasurementModeZonesEmptyError(ArchSpecError):
    def __init__(self):
        super().__init__("measurement_mode_zones must not be empty")


class MeasurementModeFirstNotZone0Error(ArchSpecError):
    def __init__(self, got: int):
        self.got = got
        super().__init__(f"measurement_mode_zones[0] must be zone 0, got {got}")


class InvalidEntanglingZoneError(ArchSpecError):
    def __init__(self, id: int):
        self.id = id
        super().__init__(f"entangling_zones contains invalid zone ID {id}")


class InvalidMeasurementModeZoneError(ArchSpecError):
    def __init__(self, id: int):
        self.id = id
        super().__init__(f"measurement_mode_zones contains invalid zone ID {id}")


class WrongSiteCountError(ArchSpecError):
    def __init__(self, word_id: int, expected: int, got: int):
        self.word_id = word_id
        self.expected = expected
        self.got = got
        super().__init__(
            f"word {word_id} has {got} sites, expected {expected} (sites_per_word)"
        )


class WrongCzPairsCountError(ArchSpecError):
    def __init__(self, word_id: int, expected: int, got: int):
        self.word_id = word_id
        self.expected = expected
        self.got = got
        super().__init__(
            f"word {word_id} has {got} cz_pairs, expected {expected} (sites_per_word)"
        )


class NonFiniteGridValueError(ArchSpecError):
    def __init__(self, word_id: int, field: str):
        self.word_id = word_id
        self.field = field
        super().__init__(f"word {word_id} grid contains non-finite value in {field}")


class NonFiniteWaypointError(ArchSpecError):
    def __init__(self, index: int):
        self.index = index
        super().__init__(f"paths[{index}]: waypoint contains non-finite coordinate")


class SiteXIndexOutOfRangeError(ArchSpecError):
    def __init__(self, word_id: int, site_idx: int, x_idx: int, x_len: int):
        self.word_id = word_id
        self.site_idx = site_idx
        self.x_idx = x_idx
        self.x_len = x_len
        super().__init__(
            f"word {word_id}, site {site_idx}: x_idx {x_idx} out of range "
            f"(grid has {x_len} x_positions)"
        )


class SiteYIndexOutOfRangeError(ArchSpecError):
    def __init__(self, word_id: int, site_idx: int, y_idx: int, y_len: int):
        self.word_id = word_id
        self.site_idx = site_idx
        self.y_idx = y_idx
        self.y_len = y_len
        super().__init__(
            f"word {word_id}, site {site_idx}: y_idx {y_idx} out of range "
            f"(grid has {y_len} y_positions)"
        )


class SiteBusLengthMismatchError(ArchSpecError):
    def __init__(self, bus_id: int, src_len: int, dst_len: int):
        self.bus_id = bus_id
        self.src_len = src_len
        self.dst_len = dst_len
        super().__init__(
            f"site_bus {bus_id}: src length ({src_len}) != dst length ({dst_len})"
        )


class SiteBusSrcDstOverlapError(ArchSpecError):
    def __init__(self, bus_id: int, site_idx: int):
        self.bus_id = bus_id
        self.site_idx = site_idx
        super().__init__(
            f"site_bus {bus_id}: src and dst overlap at site index {site_idx}"
        )


class SiteBusIndexOutOfRangeError(ArchSpecError):
    def __init__(self, bus_id: int, site_idx: int, sites_per_word: int):
        self.bus_id = bus_id
        self.site_idx = site_idx
        self.sites_per_word = sites_per_word
        super().__init__(
            f"site_bus {bus_id}: site index {site_idx} >= "
            f"sites_per_word ({sites_per_word})"
        )


class WordBusLengthMismatchError(ArchSpecError):
    def __init__(self, bus_id: int, src_len: int, dst_len: int):
        self.bus_id = bus_id
        self.src_len = src_len
        self.dst_len = dst_len
        super().__init__(
            f"word_bus {bus_id}: src length ({src_len}) != dst length ({dst_len})"
        )


class WordBusInvalidWordIdError(ArchSpecError):
    def __init__(self, bus_id: int, word_id: int):
        self.bus_id = bus_id
        self.word_id = word_id
        super().__init__(f"word_bus {bus_id}: invalid word ID {word_id}")


class InvalidWordWithSiteBusError(ArchSpecError):
    def __init__(self, word_id: int):
        self.word_id = word_id
        super().__init__(f"words_with_site_buses: invalid word ID {word_id}")


class InvalidSiteWithWordBusError(ArchSpecError):
    def __init__(self, site_idx: int, sites_per_word: int):
        self.site_idx = site_idx
        self.sites_per_word = sites_per_word
        super().__init__(
            f"sites_with_word_buses: site index {site_idx} >= "
            f"sites_per_word ({sites_per_word})"
        )


class InvalidPathLaneError(ArchSpecError):
    def __init__(self, index: int, lane: int, message: str):
        self.index = index
        self.lane = lane
        self.message = message
        super().__init__(f"paths[{index}]: lane 0x{lane:08X} is invalid: {message}")


class PathTooFewWaypointsError(ArchSpecError):
    def __init__(self, index: int, lane: int, count: int):
        self.index = index
        self.lane = lane
        self.count = count
        super().__init__(
            f"paths[{index}]: lane 0x{lane:08X} has {count} waypoint(s), minimum is 2"
        )


class PathEndpointMismatchError(ArchSpecError):
    def __init__(
        self,
        index: int,
        lane: int,
        endpoint: str,
        expected_x: float,
        expected_y: float,
        got_x: float,
        got_y: float,
    ):
        self.index = index
        self.lane = lane
        self.endpoint = endpoint
        self.expected_x = expected_x
        self.expected_y = expected_y
        self.got_x = got_x
        self.got_y = got_y
        super().__init__(
            f"paths[{index}]: lane 0x{lane:08X} {endpoint} waypoint "
            f"({got_x}, {got_y}) does not match expected position "
            f"({expected_x}, {expected_y})"
        )


class InconsistentGridShapeError(ArchSpecError):
    def __init__(
        self,
        word_id: int,
        x_len: int,
        y_len: int,
        ref_x_len: int,
        ref_y_len: int,
    ):
        self.word_id = word_id
        self.x_len = x_len
        self.y_len = y_len
        self.ref_x_len = ref_x_len
        self.ref_y_len = ref_y_len
        super().__init__(
            f"word {word_id} grid shape ({x_len}x{y_len}) differs from "
            f"word 0 ({ref_x_len}x{ref_y_len})"
        )


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
