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


class NewArrayTooManyElementsError(ValidationError):
    """``new_array`` declares more elements than the validator will model.

    ``dim0`` and ``dim1`` are read straight out of the instruction word, so
    their product can reach 2^64; the bound keeps a malformed word from
    driving an unbounded loop.
    """

    def __init__(self, pc: int, count: int, maximum: int):
        self.pc = pc
        self.count = count
        self.maximum = maximum
        super().__init__(
            f"pc {pc}: new_array declares {count} elements, "
            f"more than the maximum of {maximum}"
        )


class CodeOutsideFunctionError(ValidationError):
    """An instruction sits outside every function's extent.

    Functions are delimited by ``func_start``/``func_end`` in the code
    stream. Anything after the last ``func_end`` belongs to no function: it
    can never be entered, and the disassembler emits it after the closing
    brace, producing text that will not re-read.
    """

    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(f"pc {pc}: instruction is outside any function")


class InvalidControlFlowTargetError(ValidationError):
    """A branch or call names an address that is not what it should be.

    ``br``/``cond_br`` targets are addresses in the code; ``call`` targets are
    function entries, which since the ``func_start``/``func_end`` layout means
    the address of a function's opening marker.
    """

    def __init__(self, pc: int, target: int, expected: str):
        self.pc = pc
        self.target = target
        self.expected = expected
        super().__init__(f"pc {pc}: control-flow target {target} is not {expected}")


class CallArityMismatchError(ValidationError):
    """A ``call``'s arity disagrees with the callee's declared parameters.

    The operand is not a hint: ``call <arity>`` sets the callee's frame base to
    ``stack.len() - arity``, so an unchecked one silently redefines the
    callee's shape per call site.
    """

    def __init__(self, pc: int, target: int, declared: int, got: int):
        self.pc = pc
        self.target = target
        self.declared = declared
        self.got = got
        super().__init__(
            f"pc {pc}: call passes {got} operand(s) but the function at "
            f"{target} declares {declared}"
        )


class ReturnCountMismatchError(ValidationError):
    """A ``ret`` keeps a different number of values than its function declares.

    Two ``ret`` instructions that disagree make every caller's post-call stack
    depth path-dependent, the same defect as a branch whose arms leave
    different depths. Each is checked against the declaration instead, which
    names the offender.
    """

    def __init__(self, pc: int, declared: int, got: int):
        self.pc = pc
        self.declared = declared
        self.got = got
        super().__init__(
            f"pc {pc}: ret keeps {got} value(s) but the function declares {declared}"
        )


class GetItemInvalidDimsError(ValidationError):
    """``get_item`` takes an index count no array can have.

    ``new_array`` carries exactly two dimension fields, so an array is at
    most 2-D and one or two indices is the only well-formed shape.
    """

    def __init__(self, pc: int, ndims: int, maximum: int):
        self.pc = pc
        self.ndims = ndims
        self.maximum = maximum
        super().__init__(f"pc {pc}: get_item takes 1..={maximum} indices, got {ndims}")


class LocalIndexOutOfRangeError(ValidationError):
    """``load``/``store`` names a local index past the maximum.

    The index is read straight out of the instruction word, and ``store``
    grows the operand stack to reach it — writing every new slot, so the
    memory is resident. A local index is a function's argument slot, so the
    bound is far above any real one; it exists to keep a malformed operand
    from becoming a multi-gigabyte allocation.
    """

    def __init__(self, pc: int, mnemonic: str, index: int, maximum: int):
        self.pc = pc
        self.mnemonic = mnemonic
        self.index = index
        self.maximum = maximum
        super().__init__(
            f"pc {pc}: {mnemonic} takes a local index 0..={maximum}, got {index}"
        )


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


class FeedForwardNotSupportedError(ValidationError):
    """Multiple measure instructions require feed_forward capability."""

    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(
            f"pc {pc}: multiple measure instructions require feed_forward capability"
        )


class AtomReloadingNotSupportedError(ValidationError):
    """Fill instruction requires atom_reloading capability."""

    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(
            f"pc {pc}: fill instruction requires atom_reloading capability"
        )


class EmptyProgramError(ValidationError):
    """Program has no instructions."""

    def __init__(self) -> None:
        super().__init__(
            "program is empty: must contain at least one instruction ending with return or halt"
        )


class MissingTerminatorError(ValidationError):
    """A path through a function runs off its end without return or halt.

    Per function, not per program: ``func_end`` is a no-op, so a function that
    falls off it runs whatever was laid out next rather than returning.
    """

    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(f"pc {pc}: function must end with return or halt")


class UnreachableInstructionError(ValidationError):
    """Instruction follows a return or halt and is unreachable."""

    def __init__(self, pc: int):
        self.pc = pc
        super().__init__(f"pc {pc}: unreachable instruction after return or halt")


class AddressValidationError(ValidationError):
    """A const_loc / const_lane / const_zone operand is invalid for the arch.

    Carries the architecture layer's own diagnostic ``message``.
    """

    def __init__(self, pc: int, message: str):
        self.pc = pc
        self.message = message
        super().__init__(f"pc {pc}: invalid address: {message}")


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
        super().__init__(f"duplicate location address 0x{address:016x}")


class InvalidLocationAddressError(LocationGroupError):
    def __init__(self, zone_id: int, word_id: int, site_id: int):
        self.zone_id = zone_id
        self.word_id = word_id
        self.site_id = site_id
        super().__init__(
            f"invalid location zone_id={zone_id}, word_id={word_id}, site_id={site_id}"
        )


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


# ── Move executability errors ──
class MoveValidationError(Exception):
    """Base class for lane-group executability errors against an atom state.

    Raised by ``AtomStateData.validate_moves`` when a lane group cannot
    execute. Carries an ``errors`` attribute with the individual error
    instances — subclasses of this class for occupancy-rule violations, and
    ``LaneGroupError`` subclasses for the static lane-group checks.
    """

    def __init__(self, message: str, errors: "list[Exception] | None" = None):
        super().__init__(message)
        self.errors: list[Exception] = errors or []


class UnresolvableLaneError(MoveValidationError):
    """A lane could not be resolved to (src, dst) endpoints."""

    def __init__(self, lane: int):
        self.lane = lane
        super().__init__(f"lane 0x{lane:016x} cannot be resolved to endpoints")


class DestinationOccupiedError(MoveValidationError):
    """A lane's destination holds an atom that does not move in this group.

    Applies uniformly to mover lanes and empty-source filler lanes: the AOD
    trap site arrives at every lane's destination either way. An occupied
    destination is only legal when its occupant vacates in the same group.
    """

    def __init__(self, lane: int, dst: int, occupant: int):
        self.lane = lane
        self.dst = dst
        self.occupant = occupant
        super().__init__(
            f"lane 0x{lane:016x} targets location 0x{dst:016x}, which is "
            f"occupied by qubit {occupant} that does not move in this group"
        )


class ContestedDestinationError(MoveValidationError):
    """Two lanes in the group share a destination."""

    def __init__(self, dst: int, first: int, second: int):
        self.dst = dst
        self.first = first
        self.second = second
        super().__init__(
            f"lanes 0x{first:016x} and 0x{second:016x} share destination "
            f"0x{dst:016x}"
        )


class StaleValidatedMovesError(MoveValidationError):
    """A ``ValidatedMoves`` token was applied to a state it was not
    validated against (or to that state after it moved on).

    The token records the mover assignments resolved at validation time, so
    applying it to a different state would desynchronize the location maps.
    """

    def __init__(self, lane: int, src: int, expected: int):
        self.lane = lane
        self.src = src
        self.expected = expected
        super().__init__(
            f"stale ValidatedMoves token: lane 0x{lane:016x} expected qubit "
            f"{expected} at location 0x{src:016x}, which no longer holds it"
        )


# ── Parse errors ──


class ParseError(Exception):
    """Base class for SST text format parse errors."""


class MissingVersionError(ParseError):
    def __init__(self):
        super().__init__("missing version header")


class BadInstructionError(ParseError):
    """A line could not be parsed as an instruction."""

    def __init__(self, line: int, text: str):
        self.line = line
        self.text = text
        super().__init__(f"line {line}: cannot parse instruction '{text}'")


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
    """Base class for ``VHBC`` binary container errors."""


class BadMagicError(ProgramError):
    def __init__(self):
        super().__init__("bad magic bytes (expected VHBC)")


class TruncatedError(ProgramError):
    def __init__(self, expected: int, got: int):
        self.expected = expected
        self.got = got
        super().__init__(f"truncated: expected {expected} bytes, got {got}")


# Deprecated: no Rust path maps here anymore. `LANES` framed a program as a
# flat list of typed sections; `VHBC` carries a section *tree* whose framing
# faults vihaco reports itself, surfacing as DecodeErrorInProgram (kept for
# backward-compatible imports).
class UnknownSectionTypeError(ProgramError):
    def __init__(self, section_type: int):
        self.section_type = section_type
        super().__init__(f"unknown section type: {section_type}")


# Deprecated: no Rust path maps here anymore; superseded by UnalignedCodeError (kept for backward-compatible imports).
class InvalidCodeSectionLengthError(ProgramError):
    def __init__(self, length: int):
        self.length = length
        super().__init__(
            f"code section length {length} is not a whole number of instruction words"
        )


class UnalignedCodeError(ProgramError):
    """Binary code region length is not a whole number of instruction words."""

    def __init__(self, length: int, width: int):
        self.length = length
        self.width = width
        super().__init__(f"code length {length} is not a multiple of {width}")


# Deprecated: no Rust path maps here anymore. A `VHBC` root section always
# has a header and a bytecode region, so neither can go missing the way a
# `LANES` section could (kept for backward-compatible imports).
class MissingMetadataSectionError(ProgramError):
    def __init__(self):
        super().__init__("missing metadata section")


# Deprecated: see MissingMetadataSectionError (kept for backward-compatible
# imports).
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


class EncodeErrorInProgram(ProgramError):
    """A program uses something the binary container cannot carry.

    Raised by ``Program.to_binary``, not by reading. The container has no
    encoding for a constant pool, source symbols, or a non-empty function
    signature, so it refuses to write one rather than dropping it silently.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(f"cannot encode program: {message}")
