import pytest

from bloqade.lanes.bytecode import (
    ArchSpec,
    Direction,
    Instruction,
    LaneAddress,
    LocationAddress,
    MoveType,
    Program,
    ValidationError,
    ZoneAddress,
)
from bloqade.lanes.bytecode.exceptions import (
    AtomReloadingNotSupportedError,
    BadMagicError,
    EmptyProgramError,
    FeedForwardNotSupportedError,
    InitialFillNotFirstError,
    MissingTerminatorError,
    MissingVersionError,
    UnreachableInstructionError,
)

# ── Address Types ──


class TestLocationAddress:
    def test_construct_and_getters(self):
        addr = LocationAddress(zone_id=0, word_id=1, site_id=2)
        assert addr.word_id == 1
        assert addr.site_id == 2

    def test_encode_decode_round_trip(self):
        addr = LocationAddress(zone_id=0, word_id=3, site_id=7)
        bits = addr.encode()
        decoded = LocationAddress.decode(bits)
        assert decoded == addr

    def test_repr(self):
        addr = LocationAddress(zone_id=0, word_id=0, site_id=1)
        assert "LocationAddress" in repr(addr)
        assert "word_id=0" in repr(addr)
        assert "site_id=1" in repr(addr)

    def test_hash(self):
        a = LocationAddress(zone_id=0, word_id=0, site_id=1)
        b = LocationAddress(zone_id=0, word_id=0, site_id=1)
        assert hash(a) == hash(b)
        d = {a: "value"}
        assert d[b] == "value"


class TestLaneAddress:
    def test_construct_and_getters(self):
        addr = LaneAddress(
            move_type=MoveType.SITE,
            zone_id=0,
            word_id=0,
            site_id=1,
            bus_id=0,
            direction=Direction.FORWARD,
        )
        assert addr.direction == Direction.FORWARD
        assert addr.move_type == MoveType.SITE
        assert addr.word_id == 0
        assert addr.site_id == 1
        assert addr.bus_id == 0

    def test_default_direction(self):
        addr = LaneAddress(
            move_type=MoveType.SITE,
            zone_id=0,
            word_id=0,
            site_id=1,
            bus_id=0,
        )
        assert addr.direction == Direction.FORWARD

    def test_encode_decode_round_trip(self):
        addr = LaneAddress(
            move_type=MoveType.WORD,
            zone_id=0,
            word_id=1,
            site_id=2,
            bus_id=3,
            direction=Direction.BACKWARD,
        )
        bits = addr.encode()
        decoded = LaneAddress.decode(bits)
        assert decoded == addr

    def test_direction_enum_values(self):
        assert int(Direction.FORWARD) == 0
        assert int(Direction.BACKWARD) == 1

    def test_move_type_enum_values(self):
        assert int(MoveType.SITE) == 0
        assert int(MoveType.WORD) == 1

    def test_hash(self):
        a = LaneAddress(
            move_type=MoveType.SITE, zone_id=0, word_id=0, site_id=1, bus_id=0
        )
        b = LaneAddress(
            move_type=MoveType.SITE, zone_id=0, word_id=0, site_id=1, bus_id=0
        )
        assert hash(a) == hash(b)
        d = {a: "value"}
        assert d[b] == "value"


class TestZoneAddress:
    def test_construct_and_getters(self):
        addr = ZoneAddress(zone_id=5)
        assert addr.zone_id == 5

    def test_encode_decode_round_trip(self):
        addr = ZoneAddress(zone_id=42)
        bits = addr.encode()
        decoded = ZoneAddress.decode(bits)
        assert decoded == addr

    def test_hash(self):
        a = ZoneAddress(zone_id=5)
        b = ZoneAddress(zone_id=5)
        assert hash(a) == hash(b)
        d = {a: "value"}
        assert d[b] == "value"


# ── Instruction ──

# New opcode packing: (instruction_code << 8) | device_code
# Device codes: Cpu=0x00, LaneConst=0x0F, AtomArrangement=0x10,
#   QuantumGate=0x11, Measurement=0x12, Array=0x13, DetectorObservable=0x14


class TestInstruction:
    # The vihaco-backed ISA assigns its own opcode bytes (no legacy packed
    # (instr<<8)|device scheme), so these check instruction identity via the
    # stable op_name() rather than a specific opcode value.
    def test_const_float(self):
        inst = Instruction.const_float(1.5)
        assert inst.op_name() == "const_float"
        assert "const_float" in repr(inst)

    def test_const_int(self):
        assert Instruction.const_int(42).op_name() == "const_int"

    def test_const_loc(self):
        inst = Instruction.const_loc(zone_id=0, word_id=0, site_id=1)
        assert inst.op_name() == "const_loc"

    def test_const_lane(self):
        inst = Instruction.const_lane(
            move_type=MoveType.SITE,
            zone_id=0,
            word_id=0,
            site_id=1,
            bus_id=0,
            direction=Direction.FORWARD,
        )
        assert inst.op_name() == "const_lane"

    def test_const_zone(self):
        assert Instruction.const_zone(zone_id=0).op_name() == "const_zone"

    def test_stack_ops(self):
        assert Instruction.pop().op_name() == "pop"
        assert Instruction.dup().op_name() == "dup"
        assert Instruction.swap().op_name() == "swap"

    def test_atom_ops(self):
        assert Instruction.initial_fill(2).op_name() == "initial_fill"
        assert Instruction.fill(1).op_name() == "fill"
        assert Instruction.move_(1).op_name() == "move"

    def test_gate_ops(self):
        assert Instruction.local_r(1).op_name() == "local_r"
        assert Instruction.local_rz(1).op_name() == "local_rz"
        assert Instruction.global_r().op_name() == "global_r"
        assert Instruction.global_rz().op_name() == "global_rz"
        assert Instruction.cz().op_name() == "cz"

    def test_measurement_ops(self):
        assert Instruction.measure(1).op_name() == "measure"
        assert Instruction.await_measure().op_name() == "await_measure"

    def test_array_ops(self):
        assert Instruction.new_array(1, 10).op_name() == "new_array"
        assert Instruction.new_array(1, 10, 20).op_name() == "new_array"
        assert Instruction.get_item(2).op_name() == "get_item"

    def test_data_ops(self):
        assert Instruction.set_detector().op_name() == "set_detector"
        assert Instruction.set_observable().op_name() == "set_observable"

    def test_control_ops(self):
        assert Instruction.return_().op_name() == "return"
        assert Instruction.halt().op_name() == "halt"

    def test_equality(self):
        a = Instruction.halt()
        b = Instruction.halt()
        c = Instruction.pop()
        assert a == b
        assert a != c


class TestInstructionAccessors:
    def test_op_name_covers_every_opcode(self):
        # Exhaustive mapping of factory → expected op_name.
        cases = [
            (Instruction.const_float(0.0), "const_float"),
            (Instruction.const_int(0), "const_int"),
            (Instruction.const_loc(0, 0, 0), "const_loc"),
            (Instruction.const_lane(MoveType.SITE, 0, 0, 0, 0), "const_lane"),
            (Instruction.const_zone(0), "const_zone"),
            (Instruction.pop(), "pop"),
            (Instruction.dup(), "dup"),
            (Instruction.swap(), "swap"),
            (Instruction.initial_fill(1), "initial_fill"),
            (Instruction.fill(1), "fill"),
            (Instruction.move_(1), "move"),
            (Instruction.local_r(1), "local_r"),
            (Instruction.local_rz(1), "local_rz"),
            (Instruction.global_r(), "global_r"),
            (Instruction.global_rz(), "global_rz"),
            (Instruction.cz(), "cz"),
            (Instruction.measure(1), "measure"),
            (Instruction.await_measure(), "await_measure"),
            (Instruction.new_array(0, 1), "new_array"),
            (Instruction.get_item(1), "get_item"),
            (Instruction.set_detector(), "set_detector"),
            (Instruction.set_observable(), "set_observable"),
            (Instruction.return_(), "return"),
            (Instruction.halt(), "halt"),
        ]
        for instr, expected in cases:
            assert instr.op_name() == expected, (instr, expected)

    def test_arity_returns_field(self):
        assert Instruction.initial_fill(3).arity() == 3
        assert Instruction.fill(4).arity() == 4
        assert Instruction.move_(5).arity() == 5
        assert Instruction.local_r(2).arity() == 2
        assert Instruction.local_rz(1).arity() == 1
        assert Instruction.measure(7).arity() == 7

    def test_arity_raises_on_inapplicable_opcodes(self):
        with pytest.raises(RuntimeError):
            Instruction.const_float(0.0).arity()
        with pytest.raises(RuntimeError):
            Instruction.pop().arity()
        with pytest.raises(RuntimeError):
            Instruction.cz().arity()

    def test_float_value(self):
        assert Instruction.const_float(3.14).float_value() == 3.14
        with pytest.raises(RuntimeError):
            Instruction.const_int(0).float_value()

    def test_int_value(self):
        assert Instruction.const_int(42).int_value() == 42
        with pytest.raises(RuntimeError):
            Instruction.const_float(0.0).int_value()

    def test_location_address(self):
        addr = Instruction.const_loc(0, 1, 2).location_address()
        assert addr == LocationAddress(0, 1, 2)
        with pytest.raises(RuntimeError):
            Instruction.const_int(0).location_address()

    def test_lane_address(self):
        addr = Instruction.const_lane(MoveType.SITE, 0, 0, 0, 0).lane_address()
        assert addr == LaneAddress(MoveType.SITE, 0, 0, 0, 0)
        with pytest.raises(RuntimeError):
            Instruction.const_int(0).lane_address()

    def test_zone_address(self):
        addr = Instruction.const_zone(3).zone_address()
        assert addr == ZoneAddress(3)
        with pytest.raises(RuntimeError):
            Instruction.const_int(0).zone_address()

    def test_new_array_accessors(self):
        instr = Instruction.new_array(7, 4, 2)
        assert instr.type_tag() == 7
        assert instr.dim0() == 4
        assert instr.dim1() == 2
        with pytest.raises(RuntimeError):
            Instruction.pop().type_tag()

    def test_get_item_ndims(self):
        assert Instruction.get_item(3).ndims() == 3
        with pytest.raises(RuntimeError):
            Instruction.pop().ndims()


class TestInstructionAddressValidation:
    """Instruction address constants validate 16-bit range."""

    def test_const_loc_negative_word_id(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.const_loc(zone_id=0, word_id=-1, site_id=0)

    def test_const_loc_negative_site_id(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.const_loc(zone_id=0, word_id=0, site_id=-1)

    def test_const_loc_overflow(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            Instruction.const_loc(zone_id=0, word_id=0x10000, site_id=0)

    def test_const_lane_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.const_lane(
                move_type=MoveType.SITE, zone_id=0, word_id=-1, site_id=0, bus_id=0
            )

    def test_const_lane_overflow(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            Instruction.const_lane(
                move_type=MoveType.SITE, zone_id=0, word_id=0, site_id=0, bus_id=0x10000
            )

    def test_const_zone_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.const_zone(zone_id=-1)

    def test_const_zone_overflow(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            Instruction.const_zone(zone_id=0x100)

    def test_max_valid_values(self):
        Instruction.const_loc(zone_id=0, word_id=0xFFFF, site_id=0xFFFF)
        Instruction.const_lane(
            move_type=MoveType.SITE,
            zone_id=0,
            word_id=0xFFFF,
            site_id=0xFFFF,
            bus_id=0xFFFF,
        )
        Instruction.const_zone(zone_id=0xFF)


class TestInstructionArityValidation:
    """Instruction arity params validate non-negative u32 range."""

    def test_initial_fill_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.initial_fill(-1)

    def test_fill_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.fill(-1)

    def test_move_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.move_(-1)

    def test_local_r_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.local_r(-1)

    def test_local_rz_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.local_rz(-1)

    def test_measure_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            Instruction.measure(-1)

    def test_valid_zero(self):
        Instruction.fill(0)
        Instruction.local_r(0)
        Instruction.measure(0)


# ── Program ──


class TestProgramConstruction:
    def test_from_instructions(self):
        program = Program(
            version=(1, 0),
            instructions=[
                Instruction.const_loc(zone_id=0, word_id=0, site_id=0),
                Instruction.initial_fill(1),
                Instruction.halt(),
            ],
        )
        assert program.version == (1, 0)
        assert len(program) == 3
        assert len(program.instructions) == 3

    def test_from_text(self):
        source = """\
.version 1.0
const_loc 0x00000000
initial_fill 1
halt
"""
        program = Program.from_text(source)
        assert program.version == (1, 0)
        assert len(program) == 3

    def test_from_text_legacy_version(self):
        source = ".version 1\nhalt\n"
        program = Program.from_text(source)
        assert program.version == (1, 0)

    def test_from_text_invalid(self):
        with pytest.raises(MissingVersionError):
            Program.from_text("halt\n")  # missing .version


class TestProgramSerialization:
    def _sample_program(self):
        return Program.from_text("""\
.version 1.0
const_loc 0x00000000
const_loc 0x00000001
initial_fill 2
halt
""")

    def test_to_text(self):
        program = self._sample_program()
        text = program.to_text()
        assert ".version 1.0" in text
        assert "initial_fill 2" in text
        assert "halt" in text

    def test_text_round_trip(self):
        program = self._sample_program()
        text = program.to_text()
        reparsed = Program.from_text(text)
        assert program == reparsed

    def test_to_binary(self):
        program = self._sample_program()
        binary = program.to_binary()
        assert isinstance(binary, bytes)
        assert binary[:5] == b"LANES"

    def test_binary_round_trip(self):
        program = self._sample_program()
        binary = program.to_binary()
        decoded = Program.from_binary(binary)
        assert program == decoded

    def test_from_binary_invalid(self):
        # 9 bytes (header length) so the magic check runs before the length check.
        with pytest.raises(BadMagicError):
            Program.from_binary(b"XXXXX\x00\x00\x00\x00")

    def test_text_binary_round_trip(self):
        program = self._sample_program()
        binary = program.to_binary()
        from_binary = Program.from_binary(binary)
        text = from_binary.to_text()
        from_text = Program.from_text(text)
        assert program == from_text


class TestProgramValidation:
    def test_structural_valid(self):
        program = Program.from_text("""\
.version 1.0
const_loc 0x00000000
initial_fill 1
halt
""")
        program.validate()  # should not raise

    def test_structural_invalid(self):
        program = Program.from_text("""\
.version 1.0
halt
const_loc 0x00000000
initial_fill 1
""")
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(
            isinstance(e, InitialFillNotFirstError) for e in exc_info.value.errors
        )

    def test_stack_validation_is_noop(self):
        # Stack-type simulation has not been ported to the vihaco backend
        # (bloqade-lanes#769); `stack=True` is accepted but performs no
        # analysis. This program would underflow under the legacy simulator
        # but is structurally valid, so validate(stack=True) does not raise.
        program = Program.from_text("""\
.version 1.0
pop
halt
""")
        program.validate(stack=True)  # no-op: does not raise

    def test_stack_type_mismatch_is_noop(self):
        # Likewise, the legacy simulator caught a float used as a location
        # here; the vihaco backend's stack sim is a deferred no-op.
        program = Program.from_text("""\
.version 1.0
const.f64 1.0
initial_fill 1
halt
""")
        program.validate(stack=True)  # no-op: does not raise

    def test_empty_program_raises_empty_program_error(self):
        program = Program.from_text(".version 1.0\n")
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(isinstance(e, EmptyProgramError) for e in exc_info.value.errors)
        assert not any(
            isinstance(e, MissingTerminatorError) for e in exc_info.value.errors
        )

    def test_missing_terminator_raises_missing_terminator_error(self):
        program = Program.from_text("""\
.version 1.0
const.i64 0
""")
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(isinstance(e, MissingTerminatorError) for e in exc_info.value.errors)

    def test_unreachable_instruction_raises_unreachable_error(self):
        program = Program.from_text("""\
.version 1.0
halt
const.i64 0
""")
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(
            isinstance(e, UnreachableInstructionError) for e in exc_info.value.errors
        )
        assert not any(
            isinstance(e, MissingTerminatorError) for e in exc_info.value.errors
        )

    def test_valid_program_with_return_no_errors(self):
        program = Program.from_text("""\
.version 1.0
const.i64 0
return
""")
        program.validate()  # should not raise

    def test_valid_program_with_halt_no_errors(self):
        program = Program.from_text("""\
.version 1.0
halt
""")
        program.validate()  # should not raise


MINIMAL_ARCH_JSON = """{
    "version": "2.0",
    "words": [
        {"sites": [[0, 0], [1, 0]]}
    ],
    "zones": [
        {
            "grid": {
                "x_start": 0.0, "y_start": 0.0,
                "x_spacing": [1.0], "y_spacing": []
            },
            "site_buses": [],
            "word_buses": [],
            "words_with_site_buses": [],
            "sites_with_word_buses": []
        }
    ],
    "zone_buses": [],
    "modes": [
        {"name": "default", "zones": [0], "bitstring_order": []}
    ]
}"""


class TestCapabilityValidation:
    def test_single_measure_allowed(self):
        arch = ArchSpec.from_json(MINIMAL_ARCH_JSON)
        program = Program.from_text("""\
.version 1.0
const_loc 0x00000000
const_loc 0x00000001
initial_fill 2
const_zone 0x00000000
measure 1
await_measure
return
""")
        program.validate(arch=arch)  # should not raise

    def test_multiple_measure_rejected_without_feed_forward(self):
        arch = ArchSpec.from_json(MINIMAL_ARCH_JSON)
        program = Program.from_text("""\
.version 1.0
const_loc 0x00000000
const_loc 0x00000001
initial_fill 2
const_zone 0x00000000
measure 1
await_measure
const_zone 0x00000000
measure 1
await_measure
return
""")
        with pytest.raises(ValidationError) as exc_info:
            program.validate(arch=arch)
        assert any(
            isinstance(e, FeedForwardNotSupportedError) for e in exc_info.value.errors
        )

    def test_multiple_measure_allowed_with_feed_forward(self):
        import json

        data = json.loads(MINIMAL_ARCH_JSON)
        data["feed_forward"] = True
        arch = ArchSpec.from_json(json.dumps(data))
        program = Program.from_text("""\
.version 1.0
const_loc 0x00000000
const_loc 0x00000001
initial_fill 2
const_zone 0x00000000
measure 1
await_measure
const_zone 0x00000000
measure 1
await_measure
return
""")
        program.validate(arch=arch)  # should not raise

    def test_fill_rejected_without_atom_reloading(self):
        arch = ArchSpec.from_json(MINIMAL_ARCH_JSON)
        program = Program.from_text("""\
.version 1.0
const_loc 0x00000000
const_loc 0x00000001
initial_fill 2
const_loc 0x00000000
fill 1
halt
""")
        with pytest.raises(ValidationError) as exc_info:
            program.validate(arch=arch)
        assert any(
            isinstance(e, AtomReloadingNotSupportedError) for e in exc_info.value.errors
        )

    def test_fill_allowed_with_atom_reloading(self):
        import json

        data = json.loads(MINIMAL_ARCH_JSON)
        data["atom_reloading"] = True
        arch = ArchSpec.from_json(json.dumps(data))
        program = Program.from_text("""\
.version 1.0
const_loc 0x00000000
const_loc 0x00000001
initial_fill 2
const_loc 0x00000000
fill 1
halt
""")
        program.validate(arch=arch)  # should not raise

    def test_initial_fill_always_allowed(self):
        arch = ArchSpec.from_json(MINIMAL_ARCH_JSON)
        program = Program.from_text("""\
.version 1.0
const_loc 0x00000000
initial_fill 1
halt
""")
        program.validate(arch=arch)  # should not raise

    def test_error_attributes(self):
        err = FeedForwardNotSupportedError(pc=5)
        assert err.pc == 5
        assert "feed_forward" in str(err)

        err2 = AtomReloadingNotSupportedError(pc=10)
        assert err2.pc == 10
        assert "atom_reloading" in str(err2)


class TestProgramRepr:
    def test_repr(self):
        program = Program.from_text(".version 1.0\nhalt\n")
        r = repr(program)
        assert "Program" in r
        assert "(1, 0)" in r
        assert "1" in r  # instruction count
