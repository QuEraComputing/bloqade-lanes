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
    FeedForwardNotSupportedError,
    InitialFillNotFirstError,
    MissingVersionError,
    StackUnderflowError,
    TypeMismatchError,
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
    def test_const_float(self):
        inst = Instruction.const_float(1.5)
        assert inst.opcode == 0x0300  # Cpu device=0x00, inst=0x03
        assert "const_float" in repr(inst)

    def test_const_int(self):
        inst = Instruction.const_int(42)
        assert inst.opcode == 0x0200  # Cpu device=0x00, inst=0x02

    def test_const_loc(self):
        inst = Instruction.const_loc(zone_id=0, word_id=0, site_id=1)
        assert inst.opcode == 0x000F  # LaneConst device=0x0F, inst=0x00

    def test_const_lane(self):
        inst = Instruction.const_lane(
            move_type=MoveType.SITE,
            zone_id=0,
            word_id=0,
            site_id=1,
            bus_id=0,
            direction=Direction.FORWARD,
        )
        assert inst.opcode == 0x010F  # LaneConst device=0x0F, inst=0x01

    def test_const_zone(self):
        inst = Instruction.const_zone(zone_id=0)
        assert inst.opcode == 0x020F  # LaneConst device=0x0F, inst=0x02

    def test_stack_ops(self):
        assert Instruction.pop().opcode == 0x0500  # Cpu device=0x00, inst=0x05
        assert Instruction.dup().opcode == 0x0400  # Cpu device=0x00, inst=0x04
        assert Instruction.swap().opcode == 0x0600  # Cpu device=0x00, inst=0x06

    def test_atom_ops(self):
        assert Instruction.initial_fill(2).opcode == 0x0010  # AA device=0x10, inst=0x00
        assert Instruction.fill(1).opcode == 0x0110  # AA device=0x10, inst=0x01
        assert Instruction.move_(1).opcode == 0x0210  # AA device=0x10, inst=0x02

    def test_gate_ops(self):
        assert Instruction.local_r(1).opcode == 0x0011  # QG device=0x11, inst=0x00
        assert Instruction.local_rz(1).opcode == 0x0111  # QG device=0x11, inst=0x01
        assert Instruction.global_r().opcode == 0x0211  # QG device=0x11, inst=0x02
        assert Instruction.global_rz().opcode == 0x0311  # QG device=0x11, inst=0x03
        assert Instruction.cz().opcode == 0x0411  # QG device=0x11, inst=0x04

    def test_measurement_ops(self):
        assert Instruction.measure(1).opcode == 0x0012  # Meas device=0x12, inst=0x00
        assert (
            Instruction.await_measure().opcode == 0x0112
        )  # Meas device=0x12, inst=0x01

    def test_array_ops(self):
        assert (
            Instruction.new_array(1, 10).opcode == 0x0013
        )  # Array device=0x13, inst=0x00
        assert Instruction.new_array(1, 10, 20).opcode == 0x0013
        assert Instruction.get_item(2).opcode == 0x0113  # Array device=0x13, inst=0x01

    def test_data_ops(self):
        assert Instruction.set_detector().opcode == 0x0014  # DO device=0x14, inst=0x00
        assert (
            Instruction.set_observable().opcode == 0x0114
        )  # DO device=0x14, inst=0x01

    def test_control_ops(self):
        assert Instruction.return_().opcode == 0x6400  # Cpu device=0x00, inst=0x64
        assert Instruction.halt().opcode == 0xFF00  # Cpu device=0x00, inst=0xFF

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
            (Instruction.move_(1), "move_"),
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
            (Instruction.return_(), "return_"),
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
        assert binary[:4] == b"BLQD"

    def test_binary_round_trip(self):
        program = self._sample_program()
        binary = program.to_binary()
        decoded = Program.from_binary(binary)
        assert program == decoded

    def test_from_binary_invalid(self):
        with pytest.raises(BadMagicError):
            Program.from_binary(b"XXXX\x00\x00\x00\x00")

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

    def test_stack_validation(self):
        program = Program.from_text("""\
.version 1.0
pop
""")
        with pytest.raises(ValidationError) as exc_info:
            program.validate(stack=True)
        assert any(isinstance(e, StackUnderflowError) for e in exc_info.value.errors)

    def test_stack_type_mismatch(self):
        program = Program.from_text("""\
.version 1.0
const_float 1.0
initial_fill 1
""")
        with pytest.raises(ValidationError) as exc_info:
            program.validate(stack=True)
        assert any(isinstance(e, TypeMismatchError) for e in exc_info.value.errors)


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
halt
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
halt
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
halt
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
