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
from bloqade.lanes.bytecode.decode import BytecodeDecoder, DecodingError
from bloqade.lanes.bytecode.exceptions import (
    AtomReloadingNotSupportedError,
    BadInstructionError,
    BadMagicError,
    EmptyProgramError,
    FeedForwardNotSupportedError,
    GetItemInvalidDimsError,
    InitialFillNotFirstError,
    LocalIndexOutOfRangeError,
    MissingTerminatorError,
    MissingVersionError,
    NewArrayTooManyElementsError,
    PopBelowFrameBaseError,
    StackDepthMismatchError,
    StackUnderflowError,
    TypeMismatchError,
    UnalignedCodeError,
    UnreachableInstructionError,
)
from bloqade.lanes.dialects import stack_move


def _sst(body: str, version: str = "1.0") -> str:
    """Wrap a program body in vihaco's ``sst v1`` section container.

    The framing is eleven lines that say nothing about the test, and it was
    pasted into every case that needed a program. Behind one helper, a
    container change is one edit rather than twenty.

    ``body`` is the ``.text(root)`` payload — normally a whole ``fn @main()``
    block, trailing newline included.
    """
    return (
        f"sst v1\n\n.section(root):\n.header(root):\nversion {version}\n"
        f".header(root).\n.text(root):\n{body}.text(root).\n.section(root).\n"
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


class TestInstruction:
    # vihaco assigns opcode bytes by variant declaration order, so they shift
    # whenever the ISA gains a variant. These check instruction identity via the
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
        # Three instructions plus the `func_start`/`func_end` delimiting `@main`.
        assert len(program) == 5
        assert len(program.instructions) == 5

    def test_from_text(self):
        source = _sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.initial_fill 1
  cpu::cpu.halt
}
""")
        program = Program.from_text(source)
        assert program.version == (1, 0)
        # Three instructions plus `@main`'s two function markers.
        assert len(program) == 5

    def test_from_text_invalid(self):
        # Well-formed container, no `.header(root)` section — spelled out
        # rather than built with `_sst`, because the missing header is the
        # thing under test.
        with pytest.raises(MissingVersionError):
            Program.from_text(
                "sst v1\n\n.section(root):\n.text(root):\n"
                "fn @main() {\n  cpu::cpu.halt\n}\n"
                ".text(root).\n.section(root).\n"
            )

    def test_from_text_rejects_non_container(self):
        # The bare `version 1.0;` + `fn @main` form is no longer accepted.
        with pytest.raises(BadInstructionError):
            Program.from_text("version 1.0;\nfn @main() {\n  cpu::cpu.halt\n}\n")


class TestProgramSerialization:
    def _sample_program(self):
        return Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.const_loc 0x00000001
  lanes::lanes.initial_fill 2
  cpu::cpu.halt
}
"""))

    def test_to_text(self):
        program = self._sample_program()
        text = program.to_text()
        assert "version 1.0" in text
        assert "fn @main()" in text
        assert "lanes::lanes.initial_fill 2" in text
        assert "cpu::cpu.halt" in text

    def test_text_round_trip(self):
        program = self._sample_program()
        text = program.to_text()
        reparsed = Program.from_text(text)
        assert program == reparsed

    def test_to_binary(self):
        program = self._sample_program()
        binary = program.to_binary()
        assert isinstance(binary, bytes)
        assert binary[:4] == b"VHBC"

    def test_binary_round_trip(self):
        program = self._sample_program()
        binary = program.to_binary()
        decoded = Program.from_binary(binary)
        assert program == decoded

    def test_from_binary_invalid(self):
        # 9 bytes (header length) so the magic check runs before the length check.
        with pytest.raises(BadMagicError):
            Program.from_binary(b"XXXXX\x00\x00\x00\x00")

    def test_bad_magic_message_mentions_vhbc(self):
        with pytest.raises(BadMagicError) as e:
            Program.from_binary(b"XXXXX\x00\x00\x00\x00")
        assert "VHBC" in str(e.value)

    def test_unaligned_binary_raises_unaligned_code_error(self):
        # Grow the bytecode region by one byte, and everything whose extent or
        # offset covers it, so the container stays well-formed and the only
        # fault is that the region is no longer a whole number of instruction
        # words. Offsets are read out of the file rather than hard-coded.
        raw = bytearray(self._sample_program().to_binary())

        def u64(at):
            return int.from_bytes(raw[at : at + 8], "little")

        def bump(at):
            raw[at : at + 8] = (u64(at) + 1).to_bytes(8, "little")

        FILE_HEADER = 16
        section = FILE_HEADER + u64(8)  # past the global context
        bytecode_len_at = section + 16 + u64(section + 8)
        bytecode_at = bytecode_len_at + 8
        code_len = u64(bytecode_len_at)

        bump(section)  # section_len
        bump(bytecode_len_at)
        raw.insert(bytecode_at + code_len, 0)

        # The child sections shifted with the insert.
        child_table = bytecode_at + code_len + 1
        child_count = int.from_bytes(raw[child_table : child_table + 4], "little")
        for i in range(child_count):
            bump(child_table + 4 + i * 12 + 4)

        with pytest.raises(UnalignedCodeError):
            Program.from_binary(bytes(raw))

    def test_text_binary_round_trip(self):
        program = self._sample_program()
        binary = program.to_binary()
        from_binary = Program.from_binary(binary)
        text = from_binary.to_text()
        from_text = Program.from_text(text)
        assert program == from_text


class TestProgramValidation:
    def test_structural_valid(self):
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.initial_fill 1
  cpu::cpu.halt
}
"""))
        program.validate()  # should not raise

    def test_structural_invalid(self):
        program = Program.from_text(_sst("""\
fn @main() {
  cpu::cpu.halt
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.initial_fill 1
}
"""))
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(
            isinstance(e, InitialFillNotFirstError) for e in exc_info.value.errors
        )

    def test_stack_validation(self):
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.pop
}
"""))
        with pytest.raises(ValidationError) as exc_info:
            program.validate(stack=True)
        assert any(isinstance(e, StackUnderflowError) for e in exc_info.value.errors)

    def test_stack_type_mismatch(self):
        program = Program.from_text(_sst("""\
fn @main() {
  cpu::cpu.const f64, 1.0
  lanes::lanes.initial_fill 1
}
"""))
        with pytest.raises(ValidationError) as exc_info:
            program.validate(stack=True)
        assert any(isinstance(e, TypeMismatchError) for e in exc_info.value.errors)

    def test_empty_program_raises_empty_program_error(self):
        program = Program.from_text(_sst("fn @main() {\n}\n"))
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(isinstance(e, EmptyProgramError) for e in exc_info.value.errors)
        assert not any(
            isinstance(e, MissingTerminatorError) for e in exc_info.value.errors
        )

    def test_missing_terminator_raises_missing_terminator_error(self):
        program = Program.from_text(_sst("""\
fn @main() {
  cpu::cpu.const i64, 0
}
"""))
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(isinstance(e, MissingTerminatorError) for e in exc_info.value.errors)

    def test_unreachable_instruction_raises_unreachable_error(self):
        program = Program.from_text(_sst("""\
fn @main() {
  cpu::cpu.halt
  cpu::cpu.const i64, 0
}
"""))
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        assert any(
            isinstance(e, UnreachableInstructionError) for e in exc_info.value.errors
        )
        assert not any(
            isinstance(e, MissingTerminatorError) for e in exc_info.value.errors
        )

    def test_valid_program_with_return_no_errors(self):
        program = Program.from_text(_sst("""\
fn @main() {
  cpu::cpu.const i64, 0
  cpu::cpu.ret 0
}
"""))
        program.validate()  # should not raise

    def test_valid_program_with_halt_no_errors(self):
        program = Program.from_text(_sst("""\
fn @main() {
  cpu::cpu.halt
}
"""))
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
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.const_loc 0x00000001
  lanes::lanes.initial_fill 2
  lanes::lanes.const_zone 0x00000000
  lanes::lanes.measure 1
  lanes::lanes.await_measure
  cpu::cpu.ret 0
}
"""))
        program.validate(arch=arch)  # should not raise

    def test_multiple_measure_rejected_without_feed_forward(self):
        arch = ArchSpec.from_json(MINIMAL_ARCH_JSON)
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.const_loc 0x00000001
  lanes::lanes.initial_fill 2
  lanes::lanes.const_zone 0x00000000
  lanes::lanes.measure 1
  lanes::lanes.await_measure
  lanes::lanes.const_zone 0x00000000
  lanes::lanes.measure 1
  lanes::lanes.await_measure
  cpu::cpu.ret 0
}
"""))
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
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.const_loc 0x00000001
  lanes::lanes.initial_fill 2
  lanes::lanes.const_zone 0x00000000
  lanes::lanes.measure 1
  lanes::lanes.await_measure
  lanes::lanes.const_zone 0x00000000
  lanes::lanes.measure 1
  lanes::lanes.await_measure
  cpu::cpu.ret 0
}
"""))
        program.validate(arch=arch)  # should not raise

    def test_fill_rejected_without_atom_reloading(self):
        arch = ArchSpec.from_json(MINIMAL_ARCH_JSON)
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.const_loc 0x00000001
  lanes::lanes.initial_fill 2
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.fill 1
  cpu::cpu.halt
}
"""))
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
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.const_loc 0x00000001
  lanes::lanes.initial_fill 2
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.fill 1
  cpu::cpu.halt
}
"""))
        program.validate(arch=arch)  # should not raise

    def test_initial_fill_always_allowed(self):
        arch = ArchSpec.from_json(MINIMAL_ARCH_JSON)
        program = Program.from_text(_sst("""\
fn @main() {
  lanes::lanes.const_loc 0x00000000
  lanes::lanes.initial_fill 1
  cpu::cpu.halt
}
"""))
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
        program = Program.from_text(_sst("fn @main() {\n  cpu::cpu.halt\n}\n"))
        r = repr(program)
        assert "Program" in r
        assert "(1, 0)" in r
        assert "1" in r  # instruction count


class TestDecoderDispatch:
    """The decoder keys handlers on ``(device, op_name)``.

    A name alone is not unique across the machine's two devices: ``get_item``
    is both vihaco-cpu's heap indexing and the lanes device's array indexing.
    Dispatching on the name sent a CPU ``get_item`` to the lanes handler,
    which failed on ``ndims()`` with a self-contradictory message rather than
    reaching the purpose-built diagnostic below.
    """

    def test_cpu_get_item_reports_no_representation(self):
        program = Program.from_text(
            _sst("fn @main() {\n  cpu::cpu.get_item\n  cpu::cpu.halt\n}\n")
        )
        # Index 1: index 0 is `@main`'s `func_start`.
        instr = program.instructions[1]
        assert (instr.device(), instr.op_name()) == ("cpu", "get_item")

        with pytest.raises(DecodingError) as exc_info:
            BytecodeDecoder().decode(program)
        assert "`cpu::get_item` has no stack_move representation" in str(exc_info.value)
        assert exc_info.value.instruction_index == 1

    def test_lanes_get_item_still_decodes(self):
        # The other half of the pair must be unaffected.
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  cpu::cpu.const i64, 1\n"
                "  lanes::lanes.new_array 1 1 0\n"
                "  cpu::cpu.const i64, 0\n"
                "  lanes::lanes.get_item 1\n"
                "  lanes::lanes.pop\n"
                "  cpu::cpu.halt\n}\n"
            )
        )
        # Index 4: the leading `func_start` shifts every body instruction.
        instr = program.instructions[4]
        assert (instr.device(), instr.op_name()) == ("lanes", "get_item")
        BytecodeDecoder().decode(program)  # should not raise

    def test_arithmetic_reports_its_device(self):
        program = Program.from_text(
            _sst("fn @main() {\n  cpu::cpu.add i64\n  cpu::cpu.halt\n}\n")
        )
        with pytest.raises(DecodingError) as exc_info:
            BytecodeDecoder().decode(program)
        assert "`cpu::add` has no stack_move representation" in str(exc_info.value)


class TestArrayOperandBounds:
    """``new_array`` and ``get_item`` operands come straight out of the
    instruction word, so the validator bounds them rather than looping on
    whatever they say."""

    def test_new_array_element_count_overflow_is_rejected(self):
        # 65536 * 65536 == 2**32, which wrapped to zero in the old u32
        # arithmetic — the program validated without examining an operand.
        program = Program.from_text(
            _sst(
                "fn @main() {\n  lanes::lanes.new_array 0 65536 65536\n  cpu::cpu.halt\n}\n"
            )
        )
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        errs = [
            e
            for e in exc_info.value.errors
            if isinstance(e, NewArrayTooManyElementsError)
        ]
        assert errs and errs[0].count == 2**32

    def test_get_item_index_count_is_bounded(self):
        # Arrays are at most 2-D, so three indices is structurally wrong.
        program = Program.from_text(
            _sst("fn @main() {\n  lanes::lanes.get_item 3\n  cpu::cpu.halt\n}\n")
        )
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        errs = [
            e for e in exc_info.value.errors if isinstance(e, GetItemInvalidDimsError)
        ]
        assert errs and errs[0].ndims == 3

    def test_two_indices_are_accepted(self):
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  cpu::cpu.const i64, 1\n"
                "  cpu::cpu.const i64, 2\n"
                "  lanes::lanes.new_array 1 1 2\n"
                "  cpu::cpu.const i64, 0\n"
                "  cpu::cpu.const i64, 0\n"
                "  lanes::lanes.get_item 2\n"
                "  lanes::lanes.pop\n"
                "  cpu::cpu.halt\n}\n"
            )
        )
        program.validate(stack=True)  # should not raise


class TestLocalIndexBounds:
    """``load``/``store`` take a local index straight out of the instruction
    word, and ``store`` grows the operand stack to reach it — so the validator
    bounds the index rather than letting it become an allocation."""

    def test_store_local_index_is_bounded(self):
        # 200_000_000 measured at 3.2 GB resident in issue #1032; u32::MAX
        # would be ~68 GB. The assertion is on the operand the validator
        # reports, not on the allocator noticing.
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  cpu::cpu.const u64, 7\n"
                "  cpu::cpu.store u64, 200000000\n"
                "  cpu::cpu.halt\n}\n"
            )
        )
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        errs = [
            e for e in exc_info.value.errors if isinstance(e, LocalIndexOutOfRangeError)
        ]
        assert errs and errs[0].index == 200_000_000
        assert errs[0].mnemonic == "store"
        # pc 2: address 0 is `@main`'s `func_start`, 1 is the `const`.
        assert errs[0].pc == 2

    def test_load_local_index_is_bounded(self):
        program = Program.from_text(
            _sst("fn @main() {\n  cpu::cpu.load u64, 4294967295\n  cpu::cpu.halt\n}\n")
        )
        with pytest.raises(ValidationError) as exc_info:
            program.validate()
        errs = [
            e for e in exc_info.value.errors if isinstance(e, LocalIndexOutOfRangeError)
        ]
        assert errs and errs[0].mnemonic == "load"

    def test_an_ordinary_local_index_is_accepted(self):
        # A function's locals are its arguments, so a small index is the
        # whole legitimate range and must keep validating.
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  cpu::cpu.const u64, 7\n"
                "  cpu::cpu.store u64, 0\n"
                "  cpu::cpu.load u64, 0\n"
                "  lanes::lanes.pop\n"
                "  cpu::cpu.halt\n}\n"
            )
        )
        program.validate()  # should not raise


class TestStackDataflow:
    """The stack simulation walks each function's control-flow graph, from a
    frame seeded by its declared parameters (#1042)."""

    def test_a_callee_consuming_its_parameter_validates(self):
        # Simulated from an empty stack, the argument looked like an underflow.
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  lanes::lanes.const_zone 0x00000000\n"
                "  cpu::cpu.call 1, helper\n"
                "  lanes::lanes.pop\n"
                "  cpu::cpu.halt\n}\n\n"
                "fn @helper(z: u32) -> heap_ref {\n"
                "  cpu::cpu.load u32, 0\n"
                "  lanes::lanes.measure 1\n"
                "  lanes::lanes.await_measure\n"
                "  cpu::cpu.ret 1\n}\n"
            )
        )
        program.validate(stack=True)  # should not raise

    def test_popping_a_callers_value_is_a_frame_underflow(self):
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  lanes::lanes.const_zone 0x00000000\n"
                "  cpu::cpu.call 0, helper\n"
                "  lanes::lanes.cz\n"
                "  cpu::cpu.halt\n}\n\n"
                "fn @helper() {\n"
                "  lanes::lanes.cz\n"
                "  cpu::cpu.ret 0\n}\n"
            )
        )
        with pytest.raises(ValidationError) as exc_info:
            program.validate(stack=True)
        errs = [
            e for e in exc_info.value.errors if isinstance(e, PopBelowFrameBaseError)
        ]
        assert [e.pc for e in errs] == [7]
        # Still catchable as the underflow it is.
        assert isinstance(errs[0], StackUnderflowError)

    def test_arms_leaving_different_depths_are_rejected(self):
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  cpu::cpu.const bool, true\n"
                "  cpu::cpu.cond_br @one, @zero\n"
                "  cpu::cpu.label @one\n"
                "  lanes::lanes.const_zone 0x00000000\n"
                "  cpu::cpu.br @done\n"
                "  cpu::cpu.label @zero\n"
                "  cpu::cpu.br @done\n"
                "  cpu::cpu.label @done\n"
                "  cpu::cpu.halt\n}\n"
            )
        )
        with pytest.raises(ValidationError) as exc_info:
            program.validate(stack=True)
        errs = [
            e for e in exc_info.value.errors if isinstance(e, StackDepthMismatchError)
        ]
        assert len(errs) == 1
        assert (errs[0].expected, errs[0].got) in {(0, 1), (1, 0)}


class TestMeasurementPipeline:
    """``measure -> await_measure -> set_detector`` must type-check.

    ``await_measure`` pushed a distinct measurement-result tag while
    ``set_detector`` popped an array ref, so the canonical pipeline could
    never validate.
    """

    def test_measure_await_set_detector_validates(self):
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  lanes::lanes.const_zone 0x00000000\n"
                "  lanes::lanes.measure 1\n"
                "  lanes::lanes.await_measure\n"
                "  lanes::lanes.set_detector\n"
                "  lanes::lanes.pop\n"
                "  cpu::cpu.halt\n}\n"
            )
        )
        program.validate(stack=True)  # should not raise

    def test_measurement_result_is_a_valid_element_tag(self):
        # Tag 9 names the *element* type of a measurement-result array, which
        # is what issue #547 asked for.
        assert 9 in stack_move.TYPE_TAG
        program = Program.from_text(
            _sst(
                "fn @main() {\n"
                "  cpu::cpu.const i64, 1\n"
                "  lanes::lanes.new_array 9 1 0\n"
                "  lanes::lanes.set_observable\n"
                "  lanes::lanes.pop\n"
                "  cpu::cpu.halt\n}\n"
            )
        )
        program.validate(stack=True)  # should not raise
