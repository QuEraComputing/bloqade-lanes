//! The Bloqade Lanes [`Instruction`] enum and its encoding width.
//!
//! The instruction set is defined once as a `#[derive(Instruction)]` enum;
//! vihaco's derive generates the binary codec
//! ([`vihaco::instruction::WriteBytes`] / [`FromBytes`](vihaco::instruction::FromBytes)).
//! Text parsing is layered on top via the [`syntax`] mirror enums (see below).
//! See the [`super`] module docs for the design rationale.

use vihaco::Instruction as InstructionCodec;

use super::syntax;

/// Width, in bytes, of every encoded instruction word: 1 opcode byte plus a
/// payload, zero-padded to the widest variant. Decoding consumes exactly this
/// many bytes per instruction, so a flat program decodes without desync.
///
/// This is derived, not chosen: [`Instruction`] carries no `#[instruction(width
/// = N)]` override, so vihaco computes it as the max over all variants. Today
/// that maximum is [`Instruction::NewArray`] (1 + 3×u32 = 13).
///
/// It was pinned to 17 while the ISA nested vihaco-cpu's 16-byte instruction
/// word; vihaco-cpu 0.4 dropped its binary codec, the nesting went with it, and
/// the width now follows our own operands. The array / measurement-result
/// representation (today a bespoke `ARRAY_REF` + `new_array`/`get_item`) is
/// still slated to move onto a heap-allocated nested `IList` — see
/// <https://github.com/QuEraComputing/bloqade-lanes/issues/776>.
pub const INSTRUCTION_WIDTH: u32 = 13;

/// The Bloqade Lanes instruction set, defined on the vihaco framework.
///
/// Operands use only the scalar types vihaco implements byte traits for
/// (`u32`, `u64`, `i64`, `f64`).
///
/// **Variant order is significant**: it is the encoded opcode order (vihaco
/// assigns opcodes by declaration position), so inserting a variant anywhere
/// but the end renumbers everything after it.
///
/// ## Stack ops are lanes-native
///
/// `const_float`, `const_int`, `dup` and `halt` mirror vihaco-cpu's stack ops
/// but are declared here rather than nested. vihaco-cpu 0.4 turned into a
/// runtime *component*: its instruction enums implement `Parse` (surface) and
/// carry runtime values, but neither implements `WriteBytes`/`FromBytes`/
/// `OpCode`, so a nested variant can no longer be encoded. They keep the `cpu.`
/// text namespace to signal their provenance.
#[derive(Debug, Clone, PartialEq, InstructionCodec)]
pub enum Instruction {
    // ---- Stack ops (lanes-native; `cpu.` text namespace) ----
    Pop,
    Swap,
    Return,
    Dup,
    Halt,
    ConstFloat(f64),
    ConstInt(i64),

    // ---- Lane constants (hex operands) ----
    ConstLoc(u64),
    ConstLane(u64),
    ConstZone(u32),

    // ---- Atom arrangement ----
    InitialFill(u32),
    Fill(u32),
    Move(u32),

    // ---- Quantum gates ----
    LocalRz(u32),
    LocalR(u32),
    GlobalRz,
    GlobalR,
    Cz,

    // ---- Measurement ----
    Measure(u32),
    AwaitMeasure,

    // ---- Arrays ----
    // `new_array <type_tag> <dim0> <dim1>` — all three operands required
    // (1-D arrays use `dim1 = 0`).
    NewArray(u32, u32, u32),
    GetItem(u32),

    // ---- Detectors / observables ----
    SetDetector,
    SetObservable,
}

impl Instruction {
    /// Canonical opcode name used for decode dispatch (the Python decoder
    /// calls `_visit_{op_name}`) and introspection.
    ///
    /// This is the mnemonic without its dialect head: the text spelling of
    /// [`Instruction::Move`] is `lanes.move`, and its `op_name` is `move`.
    /// [`Display`](std::fmt::Display) writes the head; `op_name` never does.
    pub fn op_name(&self) -> &'static str {
        match self {
            Instruction::Pop => "pop",
            Instruction::Swap => "swap",
            Instruction::Return => "return",
            Instruction::Dup => "dup",
            Instruction::Halt => "halt",
            Instruction::ConstFloat(_) => "const_float",
            Instruction::ConstInt(_) => "const_int",
            Instruction::ConstLoc(_) => "const_loc",
            Instruction::ConstLane(_) => "const_lane",
            Instruction::ConstZone(_) => "const_zone",
            Instruction::InitialFill(_) => "initial_fill",
            Instruction::Fill(_) => "fill",
            Instruction::Move(_) => "move",
            Instruction::LocalRz(_) => "local_rz",
            Instruction::LocalR(_) => "local_r",
            Instruction::GlobalRz => "global_rz",
            Instruction::GlobalR => "global_r",
            Instruction::Cz => "cz",
            Instruction::Measure(_) => "measure",
            Instruction::AwaitMeasure => "await_measure",
            Instruction::NewArray(..) => "new_array",
            Instruction::GetItem(_) => "get_item",
            Instruction::SetDetector => "set_detector",
            Instruction::SetObservable => "set_observable",
        }
    }

    /// The dialect head this instruction is spelled under in `.sst` text:
    /// `"cpu"` for the stack ops, `"lanes"` for everything else.
    pub fn dialect(&self) -> &'static str {
        match self {
            Instruction::Pop
            | Instruction::Swap
            | Instruction::Return
            | Instruction::Dup
            | Instruction::Halt
            | Instruction::ConstFloat(_)
            | Instruction::ConstInt(_) => syntax::CPU_HEAD,
            _ => syntax::LANES_HEAD,
        }
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.dialect(), self.op_name())?;
        match self {
            // Unit variants: the mnemonic alone.
            Instruction::Pop
            | Instruction::Swap
            | Instruction::Return
            | Instruction::Dup
            | Instruction::Halt
            | Instruction::GlobalRz
            | Instruction::GlobalR
            | Instruction::Cz
            | Instruction::AwaitMeasure
            | Instruction::SetDetector
            | Instruction::SetObservable => Ok(()),
            // Hex operands: fixed-width so addresses line up by eye.
            Instruction::ConstLoc(v) | Instruction::ConstLane(v) => write!(f, " 0x{v:016x}"),
            Instruction::ConstZone(v) => write!(f, " 0x{v:08x}"),
            Instruction::ConstFloat(v) => write!(f, " {v:?}"),
            Instruction::ConstInt(v) => write!(f, " {v}"),
            Instruction::InitialFill(a)
            | Instruction::Fill(a)
            | Instruction::Move(a)
            | Instruction::LocalRz(a)
            | Instruction::LocalR(a)
            | Instruction::Measure(a)
            | Instruction::GetItem(a) => write!(f, " {a}"),
            Instruction::NewArray(t, d0, d1) => write!(f, " {t} {d0} {d1}"),
        }
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // sample floats are illustrative, not math constants
mod tests {
    use super::*;
    use chumsky::Parser as _;
    use vihaco::instruction::{FromBytes, OpCode, WriteBytes};
    use vihaco_parser::Parse;

    fn parse(input: &str) -> Instruction {
        Instruction::parser()
            .parse(input)
            .into_result()
            .unwrap_or_else(|e| panic!("parse({input:?}) failed: {e:?}"))
    }

    /// A representative instruction of every shape (unit, scalar, hex,
    /// multi-field, float and signed-int).
    fn sample_program() -> Vec<Instruction> {
        vec![
            Instruction::ConstFloat(3.14159),
            Instruction::ConstInt(-42),
            Instruction::Dup,
            Instruction::ConstLoc(0x0000_0000_0100_0000),
            Instruction::ConstLane(0x0000_0000_0000_0001),
            Instruction::ConstZone(0x0000_0007),
            Instruction::InitialFill(2),
            Instruction::Fill(1),
            Instruction::Move(2),
            Instruction::LocalRz(1),
            Instruction::LocalR(3),
            Instruction::GlobalRz,
            Instruction::GlobalR,
            Instruction::Cz,
            Instruction::Measure(1),
            Instruction::AwaitMeasure,
            Instruction::NewArray(1, 3, 0),
            Instruction::GetItem(1),
            Instruction::SetDetector,
            Instruction::SetObservable,
            Instruction::Pop,
            Instruction::Swap,
            Instruction::Halt,
            Instruction::Return,
        ]
    }

    #[test]
    fn every_instruction_encodes_to_fixed_width() {
        assert_eq!(Instruction::width(), INSTRUCTION_WIDTH);
        for inst in sample_program() {
            let mut buf = Vec::new();
            inst.write_bytes(&mut buf).unwrap();
            assert_eq!(
                buf.len(),
                INSTRUCTION_WIDTH as usize,
                "{inst:?} did not encode to a full {INSTRUCTION_WIDTH}-byte word"
            );
        }
    }

    #[test]
    fn binary_round_trips_a_flat_program() {
        let program = sample_program();

        // Encode every instruction back-to-back into one buffer.
        let mut bytes = Vec::new();
        for inst in &program {
            inst.write_bytes(&mut bytes).unwrap();
        }
        assert_eq!(bytes.len(), program.len() * INSTRUCTION_WIDTH as usize);

        // Decode the stream and confirm it matches, proving fixed-width words
        // stay aligned (no desync from padding).
        let mut cursor = std::io::Cursor::new(bytes);
        let mut decoded = Vec::new();
        for _ in 0..program.len() {
            decoded.push(Instruction::from_bytes(&mut cursor).unwrap());
        }
        assert_eq!(decoded, program);
    }

    #[test]
    fn text_parses_each_shape() {
        assert_eq!(
            parse("lanes.const_loc 0x0000000001000000"),
            Instruction::ConstLoc(0x0000_0000_0100_0000)
        );
        assert_eq!(
            parse("lanes.const_lane 0x0000000000000001"),
            Instruction::ConstLane(1)
        );
        assert_eq!(
            parse("lanes.const_zone 0x00000007"),
            Instruction::ConstZone(7)
        );
        assert_eq!(parse("lanes.initial_fill 2"), Instruction::InitialFill(2));
        assert_eq!(parse("lanes.move 2"), Instruction::Move(2));
        assert_eq!(
            parse("lanes.new_array 1 3 0"),
            Instruction::NewArray(1, 3, 0)
        );
        assert_eq!(parse("lanes.get_item 1"), Instruction::GetItem(1));

        for (text, inst) in [
            ("cpu.pop", Instruction::Pop),
            ("cpu.swap", Instruction::Swap),
            ("cpu.return", Instruction::Return),
            ("lanes.global_rz", Instruction::GlobalRz),
            ("lanes.global_r", Instruction::GlobalR),
            ("lanes.cz", Instruction::Cz),
            ("lanes.await_measure", Instruction::AwaitMeasure),
            ("lanes.set_detector", Instruction::SetDetector),
            ("lanes.set_observable", Instruction::SetObservable),
        ] {
            assert_eq!(parse(text), inst, "text {text:?}");
        }
    }

    #[test]
    fn stack_ops_use_the_cpu_head() {
        assert_eq!(parse("cpu.const_int 42"), Instruction::ConstInt(42));
        assert_eq!(parse("cpu.const_float 1.5"), Instruction::ConstFloat(1.5));
        assert_eq!(parse("cpu.dup"), Instruction::Dup);
        assert_eq!(parse("cpu.halt"), Instruction::Halt);
    }

    #[test]
    fn dialect_heads_are_required() {
        // A bare mnemonic with no head is not a valid instruction, and a
        // mnemonic under the wrong head does not parse either.
        for bad in ["move 2", "halt", "cpu.move 2", "lanes.halt"] {
            assert!(
                Instruction::parser().parse(bad).into_result().is_err(),
                "{bad:?} should not parse"
            );
        }
    }

    #[test]
    fn prefix_tokens_disambiguate() {
        // `local_r` is a prefix of `local_rz`; `global_r` of `global_rz`.
        assert_eq!(parse("lanes.local_rz 1"), Instruction::LocalRz(1));
        assert_eq!(parse("lanes.local_r 3"), Instruction::LocalR(3));
        assert_eq!(parse("lanes.global_rz"), Instruction::GlobalRz);
        assert_eq!(parse("lanes.global_r"), Instruction::GlobalR);
    }

    #[test]
    fn text_then_binary_agree() {
        let from_text = parse("lanes.const_loc 0x0000000001000000");
        let mut bytes = Vec::new();
        from_text.write_bytes(&mut bytes).unwrap();
        let decoded = Instruction::from_bytes(&mut std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(from_text, decoded);
    }

    #[test]
    fn display_is_dialect_head_plus_op_name() {
        // Display's first token is always `<dialect>.<op_name>`, so op_name /
        // Display / the parser token cannot drift apart.
        for inst in sample_program() {
            let rendered = inst.to_string();
            let head = rendered.split_whitespace().next().unwrap();
            assert_eq!(
                head,
                format!("{}.{}", inst.dialect(), inst.op_name()),
                "op_name/Display drift for {inst:?}"
            );
        }
    }

    #[test]
    fn op_name_is_unique_per_variant() {
        // `_visit_{op_name}` dispatch in the Python decoder requires that no
        // two variants share a name.
        let mut names: Vec<_> = sample_program().iter().map(|i| i.op_name()).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate op_name across variants");
    }

    #[test]
    fn display_round_trips_through_the_parser() {
        // Every variant's Display must re-parse to itself.
        for inst in sample_program() {
            let text = inst.to_string();
            assert_eq!(parse(&text), inst, "Display/parse mismatch for {text:?}");
        }
    }

    #[test]
    fn const_float_round_trips_exactly() {
        // `{:?}` on f64 is round-trip-exact; `{}` is not (it drops the `.0` on
        // integral floats, which would then re-parse as an int).
        for v in [0.0, -0.0, 1.0, 3.14159, f64::MIN, f64::MAX, 1e-300] {
            let inst = Instruction::ConstFloat(v);
            assert_eq!(parse(&inst.to_string()), inst, "float {v:?}");
        }
    }
}
