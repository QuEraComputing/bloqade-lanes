//! The Bloqade Lanes [`Instruction`] enum and its fixed encoding width.
//!
//! The instruction set is defined once as a `#[derive(Instruction, Parse)]`
//! enum; vihaco's derive macros generate the binary codec
//! ([`vihaco::instruction::WriteBytes`] / [`FromBytes`](vihaco::instruction::FromBytes))
//! and the text (`.sst`) parser ([`vihaco_parser_core::Parse`]). See the
//! [`super`] module docs for the design rationale (CPU-op reuse via
//! [`Cpu`](Instruction::Cpu), native byte layout).

use vihaco::Instruction;

/// Fixed width, in bytes, of every encoded instruction word: 1 opcode byte
/// plus a payload up to the nested [`vihaco_cpu::Instruction`] word (16 bytes),
/// zero-padded. Decoding consumes exactly this many bytes per instruction, so a
/// flat program decodes without desync.
///
/// The `17` is forced entirely by nesting the 16-byte vihaco-cpu word (1 + 16);
/// every lanes-native variant needs at most 13 bytes (`NewArray` = 1 + 3×u32).
/// It is *not* an alignment choice. This is entangled with the array /
/// measurement-result representation (today a bespoke `ARRAY_REF` +
/// `new_array`/`get_item`), which is slated to move onto vihaco-cpu's heap
/// allocator as a nested `IList` — see
/// <https://github.com/QuEraComputing/bloqade-lanes/issues/776>. That refactor
/// is deliberately deferred; the width stays 17 until it lands.
pub const INSTRUCTION_WIDTH: u32 = 17;

/// The Bloqade Lanes instruction set, defined on the vihaco framework.
///
/// Device operands use only the scalar types vihaco implements byte traits for
/// (`u32`, `u64`, `i64`, `f64`); the legacy `u8`/`u16` array operands are
/// widened to `u32`. CPU ops are reused from [`vihaco_cpu`] via the nested
/// [`Cpu`](Instruction::Cpu) variant (see module docs).
///
/// **Variant order is significant.** It is both the encoded opcode order and
/// the text parser's try-order; a token that is a prefix of another must be
/// declared *after* the longer token (hence `*_rz` precedes `*_r`), and the
/// `#[delegate]` [`Cpu`](Instruction::Cpu) variant is declared **last** so
/// device-specific tokens (e.g. `get_item <n>`) win over any vihaco-cpu token
/// they would otherwise shadow.
#[derive(Debug, Clone, PartialEq, Instruction, vihaco_parser::Parse)]
#[instruction(width = 17)]
pub enum Instruction {
    // ---- Lanes-native stack ops (no round-trippable vihaco-cpu equivalent) ----
    #[token = "pop"]
    Pop,
    #[token = "swap"]
    Swap,
    #[token = "return"]
    Return,

    // ---- Lane constants (hex operands) ----
    #[token = "const_loc"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstLoc(#[parse_with = "crate::isa::parse_helpers::hex_u64"] u64),

    #[token = "const_lane"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstLane(#[parse_with = "crate::isa::parse_helpers::hex_u64"] u64),

    #[token = "const_zone"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstZone(#[parse_with = "crate::isa::parse_helpers::hex_u32"] u32),

    // ---- Atom arrangement ----
    #[token = "initial_fill"]
    #[delimiters(open = "", close = "", separator = "")]
    InitialFill(u32),
    #[token = "fill"]
    #[delimiters(open = "", close = "", separator = "")]
    Fill(u32),
    #[token = "move"]
    #[delimiters(open = "", close = "", separator = "")]
    Move(u32),

    // ---- Quantum gates (`*_rz` before `*_r`: token-prefix ordering) ----
    #[token = "local_rz"]
    #[delimiters(open = "", close = "", separator = "")]
    LocalRz(u32),
    #[token = "local_r"]
    #[delimiters(open = "", close = "", separator = "")]
    LocalR(u32),
    #[token = "global_rz"]
    GlobalRz,
    #[token = "global_r"]
    GlobalR,
    #[token = "cz"]
    Cz,

    // ---- Measurement ----
    #[token = "measure"]
    #[delimiters(open = "", close = "", separator = "")]
    Measure(u32),
    #[token = "await_measure"]
    AwaitMeasure,

    // ---- Arrays ----
    // `new_array <type_tag> <dim0> <dim1>` — all three operands required
    // (1-D arrays use `dim1 = 0`). Legacy `u8`/`u16` widened to `u32`.
    #[token = "new_array"]
    #[delimiters(open = "", close = "", separator = " ")]
    NewArray(u32, u32, u32),
    #[token = "get_item"]
    #[delimiters(open = "", close = "", separator = "")]
    GetItem(u32),

    // ---- Detectors / observables ----
    #[token = "set_detector"]
    SetDetector,
    #[token = "set_observable"]
    SetObservable,

    // ---- CPU / stack ops, reused wholesale from vihaco-cpu ----
    // Declared LAST: parsing is `#[delegate]`d to vihaco-cpu's parser, so
    // device tokens above are tried first and win on any shared prefix.
    #[delegate]
    Cpu(vihaco_cpu::Instruction),
}

impl Instruction {
    /// Canonical opcode name used for decode dispatch (the Python decoder
    /// calls `_visit_{op_name}`) and introspection.
    ///
    /// For lanes-native ops this equals the parser `#[token]` / `Display`
    /// mnemonic. For nested vihaco-cpu ops it returns the **decode-handler**
    /// name (`const_float`/`const_int`/`dup`/`halt`), which deliberately
    /// differs from the vihaco-cpu *text* syntax (`const.f64`, …) because
    /// decode handler method names cannot contain `.`.
    pub fn op_name(&self) -> &'static str {
        match self {
            Instruction::Pop => "pop",
            Instruction::Swap => "swap",
            Instruction::Return => "return",
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
            Instruction::Cpu(cpu) => cpu_op_name(cpu),
        }
    }
}

/// Decode-handler dispatch name for a nested vihaco-cpu instruction. NOT the
/// text mnemonic (see [`Instruction::op_name`] docs).
fn cpu_op_name(cpu: &vihaco_cpu::Instruction) -> &'static str {
    use vihaco::value::Value;
    use vihaco_cpu::Instruction as Cpu;
    match cpu {
        Cpu::Const(Value::F64(_)) => "const_float",
        Cpu::Const(Value::I64(_)) => "const_int",
        Cpu::Const(_) => "const",
        Cpu::Dup => "dup",
        Cpu::Halt => "halt",
        _ => "cpu",
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Pop => f.write_str("pop"),
            Instruction::Swap => f.write_str("swap"),
            Instruction::Return => f.write_str("return"),
            Instruction::ConstLoc(v) => write!(f, "const_loc 0x{v:016x}"),
            Instruction::ConstLane(v) => write!(f, "const_lane 0x{v:016x}"),
            Instruction::ConstZone(v) => write!(f, "const_zone 0x{v:08x}"),
            Instruction::InitialFill(a) => write!(f, "initial_fill {a}"),
            Instruction::Fill(a) => write!(f, "fill {a}"),
            Instruction::Move(a) => write!(f, "move {a}"),
            Instruction::LocalRz(a) => write!(f, "local_rz {a}"),
            Instruction::LocalR(a) => write!(f, "local_r {a}"),
            Instruction::GlobalRz => f.write_str("global_rz"),
            Instruction::GlobalR => f.write_str("global_r"),
            Instruction::Cz => f.write_str("cz"),
            Instruction::Measure(a) => write!(f, "measure {a}"),
            Instruction::AwaitMeasure => f.write_str("await_measure"),
            Instruction::NewArray(t, d0, d1) => write!(f, "new_array {t} {d0} {d1}"),
            Instruction::GetItem(n) => write!(f, "get_item {n}"),
            Instruction::SetDetector => f.write_str("set_detector"),
            Instruction::SetObservable => f.write_str("set_observable"),
            // vihaco-cpu owns its own Display (const.f64 / dup / halt / …).
            Instruction::Cpu(cpu) => write!(f, "{cpu}"),
        }
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // sample floats are illustrative, not math constants
mod tests {
    use super::*;
    use chumsky::Parser as _;
    use vihaco::instruction::{FromBytes, OpCode, WriteBytes};
    use vihaco::value::Value;
    use vihaco_cpu::Instruction as Cpu;
    use vihaco_parser_core::Parse;

    fn parse(input: &str) -> Instruction {
        Instruction::parser()
            .parse(input)
            .into_result()
            .unwrap_or_else(|e| panic!("parse({input:?}) failed: {e:?}"))
    }

    /// A representative instruction of every shape (unit, scalar, hex,
    /// multi-field, and nested vihaco-cpu).
    fn sample_program() -> Vec<Instruction> {
        vec![
            Instruction::Cpu(Cpu::Const(Value::F64(3.14159))),
            Instruction::Cpu(Cpu::Const(Value::I64(-42))),
            Instruction::Cpu(Cpu::Dup),
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
            Instruction::Cpu(Cpu::Halt),
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
            parse("const_loc 0x0000000001000000"),
            Instruction::ConstLoc(0x0000_0000_0100_0000)
        );
        assert_eq!(
            parse("const_lane 0x0000000000000001"),
            Instruction::ConstLane(1)
        );
        assert_eq!(parse("const_zone 0x00000007"), Instruction::ConstZone(7));
        assert_eq!(parse("initial_fill 2"), Instruction::InitialFill(2));
        assert_eq!(parse("move 2"), Instruction::Move(2));
        assert_eq!(parse("new_array 1 3 0"), Instruction::NewArray(1, 3, 0));
        assert_eq!(parse("get_item 1"), Instruction::GetItem(1));

        for (text, inst) in [
            ("pop", Instruction::Pop),
            ("swap", Instruction::Swap),
            ("return", Instruction::Return),
            ("global_rz", Instruction::GlobalRz),
            ("global_r", Instruction::GlobalR),
            ("cz", Instruction::Cz),
            ("await_measure", Instruction::AwaitMeasure),
            ("set_detector", Instruction::SetDetector),
            ("set_observable", Instruction::SetObservable),
        ] {
            assert_eq!(parse(text), inst, "text {text:?}");
        }
    }

    #[test]
    fn cpu_ops_delegate_to_vihaco_cpu() {
        // CPU mnemonics use vihaco-cpu's syntax and route through the nested
        // `Cpu` variant.
        assert_eq!(
            parse("const.i64 42"),
            Instruction::Cpu(Cpu::Const(Value::I64(42)))
        );
        assert_eq!(
            parse("const.f64 1.5"),
            Instruction::Cpu(Cpu::Const(Value::F64(1.5)))
        );
        assert_eq!(parse("dup"), Instruction::Cpu(Cpu::Dup));
        assert_eq!(parse("halt"), Instruction::Cpu(Cpu::Halt));
    }

    #[test]
    fn device_token_wins_over_delegated_cpu() {
        // vihaco-cpu also defines `get_item` (unit), but the lanes array
        // `get_item <n>` is declared first and must win on a full line.
        assert_eq!(parse("get_item 2"), Instruction::GetItem(2));
    }

    #[test]
    fn prefix_tokens_disambiguate() {
        // `local_r` is a prefix of `local_rz`; `global_r` of `global_rz`.
        assert_eq!(parse("local_rz 1"), Instruction::LocalRz(1));
        assert_eq!(parse("local_r 3"), Instruction::LocalR(3));
        assert_eq!(parse("global_rz"), Instruction::GlobalRz);
        assert_eq!(parse("global_r"), Instruction::GlobalR);
    }

    #[test]
    fn text_then_binary_agree() {
        let from_text = parse("const_loc 0x0000000001000000");
        let mut bytes = Vec::new();
        from_text.write_bytes(&mut bytes).unwrap();
        let decoded = Instruction::from_bytes(&mut std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(from_text, decoded);
    }

    #[test]
    fn op_name_matches_display_leading_token_for_native_ops() {
        // For lanes-native ops, op_name() == the first whitespace token of Display
        // (Display is itself pinned to the parser #[token] by the round-trip test),
        // so op_name / Display / parser token cannot drift apart.
        let native = [
            Instruction::Pop,
            Instruction::Swap,
            Instruction::Return,
            Instruction::ConstLoc(0),
            Instruction::ConstLane(0),
            Instruction::ConstZone(0),
            Instruction::InitialFill(1),
            Instruction::Fill(1),
            Instruction::Move(1),
            Instruction::LocalRz(1),
            Instruction::LocalR(1),
            Instruction::GlobalRz,
            Instruction::GlobalR,
            Instruction::Cz,
            Instruction::Measure(1),
            Instruction::AwaitMeasure,
            Instruction::NewArray(1, 1, 0),
            Instruction::GetItem(1),
            Instruction::SetDetector,
            Instruction::SetObservable,
        ];
        for inst in native {
            let display_head = inst.to_string();
            let head = display_head.split_whitespace().next().unwrap();
            assert_eq!(inst.op_name(), head, "op_name/Display drift for {inst:?}");
        }
    }

    #[test]
    fn cpu_op_name_uses_decode_handler_names() {
        assert_eq!(
            Instruction::Cpu(Cpu::Const(Value::F64(0.0))).op_name(),
            "const_float"
        );
        assert_eq!(
            Instruction::Cpu(Cpu::Const(Value::I64(0))).op_name(),
            "const_int"
        );
        assert_eq!(Instruction::Cpu(Cpu::Dup).op_name(), "dup");
        assert_eq!(Instruction::Cpu(Cpu::Halt).op_name(), "halt");
    }

    #[test]
    fn display_matches_parser_tokens() {
        use std::string::ToString;
        // Every non-CPU variant's Display must re-parse to itself.
        let samples = [
            Instruction::ConstLoc(0x0100_0000),
            Instruction::Move(2),
            Instruction::LocalRz(1),
            Instruction::GlobalR,
            Instruction::NewArray(1, 3, 0),
            Instruction::Pop,
            Instruction::Return,
            Instruction::Cpu(Cpu::Const(Value::F64(1.5))),
            Instruction::Cpu(Cpu::Halt),
        ];
        for inst in samples {
            let text = inst.to_string();
            assert_eq!(parse(&text), inst, "Display/parse mismatch for {text:?}");
        }
    }
}
