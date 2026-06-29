//! Vihaco-backed instruction set (ISA) for the Bloqade Lanes bytecode.
//!
//! This is the in-progress migration of the hand-rolled bytecode (see the
//! [`crate::bytecode`] module) onto the [`vihaco`] virtual-ISA framework, per
//! <https://github.com/QuEraComputing/bloqade-lanes/issues/769>.
//!
//! The instruction set is defined once as a `#[derive(Instruction, Parse)]`
//! enum; vihaco's derive macros then generate:
//!
//! - binary encode/decode ([`vihaco::instruction::WriteBytes`] /
//!   [`vihaco::instruction::FromBytes`]) — a 1-byte opcode followed by a
//!   little-endian payload, zero-padded to a fixed [`INSTRUCTION_WIDTH`]-byte
//!   word, so a program is simply N concatenated words;
//! - a text (`.sst`) parser ([`vihaco_parser_core::Parse`]).
//!
//! This adopts vihaco's **native** byte layout (the issue #769 decision); it is
//! intentionally *not* compatible with the legacy `BLQD` container in
//! [`crate::bytecode::program`].

pub mod parse_helpers;

use vihaco::Instruction;

/// Fixed width, in bytes, of every encoded instruction word: 1 opcode byte
/// plus up to 8 payload bytes (the widest operand is a 64-bit address), with
/// the remainder zero-padded. Decoding consumes exactly this many bytes per
/// instruction, so a flat program decodes without desync.
pub const INSTRUCTION_WIDTH: u32 = 16;

/// The Bloqade Lanes instruction set, defined on the vihaco framework.
///
/// Operands use only the scalar types vihaco implements byte traits for
/// (`u32`, `u64`, `i64`, `f64`); the legacy `u8`/`u16` array operands are
/// widened to `u32`.
///
/// **Variant order is significant.** It is both the encoded opcode order and
/// the text parser's try-order; a token that is a prefix of another must be
/// declared *after* the longer token. Hence `*_rz` precedes `*_r`.
#[derive(Debug, Clone, PartialEq, Instruction, vihaco_parser::Parse)]
#[instruction(width = 16)]
pub enum Instruction {
    // ---- CPU / stack ----
    #[token = "const_float"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstFloat(f64),

    #[token = "const_int"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstInt(i64),

    #[token = "pop"]
    Pop,
    #[token = "dup"]
    Dup,
    #[token = "swap"]
    Swap,
    #[token = "return"]
    Return,
    #[token = "halt"]
    Halt,

    // ---- Lane constants (hex operands) ----
    #[token = "const_loc"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstLoc(#[parse_with = "crate::vihaco_isa::parse_helpers::hex_u64"] u64),

    #[token = "const_lane"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstLane(#[parse_with = "crate::vihaco_isa::parse_helpers::hex_u64"] u64),

    #[token = "const_zone"]
    #[delimiters(open = "", close = "", separator = "")]
    ConstZone(#[parse_with = "crate::vihaco_isa::parse_helpers::hex_u32"] u32),

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
}

#[cfg(test)]
#[allow(clippy::approx_constant)] // sample floats are illustrative, not math constants
mod tests {
    use super::*;
    use chumsky::Parser as _;
    use vihaco::instruction::{FromBytes, OpCode, WriteBytes};
    use vihaco_parser_core::Parse;

    fn parse(input: &str) -> Instruction {
        Instruction::parser()
            .parse(input)
            .into_result()
            .unwrap_or_else(|e| panic!("parse({input:?}) failed: {e:?}"))
    }

    /// A representative instruction of every shape (unit, scalar, hex, multi-field).
    fn sample_program() -> Vec<Instruction> {
        vec![
            Instruction::ConstFloat(3.14159),
            Instruction::ConstInt(-42),
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
            Instruction::Dup,
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
            parse("const_float 3.14159"),
            Instruction::ConstFloat(3.14159)
        );
        assert_eq!(parse("const_int -42"), Instruction::ConstInt(-42));
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
            ("dup", Instruction::Dup),
            ("swap", Instruction::Swap),
            ("return", Instruction::Return),
            ("halt", Instruction::Halt),
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
}
