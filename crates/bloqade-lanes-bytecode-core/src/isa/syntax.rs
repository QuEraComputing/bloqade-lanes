//! Text (`.sst`) surface syntax for [`Instruction`].
//!
//! vihaco 0.4's pattern parser derives a [`Parse`] impl per *dialect*: one enum
//! per `#[syntax_class(instruction, head = "…")]`, whose variants spell their
//! own concrete syntax with `#[pattern]`. Our instruction set spans two
//! dialects ([`LANES_HEAD`] and [`CPU_HEAD`]) but is a single flat
//! [`Instruction`] enum, so this module holds two *mirror* enums — one per
//! dialect — plus the [`Parse`] impl that tries both and folds the result back
//! into [`Instruction`].
//!
//! The mirrors exist only to carry patterns. They are private, they never
//! escape this module, and [`Instruction`] keeps plain scalar operands (`u64`,
//! not `HexU64`), so nothing downstream — validation, the PyO3 bindings, the
//! Python decoder — sees the surface types.
//!
//! `mirrors_cover_every_instruction` pins the two lists together: every
//! [`Instruction`] variant must round-trip through `Display` → `parse`, so a
//! variant added to the ISA without a pattern here fails the test rather than
//! silently losing its text syntax.

use chumsky::prelude::*;
use vihaco_parser::Parse;

use super::Instruction;

/// Dialect head for the lanes-native device instructions (`lanes.move 2`).
pub const LANES_HEAD: &str = "lanes";

/// Dialect head for the stack ops (`cpu.halt`). These are declared natively in
/// [`Instruction`] — vihaco-cpu 0.4 has no binary codec to nest — but keep the
/// `cpu.` namespace to signal that their semantics mirror vihaco-cpu's.
pub const CPU_HEAD: &str = "cpu";

type E<'src> = chumsky::extra::Err<chumsky::error::Simple<'src, char>>;

// ── Hex operands ──────────────────────────────────────────────────────────────

/// A `0x`-prefixed [`u64`] operand (`const_loc`, `const_lane`).
///
/// vihaco's built-in integer parsers are decimal only, and 0.4 removed the
/// field-level `#[parse_with]` escape hatch — the documented replacement is to
/// give domain-specific field syntax its own type with its own [`Parse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexU64(pub u64);

/// A `0x`-prefixed [`u32`] operand (`const_zone`). See [`HexU64`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexU32(pub u32);

impl<'src> Parse<'src> for HexU64 {
    fn parser() -> impl Parser<'src, &'src str, Self, E<'src>> {
        super::parse_helpers::hex_u64().map(HexU64)
    }
}

impl<'src> Parse<'src> for HexU32 {
    fn parser() -> impl Parser<'src, &'src str, Self, E<'src>> {
        super::parse_helpers::hex_u32().map(HexU32)
    }
}

// ── Dialect mirrors ───────────────────────────────────────────────────────────

/// Mirror of the `lanes.*` half of [`Instruction`], carrying its text patterns.
///
/// **Variant order matters**: the derive builds an alternation in declaration
/// order, so a token that is a prefix of another must come *after* the longer
/// one (hence `local_rz` precedes `local_r`).
#[derive(Debug, Clone, PartialEq, vihaco::Parse)]
#[syntax_class(instruction, head = "lanes")]
enum LanesSyntax {
    #[pattern = "'const_loc $0"]
    ConstLoc(HexU64),
    #[pattern = "'const_lane $0"]
    ConstLane(HexU64),
    #[pattern = "'const_zone $0"]
    ConstZone(HexU32),

    #[pattern = "'initial_fill $0"]
    InitialFill(u32),
    #[pattern = "'fill $0"]
    Fill(u32),
    #[pattern = "'move $0"]
    Move(u32),

    #[pattern = "'local_rz $0"]
    LocalRz(u32),
    #[pattern = "'local_r $0"]
    LocalR(u32),
    #[pattern = "'global_rz"]
    GlobalRz,
    #[pattern = "'global_r"]
    GlobalR,
    #[pattern = "'cz"]
    Cz,

    #[pattern = "'measure $0"]
    Measure(u32),
    #[pattern = "'await_measure"]
    AwaitMeasure,

    #[pattern = "'new_array $0 $1 $2"]
    NewArray(u32, u32, u32),
    #[pattern = "'get_item $0"]
    GetItem(u32),

    #[pattern = "'set_detector"]
    SetDetector,
    #[pattern = "'set_observable"]
    SetObservable,
}

/// Mirror of the `cpu.*` stack ops. See [`LanesSyntax`] for the ordering rule.
#[derive(Debug, Clone, PartialEq, vihaco::Parse)]
#[syntax_class(instruction, head = "cpu")]
enum CpuSyntax {
    #[pattern = "'const_float $0"]
    ConstFloat(f64),
    #[pattern = "'const_int $0"]
    ConstInt(i64),
    #[pattern = "'pop"]
    Pop,
    #[pattern = "'swap"]
    Swap,
    #[pattern = "'return"]
    Return,
    #[pattern = "'dup"]
    Dup,
    #[pattern = "'halt"]
    Halt,
}

impl From<LanesSyntax> for Instruction {
    fn from(s: LanesSyntax) -> Self {
        match s {
            LanesSyntax::ConstLoc(HexU64(v)) => Instruction::ConstLoc(v),
            LanesSyntax::ConstLane(HexU64(v)) => Instruction::ConstLane(v),
            LanesSyntax::ConstZone(HexU32(v)) => Instruction::ConstZone(v),
            LanesSyntax::InitialFill(a) => Instruction::InitialFill(a),
            LanesSyntax::Fill(a) => Instruction::Fill(a),
            LanesSyntax::Move(a) => Instruction::Move(a),
            LanesSyntax::LocalRz(a) => Instruction::LocalRz(a),
            LanesSyntax::LocalR(a) => Instruction::LocalR(a),
            LanesSyntax::GlobalRz => Instruction::GlobalRz,
            LanesSyntax::GlobalR => Instruction::GlobalR,
            LanesSyntax::Cz => Instruction::Cz,
            LanesSyntax::Measure(a) => Instruction::Measure(a),
            LanesSyntax::AwaitMeasure => Instruction::AwaitMeasure,
            LanesSyntax::NewArray(t, d0, d1) => Instruction::NewArray(t, d0, d1),
            LanesSyntax::GetItem(n) => Instruction::GetItem(n),
            LanesSyntax::SetDetector => Instruction::SetDetector,
            LanesSyntax::SetObservable => Instruction::SetObservable,
        }
    }
}

impl From<CpuSyntax> for Instruction {
    fn from(s: CpuSyntax) -> Self {
        match s {
            CpuSyntax::ConstFloat(v) => Instruction::ConstFloat(v),
            CpuSyntax::ConstInt(v) => Instruction::ConstInt(v),
            CpuSyntax::Pop => Instruction::Pop,
            CpuSyntax::Swap => Instruction::Swap,
            CpuSyntax::Return => Instruction::Return,
            CpuSyntax::Dup => Instruction::Dup,
            CpuSyntax::Halt => Instruction::Halt,
        }
    }
}

// ── The composed instruction parser ───────────────────────────────────────────

impl<'src> Parse<'src> for Instruction {
    fn parser() -> impl Parser<'src, &'src str, Self, E<'src>> {
        // The two dialects have disjoint heads, so the alternation is
        // unambiguous and order between them is irrelevant.
        choice((
            LanesSyntax::parser().map(Instruction::from),
            CpuSyntax::parser().map(Instruction::from),
        ))
    }
}

/// Marker: [`Instruction`] is the surface instruction type of our `.sst`
/// grammar, so `ParsedFunction<Instruction, _>` can parse a function body.
impl vihaco::SurfaceInstruction for Instruction {}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(input: &str) -> Instruction {
        Instruction::parser()
            .parse(input)
            .into_result()
            .unwrap_or_else(|e| panic!("parse({input:?}) failed: {e:?}"))
    }

    #[test]
    fn heads_match_the_dialect_constants() {
        // `#[syntax_class(head = …)]` needs a literal, so the constants and the
        // attributes are written twice; this keeps them honest.
        assert_eq!(parse("lanes.cz").dialect(), LANES_HEAD);
        assert_eq!(parse("cpu.halt").dialect(), CPU_HEAD);
    }

    #[test]
    fn mirrors_cover_every_instruction() {
        // Exhaustive by construction: the match below has no catch-all, so a
        // new `Instruction` variant fails to compile until it is listed, and
        // the round-trip then proves it has a working pattern.
        let all = [
            Instruction::Pop,
            Instruction::Swap,
            Instruction::Return,
            Instruction::Dup,
            Instruction::Halt,
            Instruction::ConstFloat(1.5),
            Instruction::ConstInt(-42),
            Instruction::ConstLoc(0x0100_0000),
            Instruction::ConstLane(1),
            Instruction::ConstZone(7),
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
        ];
        for inst in &all {
            // Compile-time exhaustiveness check over the ISA.
            match inst {
                Instruction::Pop
                | Instruction::Swap
                | Instruction::Return
                | Instruction::Dup
                | Instruction::Halt
                | Instruction::ConstFloat(_)
                | Instruction::ConstInt(_)
                | Instruction::ConstLoc(_)
                | Instruction::ConstLane(_)
                | Instruction::ConstZone(_)
                | Instruction::InitialFill(_)
                | Instruction::Fill(_)
                | Instruction::Move(_)
                | Instruction::LocalRz(_)
                | Instruction::LocalR(_)
                | Instruction::GlobalRz
                | Instruction::GlobalR
                | Instruction::Cz
                | Instruction::Measure(_)
                | Instruction::AwaitMeasure
                | Instruction::NewArray(..)
                | Instruction::GetItem(_)
                | Instruction::SetDetector
                | Instruction::SetObservable => {}
            }
            assert_eq!(parse(&inst.to_string()), *inst, "round-trip {inst:?}");
        }
    }

    #[test]
    fn hex_operands_require_the_prefix() {
        // Decimal is not accepted where the format specifies hex.
        assert!(
            Instruction::parser()
                .parse("lanes.const_loc 16777216")
                .into_result()
                .is_err()
        );
    }

    #[test]
    fn wrong_dialect_head_is_rejected() {
        for bad in ["cpu.move 2", "lanes.halt", "move 2", "halt"] {
            assert!(
                Instruction::parser().parse(bad).into_result().is_err(),
                "{bad:?} should not parse"
            );
        }
    }
}
