//! Hex operand types for the lanes device's surface syntax.
//!
//! vihaco's built-in integer parsers are decimal only, and 0.4 removed the
//! field-level `#[parse_with]` escape hatch — the documented replacement is to
//! give domain-specific field syntax its own type with its own [`Parse`].
//!
//! These are parse-only. [`LanesInstruction`](super::device::LanesInstruction)
//! carries plain `u64`/`u32`, so nothing downstream — validation, the machine,
//! the PyO3 bindings — ever sees them.

use chumsky::prelude::*;
use vihaco_parser::Parse;

type E<'src> = chumsky::extra::Err<chumsky::error::Simple<'src, char>>;

/// A `0x`-prefixed [`u64`] operand (`const_loc`, `const_lane`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HexU64(pub u64);

/// A `0x`-prefixed [`u32`] operand (`const_zone`).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_operands_require_the_prefix_and_reject_overflow() {
        assert_eq!(
            HexU64::parser().parse("0xdeadbeef").into_result().unwrap(),
            HexU64(0xdead_beef)
        );
        assert_eq!(
            HexU32::parser().parse("0x00000007").into_result().unwrap(),
            HexU32(7)
        );
        // Decimal is not hex, and 9 hex digits overflow a u32.
        assert!(HexU64::parser().parse("16777216").into_result().is_err());
        assert!(HexU32::parser().parse("0x1ffffffff").into_result().is_err());
    }
}
