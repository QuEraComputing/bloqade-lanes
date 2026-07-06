//! Field-level `#[parse_with]` helpers for [`super::Instruction`].
//!
//! The `.sst` text format writes location/lane/zone constants as `0x`-prefixed
//! hexadecimal; vihaco's built-in integer parsers are decimal only, so these
//! helpers parse the hex operands.

use chumsky::error::Simple;
use chumsky::extra;
use chumsky::prelude::*;

type E<'src> = extra::Err<Simple<'src, char>>;

/// `0x` followed by one or more ASCII hex digits, collected as a `String`.
fn hex_digits<'src>() -> impl Parser<'src, &'src str, String, E<'src>> {
    just("0x").ignore_then(
        any()
            .filter(|c: &char| c.is_ascii_hexdigit())
            .repeated()
            .at_least(1)
            .collect::<String>(),
    )
}

/// Parse a `0x`-prefixed hexadecimal [`u64`] (`const_loc`, `const_lane`).
pub fn hex_u64<'src>() -> impl Parser<'src, &'src str, u64, E<'src>> {
    hex_digits().try_map(|s, span| u64::from_str_radix(&s, 16).map_err(|_| Simple::new(None, span)))
}

/// Parse a `0x`-prefixed hexadecimal [`u32`] (`const_zone`).
pub fn hex_u32<'src>() -> impl Parser<'src, &'src str, u32, E<'src>> {
    hex_digits().try_map(|s, span| u32::from_str_radix(&s, 16).map_err(|_| Simple::new(None, span)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_u64_parses_full_range() {
        assert_eq!(hex_u64().parse("0x0").into_result().unwrap(), 0);
        assert_eq!(
            hex_u64().parse("0xffffffffffffffff").into_result().unwrap(),
            u64::MAX
        );
    }

    #[test]
    fn hex_u64_rejects_overflow() {
        // 17 hex digits exceed u64, so `from_str_radix` fails and the parser
        // surfaces an error rather than silently truncating.
        assert!(
            hex_u64()
                .parse("0x1ffffffffffffffff")
                .into_result()
                .is_err()
        );
    }

    #[test]
    fn hex_u32_parses_and_rejects_overflow() {
        assert_eq!(
            hex_u32().parse("0xdeadbeef").into_result().unwrap(),
            0xdead_beef
        );
        // 9 hex digits exceed u32.
        assert!(hex_u32().parse("0x1ffffffff").into_result().is_err());
    }

    #[test]
    fn hex_requires_prefix_and_digits() {
        // Missing `0x` prefix, and a lone prefix with no digits, both fail.
        assert!(hex_u64().parse("1234").into_result().is_err());
        assert!(hex_u64().parse("0x").into_result().is_err());
    }
}
