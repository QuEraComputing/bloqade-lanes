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
