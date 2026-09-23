//! Binary codec for [`MachineInstruction`].
//!
//! ## Why a mirror ISA
//!
//! Neither half of the composite can be encoded directly. vihaco-cpu's
//! instruction enums implement `Parse` and carry runtime values but no
//! `WriteBytes`/`FromBytes`/`OpCode`, and the enum `#[composite]` generates
//! derives only `Debug`/`Clone`. So the encodable form is declared here as a
//! parallel enum that *does* derive [`vihaco::Instruction`], with explicit
//! conversions either way.
//!
//! This is the same shape PPVM uses (`BytecodeCpu` / `BytecodeCircuit` /
//! `BytecodeInstruction`), and for the same reason. It is duplication, but it
//! is duplication vihaco currently forces on every consumer that wants both a
//! CPU component and a binary format.
//!
//! The round-trip test is what keeps the mirror honest: every variant of
//! [`MachineInstruction`] must survive `encode` → bytes → `decode`.
//!
//! ## Layout
//!
//! vihaco assigns opcodes by declaration position, so [`BytecodeInstruction`]'s
//! two variants tag the device (`0x00` cpu, `0x01` lanes) and the nested enum's
//! opcode follows. Width is derived as the widest variant.

use vihaco::instruction::OpCode;
use vihaco::{Type, Value};
use vihaco_cpu::RuntimeInstruction as CpuInstruction;

use super::device::LanesInstruction;
use super::machine::MachineInstruction;

/// Width, in bytes, of every encoded instruction word.
///
/// Derived, not chosen: vihaco computes it as `1 + Σ field widths` of the
/// widest variant. Today that is 14 — a device byte, an opcode byte, and
/// three `u32`s — reached by both `Cpu(Span(u32, u32, u32))` and
/// `Lanes(NewArray(u32, u32, u32))`. (`Const(BytecodeType, BytecodeValue)` is
/// only 12.) Every narrower instruction is zero-padded to match, so a program
/// is N concatenated words.
pub fn instruction_width() -> u32 {
    BytecodeInstruction::width()
}

// ── Mirrors of vihaco's Type and Value ────────────────────────────────────────

/// Encodable mirror of [`vihaco::Type`].
#[derive(Debug, Clone, PartialEq, vihaco::Instruction)]
pub enum BytecodeType {
    Undefined,
    String,
    Bool,
    I64,
    U32,
    U64,
    F64,
    FunctionRef,
    HeapRef,
}

/// Encodable mirror of [`vihaco::Value`].
#[derive(Debug, Clone, PartialEq, vihaco::Instruction)]
pub enum BytecodeValue {
    Undefined,
    String(u32),
    Bool(bool),
    I64(i64),
    U32(u32),
    U64(u64),
    F64(f64),
    FunctionRef(u32),
    HeapRef(u32),
}

/// Generate the two-way mapping for a mirror whose variants match one-to-one.
macro_rules! mirror {
    ($source:ident, $encoded:ident, $encode:ident, $decode:ident;
     $($variant:ident $(($binding:ident))?),+ $(,)?) => {
        fn $encode(value: $source) -> $encoded {
            match value {
                $( $source::$variant $(($binding))? => $encoded::$variant $(($binding))?, )+
            }
        }

        fn $decode(value: $encoded) -> $source {
            match value {
                $( $encoded::$variant $(($binding))? => $source::$variant $(($binding))?, )+
            }
        }
    };
}

mirror! {
    Type, BytecodeType, encode_type, decode_type;
    Undefined, String, Bool, I64, U32, U64, F64, FunctionRef, HeapRef,
}

mirror! {
    Value, BytecodeValue, encode_value, decode_value;
    Undefined, String(v), Bool(v), I64(v), U32(v), U64(v), F64(v), FunctionRef(v), HeapRef(v),
}

// ── Mirrors of the two device instruction sets ────────────────────────────────

/// Encodable mirror of [`vihaco_cpu::RuntimeInstruction`].
///
/// `Label` is absent on purpose: it carries an interned `Ident`, which has no
/// meaning outside the parse that produced it. Encoding one is an error.
#[derive(Debug, Clone, PartialEq, vihaco::Instruction)]
pub enum BytecodeCpu {
    Span(u32, u32, u32),
    FunctionStart,
    FunctionEnd,
    Breakpoint,
    Branch(u32),
    ConditionalBranch(u32, u32),
    Return(u32),
    IndirectCall,
    Call(u32, u32),
    Halt,
    Print,
    Load(BytecodeType, u32),
    Store(BytecodeType, u32),
    Dup,
    HeapAlloc(u32),
    GetItem,
    HeapDealloc,
    Const(BytecodeType, BytecodeValue),
    Add(BytecodeType),
    Sub(BytecodeType),
    Mul(BytecodeType),
    Div(BytecodeType),
    Rem(BytecodeType),
    Neg(BytecodeType),
    Shl(BytecodeType),
    Shr(BytecodeType),
    Rol(BytecodeType),
    Ror(BytecodeType),
    BitAnd(BytecodeType),
    BitOr(BytecodeType),
    BitXor(BytecodeType),
    Not,
    And,
    Or,
    Xor,
    Eq(BytecodeType),
    Ne(BytecodeType),
    Lt(BytecodeType),
    Gt(BytecodeType),
    Le(BytecodeType),
    Ge(BytecodeType),
}

/// Encodable mirror of [`LanesInstruction`].
#[derive(Debug, Clone, PartialEq, vihaco::Instruction)]
pub enum BytecodeLanes {
    ConstLoc(u64),
    ConstLane(u64),
    ConstZone(u32),
    InitialFill(u32),
    Fill(u32),
    Move(u32),
    LocalRz(u32),
    LocalR(u32),
    GlobalRz,
    GlobalR,
    Cz,
    Measure(u32),
    AwaitMeasure,
    NewArray(u32, u32, u32),
    GetItem(u32),
    SetDetector,
    SetObservable,
}

/// The encodable form of a whole program's instruction: device tag + payload.
#[derive(Debug, Clone, PartialEq, vihaco::Instruction)]
pub enum BytecodeInstruction {
    Cpu(BytecodeCpu),
    Lanes(BytecodeLanes),
}

// ── Conversions ───────────────────────────────────────────────────────────────

/// A 16-bit opcode that identifies an instruction across both devices:
/// `(device_code << 8) | instruction_code`.
///
/// [`BytecodeInstruction`]'s own opcode is just the device tag — `0x00` cpu,
/// `0x01` lanes — because the instruction's own code sits in the nested enum.
/// On its own that does not distinguish `move` from `cz`, so the two are packed
/// together here. This is what the Python `Instruction.opcode` reports.
///
/// Both halves shift when either instruction set gains a variant, so compare
/// identity with `op_name()` rather than a literal.
///
/// Neither half is written down here. The device tag is read back off the
/// constructed [`BytecodeInstruction`], so it is by construction the byte that
/// goes on disk — literals would have drifted silently the first time either
/// the variant order or a `#[device(..)]` attribute changed, and every test
/// compares via `op_name`, so nothing would have caught it.
///
/// An instruction with no encodable form — today just a runtime label, whose
/// identifier is meaningless outside its parse — reports
/// [`NO_ENCODABLE_FORM`] rather than colliding with a real opcode.
pub fn packed_opcode(inst: &MachineInstruction) -> u16 {
    let Ok(encoded) = encode(inst) else {
        return NO_ENCODABLE_FORM;
    };
    let device = OpCode::opcode(&encoded) as u16;
    let code = match &encoded {
        BytecodeInstruction::Cpu(i) => OpCode::opcode(i),
        BytecodeInstruction::Lanes(i) => OpCode::opcode(i),
    } as u16;
    (device << 8) | code
}

/// Reported by [`packed_opcode`] for an instruction that cannot be encoded.
///
/// `0xFFFF` is outside the packed space — the device tag is a
/// [`BytecodeInstruction`] variant index and the code an inner variant index,
/// so neither half can reach `0xFF`. A plain `0` would have collided with the
/// first CPU instruction's real opcode.
pub const NO_ENCODABLE_FORM: u16 = 0xFFFF;

/// Map a runtime instruction to its encodable mirror.
pub fn encode(inst: &MachineInstruction) -> eyre::Result<BytecodeInstruction> {
    Ok(match inst {
        MachineInstruction::Cpu(inst) => BytecodeInstruction::Cpu(encode_cpu(inst)?),
        MachineInstruction::Lanes(inst) => BytecodeInstruction::Lanes(encode_lanes(inst)),
    })
}

/// Map an encodable mirror back to its runtime instruction.
pub fn decode(inst: BytecodeInstruction) -> MachineInstruction {
    match inst {
        BytecodeInstruction::Cpu(inst) => MachineInstruction::Cpu(decode_cpu(inst)),
        BytecodeInstruction::Lanes(inst) => MachineInstruction::Lanes(decode_lanes(inst)),
    }
}

fn encode_cpu(inst: &CpuInstruction) -> eyre::Result<BytecodeCpu> {
    use CpuInstruction as C;
    Ok(match inst {
        C::Span(a, b, c) => BytecodeCpu::Span(*a, *b, *c),
        C::FunctionStart => BytecodeCpu::FunctionStart,
        C::FunctionEnd => BytecodeCpu::FunctionEnd,
        C::Breakpoint => BytecodeCpu::Breakpoint,
        C::Branch(v) => BytecodeCpu::Branch(*v),
        C::ConditionalBranch(a, b) => BytecodeCpu::ConditionalBranch(*a, *b),
        C::Return(v) => BytecodeCpu::Return(*v),
        C::IndirectCall => BytecodeCpu::IndirectCall,
        C::Call(a, b) => BytecodeCpu::Call(*a, *b),
        C::Halt => BytecodeCpu::Halt,
        C::Print => BytecodeCpu::Print,
        C::Load(t, v) => BytecodeCpu::Load(encode_type(*t), *v),
        C::Store(t, v) => BytecodeCpu::Store(encode_type(*t), *v),
        C::Dup => BytecodeCpu::Dup,
        C::HeapAlloc(v) => BytecodeCpu::HeapAlloc(*v),
        C::GetItem => BytecodeCpu::GetItem,
        C::HeapDealloc => BytecodeCpu::HeapDealloc,
        C::Const(t, v) => BytecodeCpu::Const(encode_type(*t), encode_value(*v)),
        C::Add(t) => BytecodeCpu::Add(encode_type(*t)),
        C::Sub(t) => BytecodeCpu::Sub(encode_type(*t)),
        C::Mul(t) => BytecodeCpu::Mul(encode_type(*t)),
        C::Div(t) => BytecodeCpu::Div(encode_type(*t)),
        C::Rem(t) => BytecodeCpu::Rem(encode_type(*t)),
        C::Neg(t) => BytecodeCpu::Neg(encode_type(*t)),
        C::Shl(t) => BytecodeCpu::Shl(encode_type(*t)),
        C::Shr(t) => BytecodeCpu::Shr(encode_type(*t)),
        C::Rol(t) => BytecodeCpu::Rol(encode_type(*t)),
        C::Ror(t) => BytecodeCpu::Ror(encode_type(*t)),
        C::BitAnd(t) => BytecodeCpu::BitAnd(encode_type(*t)),
        C::BitOr(t) => BytecodeCpu::BitOr(encode_type(*t)),
        C::BitXor(t) => BytecodeCpu::BitXor(encode_type(*t)),
        C::Not => BytecodeCpu::Not,
        C::And => BytecodeCpu::And,
        C::Or => BytecodeCpu::Or,
        C::Xor => BytecodeCpu::Xor,
        C::Eq(t) => BytecodeCpu::Eq(encode_type(*t)),
        C::Ne(t) => BytecodeCpu::Ne(encode_type(*t)),
        C::Lt(t) => BytecodeCpu::Lt(encode_type(*t)),
        C::Gt(t) => BytecodeCpu::Gt(encode_type(*t)),
        C::Le(t) => BytecodeCpu::Le(encode_type(*t)),
        C::Ge(t) => BytecodeCpu::Ge(encode_type(*t)),
        C::Label(_) => {
            return Err(eyre::eyre!(
                "a runtime label carries a parse-local identifier and cannot be encoded"
            ));
        }
    })
}

fn decode_cpu(inst: BytecodeCpu) -> CpuInstruction {
    use BytecodeCpu as B;
    match inst {
        B::Span(a, b, c) => CpuInstruction::Span(a, b, c),
        B::FunctionStart => CpuInstruction::FunctionStart,
        B::FunctionEnd => CpuInstruction::FunctionEnd,
        B::Breakpoint => CpuInstruction::Breakpoint,
        B::Branch(v) => CpuInstruction::Branch(v),
        B::ConditionalBranch(a, b) => CpuInstruction::ConditionalBranch(a, b),
        B::Return(v) => CpuInstruction::Return(v),
        B::IndirectCall => CpuInstruction::IndirectCall,
        B::Call(a, b) => CpuInstruction::Call(a, b),
        B::Halt => CpuInstruction::Halt,
        B::Print => CpuInstruction::Print,
        B::Load(t, v) => CpuInstruction::Load(decode_type(t), v),
        B::Store(t, v) => CpuInstruction::Store(decode_type(t), v),
        B::Dup => CpuInstruction::Dup,
        B::HeapAlloc(v) => CpuInstruction::HeapAlloc(v),
        B::GetItem => CpuInstruction::GetItem,
        B::HeapDealloc => CpuInstruction::HeapDealloc,
        B::Const(t, v) => CpuInstruction::Const(decode_type(t), decode_value(v)),
        B::Add(t) => CpuInstruction::Add(decode_type(t)),
        B::Sub(t) => CpuInstruction::Sub(decode_type(t)),
        B::Mul(t) => CpuInstruction::Mul(decode_type(t)),
        B::Div(t) => CpuInstruction::Div(decode_type(t)),
        B::Rem(t) => CpuInstruction::Rem(decode_type(t)),
        B::Neg(t) => CpuInstruction::Neg(decode_type(t)),
        B::Shl(t) => CpuInstruction::Shl(decode_type(t)),
        B::Shr(t) => CpuInstruction::Shr(decode_type(t)),
        B::Rol(t) => CpuInstruction::Rol(decode_type(t)),
        B::Ror(t) => CpuInstruction::Ror(decode_type(t)),
        B::BitAnd(t) => CpuInstruction::BitAnd(decode_type(t)),
        B::BitOr(t) => CpuInstruction::BitOr(decode_type(t)),
        B::BitXor(t) => CpuInstruction::BitXor(decode_type(t)),
        B::Not => CpuInstruction::Not,
        B::And => CpuInstruction::And,
        B::Or => CpuInstruction::Or,
        B::Xor => CpuInstruction::Xor,
        B::Eq(t) => CpuInstruction::Eq(decode_type(t)),
        B::Ne(t) => CpuInstruction::Ne(decode_type(t)),
        B::Lt(t) => CpuInstruction::Lt(decode_type(t)),
        B::Gt(t) => CpuInstruction::Gt(decode_type(t)),
        B::Le(t) => CpuInstruction::Le(decode_type(t)),
        B::Ge(t) => CpuInstruction::Ge(decode_type(t)),
    }
}

fn encode_lanes(inst: &LanesInstruction) -> BytecodeLanes {
    use LanesInstruction as L;
    match inst {
        L::ConstLoc(v) => BytecodeLanes::ConstLoc(*v),
        L::ConstLane(v) => BytecodeLanes::ConstLane(*v),
        L::ConstZone(v) => BytecodeLanes::ConstZone(*v),
        L::InitialFill(a) => BytecodeLanes::InitialFill(*a),
        L::Fill(a) => BytecodeLanes::Fill(*a),
        L::Move(a) => BytecodeLanes::Move(*a),
        L::LocalRz(a) => BytecodeLanes::LocalRz(*a),
        L::LocalR(a) => BytecodeLanes::LocalR(*a),
        L::GlobalRz => BytecodeLanes::GlobalRz,
        L::GlobalR => BytecodeLanes::GlobalR,
        L::Cz => BytecodeLanes::Cz,
        L::Measure(a) => BytecodeLanes::Measure(*a),
        L::AwaitMeasure => BytecodeLanes::AwaitMeasure,
        L::NewArray(t, d0, d1) => BytecodeLanes::NewArray(*t, *d0, *d1),
        L::GetItem(n) => BytecodeLanes::GetItem(*n),
        L::SetDetector => BytecodeLanes::SetDetector,
        L::SetObservable => BytecodeLanes::SetObservable,
    }
}

fn decode_lanes(inst: BytecodeLanes) -> LanesInstruction {
    use BytecodeLanes as B;
    match inst {
        B::ConstLoc(v) => LanesInstruction::ConstLoc(v),
        B::ConstLane(v) => LanesInstruction::ConstLane(v),
        B::ConstZone(v) => LanesInstruction::ConstZone(v),
        B::InitialFill(a) => LanesInstruction::InitialFill(a),
        B::Fill(a) => LanesInstruction::Fill(a),
        B::Move(a) => LanesInstruction::Move(a),
        B::LocalRz(a) => LanesInstruction::LocalRz(a),
        B::LocalR(a) => LanesInstruction::LocalR(a),
        B::GlobalRz => LanesInstruction::GlobalRz,
        B::GlobalR => LanesInstruction::GlobalR,
        B::Cz => LanesInstruction::Cz,
        B::Measure(a) => LanesInstruction::Measure(a),
        B::AwaitMeasure => LanesInstruction::AwaitMeasure,
        B::NewArray(t, d0, d1) => LanesInstruction::NewArray(t, d0, d1),
        B::GetItem(n) => LanesInstruction::GetItem(n),
        B::SetDetector => LanesInstruction::SetDetector,
        B::SetObservable => LanesInstruction::SetObservable,
    }
}

/// The exhaustive instruction list the tests walk.
///
/// It lives outside `mod tests` because `machine`'s renderer tests walk the
/// same list: there is one place to keep exhaustive, not two that can drift
/// apart while both look thorough.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    /// **Every** variant of both instruction sets.
    ///
    /// The mirror exists to cover the two device enums exactly; anything missing
    /// here is a variant whose encoding nothing checks. The only omission is
    /// `Label`, which carries a parse-local identifier and deliberately has no
    /// encodable form — `a_runtime_label_cannot_be_encoded` covers that.
    pub(crate) fn every_instruction() -> Vec<MachineInstruction> {
        // One instance of each typed op per type, so the Type mirror is covered
        // in both directions too.
        let tys = [
            Type::Undefined,
            Type::String,
            Type::Bool,
            Type::I64,
            Type::U32,
            Type::U64,
            Type::F64,
            Type::FunctionRef,
            Type::HeapRef,
        ];
        let mut cpu = vec![
            CpuInstruction::Span(1, 2, 3),
            CpuInstruction::FunctionStart,
            CpuInstruction::FunctionEnd,
            CpuInstruction::Breakpoint,
            CpuInstruction::Branch(4),
            CpuInstruction::ConditionalBranch(5, 6),
            CpuInstruction::Return(0),
            CpuInstruction::IndirectCall,
            CpuInstruction::Call(1, 2),
            CpuInstruction::Halt,
            CpuInstruction::Print,
            CpuInstruction::Dup,
            CpuInstruction::HeapAlloc(3),
            CpuInstruction::GetItem,
            CpuInstruction::HeapDealloc,
            CpuInstruction::Not,
            CpuInstruction::And,
            CpuInstruction::Or,
            CpuInstruction::Xor,
        ];
        // Every Value variant, so the Value mirror is covered too.
        cpu.extend([
            CpuInstruction::Const(Type::Undefined, Value::Undefined),
            CpuInstruction::Const(Type::String, Value::String(2)),
            CpuInstruction::Const(Type::Bool, Value::Bool(true)),
            CpuInstruction::Const(Type::I64, Value::I64(-42)),
            CpuInstruction::Const(Type::U32, Value::U32(3)),
            CpuInstruction::Const(Type::U64, Value::U64(7)),
            CpuInstruction::Const(Type::F64, Value::F64(1.5)),
            CpuInstruction::Const(Type::FunctionRef, Value::FunctionRef(4)),
            CpuInstruction::Const(Type::HeapRef, Value::HeapRef(5)),
        ]);
        for ty in tys {
            cpu.extend([
                CpuInstruction::Load(ty, 7),
                CpuInstruction::Store(ty, 9),
                CpuInstruction::Add(ty),
                CpuInstruction::Sub(ty),
                CpuInstruction::Mul(ty),
                CpuInstruction::Div(ty),
                CpuInstruction::Rem(ty),
                CpuInstruction::Neg(ty),
                CpuInstruction::Shl(ty),
                CpuInstruction::Shr(ty),
                CpuInstruction::Rol(ty),
                CpuInstruction::Ror(ty),
                CpuInstruction::BitAnd(ty),
                CpuInstruction::BitOr(ty),
                CpuInstruction::BitXor(ty),
                CpuInstruction::Eq(ty),
                CpuInstruction::Ne(ty),
                CpuInstruction::Lt(ty),
                CpuInstruction::Gt(ty),
                CpuInstruction::Le(ty),
                CpuInstruction::Ge(ty),
            ]);
        }

        let lanes = [
            LanesInstruction::ConstLoc(0x0100_0000),
            LanesInstruction::ConstLane(1),
            LanesInstruction::ConstZone(7),
            LanesInstruction::InitialFill(2),
            LanesInstruction::Fill(1),
            LanesInstruction::Move(2),
            LanesInstruction::LocalRz(1),
            LanesInstruction::LocalR(3),
            LanesInstruction::GlobalRz,
            LanesInstruction::GlobalR,
            LanesInstruction::Cz,
            LanesInstruction::Measure(1),
            LanesInstruction::AwaitMeasure,
            LanesInstruction::NewArray(1, 3, 0),
            LanesInstruction::GetItem(1),
            LanesInstruction::SetDetector,
            LanesInstruction::SetObservable,
        ];
        cpu.into_iter()
            .map(MachineInstruction::Cpu)
            .chain(lanes.into_iter().map(MachineInstruction::Lanes))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::every_instruction as samples;
    use super::*;
    use vihaco::instruction::{FromBytes, WriteBytes};
    use vihaco_parser::Ident;

    #[test]
    fn every_instruction_round_trips_through_bytes() {
        for inst in samples() {
            let encoded = encode(&inst).expect("sample should encode");
            let mut buf = Vec::new();
            encoded.write_bytes(&mut buf).unwrap();
            assert_eq!(
                buf.len(),
                instruction_width() as usize,
                "{inst:?} did not fill a full word"
            );

            let read = BytecodeInstruction::from_bytes(&mut std::io::Cursor::new(buf)).unwrap();
            assert_eq!(read, encoded, "byte round-trip changed {inst:?}");
            // And the mirror maps back to the same runtime instruction.
            assert_eq!(format!("{:?}", decode(read)), format!("{inst:?}"));
        }
    }

    #[test]
    fn a_flat_stream_decodes_without_desync() {
        let program = samples();
        let mut bytes = Vec::new();
        for inst in &program {
            encode(inst).unwrap().write_bytes(&mut bytes).unwrap();
        }
        assert_eq!(bytes.len(), program.len() * instruction_width() as usize);

        let mut cursor = std::io::Cursor::new(bytes);
        for inst in &program {
            let read = BytecodeInstruction::from_bytes(&mut cursor).unwrap();
            assert_eq!(format!("{:?}", decode(read)), format!("{inst:?}"));
        }
    }

    #[test]
    fn a_runtime_label_cannot_be_encoded() {
        // It carries an identifier that only means anything to the parse that
        // produced it, so encoding is refused rather than silently lossy.
        let label = MachineInstruction::Cpu(CpuInstruction::Label(Ident("loop".into())));
        let err = encode(&label).unwrap_err().to_string();
        assert!(err.contains("cannot be encoded"), "got {err}");
        // And it reports a code no real instruction can occupy, rather than
        // colliding with the first CPU opcode.
        assert_eq!(packed_opcode(&label), NO_ENCODABLE_FORM);
    }

    /// The opcode Python reports must be the bytes that went to disk.
    ///
    /// Nothing else checks this: every other test compares instructions via
    /// `op_name`, so a `packed_opcode` that disagreed with the encoding would
    /// stay green. The first two bytes of a word are the device tag and the
    /// instruction's own code, which is exactly what the packing claims to be.
    #[test]
    fn the_packed_opcode_is_the_first_two_bytes_on_disk() {
        for inst in samples() {
            let mut buf = Vec::new();
            encode(&inst).unwrap().write_bytes(&mut buf).unwrap();
            assert_eq!(
                packed_opcode(&inst),
                ((buf[0] as u16) << 8) | buf[1] as u16,
                "{inst:?}: packed opcode disagrees with its encoding {:02x?}",
                &buf[..2]
            );
        }
    }

    /// The word width is what the doc on `instruction_width` claims it is.
    ///
    /// 14 = 1 device byte + 1 opcode byte + three `u32`s, reached by the
    /// widest variant on *either* device. It is derived, so it moves the
    /// moment either gains a wider operand — which is worth noticing, since
    /// it is the on-disk format.
    #[test]
    fn the_word_width_comes_from_the_widest_variant() {
        assert_eq!(instruction_width(), 14);
        for widest in [
            BytecodeInstruction::Cpu(BytecodeCpu::Span(1, 2, 3)),
            BytecodeInstruction::Lanes(BytecodeLanes::NewArray(1, 2, 3)),
        ] {
            let mut buf = Vec::new();
            widest.write_bytes(&mut buf).unwrap();
            assert_eq!(buf.len(), 14, "{widest:?}");
        }
        // The `const` word is narrower, and is padded out to match.
        let mut buf = Vec::new();
        BytecodeInstruction::Cpu(BytecodeCpu::Const(
            BytecodeType::F64,
            BytecodeValue::F64(1.5),
        ))
        .write_bytes(&mut buf)
        .unwrap();
        assert_eq!(buf.len(), 14);
        assert_eq!(&buf[12..], &[0, 0], "the last two bytes should be padding");
    }

    /// Distinct instructions must get distinct opcodes — the defect the
    /// packing exists to fix was every lanes op reporting the device tag.
    #[test]
    fn each_instruction_has_its_own_packed_opcode() {
        let mut seen: std::collections::HashMap<u16, MachineInstruction> = Default::default();
        for inst in samples() {
            let code = packed_opcode(&inst);
            // The type is an operand, not part of the opcode, so `add i64`
            // and `add f64` — and all nine `const` spellings — legitimately
            // share one. Key on the mnemonic, which is what an opcode names.
            let mnemonic = |i: &MachineInstruction| {
                super::super::machine::to_sst_text(i)
                    .split_whitespace()
                    .next()
                    .unwrap()
                    .to_owned()
            };
            if let Some(other) = seen.insert(code, inst.clone()) {
                assert_eq!(
                    mnemonic(&other),
                    mnemonic(&inst),
                    "{inst:?} and {other:?} share opcode {code:#06x}"
                );
            }
        }
        // Sanity: the two devices land in different high bytes.
        assert_ne!(
            packed_opcode(&MachineInstruction::Cpu(CpuInstruction::Halt)) >> 8,
            packed_opcode(&MachineInstruction::Lanes(LanesInstruction::Cz)) >> 8,
        );
    }
}
