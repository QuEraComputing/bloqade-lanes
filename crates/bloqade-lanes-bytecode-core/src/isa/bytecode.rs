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
/// Derived, not chosen: vihaco computes it as the widest variant. Today that is
/// `Cpu(Const(BytecodeType, BytecodeValue))`.
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
    Pop,
    Swap,
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
pub fn packed_opcode(inst: &MachineInstruction) -> u16 {
    let (device, code) = match inst {
        MachineInstruction::Cpu(_) => (0u16, encode_cpu_opcode(inst)),
        MachineInstruction::Lanes(i) => (1u16, OpCode::opcode(&encode_lanes(i))),
    };
    (device << 8) | code as u16
}

/// The nested opcode of a CPU instruction, or `0` for one with no encodable
/// form (a runtime label).
fn encode_cpu_opcode(inst: &MachineInstruction) -> u8 {
    match inst {
        MachineInstruction::Cpu(i) => encode_cpu(i).map(|e| OpCode::opcode(&e)).unwrap_or(0),
        _ => 0,
    }
}

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
        L::Pop => BytecodeLanes::Pop,
        L::Swap => BytecodeLanes::Swap,
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
        B::Pop => LanesInstruction::Pop,
        B::Swap => LanesInstruction::Swap,
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

#[cfg(test)]
mod tests {
    use super::*;
    use vihaco::instruction::{FromBytes, WriteBytes};
    use vihaco_parser::Ident;

    /// Every shape in both devices.
    fn samples() -> Vec<MachineInstruction> {
        let cpu = [
            CpuInstruction::Halt,
            CpuInstruction::Dup,
            CpuInstruction::Const(Type::F64, Value::F64(1.5)),
            CpuInstruction::Const(Type::I64, Value::I64(-42)),
            CpuInstruction::Const(Type::U64, Value::U64(7)),
            CpuInstruction::Span(1, 2, 3),
            CpuInstruction::Branch(4),
            CpuInstruction::ConditionalBranch(5, 6),
            CpuInstruction::Call(1, 2),
            CpuInstruction::Return(0),
            CpuInstruction::HeapAlloc(3),
            CpuInstruction::GetItem,
            CpuInstruction::HeapDealloc,
            CpuInstruction::Add(Type::I64),
            CpuInstruction::Ge(Type::F64),
            CpuInstruction::Load(Type::U32, 9),
        ];
        let lanes = [
            LanesInstruction::Pop,
            LanesInstruction::Swap,
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
    }
}
