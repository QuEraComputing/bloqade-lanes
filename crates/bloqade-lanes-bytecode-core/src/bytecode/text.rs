use std::fmt;

use super::instruction::{
    ArrayInstruction, AtomArrangementInstruction, CpuInstruction, DetectorObservableInstruction,
    Instruction, LaneConstInstruction, MeasurementInstruction, QuantumGateInstruction,
};
use super::program::Program;
use crate::version::Version;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    MissingVersion,
    InvalidVersion(String),
    UnknownMnemonic { line: usize, mnemonic: String },
    MissingOperand { line: usize, mnemonic: String },
    InvalidOperand { line: usize, message: String },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::MissingVersion => write!(f, "missing .version directive"),
            ParseError::InvalidVersion(s) => write!(f, "invalid version: {}", s),
            ParseError::UnknownMnemonic { line, mnemonic } => {
                write!(f, "line {}: unknown mnemonic '{}'", line, mnemonic)
            }
            ParseError::MissingOperand { line, mnemonic } => {
                write!(f, "line {}: missing operand for '{}'", line, mnemonic)
            }
            ParseError::InvalidOperand { line, message } => {
                write!(f, "line {}: {}", line, message)
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse assembly text into a Program.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    let mut version: Option<Version> = None;
    let mut instructions = Vec::new();

    for (line_idx, raw_line) in source.lines().enumerate() {
        let line_num = line_idx + 1;

        // Strip comments
        let line = match raw_line.find(';') {
            Some(pos) => &raw_line[..pos],
            None => raw_line,
        };
        let line = line.trim();

        if line.is_empty() {
            continue;
        }

        // Directive
        if let Some(rest) = line.strip_prefix('.') {
            let mut parts = rest.split_whitespace();
            let directive = parts.next().unwrap_or("");
            if directive == "version" {
                let val = parts
                    .next()
                    .ok_or(ParseError::InvalidVersion("missing value".to_string()))?;
                version = Some(parse_version(val)?);
            }
            continue;
        }

        // Instruction
        let mut parts = line.split_whitespace();
        let mnemonic = parts.next().unwrap();
        let instr = parse_instruction(mnemonic, &mut parts, line_num)?;
        instructions.push(instr);
    }

    let version = version.ok_or(ParseError::MissingVersion)?;
    Ok(Program {
        version,
        instructions,
    })
}

fn parse_instruction<'a>(
    mnemonic: &str,
    operands: &mut impl Iterator<Item = &'a str>,
    line: usize,
) -> Result<Instruction, ParseError> {
    match mnemonic {
        "const_float" => {
            let val = require_operand(operands, line, mnemonic)?;
            let f: f64 =
                val.parse()
                    .map_err(|e: std::num::ParseFloatError| ParseError::InvalidOperand {
                        line,
                        message: e.to_string(),
                    })?;
            Ok(Instruction::Cpu(CpuInstruction::ConstFloat(f)))
        }
        "const_int" => {
            let val = require_operand(operands, line, mnemonic)?;
            let n = parse_i64(val, line)?;
            Ok(Instruction::Cpu(CpuInstruction::ConstInt(n)))
        }
        "const_loc" => {
            let val = require_operand(operands, line, mnemonic)?;
            let n = parse_hex_u32(val, line)?;
            Ok(Instruction::LaneConst(LaneConstInstruction::ConstLoc(n)))
        }
        "const_lane" => {
            let val = require_operand(operands, line, mnemonic)?;
            let n = parse_hex_u64(val, line)?;
            let data0 = n as u32;
            let data1 = (n >> 32) as u32;
            Ok(Instruction::LaneConst(LaneConstInstruction::ConstLane(
                data0, data1,
            )))
        }
        "const_zone" => {
            let val = require_operand(operands, line, mnemonic)?;
            let n = parse_hex_u32(val, line)?;
            Ok(Instruction::LaneConst(LaneConstInstruction::ConstZone(n)))
        }
        "pop" => Ok(Instruction::Cpu(CpuInstruction::Pop)),
        "dup" => Ok(Instruction::Cpu(CpuInstruction::Dup)),
        "swap" => Ok(Instruction::Cpu(CpuInstruction::Swap)),
        "initial_fill" => {
            let val = require_operand(operands, line, mnemonic)?;
            let arity = parse_u32(val, line)?;
            Ok(Instruction::AtomArrangement(
                AtomArrangementInstruction::InitialFill { arity },
            ))
        }
        "fill" => {
            let val = require_operand(operands, line, mnemonic)?;
            let arity = parse_u32(val, line)?;
            Ok(Instruction::AtomArrangement(
                AtomArrangementInstruction::Fill { arity },
            ))
        }
        "move" => {
            let val = require_operand(operands, line, mnemonic)?;
            let arity = parse_u32(val, line)?;
            Ok(Instruction::AtomArrangement(
                AtomArrangementInstruction::Move { arity },
            ))
        }
        "local_r" => {
            let val = require_operand(operands, line, mnemonic)?;
            let arity = parse_u32(val, line)?;
            Ok(Instruction::QuantumGate(QuantumGateInstruction::LocalR {
                arity,
            }))
        }
        "local_rz" => {
            let val = require_operand(operands, line, mnemonic)?;
            let arity = parse_u32(val, line)?;
            Ok(Instruction::QuantumGate(QuantumGateInstruction::LocalRz {
                arity,
            }))
        }
        "global_r" => Ok(Instruction::QuantumGate(QuantumGateInstruction::GlobalR)),
        "global_rz" => Ok(Instruction::QuantumGate(QuantumGateInstruction::GlobalRz)),
        "cz" => Ok(Instruction::QuantumGate(QuantumGateInstruction::CZ)),
        "measure" => {
            let val = require_operand(operands, line, mnemonic)?;
            let arity = parse_u32(val, line)?;
            Ok(Instruction::Measurement(MeasurementInstruction::Measure {
                arity,
            }))
        }
        "await_measure" => Ok(Instruction::Measurement(
            MeasurementInstruction::AwaitMeasure,
        )),
        "new_array" => {
            let type_str = require_operand(operands, line, mnemonic)?;
            let type_tag = parse_u8(type_str, line)?;
            let dim0_str = require_operand(operands, line, "new_array dim0")?;
            let dim0 = parse_u16(dim0_str, line)?;
            let dim1 = match operands.next() {
                Some(s) => parse_u16(s, line)?,
                None => 0,
            };
            Ok(Instruction::Array(ArrayInstruction::NewArray {
                type_tag,
                dim0,
                dim1,
            }))
        }
        "get_item" => {
            let val = require_operand(operands, line, mnemonic)?;
            let ndims = parse_u16(val, line)?;
            Ok(Instruction::Array(ArrayInstruction::GetItem { ndims }))
        }
        "set_detector" => Ok(Instruction::DetectorObservable(
            DetectorObservableInstruction::SetDetector,
        )),
        "set_observable" => Ok(Instruction::DetectorObservable(
            DetectorObservableInstruction::SetObservable,
        )),
        "return" => Ok(Instruction::Cpu(CpuInstruction::Return)),
        "halt" => Ok(Instruction::Cpu(CpuInstruction::Halt)),
        _ => Err(ParseError::UnknownMnemonic {
            line,
            mnemonic: mnemonic.to_string(),
        }),
    }
}

fn require_operand<'a>(
    operands: &mut impl Iterator<Item = &'a str>,
    line: usize,
    mnemonic: &str,
) -> Result<&'a str, ParseError> {
    operands.next().ok_or(ParseError::MissingOperand {
        line,
        mnemonic: mnemonic.to_string(),
    })
}

fn parse_hex_u32(s: &str, line: usize) -> Result<u32, ParseError> {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u32::from_str_radix(s, 16).map_err(|e| ParseError::InvalidOperand {
        line,
        message: format!("invalid hex u32: {}", e),
    })
}

fn parse_i64(s: &str, line: usize) -> Result<i64, ParseError> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map(|v| v as i64)
    } else {
        s.parse::<i64>()
    }
    .map_err(|e| ParseError::InvalidOperand {
        line,
        message: format!("invalid i64: {}", e),
    })
}

fn parse_hex_u64(s: &str, line: usize) -> Result<u64, ParseError> {
    let s = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    u64::from_str_radix(s, 16).map_err(|e| ParseError::InvalidOperand {
        line,
        message: format!("invalid hex u64: {}", e),
    })
}

fn parse_u16(s: &str, line: usize) -> Result<u16, ParseError> {
    s.parse::<u16>().map_err(|e| ParseError::InvalidOperand {
        line,
        message: format!("invalid u16: {}", e),
    })
}

fn parse_u32(s: &str, line: usize) -> Result<u32, ParseError> {
    s.parse::<u32>().map_err(|e| ParseError::InvalidOperand {
        line,
        message: format!("invalid u32: {}", e),
    })
}

/// Parse a version string: either "major.minor" (e.g. "1.0") or just "major" (e.g. "1" → Version { major: 1, minor: 0 }).
fn parse_version(s: &str) -> Result<Version, ParseError> {
    if let Some((major_str, minor_str)) = s.split_once('.') {
        let major = major_str
            .parse::<u16>()
            .map_err(|e| ParseError::InvalidVersion(e.to_string()))?;
        let minor = minor_str
            .parse::<u16>()
            .map_err(|e| ParseError::InvalidVersion(e.to_string()))?;
        Ok(Version::new(major, minor))
    } else {
        let major = s
            .parse::<u16>()
            .map_err(|e| ParseError::InvalidVersion(e.to_string()))?;
        Ok(Version::new(major, 0))
    }
}

fn parse_u8(s: &str, line: usize) -> Result<u8, ParseError> {
    s.parse::<u8>().map_err(|e| ParseError::InvalidOperand {
        line,
        message: format!("invalid u8: {}", e),
    })
}

/// Print a Program as assembly text.
pub fn print(program: &Program) -> String {
    let mut out = format!(
        ".version {}.{}\n",
        program.version.major, program.version.minor
    );

    for instr in &program.instructions {
        out.push_str(&print_instruction(instr));
        out.push('\n');
    }

    out
}

fn print_instruction(instr: &Instruction) -> String {
    match instr {
        Instruction::Cpu(cpu) => match cpu {
            CpuInstruction::ConstFloat(f) => format!("const_float {}", format_float(*f)),
            CpuInstruction::ConstInt(n) => format!("const_int {}", n),
            CpuInstruction::Pop => "pop".to_string(),
            CpuInstruction::Dup => "dup".to_string(),
            CpuInstruction::Swap => "swap".to_string(),
            CpuInstruction::Return => "return".to_string(),
            CpuInstruction::Halt => "halt".to_string(),
        },
        Instruction::LaneConst(lc) => match lc {
            LaneConstInstruction::ConstLoc(v) => format!("const_loc 0x{:08x}", v),
            LaneConstInstruction::ConstLane(d0, d1) => {
                let combined = (*d0 as u64) | ((*d1 as u64) << 32);
                format!("const_lane 0x{:016x}", combined)
            }
            LaneConstInstruction::ConstZone(v) => format!("const_zone 0x{:08x}", v),
        },
        Instruction::AtomArrangement(aa) => match aa {
            AtomArrangementInstruction::InitialFill { arity } => {
                format!("initial_fill {}", arity)
            }
            AtomArrangementInstruction::Fill { arity } => format!("fill {}", arity),
            AtomArrangementInstruction::Move { arity } => format!("move {}", arity),
        },
        Instruction::QuantumGate(qg) => match qg {
            QuantumGateInstruction::LocalR { arity } => format!("local_r {}", arity),
            QuantumGateInstruction::LocalRz { arity } => format!("local_rz {}", arity),
            QuantumGateInstruction::GlobalR => "global_r".to_string(),
            QuantumGateInstruction::GlobalRz => "global_rz".to_string(),
            QuantumGateInstruction::CZ => "cz".to_string(),
        },
        Instruction::Measurement(m) => match m {
            MeasurementInstruction::Measure { arity } => format!("measure {}", arity),
            MeasurementInstruction::AwaitMeasure => "await_measure".to_string(),
        },
        Instruction::Array(arr) => match arr {
            ArrayInstruction::NewArray {
                type_tag,
                dim0,
                dim1,
            } => {
                if *dim1 == 0 {
                    format!("new_array {} {}", type_tag, dim0)
                } else {
                    format!("new_array {} {} {}", type_tag, dim0, dim1)
                }
            }
            ArrayInstruction::GetItem { ndims } => format!("get_item {}", ndims),
        },
        Instruction::DetectorObservable(dob) => match dob {
            DetectorObservableInstruction::SetDetector => "set_detector".to_string(),
            DetectorObservableInstruction::SetObservable => "set_observable".to_string(),
        },
    }
}

/// Format a float for text output, ensuring it always has a decimal point.
fn format_float(f: f64) -> String {
    let s = format!("{}", f);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{}.0", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_program() {
        let source = ".version 1\nhalt\n";
        let program = parse(source).unwrap();
        assert_eq!(program.version, Version::new(1, 0));
        assert_eq!(program.instructions.len(), 1);
        assert_eq!(
            program.instructions[0],
            Instruction::Cpu(CpuInstruction::Halt)
        );
    }

    #[test]
    fn test_parse_with_comments_and_blanks() {
        let source = r#"
; This is a comment
.version 2

const_int 42  ; inline comment
halt
"#;
        let program = parse(source).unwrap();
        assert_eq!(program.version, Version::new(2, 0));
        assert_eq!(program.instructions.len(), 2);
        assert_eq!(
            program.instructions[0],
            Instruction::Cpu(CpuInstruction::ConstInt(42))
        );
    }

    #[test]
    fn test_parse_all_mnemonics() {
        let source = r#".version 1
const_float 1.5
const_int 42
const_loc 0x00010002
const_lane 0x0000000080010200
const_zone 0x00000003
pop
dup
swap
initial_fill 3
fill 2
move 1
local_r 4
local_rz 2
global_r
global_rz
cz
measure 1
await_measure
new_array 2 10 20
get_item 2
set_detector
set_observable
return
halt
"#;
        let program = parse(source).unwrap();
        assert_eq!(program.instructions.len(), 24);
    }

    #[test]
    fn test_parse_new_array_1d() {
        let source = ".version 1\nnew_array 1 10\n";
        let program = parse(source).unwrap();
        assert_eq!(
            program.instructions[0],
            Instruction::Array(ArrayInstruction::NewArray {
                type_tag: 1,
                dim0: 10,
                dim1: 0,
            })
        );
    }

    #[test]
    fn test_print_round_trip() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![
                Instruction::Cpu(CpuInstruction::ConstFloat(1.5)),
                Instruction::Cpu(CpuInstruction::ConstInt(42)),
                Instruction::LaneConst(LaneConstInstruction::ConstLoc(0x00010002)),
                Instruction::LaneConst(LaneConstInstruction::ConstLane(0x80010200, 0x00000000)),
                Instruction::LaneConst(LaneConstInstruction::ConstZone(3)),
                Instruction::Cpu(CpuInstruction::Pop),
                Instruction::Cpu(CpuInstruction::Dup),
                Instruction::Cpu(CpuInstruction::Swap),
                Instruction::AtomArrangement(AtomArrangementInstruction::InitialFill { arity: 3 }),
                Instruction::AtomArrangement(AtomArrangementInstruction::Fill { arity: 2 }),
                Instruction::AtomArrangement(AtomArrangementInstruction::Move { arity: 1 }),
                Instruction::QuantumGate(QuantumGateInstruction::LocalR { arity: 4 }),
                Instruction::QuantumGate(QuantumGateInstruction::LocalRz { arity: 2 }),
                Instruction::QuantumGate(QuantumGateInstruction::GlobalR),
                Instruction::QuantumGate(QuantumGateInstruction::GlobalRz),
                Instruction::QuantumGate(QuantumGateInstruction::CZ),
                Instruction::Measurement(MeasurementInstruction::Measure { arity: 1 }),
                Instruction::Measurement(MeasurementInstruction::AwaitMeasure),
                Instruction::Array(ArrayInstruction::NewArray {
                    type_tag: 2,
                    dim0: 10,
                    dim1: 20,
                }),
                Instruction::Array(ArrayInstruction::GetItem { ndims: 2 }),
                Instruction::DetectorObservable(DetectorObservableInstruction::SetDetector),
                Instruction::DetectorObservable(DetectorObservableInstruction::SetObservable),
                Instruction::Cpu(CpuInstruction::Return),
                Instruction::Cpu(CpuInstruction::Halt),
            ],
        };

        let text = print(&program);
        let parsed = parse(&text).unwrap();
        assert_eq!(program, parsed);
    }

    #[test]
    fn test_text_binary_round_trip() {
        let source = r#".version 1
const_loc 0x00000000
const_loc 0x00000001
initial_fill 2
const_lane 0x0000000000000100
move 1
const_float 1.5708
global_rz
const_zone 0x00000000
cz
measure 1
await_measure
return
"#;
        let program = parse(source).unwrap();
        let binary = program.to_binary();
        let from_binary = Program::from_binary(&binary).unwrap();
        let text_again = print(&from_binary);
        let reparsed = parse(&text_again).unwrap();
        assert_eq!(program, reparsed);
    }

    #[test]
    fn test_missing_version() {
        let source = "halt\n";
        assert_eq!(parse(source), Err(ParseError::MissingVersion));
    }

    #[test]
    fn test_unknown_mnemonic() {
        let source = ".version 1\nfoobar\n";
        assert!(matches!(
            parse(source),
            Err(ParseError::UnknownMnemonic { line: 2, .. })
        ));
    }

    #[test]
    fn test_missing_operand() {
        let source = ".version 1\nconst_int\n";
        assert!(matches!(
            parse(source),
            Err(ParseError::MissingOperand { line: 2, .. })
        ));
    }

    #[test]
    fn test_invalid_operand() {
        let source = ".version 1\nconst_int abc\n";
        assert!(matches!(
            parse(source),
            Err(ParseError::InvalidOperand { line: 2, .. })
        ));
    }

    #[test]
    fn test_print_new_array_1d() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::Array(ArrayInstruction::NewArray {
                type_tag: 1,
                dim0: 5,
                dim1: 0,
            })],
        };
        let text = print(&program);
        assert!(text.contains("new_array 1 5\n"));
        assert!(!text.contains("new_array 1 5 0"));
    }

    #[test]
    fn test_print_new_array_2d() {
        let program = Program {
            version: Version::new(1, 0),
            instructions: vec![Instruction::Array(ArrayInstruction::NewArray {
                type_tag: 2,
                dim0: 10,
                dim1: 20,
            })],
        };
        let text = print(&program);
        assert!(text.contains("new_array 2 10 20"));
    }

    #[test]
    fn test_design_doc_example() {
        let source = r#".version 1

const_loc 0x00000000    ; word 0, site 0
const_loc 0x00000001    ; word 0, site 1
initial_fill 2
const_lane 0x0000000000010000   ; fwd, site_bus, word 0, site 1, bus 0
move 1
const_loc 0x00000000    ; word 0, site 0
const_float 1.5708      ; axis_angle
const_float 3.14159     ; rotation_angle
local_r 1
const_float 0.7854      ; rotation_angle
global_rz
const_zone 0x00000000
cz
const_zone 0x00000000
measure 1
await_measure
return
"#;
        let program = parse(source).unwrap();
        assert_eq!(program.version, Version::new(1, 0));
        assert_eq!(program.instructions.len(), 17);
    }

    #[test]
    fn test_parse_version_major_minor() {
        let source = ".version 2.3\nhalt\n";
        let program = parse(source).unwrap();
        assert_eq!(program.version, Version::new(2, 3));
    }

    #[test]
    fn test_print_version_format() {
        let program = Program {
            version: Version::new(2, 3),
            instructions: vec![Instruction::Cpu(CpuInstruction::Halt)],
        };
        let text = print(&program);
        assert!(text.starts_with(".version 2.3\n"));
    }

    #[test]
    fn test_const_int_negative() {
        let source = ".version 1\nconst_int -42\nhalt\n";
        let program = parse(source).unwrap();
        assert_eq!(
            program.instructions[0],
            Instruction::Cpu(CpuInstruction::ConstInt(-42))
        );
        let text = print(&program);
        assert!(text.contains("const_int -42"));
    }
}
