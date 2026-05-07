//! Shared output formatting for `eval-policy` and `trace-policy`.

#![allow(dead_code)]

use serde::Serialize;

#[derive(Serialize)]
pub struct EvalEnvelope<'a> {
    pub v: u32,
    pub kind: &'static str, // "move" or "target"
    pub policy: &'a str,
    pub problem: &'a str,
    pub status: &'a str,
    pub halt_reason: Option<&'a str>,
    pub expansions: u64,
    pub max_depth: u32,
    pub wall_time_ms: f64,
}

#[derive(Serialize)]
pub struct TargetEvalEnvelope<'a> {
    pub v: u32,
    pub kind: &'static str, // "target"
    pub policy: &'a str,
    pub problem: &'a str,
    pub ok: bool,
    pub num_candidates: usize,
    pub first_candidate_size: usize,
    pub wall_time_ms: f64,
}

pub fn print_human_move(env: &EvalEnvelope) {
    println!("status         {}", env.status);
    if let Some(r) = env.halt_reason {
        println!("halt_reason    {r}");
    }
    println!("expansions     {}", env.expansions);
    println!("max_depth      {}", env.max_depth);
    println!("wall_time      {:.1} ms", env.wall_time_ms);
}

pub fn print_human_target(env: &TargetEvalEnvelope) {
    println!("ok                       {}", env.ok);
    println!("num_candidates           {}", env.num_candidates);
    println!("first_candidate_size     {}", env.first_candidate_size);
    println!("wall_time                {:.1} ms", env.wall_time_ms);
}
