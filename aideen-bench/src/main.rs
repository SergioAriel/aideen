// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Sergio Ariel Solis and Juan Patricio Marchetto

//! bench_harness — AIDEEN (DEQ+SSM) vs Transformer: benchmark defensible.
//!
//! Three coordinated experiments:
//!   EXP 1 · Iso-data  : same tokens seen → compares val_loss and efficiency
//!   EXP 2 · Iso-time  : same wall time → compares tokens achieved and val_loss
//!   EXP 3 · Inference : teacher-forcing NLL + throughput (on EXP 1 checkpoints)
//!
//! Protocol guarantees:
//!   • eval_*() is pure &self — does not mutate weights, optimizer, or state
//!   • Both models see the same batches in the same order
//!   • tokens_seen counted exactly: tokens_in.len() per step
//!   • training wall_clock accumulated separately from eval time
//!   • Declared limitation: AIDEEN does T micro-steps Adam per batch
//!
//! Uso:
//!   cargo run -p aideen-bench --release

mod dataset;
mod transformer_candle;

use std::time::{Duration, Instant};
use std::{env, fs};

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;
use dataset::Dataset;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use statrs::distribution::{ContinuousCDF, StudentsT};
use transformer_candle::{CandleBackend, CandleTransformer, CandleTransformerConfig};

// ── Experiment budget ────────────────────────────────────────────────────────
/// Context per batch — both models see the same length.
const CTX_LEN: usize = 128;
/// DEQ hidden dimension → ~283 K params with vocab=65.
const D_R_AIDEEN: usize = 192;
/// Shared learning rate.
const LR: f32 = 1e-3;
/// Iso-data: tokens que ve cada modelo (EXP 1).
const TOTAL_TOKENS: usize = 1_000_000;
/// Iso-time: segundos de pared por experimento (EXP 2).
const ISO_TIME_SECS: u64 = 600;
/// Evaluar val_loss cada N tokens de training.
const EVAL_EVERY: usize = 20_000;
/// Validation tokens per checkpoint (teacher-forcing).
const VAL_TOKENS: usize = 16_384;
const PARAM_TOLERANCE_PCT: f32 = 5.0;
const DEFAULT_SEEDS: [u64; 5] = [42, 1337, 2026, 31415, 27182];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }
}

#[derive(Clone, Copy)]
struct BenchConfig {
    total_tokens: usize,
    iso_time_secs: u64,
    eval_every: usize,
    iso_data_unit: IsoDataUnit,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            total_tokens: TOTAL_TOKENS,
            iso_time_secs: ISO_TIME_SECS,
            eval_every: EVAL_EVERY,
            iso_data_unit: IsoDataUnit::Updates,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum IsoDataUnit {
    Tokens,
    Updates,
}

impl IsoDataUnit {
    fn as_str(self) -> &'static str {
        match self {
            IsoDataUnit::Tokens => "tokens",
            IsoDataUnit::Updates => "updates",
        }
    }
}

#[derive(Clone, Copy)]
struct SeedResult {
    seed: u64,
    exp1_ai_best: f32,
    exp1_tf_best: f32,
    exp1_ai_tps: f64,
    exp1_tf_tps: f64,
    exp2_ai_best: f32,
    exp2_tf_best: f32,
    exp2_ai_tokens: usize,
    exp2_tf_tokens: usize,
    exp1_ai_updates: usize,
    exp1_tf_updates: usize,
    exp2_ai_updates: usize,
    exp2_tf_updates: usize,
    inf_ai_nll: f32,
    inf_tf_nll: f32,
    inf_ai_tps: f64,
    inf_tf_tps: f64,
}

#[derive(Clone, Copy)]
struct InferenceMetrics {
    ai_nll: f32,
    tf_nll: f32,
    ai_tps: f64,
    tf_tps: f64,
}

// ── Estructuras de métricas ───────────────────────────────────────────────────

#[derive(Clone)]
struct EvalPoint {
    tokens: usize,
    wall_secs: f64,
    val_loss: f32,
    train_loss: f32,
    tps: f32, // tokens/s de training (sin contar tiempo de eval)
}

struct ModelRun {
    name: &'static str,
    params: usize,
    log: Vec<EvalPoint>,
    best_val: f32,
    tokens_total: usize,
    updates_total: usize,
    secs_train: f64, // solo tiempo de training, excluye eval
}

impl ModelRun {
    fn new(name: &'static str, params: usize) -> Self {
        Self {
            name,
            params,
            log: vec![],
            best_val: f32::INFINITY,
            tokens_total: 0,
            updates_total: 0,
            secs_train: 0.0,
        }
    }
}

// ── Model construction ───────────────────────────────────────────────────────

fn build_aideen(vocab_size: usize, vocab: Vec<char>, seed: u64, backend: Backend) -> Trainer {
    let mut cfg = ArchitectureConfig::default();
    cfg.d_r = D_R_AIDEEN;
    cfg.vocab_size = vocab_size;
    cfg.ctx_len = CTX_LEN;
    cfg.max_deq_iters = 16; // match training defaults (was 6, too low to converge from random)
    cfg.adj_iters = 6;
    cfg.train_deq = true;
    cfg.deq_grad_scale = 0.01;
    cfg.renorm_every_steps = 4; // match training defaults (was 50, spectral norm applied too infrequently)

    let mut tok = Tokenizer::new_empty(vocab_size, cfg.clone());
    tok.vocab = vocab;
    {
        use nalgebra::DMatrix;
        let mut rng = StdRng::seed_from_u64(seed);
        tok.embeddings = DMatrix::from_fn(vocab_size, D_R_AIDEEN, |_, _| {
            (rng.gen::<f32>() - 0.5) * 0.02
        });
    }
    let mut trainer = Trainer::from_tokenizer_seeded(tok, LR, seed);
    trainer.config = cfg;
    trainer.training_config.lr = LR;
    match backend {
        Backend::Gpu => {
            if trainer.gpu_deq.is_none() {
                panic!("AIDEEN backend=gpu but trainer.gpu_deq=None.");
            }
        }
        Backend::Cpu => {
            trainer.gpu_deq = None;
            trainer.gpu_lm = None;
            trainer.gpu_emb = None;
        }
    }
    trainer
}

fn aideen_params(vocab_size: usize) -> usize {
    // DEQ: 7 weight matrices d_r×d_r + 2 vectors d_r
    // Embedding: vocab × d_r
    // LM head: vocab × d_r + vocab (bias)
    7 * D_R_AIDEEN * D_R_AIDEEN
        + 2 * D_R_AIDEEN
        + vocab_size * D_R_AIDEEN
        + vocab_size * D_R_AIDEEN
        + vocab_size
}

fn tf_candle_config(vocab_size: usize) -> CandleTransformerConfig {
    CandleTransformerConfig {
        vocab_size,
        d_model: 128,
        n_heads: 4,
        d_ff: 256,
        ctx_len: CTX_LEN,
        n_layers: 2,
    }
}

// ── Eval puro — &self garantizado, no muta nada ───────────────────────────────

/// Evaluates AIDEEN on VAL_CHUNKS windows of the val set.
/// Uses `eval_loss` which is `&self` (forward DEQ without backprop, without optimizer.tick()).
fn eval_aideen(trainer: &Trainer, val: &[u32]) -> f32 {
    let stride = CTX_LEN + 1;
    let n_target = (VAL_TOKENS / CTX_LEN).max(1);
    let n = n_target.min(val.len() / stride);
    if n == 0 {
        return f32::NAN;
    }
    let sum: f32 = (0..n)
        .map(|i| trainer.eval_loss(&val[i * stride..i * stride + stride]))
        .filter(|v| v.is_finite())
        .sum();
    sum / n as f32
}

/// Evaluates Transformer on VAL_CHUNKS windows.
/// Uses `val_loss` which is `&self` (forward_only, without optimizer.tick()).
fn eval_tf(tf: &CandleTransformer, val: &[u32]) -> f32 {
    let stride = CTX_LEN + 1;
    let n_target = (VAL_TOKENS / CTX_LEN).max(1);
    let n = n_target.min(val.len() / stride);
    if n == 0 {
        return f32::NAN;
    }
    let sum: f32 = (0..n)
        .map(|i| {
            tf.val_loss(&val[i * stride..i * stride + stride])
                .unwrap_or(f32::NAN)
        })
        .filter(|v| v.is_finite())
        .sum();
    sum / n as f32
}

// ── Stop condition ───────────────────────────────────────────────────────────

enum Budget {
    Tokens(usize),
    Updates(usize),
    Time(Duration),
}

// ── Shared training loop ────────────────────────────────────────────────────
//
// Both models receive the same batches in the same order.
// Eval time is excluded from secs_train for each model.

fn train_loop(
    eval_every: usize,
    budget: &Budget,
    ds: &Dataset,
    aideen: &mut Trainer,
    tf: &mut CandleTransformer,
    ai: &mut ModelRun,
    tfr: &mut ModelRun,
    sampler_seed: u64,
) {
    let stride = CTX_LEN + 1;
    let train = &ds.train;
    let val = &ds.val;
    let n_train = train.len();

    let wall_start = Instant::now();
    let mut sampler = StdRng::seed_from_u64(sampler_seed);

    // Accumulators between evals
    let mut ai_loss_sum = 0.0f32;
    let mut tf_loss_sum = 0.0f32;
    let mut ai_tok_int = 0usize;
    let mut tf_tok_int = 0usize;
    let mut ai_steps_int = 0usize;
    let mut tf_steps_int = 0usize;
    let mut ai_t_int = Duration::ZERO;
    let mut tf_t_int = Duration::ZERO;
    let mut last_ai_eval = 0usize;
    let mut last_tf_eval = 0usize;

    loop {
        let elapsed = wall_start.elapsed();
        let done = match budget {
            Budget::Tokens(max) => ai.tokens_total >= *max && tfr.tokens_total >= *max,
            Budget::Updates(max) => ai.updates_total >= *max && tfr.updates_total >= *max,
            Budget::Time(dur) => elapsed >= *dur,
        };
        if done {
            break;
        }

        // Mismo batch para ambos modelos
        let max_off = n_train.saturating_sub(stride + 1);
        if max_off == 0 {
            break;
        }
        let off = sampler.gen_range(0..max_off);
        let end = (off + stride).min(n_train);
        let batch = &train[off..end];

        if batch.len() < 2 {
            continue;
        }

        let tokens_in = &batch[..batch.len() - 1];
        let targets = &batch[1..];
        let n = tokens_in.len();

        // ── AIDEEN step ──────────────────────────────────────────────────────
        let ai_active = match budget {
            Budget::Tokens(max) => ai.tokens_total < *max,
            Budget::Updates(max) => ai.updates_total < *max,
            Budget::Time(_) => true,
        };
        if ai_active {
            let t0 = Instant::now();
            let loss = aideen.train_sequence(tokens_in, targets, true, aideen.config.deq_epsilon);
            let dt = t0.elapsed();
            if loss.is_finite() {
                ai.tokens_total += n;
                ai.updates_total += 1;
                ai.secs_train += dt.as_secs_f64();
                ai_loss_sum += loss;
                ai_tok_int += n;
                ai_steps_int += 1;
                ai_t_int += dt;
            }
        }

        // ── Transformer step ─────────────────────────────────────────────────
        let tf_active = match budget {
            Budget::Tokens(max) => tfr.tokens_total < *max,
            Budget::Updates(max) => tfr.updates_total < *max,
            Budget::Time(_) => true,
        };
        if tf_active {
            let t0 = Instant::now();
            let loss = tf.train_step(batch).unwrap_or(f32::NAN);
            let dt = t0.elapsed();
            if loss.is_finite() {
                tfr.tokens_total += n;
                tfr.updates_total += 1;
                tfr.secs_train += dt.as_secs_f64();
                tf_loss_sum += loss;
                tf_tok_int += n;
                tf_steps_int += 1;
                tf_t_int += dt;
            }
        }

        // ── Eval checkpoint ──────────────────────────────────────────────────
        let ai_eval = ai_tok_int > 0 && ai.tokens_total.saturating_sub(last_ai_eval) >= eval_every;
        let tf_eval = tf_tok_int > 0 && tfr.tokens_total.saturating_sub(last_tf_eval) >= eval_every;

        if ai_eval || tf_eval {
            let wall = wall_start.elapsed().as_secs_f64();

            if ai_eval {
                let vl = eval_aideen(aideen, val);
                let tps = ai_tok_int as f32 / ai_t_int.as_secs_f32().max(1e-9);
                let pt = EvalPoint {
                    tokens: ai.tokens_total,
                    wall_secs: wall,
                    val_loss: vl,
                    train_loss: ai_loss_sum / ai_steps_int.max(1) as f32,
                    tps,
                };
                if vl.is_finite() && vl < ai.best_val {
                    ai.best_val = vl;
                }
                ai.log.push(pt);
                ai_loss_sum = 0.0;
                ai_tok_int = 0;
                ai_steps_int = 0;
                ai_t_int = Duration::ZERO;
                last_ai_eval = ai.tokens_total;
            }

            if tf_eval {
                let vl = eval_tf(tf, val);
                let tps = tf_tok_int as f32 / tf_t_int.as_secs_f32().max(1e-9);
                let pt = EvalPoint {
                    tokens: tfr.tokens_total,
                    wall_secs: wall,
                    val_loss: vl,
                    train_loss: tf_loss_sum / tf_steps_int.max(1) as f32,
                    tps,
                };
                if vl.is_finite() && vl < tfr.best_val {
                    tfr.best_val = vl;
                }
                tfr.log.push(pt);
                tf_loss_sum = 0.0;
                tf_tok_int = 0;
                tf_steps_int = 0;
                tf_t_int = Duration::ZERO;
                last_tf_eval = tfr.tokens_total;
            }
        }
    }
}

// ── Salida de curvas ──────────────────────────────────────────────────────────

fn print_curve(ai: &ModelRun, tf: &ModelRun, x_is_tokens: bool) {
    let x_label = if x_is_tokens { "tokens" } else { "wall(s)" };
    println!();
    println!(
        "  {:>10} │ {:^10} {:^10} {:^8} │ {:^10} {:^10} {:^8} │ {}",
        x_label, "AI-val", "AI-train", "AI t/s", "TF-val", "TF-train", "TF t/s", "win"
    );
    println!("  {}", "─".repeat(80));

    let n = ai.log.len().max(tf.log.len());
    for i in 0..n {
        let ai_p = ai.log.get(i);
        let tf_p = tf.log.get(i);

        let x = if x_is_tokens {
            let tok = ai_p.or(tf_p).map(|p| p.tokens).unwrap_or(0);
            fmt_tokens(tok)
        } else {
            let s = ai_p.or(tf_p).map(|p| p.wall_secs).unwrap_or(0.0);
            format!("{:.0}", s)
        };

        let ai_val = ai_p.map_or(f32::NAN, |p| p.val_loss);
        let ai_tr = ai_p.map_or(f32::NAN, |p| p.train_loss);
        let ai_tps = ai_p.map_or(0.0, |p| p.tps);
        let tf_val = tf_p.map_or(f32::NAN, |p| p.val_loss);
        let tf_tr = tf_p.map_or(f32::NAN, |p| p.train_loss);
        let tf_tps = tf_p.map_or(0.0, |p| p.tps);

        let win = match (ai_val.is_finite(), tf_val.is_finite()) {
            (true, true) => {
                if ai_val < tf_val {
                    "AI ←"
                } else {
                    "TF →"
                }
            }
            _ => "    ",
        };

        println!(
            "  {:>10} │ {:>10.4} {:>10.4} {:>8.0} │ {:>10.4} {:>10.4} {:>8.0} │ {}",
            x, ai_val, ai_tr, ai_tps, tf_val, tf_tr, tf_tps, win,
        );
    }

    println!("  {}", "─".repeat(80));
    println!(
        "  best_val   │ {:>10.4} {:^30} │ {:>10.4} {:^30} │",
        ai.best_val, "", tf.best_val, ""
    );
    println!(
        "  t_training │ {:>10} {:^30} │ {:>10} {:^30} │",
        fmt_secs(ai.secs_train),
        "",
        fmt_secs(tf.secs_train),
        ""
    );
    println!(
        "  tok_seen   │ {:>10} {:^30} │ {:>10} {:^30} │",
        fmt_tokens(ai.tokens_total),
        "",
        fmt_tokens(tf.tokens_total),
        ""
    );
}

// ── Inference bench ───────────────────────────────────────────────────────────

fn inference_metrics(aideen: &Trainer, tf: &CandleTransformer, val: &[u32]) -> InferenceMetrics {
    let stride = CTX_LEN + 1;
    let n_chunks = ((VAL_TOKENS / CTX_LEN).max(1)).min(val.len() / stride);
    let tokens_measured = n_chunks * CTX_LEN;

    let t0 = Instant::now();
    let ai_nll: f32 = {
        let sum: f32 = (0..n_chunks)
            .map(|i| aideen.eval_loss(&val[i * stride..i * stride + stride]))
            .filter(|v| v.is_finite())
            .sum();
        if n_chunks > 0 {
            sum / n_chunks as f32
        } else {
            f32::NAN
        }
    };
    let ai_secs = t0.elapsed().as_secs_f64().max(1e-9);

    let t0 = Instant::now();
    let tf_nll: f32 = {
        let sum: f32 = (0..n_chunks)
            .map(|i| {
                tf.val_loss(&val[i * stride..i * stride + stride])
                    .unwrap_or(f32::NAN)
            })
            .filter(|v| v.is_finite())
            .sum();
        if n_chunks > 0 {
            sum / n_chunks as f32
        } else {
            f32::NAN
        }
    };
    let tf_secs = t0.elapsed().as_secs_f64().max(1e-9);

    InferenceMetrics {
        ai_nll,
        tf_nll,
        ai_tps: tokens_measured as f64 / ai_secs,
        tf_tps: tokens_measured as f64 / tf_secs,
    }
}

fn inference_bench(aideen: &Trainer, tf: &CandleTransformer, val: &[u32]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXP 3: Inference — teacher-forcing NLL + throughput        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  • forward-only (&self): no sampling, no backprop, no tick");
    println!(
        "  • Same chunks from the val set (target_tokens={}, stride=CTX_LEN+1)",
        VAL_TOKENS
    );
    println!();

    let m = inference_metrics(aideen, tf, val);

    println!(
        "  {:>14} │ {:>8} │ {:>10} │ {:>10} │ {:>10}",
        "Model", "NLL", "tok/sec", "ms/token", "#params"
    );
    println!("  {}", "─".repeat(62));
    println!(
        "  {:>14} │ {:>8.4} │ {:>10.0} │ {:>10.3} │ {:>10}",
        "AIDEEN",
        m.ai_nll,
        m.ai_tps,
        1000.0 / m.ai_tps.max(1e-9),
        fmt_tokens(aideen_params(aideen.tokenizer.vocab_size()))
    );
    println!(
        "  {:>14} │ {:>8.4} │ {:>10.0} │ {:>10.3} │ {:>10}",
        "Transformer",
        m.tf_nll,
        m.tf_tps,
        1000.0 / m.tf_tps.max(1e-9),
        fmt_tokens(tf.param_count())
    );
    println!();
    let speedup = m.tf_tps / m.ai_tps.max(1e-9);
    println!("  Speedup inferencia Transformer/AIDEEN : {:.1}×", speedup);
    let nll_delta = m.tf_nll - m.ai_nll;
    if nll_delta > 0.005 {
        println!(
            "  NLL: AIDEEN wins by {:.4} nats ({:.1}% better)",
            nll_delta,
            100.0 * nll_delta / m.tf_nll
        );
    } else if nll_delta < -0.005 {
        println!(
            "  NLL: Transformer wins by {:.4} nats ({:.1}% better)",
            nll_delta.abs(),
            100.0 * nll_delta.abs() / m.ai_nll
        );
    } else {
        println!(
            "  NLL: statistically similar (|delta|={:.4} nats)",
            nll_delta.abs()
        );
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn fmt_tokens(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f32 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.0}K", n as f32 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn fmt_secs(s: f64) -> String {
    if s >= 60.0 {
        format!("{:.1}m", s / 60.0)
    } else {
        format!("{:.1}s", s)
    }
}

fn parse_usize_env(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_u64_env(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn parse_bool_env(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(v) => matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => default,
    }
}

fn parse_seeds() -> Vec<u64> {
    if let Ok(raw) = env::var("AIDEEN_BENCH_SEEDS") {
        let parsed: Vec<u64> = raw
            .split(',')
            .filter_map(|x| x.trim().parse::<u64>().ok())
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    DEFAULT_SEEDS.to_vec()
}

fn parse_backend_env(name: &str, default: Backend) -> Backend {
    match env::var(name)
        .unwrap_or_else(|_| default.as_str().to_string())
        .to_lowercase()
        .as_str()
    {
        "cpu" => Backend::Cpu,
        "gpu" => Backend::Gpu,
        other => {
            eprintln!(
                "Invalid value for {}='{}'. Using '{}'.",
                name,
                other,
                default.as_str()
            );
            default
        }
    }
}

fn parse_iso_data_unit_env(name: &str, default: IsoDataUnit) -> IsoDataUnit {
    match env::var(name)
        .unwrap_or_else(|_| default.as_str().to_string())
        .to_lowercase()
        .as_str()
    {
        "tokens" => IsoDataUnit::Tokens,
        "updates" => IsoDataUnit::Updates,
        other => {
            eprintln!(
                "Invalid value for {}='{}'. Using '{}'.",
                name,
                other,
                default.as_str()
            );
            default
        }
    }
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    if vals.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    if vals.len() == 1 {
        return (mean, 0.0);
    }
    let var = vals
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (vals.len() as f64 - 1.0);
    (mean, var.sqrt())
}

fn paired_t_test_and_effect(ai: &[f64], tf: &[f64]) -> Option<(f64, f64, f64)> {
    if ai.len() != tf.len() || ai.len() < 2 {
        return None;
    }
    let diffs: Vec<f64> = ai.iter().zip(tf.iter()).map(|(a, t)| a - t).collect();
    let (mean_d, sd_d) = mean_std(&diffs);
    if !mean_d.is_finite() || !sd_d.is_finite() || sd_d <= 0.0 {
        return None;
    }
    let n = diffs.len() as f64;
    let t_stat = mean_d / (sd_d / n.sqrt());
    let dof = n - 1.0;
    let dist = StudentsT::new(0.0, 1.0, dof).ok()?;
    let p = 2.0 * (1.0 - dist.cdf(t_stat.abs()));
    // Cohen's dz para muestras pareadas.
    let dz = mean_d / sd_d;
    Some((t_stat, p, dz))
}

fn write_seed_csv(
    results: &[SeedResult],
    ai_backend: Backend,
    tf_backend: Backend,
    bench: BenchConfig,
) -> std::io::Result<String> {
    let out_dir = "aideen-bench/results";
    fs::create_dir_all(out_dir)?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = format!("{out_dir}/arch_bench_{ts}.csv");
    let mut csv = String::from(
        "seed,aideen_backend,transformer_backend,transformer_impl,iso_data_unit,exp1_ai_best,exp1_tf_best,exp1_ai_tps,exp1_tf_tps,exp2_ai_best,exp2_tf_best,exp2_ai_tokens,exp2_tf_tokens,exp1_ai_updates,exp1_tf_updates,exp2_ai_updates,exp2_tf_updates,inf_ai_nll,inf_tf_nll,inf_ai_tps,inf_tf_tps\n",
    );
    for r in results {
        csv.push_str(&format!(
            "{},{},{},candle_gpt,{},{:.6},{:.6},{:.3},{:.3},{:.6},{:.6},{},{},{},{},{},{},{:.6},{:.6},{:.3},{:.3}\n",
            r.seed,
            ai_backend.as_str(),
            tf_backend.as_str(),
            bench.iso_data_unit.as_str(),
            r.exp1_ai_best,
            r.exp1_tf_best,
            r.exp1_ai_tps,
            r.exp1_tf_tps,
            r.exp2_ai_best,
            r.exp2_tf_best,
            r.exp2_ai_tokens,
            r.exp2_tf_tokens,
            r.exp1_ai_updates,
            r.exp1_tf_updates,
            r.exp2_ai_updates,
            r.exp2_tf_updates,
            r.inf_ai_nll,
            r.inf_tf_nll,
            r.inf_ai_tps,
            r.inf_tf_tps
        ));
    }
    fs::write(&path, csv)?;
    Ok(path)
}

fn run_single_seed(
    seed: u64,
    ds: &Dataset,
    cfg_tf: &CandleTransformerConfig,
    bench: BenchConfig,
    ai_backend: Backend,
    tf_backend: Backend,
    verbose: bool,
) -> SeedResult {
    let vocab_size = ds.vocab_size();
    let tf_cb = match tf_backend {
        Backend::Cpu => CandleBackend::Cpu,
        Backend::Gpu => CandleBackend::Metal,
    };

    // ── Spectral warm-up phase (DEQ only) ─────────────────────────────────
    // DEQ requires spectral pre-conditioning before convergence.
    // Warm-up is NOT counted in the iso-data comparison.
    let warmup_steps: usize = std::env::var("AIDEEN_BENCH_WARMUP_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let mut aideen1 = build_aideen(vocab_size, ds.vocab.clone(), seed, ai_backend);
    if warmup_steps > 0 {
        println!(
            "  [WARM-UP] DEQ spectral conditioning: {warmup_steps} steps (not counted in comparison)"
        );
        let mut rng_wu = StdRng::seed_from_u64(seed ^ 0x0A41_0000);
        for step in 0..warmup_steps {
            let start = rng_wu.gen_range(0..ds.train.len().saturating_sub(CTX_LEN + 1));
            let tokens_in = &ds.train[start..start + CTX_LEN];
            let targets = &ds.train[start + 1..start + CTX_LEN + 1];
            let _loss =
                aideen1.train_sequence(tokens_in, targets, true, aideen1.config.deq_epsilon);
            if step % 200 == 0 {
                if let Some(ref gpu) = aideen1.gpu_deq {
                    let fw = gpu.read_debug_buffer();
                    let contr = if fw.len() > 21 { fw[21] } else { 0.0 };
                    let iters = if fw.len() > 13 { fw[13] } else { 0.0 };
                    println!(
                        "    [WARM-UP] step {step}: contractivity={contr:.3}, avg_iters={iters:.1}"
                    );
                }
            }
        }
        println!("  [WARM-UP] Complete. DEQ pre-conditioned for benchmark.");
    }

    // EXP 1: iso-data
    let mut tf1 =
        CandleTransformer::new(cfg_tf.clone(), LR as f64, tf_cb).expect("tf candle init exp1");
    let mut ai1 = ModelRun::new("AIDEEN", aideen_params(vocab_size));
    let mut tf1r = ModelRun::new("Transformer", cfg_tf.param_count());
    let exp1_budget = match bench.iso_data_unit {
        IsoDataUnit::Tokens => Budget::Tokens(bench.total_tokens),
        IsoDataUnit::Updates => Budget::Updates((bench.total_tokens / CTX_LEN).max(1)),
    };
    train_loop(
        bench.eval_every,
        &exp1_budget,
        ds,
        &mut aideen1,
        &mut tf1,
        &mut ai1,
        &mut tf1r,
        seed ^ 0xA11D_EE11,
    );
    if verbose {
        print_curve(&ai1, &tf1r, true);
    }

    // EXP 2: iso-time
    let mut aideen2 = build_aideen(
        vocab_size,
        ds.vocab.clone(),
        seed.wrapping_add(2),
        ai_backend,
    );
    let mut tf2 =
        CandleTransformer::new(cfg_tf.clone(), LR as f64, tf_cb).expect("tf candle init exp2");
    let mut ai2 = ModelRun::new("AIDEEN", aideen_params(vocab_size));
    let mut tf2r = ModelRun::new("Transformer", cfg_tf.param_count());
    train_loop(
        bench.eval_every,
        &Budget::Time(Duration::from_secs(bench.iso_time_secs)),
        ds,
        &mut aideen2,
        &mut tf2,
        &mut ai2,
        &mut tf2r,
        seed.wrapping_add(2) ^ 0x71CE_5EED,
    );
    if verbose {
        print_curve(&ai2, &tf2r, false);
        inference_bench(&aideen1, &tf1, &ds.val);
    }
    let inf = inference_metrics(&aideen1, &tf1, &ds.val);

    SeedResult {
        seed,
        exp1_ai_best: ai1.best_val,
        exp1_tf_best: tf1r.best_val,
        exp1_ai_tps: if ai1.secs_train > 0.0 {
            ai1.tokens_total as f64 / ai1.secs_train
        } else {
            0.0
        },
        exp1_tf_tps: if tf1r.secs_train > 0.0 {
            tf1r.tokens_total as f64 / tf1r.secs_train
        } else {
            0.0
        },
        exp2_ai_best: ai2.best_val,
        exp2_tf_best: tf2r.best_val,
        exp2_ai_tokens: ai2.tokens_total,
        exp2_tf_tokens: tf2r.tokens_total,
        exp1_ai_updates: ai1.updates_total,
        exp1_tf_updates: tf1r.updates_total,
        exp2_ai_updates: ai2.updates_total,
        exp2_tf_updates: tf2r.updates_total,
        inf_ai_nll: inf.ai_nll,
        inf_tf_nll: inf.tf_nll,
        inf_ai_tps: inf.ai_tps,
        inf_tf_tps: inf.tf_tps,
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let bench_cfg = BenchConfig {
        total_tokens: parse_usize_env("AIDEEN_BENCH_TOTAL_TOKENS", TOTAL_TOKENS),
        iso_time_secs: parse_u64_env("AIDEEN_BENCH_ISO_TIME_SECS", ISO_TIME_SECS),
        eval_every: parse_usize_env("AIDEEN_BENCH_EVAL_EVERY", EVAL_EVERY),
        iso_data_unit: parse_iso_data_unit_env("AIDEEN_BENCH_ISO_DATA_UNIT", IsoDataUnit::Updates),
    };
    let seeds = parse_seeds();
    let ai_backend = parse_backend_env("AIDEEN_BENCH_AIDEEN_BACKEND", Backend::Gpu);
    let tf_backend = parse_backend_env("AIDEEN_BENCH_TF_BACKEND", Backend::Cpu);

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AIDEEN vs Transformer — Benchmark (Multi-Seed)             ║");
    println!("║  EXP 1: Iso-Data | EXP 2: Iso-Time | EXP 3: Inference      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Protocol guarantees:");
    println!("  ✓ eval_*() is pure &self — does not mutate weights, optimizer, or state");
    println!("  ✓ Both models see the same batches in the same order");
    println!("  ✓ tokens_seen counted exactly (tokens_in.len() per step)");
    println!("  ✓ training wall_clock accumulated WITHOUT counting eval time");
    println!("  ✓ Same tokenizer and train/val split for both models");
    println!();
    println!(
        "  Config: iso_data_budget={} {} | iso_time={}s eval_every={} val_tokens={} seeds={:?} | backends: aideen={} transformer={}",
        bench_cfg.total_tokens,
        bench_cfg.iso_data_unit.as_str(),
        bench_cfg.iso_time_secs,
        bench_cfg.eval_every,
        VAL_TOKENS,
        seeds,
        ai_backend.as_str(),
        tf_backend.as_str()
    );
    let allow_mixed = parse_bool_env("AIDEEN_BENCH_ALLOW_MIXED", false);
    if ai_backend != tf_backend && !allow_mixed {
        panic!(
            "Benchmark aborted: mixed backend ({} vs {}). Use both on cpu/gpu or export AIDEEN_BENCH_ALLOW_MIXED=1.",
            ai_backend.as_str(),
            tf_backend.as_str()
        );
    }
    if ai_backend != tf_backend && allow_mixed {
        println!(
            "  [WARN] Mixed backend comparison ({} vs {}). For symmetric benchmark use CPUvsCPU.",
            ai_backend.as_str(),
            tf_backend.as_str()
        );
    }
    println!();

    // ── Dataset
    println!("① Loading dataset...");
    let ds = Dataset::load();
    let vocab_size = ds.vocab_size();
    println!();

    // ── Param counts
    let ai_p = aideen_params(vocab_size);
    let cfg_tf = tf_candle_config(vocab_size);
    let tf_p = cfg_tf.param_count();
    println!(
        "  AIDEEN     : {} params  (D_R={}, backend={})",
        fmt_tokens(ai_p),
        D_R_AIDEEN,
        ai_backend.as_str()
    );
    println!(
        "  Transformer: {} params  (d={}, {}-layer GPT, ctx={})",
        fmt_tokens(tf_p),
        cfg_tf.d_model,
        cfg_tf.n_layers,
        CTX_LEN
    );
    let param_diff_pct = 100.0 * (ai_p as f32 - tf_p as f32).abs() / tf_p as f32;
    println!(
        "  Difference : {:.1}%  (target iso-params <{:.1}%)",
        param_diff_pct, PARAM_TOLERANCE_PCT
    );
    if param_diff_pct > PARAM_TOLERANCE_PCT {
        eprintln!(
            "❌ Aborted: iso-params out of tolerance ({:.2}% > {:.2}%). Adjust D_R_AIDEEN or tf_config().",
            param_diff_pct, PARAM_TOLERANCE_PCT
        );
        std::process::exit(2);
    }
    println!();

    let mut results = Vec::new();
    for (i, seed) in seeds.iter().enumerate() {
        println!();
        println!(
            "══════════ Seed {} / {} ({}) ══════════",
            i + 1,
            seeds.len(),
            seed
        );
        let verbose = i == 0;
        let r = run_single_seed(
            *seed, &ds, &cfg_tf, bench_cfg, ai_backend, tf_backend, verbose,
        );
        results.push(r);
    }

    let exp1_ai: Vec<f64> = results.iter().map(|r| r.exp1_ai_best as f64).collect();
    let exp1_tf: Vec<f64> = results.iter().map(|r| r.exp1_tf_best as f64).collect();
    let exp2_ai: Vec<f64> = results.iter().map(|r| r.exp2_ai_best as f64).collect();
    let exp2_tf: Vec<f64> = results.iter().map(|r| r.exp2_tf_best as f64).collect();
    let inf_ai: Vec<f64> = results.iter().map(|r| r.inf_ai_nll as f64).collect();
    let inf_tf: Vec<f64> = results.iter().map(|r| r.inf_tf_nll as f64).collect();
    let exp2_ai_tok: Vec<f64> = results.iter().map(|r| r.exp2_ai_tokens as f64).collect();
    let exp2_tf_tok: Vec<f64> = results.iter().map(|r| r.exp2_tf_tokens as f64).collect();
    let exp1_ai_upd: Vec<f64> = results.iter().map(|r| r.exp1_ai_updates as f64).collect();
    let exp1_tf_upd: Vec<f64> = results.iter().map(|r| r.exp1_tf_updates as f64).collect();

    let (exp1_ai_m, exp1_ai_s) = mean_std(&exp1_ai);
    let (exp1_tf_m, exp1_tf_s) = mean_std(&exp1_tf);
    let (exp2_ai_m, exp2_ai_s) = mean_std(&exp2_ai);
    let (exp2_tf_m, exp2_tf_s) = mean_std(&exp2_tf);
    let (inf_ai_m, inf_ai_s) = mean_std(&inf_ai);
    let (inf_tf_m, inf_tf_s) = mean_std(&inf_tf);
    let (exp2_ai_tok_m, exp2_ai_tok_s) = mean_std(&exp2_ai_tok);
    let (exp2_tf_tok_m, exp2_tf_tok_s) = mean_std(&exp2_tf_tok);
    let (exp1_ai_upd_m, exp1_ai_upd_s) = mean_std(&exp1_ai_upd);
    let (exp1_tf_upd_m, exp1_tf_upd_s) = mean_std(&exp1_tf_upd);

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Aggregated summary (mean ± std, multi-seed)                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!(
        "EXP1 Iso-Data best_val:   AIDEEN {:.4} ± {:.4} | TF {:.4} ± {:.4}",
        exp1_ai_m, exp1_ai_s, exp1_tf_m, exp1_tf_s
    );
    println!(
        "EXP2 Iso-Time best_val:   AIDEEN {:.4} ± {:.4} | TF {:.4} ± {:.4}",
        exp2_ai_m, exp2_ai_s, exp2_tf_m, exp2_tf_s
    );
    println!(
        "EXP2 tokens seen:         AIDEEN {:.0} ± {:.0} | TF {:.0} ± {:.0}",
        exp2_ai_tok_m, exp2_ai_tok_s, exp2_tf_tok_m, exp2_tf_tok_s
    );
    println!(
        "EXP1 updates:             AIDEEN {:.0} ± {:.0} | TF {:.0} ± {:.0}",
        exp1_ai_upd_m, exp1_ai_upd_s, exp1_tf_upd_m, exp1_tf_upd_s
    );
    println!(
        "EXP3 Inference NLL:       AIDEEN {:.4} ± {:.4} | TF {:.4} ± {:.4}",
        inf_ai_m, inf_ai_s, inf_tf_m, inf_tf_s
    );
    if let Some((t, p, dz)) = paired_t_test_and_effect(&exp1_ai, &exp1_tf) {
        println!(
            "EXP1 paired t-test (AI-TF): t={:.3}, p={:.4}, cohen_dz={:.3}",
            t, p, dz
        );
    } else {
        println!("EXP1 paired t-test: insufficient n or zero variance.");
    }
    if let Some((t, p, dz)) = paired_t_test_and_effect(&exp2_ai, &exp2_tf) {
        println!(
            "EXP2 paired t-test (AI-TF): t={:.3}, p={:.4}, cohen_dz={:.3}",
            t, p, dz
        );
    } else {
        println!("EXP2 paired t-test: insufficient n or zero variance.");
    }
    if let Some((t, p, dz)) = paired_t_test_and_effect(&inf_ai, &inf_tf) {
        println!(
            "EXP3 paired t-test (AI-TF): t={:.3}, p={:.4}, cohen_dz={:.3}",
            t, p, dz
        );
    } else {
        println!("EXP3 paired t-test: insufficient n or zero variance.");
    }

    match write_seed_csv(&results, ai_backend, tf_backend, bench_cfg) {
        Ok(path) => println!("Seeds CSV: {}", path),
        Err(e) => eprintln!("Could not write CSV: {}", e),
    }
}
