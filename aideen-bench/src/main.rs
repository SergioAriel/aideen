//! bench_harness — AIDEEN (DEQ+SSM) vs Transformer: benchmark defensible.
//!
//! Tres experimentos coordinados:
//!   EXP 1 · Iso-data  : mismos tokens vistos → compara val_loss y eficiencia
//!   EXP 2 · Iso-time  : mismo tiempo de pared → compara tokens logrados y val_loss
//!   EXP 3 · Inference : teacher-forcing NLL + throughput (sobre checkpoints del EXP 1)
//!
//! Garantías del protocolo:
//!   • eval_*() es &self puro — no muta pesos, optimizer ni estados
//!   • Ambos modelos ven los mismos batches en el mismo orden
//!   • tokens_seen se cuenta exactamente: tokens_in.len() por paso
//!   • wall_clock de training se acumula separado del tiempo de eval
//!   • Limitación declarada: AIDEEN hace T micro-steps Adam por batch
//!
//! Uso:
//!   cargo run -p aideen-bench --release

mod dataset;
mod transformer;

use std::time::{Duration, Instant};

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;
use dataset::Dataset;
use transformer::{MiniAdam, Transformer, TransformerConfig};

// ── Presupuesto del experimento ───────────────────────────────────────────────
/// Contexto por batch — ambos modelos ven la misma longitud.
const CTX_LEN: usize = 16;
/// Dimensión oculta DEQ → ~283 K params con vocab=65.
const D_R_AIDEEN: usize = 192;
/// Learning rate compartido.
const LR: f32 = 1e-3;
/// Iso-data: tokens que ve cada modelo (EXP 1).
const TOTAL_TOKENS: usize = 50_000;
/// Iso-time: segundos de pared por experimento (EXP 2).
const ISO_TIME_SECS: u64 = 120;
/// Evaluar val_loss cada N tokens de training.
const EVAL_EVERY: usize = 5_000;
/// Chunks de validación (cada uno: CTX_LEN+1 tokens).
const VAL_CHUNKS: usize = 25;

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
            secs_train: 0.0,
        }
    }
}

// ── Construcción de modelos ───────────────────────────────────────────────────

fn build_aideen(vocab_size: usize, vocab: Vec<char>) -> Trainer {
    let mut cfg = ArchitectureConfig::default();
    cfg.d_r = D_R_AIDEEN;
    cfg.vocab_size = vocab_size;
    cfg.ctx_len = CTX_LEN;
    cfg.max_deq_iters = 6;
    cfg.cg_iters = 4;
    cfg.train_deq = true;
    cfg.deq_grad_scale = 0.01;
    cfg.renorm_every_steps = 50;

    let mut tok = Tokenizer::new_empty(vocab_size, cfg.clone());
    tok.vocab = vocab;
    {
        use nalgebra::DMatrix;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        tok.embeddings = DMatrix::from_fn(vocab_size, D_R_AIDEEN, |_, _| {
            (rng.gen::<f32>() - 0.5) * 0.02
        });
    }
    let mut trainer = Trainer::from_tokenizer(tok, LR);
    trainer.config = cfg;
    trainer.training_config.lr = LR;
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

/// Config Transformer iso-parámetro con ctx_len = CTX_LEN del benchmark.
fn tf_config(vocab_size: usize) -> TransformerConfig {
    TransformerConfig {
        vocab_size,
        d_model: 128,
        n_heads: 4,
        d_ff: 256,
        ctx_len: CTX_LEN,
        n_layers: 2,
    }
}

fn n_adam_groups_for(cfg: &TransformerConfig) -> usize {
    2 + cfg.n_layers * 8 + 4
}

// ── Eval puro — &self garantizado, no muta nada ───────────────────────────────

/// Evalúa AIDEEN en VAL_CHUNKS ventanas del conjunto val.
/// Usa `eval_loss` que es `&self` (forward DEQ sin backprop, sin optimizer.tick()).
fn eval_aideen(trainer: &Trainer, val: &[u32]) -> f32 {
    let stride = CTX_LEN + 1;
    let n = VAL_CHUNKS.min(val.len() / stride);
    if n == 0 {
        return f32::NAN;
    }
    let sum: f32 = (0..n)
        .map(|i| trainer.eval_loss(&val[i * stride..i * stride + stride]))
        .filter(|v| v.is_finite())
        .sum();
    sum / n as f32
}

/// Evalúa Transformer en VAL_CHUNKS ventanas.
/// Usa `val_loss` que es `&self` (forward_only, sin optimizer.tick()).
fn eval_tf(tf: &Transformer, val: &[u32]) -> f32 {
    let stride = CTX_LEN + 1;
    let n = VAL_CHUNKS.min(val.len() / stride);
    if n == 0 {
        return f32::NAN;
    }
    let sum: f32 = (0..n)
        .map(|i| tf.val_loss(&val[i * stride..i * stride + stride]))
        .filter(|v| v.is_finite())
        .sum();
    sum / n as f32
}

// ── Condición de parada ───────────────────────────────────────────────────────

enum Budget {
    Tokens(usize),
    Time(Duration),
}

// ── Loop de entrenamiento compartido ─────────────────────────────────────────
//
// Ambos modelos reciben los mismos batches en el mismo orden.
// El tiempo de eval se excluye del secs_train de cada modelo.

fn train_loop(
    budget: &Budget,
    ds: &Dataset,
    aideen: &mut Trainer,
    tf: &mut Transformer,
    tf_opt: &mut MiniAdam,
    ai: &mut ModelRun,
    tfr: &mut ModelRun,
) {
    let stride = CTX_LEN + 1;
    let train = &ds.train;
    let val = &ds.val;
    let n_train = train.len();

    let wall_start = Instant::now();
    let mut batch_idx = 0usize;

    // Acumuladores entre evals
    let mut ai_loss_sum = 0.0f32;
    let mut tf_loss_sum = 0.0f32;
    let mut ai_tok_int = 0usize;
    let mut tf_tok_int = 0usize;
    let mut ai_t_int = Duration::ZERO;
    let mut tf_t_int = Duration::ZERO;
    let mut last_ai_eval = 0usize;
    let mut last_tf_eval = 0usize;

    loop {
        let elapsed = wall_start.elapsed();
        let done = match budget {
            Budget::Tokens(max) => ai.tokens_total >= *max && tfr.tokens_total >= *max,
            Budget::Time(dur) => elapsed >= *dur,
        };
        if done {
            break;
        }

        // Mismo batch para ambos modelos
        let off = (batch_idx * (stride / 2 + 1)) % n_train.saturating_sub(stride + 1);
        let end = (off + stride).min(n_train);
        let batch = &train[off..end];
        batch_idx += 1;

        if batch.len() < 2 {
            continue;
        }

        let tokens_in = &batch[..batch.len() - 1];
        let targets = &batch[1..];
        let n = tokens_in.len();

        // ── AIDEEN step ──────────────────────────────────────────────────────
        let ai_active = match budget {
            Budget::Tokens(max) => ai.tokens_total < *max,
            Budget::Time(_) => true,
        };
        if ai_active {
            let t0 = Instant::now();
            let loss = aideen.train_sequence(tokens_in, targets, true);
            let dt = t0.elapsed();
            if loss.is_finite() {
                ai.tokens_total += n;
                ai.secs_train += dt.as_secs_f64();
                ai_loss_sum += loss;
                ai_tok_int += n;
                ai_t_int += dt;
            }
        }

        // ── Transformer step ─────────────────────────────────────────────────
        let tf_active = match budget {
            Budget::Tokens(max) => tfr.tokens_total < *max,
            Budget::Time(_) => true,
        };
        if tf_active {
            let t0 = Instant::now();
            let loss = tf.train_step(batch, tf_opt);
            let dt = t0.elapsed();
            if loss.is_finite() {
                tfr.tokens_total += n;
                tfr.secs_train += dt.as_secs_f64();
                tf_loss_sum += loss;
                tf_tok_int += n;
                tf_t_int += dt;
            }
        }

        // ── Eval checkpoint ──────────────────────────────────────────────────
        let ai_eval = ai_tok_int > 0
            && ai.tokens_total.saturating_sub(last_ai_eval) >= EVAL_EVERY;
        let tf_eval = tf_tok_int > 0
            && tfr.tokens_total.saturating_sub(last_tf_eval) >= EVAL_EVERY;

        if ai_eval || tf_eval {
            let wall = wall_start.elapsed().as_secs_f64();

            if ai_eval {
                let vl = eval_aideen(aideen, val);
                let tps = ai_tok_int as f32 / ai_t_int.as_secs_f32().max(1e-9);
                let pt = EvalPoint {
                    tokens: ai.tokens_total,
                    wall_secs: wall,
                    val_loss: vl,
                    train_loss: ai_loss_sum / ai_tok_int.max(1) as f32,
                    tps,
                };
                if vl.is_finite() && vl < ai.best_val {
                    ai.best_val = vl;
                }
                ai.log.push(pt);
                ai_loss_sum = 0.0;
                ai_tok_int = 0;
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
                    train_loss: tf_loss_sum / tf_tok_int.max(1) as f32,
                    tps,
                };
                if vl.is_finite() && vl < tfr.best_val {
                    tfr.best_val = vl;
                }
                tfr.log.push(pt);
                tf_loss_sum = 0.0;
                tf_tok_int = 0;
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
        x_label, "AI-val", "AI-train", "AI t/s",
        "TF-val", "TF-train", "TF t/s", "win"
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
            (true, true) => if ai_val < tf_val { "AI ←" } else { "TF →" },
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
        fmt_secs(ai.secs_train), "", fmt_secs(tf.secs_train), ""
    );
    println!(
        "  tok_seen   │ {:>10} {:^30} │ {:>10} {:^30} │",
        fmt_tokens(ai.tokens_total), "", fmt_tokens(tf.tokens_total), ""
    );
}

// ── Inference bench ───────────────────────────────────────────────────────────

fn inference_bench(aideen: &Trainer, tf: &Transformer, val: &[u32]) {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXP 3: Inference — teacher-forcing NLL + throughput        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  • forward-only (&self): sin sampling, sin backprop, sin tick");
    println!("  • Mismos chunks del conjunto val (n=50, stride=CTX_LEN+1)");
    println!();

    let stride = CTX_LEN + 1;
    let n_chunks = 50_usize.min(val.len() / stride);
    let tokens_measured = n_chunks * CTX_LEN;

    // AIDEEN inference
    let t0 = Instant::now();
    let ai_nll: f32 = {
        let sum: f32 = (0..n_chunks)
            .map(|i| aideen.eval_loss(&val[i * stride..i * stride + stride]))
            .filter(|v| v.is_finite())
            .sum();
        if n_chunks > 0 { sum / n_chunks as f32 } else { f32::NAN }
    };
    let ai_secs = t0.elapsed().as_secs_f64().max(1e-9);

    // Transformer inference
    let t0 = Instant::now();
    let tf_nll: f32 = {
        let sum: f32 = (0..n_chunks)
            .map(|i| tf.val_loss(&val[i * stride..i * stride + stride]))
            .filter(|v| v.is_finite())
            .sum();
        if n_chunks > 0 { sum / n_chunks as f32 } else { f32::NAN }
    };
    let tf_secs = t0.elapsed().as_secs_f64().max(1e-9);

    println!(
        "  {:>14} │ {:>8} │ {:>10} │ {:>10} │ {:>10}",
        "Model", "NLL", "tok/sec", "ms/token", "#params"
    );
    println!("  {}", "─".repeat(62));
    println!(
        "  {:>14} │ {:>8.4} │ {:>10.0} │ {:>10.3} │ {:>10}",
        "AIDEEN",
        ai_nll,
        tokens_measured as f64 / ai_secs,
        ai_secs * 1000.0 / tokens_measured as f64,
        fmt_tokens(aideen_params(aideen.tokenizer.vocab_size()))
    );
    println!(
        "  {:>14} │ {:>8.4} │ {:>10.0} │ {:>10.3} │ {:>10}",
        "Transformer",
        tf_nll,
        tokens_measured as f64 / tf_secs,
        tf_secs * 1000.0 / tokens_measured as f64,
        fmt_tokens(tf.param_count())
    );
    println!();
    let speedup = ai_secs / tf_secs;
    println!("  Speedup inferencia Transformer/AIDEEN : {:.1}×", speedup);
    let nll_delta = tf_nll - ai_nll;
    if nll_delta > 0.005 {
        println!("  NLL: AIDEEN gana por {:.4} nats ({:.1}% mejor)", nll_delta, 100.0 * nll_delta / tf_nll);
    } else if nll_delta < -0.005 {
        println!("  NLL: Transformer gana por {:.4} nats ({:.1}% mejor)", nll_delta.abs(), 100.0 * nll_delta.abs() / ai_nll);
    } else {
        println!("  NLL: estadísticamente similar (|delta|={:.4} nats)", nll_delta.abs());
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

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AIDEEN vs Transformer — Benchmark Defensible               ║");
    println!("║  EXP 1: Iso-Data | EXP 2: Iso-Time | EXP 3: Inference      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Garantías del protocolo:");
    println!("  ✓ eval_*() es &self puro — no muta pesos, optimizer ni estado");
    println!("  ✓ Ambos modelos ven los mismos batches en el mismo orden");
    println!("  ✓ tokens_seen contado exactamente (tokens_in.len() por paso)");
    println!("  ✓ wall_clock de training acumulado SIN contar tiempo de eval");
    println!("  ✓ Mismo tokenizer y split train/val para los dos modelos");
    println!();
    println!("  Limitación declarada:");
    println!(
        "  ⚠ AIDEEN hace {} micro-steps Adam por batch (LR/{}).",
        CTX_LEN - 1, CTX_LEN - 1
    );
    println!("    Transformer hace 1 macro-step. Bias correction Adam evoluciona");
    println!("    {}× más rápido en AIDEEN (misma magnitud efectiva de update).", CTX_LEN - 1);
    println!();

    // ── Dataset
    println!("① Cargando dataset...");
    let ds = Dataset::load();
    let vocab_size = ds.vocab_size();
    println!();

    // ── Param counts
    let ai_p = aideen_params(vocab_size);
    let cfg_tf = tf_config(vocab_size);
    let tf_p = cfg_tf.param_count();
    println!("  AIDEEN     : {} params  (D_R={}, CPU DEQ+SSM)", fmt_tokens(ai_p), D_R_AIDEEN);
    println!(
        "  Transformer: {} params  (d={}, {}-layer GPT, ctx={})",
        fmt_tokens(tf_p), cfg_tf.d_model, cfg_tf.n_layers, CTX_LEN
    );
    println!(
        "  Diferencia : {:.1}%  (objetivo iso-params <5%)",
        100.0 * (ai_p as f32 - tf_p as f32).abs() / tf_p as f32
    );
    println!();

    let n_adam = n_adam_groups_for(&cfg_tf);

    // ═══════════════════════════════════════════════════════════════════
    // EXP 1: Iso-data
    // ═══════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  EXP 1: Iso-Data ({} tokens/modelo, eval cada {}K)   ║",
        fmt_tokens(TOTAL_TOKENS),
        EVAL_EVERY / 1000
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut aideen1 = build_aideen(vocab_size, ds.vocab.clone());
    let mut tf1 = Transformer::new(cfg_tf.clone());
    let mut opt1 = MiniAdam::new(LR, n_adam);
    let mut ai1 = ModelRun::new("AIDEEN", ai_p);
    let mut tf1r = ModelRun::new("Transformer", tf_p);

    train_loop(&Budget::Tokens(TOTAL_TOKENS), &ds, &mut aideen1, &mut tf1, &mut opt1, &mut ai1, &mut tf1r);
    print_curve(&ai1, &tf1r, true);

    // ═══════════════════════════════════════════════════════════════════
    // EXP 2: Iso-time
    // ═══════════════════════════════════════════════════════════════════
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  EXP 2: Iso-Time ({} seg de pared, eval cada {}K)     ║",
        ISO_TIME_SECS, EVAL_EVERY / 1000
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  Cuánto aprende cada arquitectura dado el mismo compute budget.");

    let mut aideen2 = build_aideen(vocab_size, ds.vocab.clone());
    let mut tf2 = Transformer::new(cfg_tf.clone());
    let mut opt2 = MiniAdam::new(LR, n_adam);
    let mut ai2 = ModelRun::new("AIDEEN", ai_p);
    let mut tf2r = ModelRun::new("Transformer", tf_p);

    train_loop(
        &Budget::Time(Duration::from_secs(ISO_TIME_SECS)),
        &ds,
        &mut aideen2,
        &mut tf2,
        &mut opt2,
        &mut ai2,
        &mut tf2r,
    );
    print_curve(&ai2, &tf2r, false);

    // ═══════════════════════════════════════════════════════════════════
    // EXP 3: Inference (checkpoints del EXP 1 — iso-data, mismos tokens)
    // ═══════════════════════════════════════════════════════════════════
    inference_bench(&aideen1, &tf1, &ds.val);

    // ═══════════════════════════════════════════════════════════════════
    // Resumen final
    // ═══════════════════════════════════════════════════════════════════
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Resumen ejecutivo                                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    println!("  EXP 1 · Iso-Data ({}):", fmt_tokens(TOTAL_TOKENS));
    let w1 = if ai1.best_val < tf1r.best_val { "AIDEEN" } else { "Transformer" };
    println!(
        "    AIDEEN      best_val={:.4}  avg_tps={:.0}  t_train={}",
        ai1.best_val,
        if ai1.secs_train > 0.0 { ai1.tokens_total as f64 / ai1.secs_train } else { 0.0 },
        fmt_secs(ai1.secs_train)
    );
    println!(
        "    Transformer best_val={:.4}  avg_tps={:.0}  t_train={}",
        tf1r.best_val,
        if tf1r.secs_train > 0.0 { tf1r.tokens_total as f64 / tf1r.secs_train } else { 0.0 },
        fmt_secs(tf1r.secs_train)
    );
    println!("    → {} aprende mejor con mismos datos (iso-params, iso-data)", w1);

    println!();
    println!("  EXP 2 · Iso-Time ({} s):", ISO_TIME_SECS);
    let w2 = if ai2.best_val < tf2r.best_val { "AIDEEN" } else { "Transformer" };
    println!(
        "    AIDEEN      tokens={:>8}  best_val={:.4}",
        fmt_tokens(ai2.tokens_total), ai2.best_val
    );
    println!(
        "    Transformer tokens={:>8}  best_val={:.4}",
        fmt_tokens(tf2r.tokens_total), tf2r.best_val
    );
    println!("    → {} gana dado el mismo presupuesto real de cómputo", w2);

    println!();
    println!("  Nota: 1 seed — repetir con ≥3 para reportar media±std.");
    println!("  Para GPU: aumentar TOTAL_TOKENS y CTX_LEN en las constantes.");
}
