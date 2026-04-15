//! Associative Recall Benchmark for AIDEEN.
//!
//! Tests whether FPM (Fixed Point Memory) retains key→value associations
//! across token gaps longer than the context window.
//!
//! ── Sequence format ───────────────────────────────────────────────────────────
//!   [KEY_i, VAL_i, FILLER×gap, KEY_i, QUERY, VAL_i]
//!
//! The model must predict VAL_i after KEY_i+QUERY, even though the KV pair
//! was established `gap` tokens ago — possibly many context windows away.
//!
//! ── Design ───────────────────────────────────────────────────────────────────
//! One trainer (one GPU init = one Metal shader compilation) is created and
//! trained on a MIXTURE of gap lengths.  After training, each gap length is
//! evaluated independently.  This isolates the memory contribution: the model
//! sees every gap at training time so we can compare answer-loss per gap.
//!
//! ── Metric ────────────────────────────────────────────────────────────────────
//!   answer_loss  = -log P(VAL_i | full_prefix…KEY_i…QUERY)
//!   random_base  = log(N_VALS) ≈ 2.77
//!   FPM should achieve answer_loss << random_base for gap > ctx_len;
//!   DEQ-only cannot (no cross-chunk memory) and should stay near random_base.
//!
//! ── Usage ─────────────────────────────────────────────────────────────────────
//!   # FPM (recommended):
//!   AIDEEN_NO_SLOT_ATTN=1 AIDEEN_CTX_LEN=128 \
//!     cargo run --release --features wgpu -p aideen-training --bin assoc_recall
//!
//!   # DEQ-only baseline:
//!   AIDEEN_DEQ_ONLY=1 AIDEEN_CTX_LEN=128 \
//!     cargo run --release --features wgpu -p aideen-training --bin assoc_recall
//!
//!   # Custom gaps (comma-separated) and training size:
//!   AR_GAPS=0,128,256,512 AR_N_TRAIN=800 AR_N_EVAL=100 \
//!   AIDEEN_NO_SLOT_ATTN=1 AIDEEN_CTX_LEN=128 \
//!     cargo run --release --features wgpu -p aideen-training --bin assoc_recall

use std::io::Write as IoWrite;
use std::time::Instant;

use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::state::ArchitectureConfig;
use aideen_training::trainer::Trainer;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ── Vocabulary (50 tokens) ────────────────────────────────────────────────────
//   1..=16  → KEY tokens
//  17..=32  → VAL tokens
//      33   → QUERY marker
//  34..=49  → FILLER (random noise during gap)
const N_KEYS: u32 = 16;
const N_VALS: u32 = 16;
const TOK_KEY_BASE: u32 = 1;
const TOK_VAL_BASE: u32 = 17;
const TOK_QUERY: u32 = 33;
const TOK_FILLER_BASE: u32 = 34;
const N_FILLER: u32 = 16;
const VOCAB_SIZE: usize = 50;

// ── Builds one AR sequence ────────────────────────────────────────────────────
fn make_ar_sequence(key: u32, val: u32, gap: usize, rng: &mut StdRng) -> Vec<u32> {
    // Format: KEY_i VAL_i FILLER×gap KEY_i QUERY VAL_i
    let mut seq = Vec::with_capacity(gap + 6);
    seq.push(TOK_KEY_BASE + key);
    seq.push(TOK_VAL_BASE + val);
    for _ in 0..gap {
        seq.push(TOK_FILLER_BASE + rng.gen_range(0..N_FILLER));
    }
    seq.push(TOK_KEY_BASE + key);
    seq.push(TOK_QUERY);
    seq.push(TOK_VAL_BASE + val); // answer token (last)
    seq
}

// ── Chunked forward: carry M across context-window boundaries ─────────────────

/// Trains on one AR sequence, chunked so M state carries across chunks.
/// Returns average cross-entropy over all predicted positions.
fn train_ar_sequence(trainer: &mut Trainer, seq: &[u32], ctx_len: usize, eps: f32) -> f32 {
    if seq.len() < 2 {
        return f32::NAN;
    }
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];

    let mut total_loss = 0.0f32;
    let mut n_chunks = 0usize;
    let mut first = true;
    let mut start = 0;

    while start < inputs.len() {
        let end = (start + ctx_len).min(inputs.len());
        let loss = trainer.train_sequence(&inputs[start..end], &targets[start..end], first, eps);
        first = false;
        if loss.is_finite() {
            total_loss += loss;
            n_chunks += 1;
        }
        start = end;
    }

    if n_chunks > 0 {
        total_loss / n_chunks as f32
    } else {
        f32::NAN
    }
}

/// Evaluates loss at ONLY the final answer token, carrying M state across chunks.
/// Processes all chunks except the last token with no weight updates (eval_mode).
/// Returns -log P(VAL_i | full_prefix…QUERY).
fn eval_answer_loss(trainer: &mut Trainer, seq: &[u32], ctx_len: usize, eps: f32) -> f32 {
    if seq.len() < 2 {
        return f32::NAN;
    }
    // inputs  = seq[0 .. len-1]   (everything except the answer)
    // targets = seq[1 .. len]     (shifted by 1)
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];
    let total_len = inputs.len(); // = seq.len() - 1

    // Process all positions before the final one in chunks (no weight update).
    let prefix_len = if total_len > 1 { total_len - 1 } else { 0 };
    let mut first = true;
    let mut start = 0;
    while start < prefix_len {
        let end = (start + ctx_len).min(prefix_len);
        trainer.train_sequence(&inputs[start..end], &targets[start..end], first, eps);
        first = false;
        start = end;
    }

    // Final step: input=QUERY, target=VAL_answer.
    // reset=first: true only if the whole prefix fit in zero chunks (gap=0 edge).
    trainer.train_sequence(
        &inputs[total_len - 1..total_len],
        &targets[total_len - 1..total_len],
        first, // carry M from previous chunks (false unless prefix was empty)
        eps,
    )
}

fn flush() {
    let _ = std::io::stdout().flush();
}

fn main() {
    let ctx_len: usize = std::env::var("AIDEEN_CTX_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    let gaps: Vec<usize> = std::env::var("AR_GAPS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0, 64, 128, 256]);

    let n_train: usize = std::env::var("AR_N_TRAIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(600);

    let n_eval: usize = std::env::var("AR_N_EVAL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);

    let mode_label = if std::env::var("AIDEEN_DEQ_ONLY").ok().as_deref() == Some("1") {
        "DEQ-only (no memory)"
    } else if std::env::var("AIDEEN_NO_SLOT_ATTN").ok().as_deref() == Some("1") {
        "FPM (no slot-attn)"
    } else {
        "FPM+SlotAttn"
    };

    let random_baseline = (N_VALS as f32).ln(); // log(16) ≈ 2.773

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              AIDEEN  Associative Recall Benchmark                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("  vocab={}  keys={}  vals={}", VOCAB_SIZE, N_KEYS, N_VALS);
    println!("  ctx_len={}  mode={}", ctx_len, mode_label);
    println!("  n_train={}  n_eval={}  gaps={:?}", n_train, n_eval, gaps);
    println!("  random_baseline = log({}) = {:.4}", N_VALS, random_baseline);
    println!();
    flush();

    // ── Build ONE trainer (one GPU init, one Metal shader compilation) ─────────
    let mut arch = ArchitectureConfig::default();
    arch.vocab_size = VOCAB_SIZE;
    arch.d_r = std::env::var("AIDEEN_D_MODEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512); // default 512 to hit the Metal pipeline cache from training runs
    arch.h_slots = std::env::var("AIDEEN_H_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    arch.ctx_len = ctx_len;
    arch.train_deq = true;
    arch.adj_iters = std::env::var("AIDEEN_ADJ_ITERS_OVERRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    arch.max_deq_iters = 12;
    // With small vocab, num_samples must be <= vocab_size.
    // Default is 512, which causes an infinite rejection-sampling loop when
    // vocab_size=50 (can never fill 512 unique indices from a pool of 50).
    // Set to vocab_size to get full softmax over all tokens.
    arch.num_samples = VOCAB_SIZE;

    println!("  d_r={}  h_slots={}", arch.d_r, arch.h_slots);
    println!("  Initializing GPU (Metal shader compilation may take a few minutes)…");
    flush();

    let mut tok = Tokenizer::new_empty(VOCAB_SIZE, arch);
    // new_empty initialises embeddings to zero → DEQ receives zero input → NaN.
    // Initialise with small random values (same scale as from_text / from_hf).
    {
        let d = tok.embeddings.ncols();
        let scale = (d as f32).sqrt().recip(); // 1/sqrt(d_r) ≈ 0.044 for d=512
        let mut rng_emb = StdRng::seed_from_u64(1234);
        for v in 0..VOCAB_SIZE {
            for j in 0..d {
                tok.embeddings[(v, j)] = rng_emb.gen_range(-scale..scale);
            }
        }
    }
    let mut trainer = Trainer::from_tokenizer_seeded(tok, 3e-4, 42);
    let eps = trainer.config.deq_epsilon;

    println!("  GPU ready.");
    flush();

    // ── Training: interleaved across all gap lengths ───────────────────────────
    // Each step draws a random gap from the configured set and trains one sequence.
    // This ensures the model learns to handle all gap lengths with one training run.
    println!("\n  Training ({} sequences, gaps={:?})…", n_train, gaps);
    flush();

    let t_train = Instant::now();
    let mut rng_train = StdRng::seed_from_u64(42);
    let mut train_loss_sum = 0.0f32;
    let mut train_valid = 0usize;
    let n_gaps = gaps.len();

    for i in 0..n_train {
        // Cycle through gaps: each n_gaps sequences covers every gap once.
        let gap = gaps[i % n_gaps];
        let key = rng_train.gen_range(0..N_KEYS);
        let val = rng_train.gen_range(0..N_VALS);
        let seq = make_ar_sequence(key, val, gap, &mut rng_train);
        let loss = train_ar_sequence(&mut trainer, &seq, ctx_len, eps);
        if loss.is_finite() {
            train_loss_sum += loss;
            train_valid += 1;
        }

        if (i + 1) % 100 == 0 {
            let avg = if train_valid > 0 {
                train_loss_sum / train_valid as f32
            } else {
                f32::NAN
            };
            println!(
                "  train {:>4}/{} | avg_loss={:.4} | {:.1}s",
                i + 1,
                n_train,
                avg,
                t_train.elapsed().as_secs_f32()
            );
            flush();
        }
    }

    let train_avg = if train_valid > 0 {
        train_loss_sum / train_valid as f32
    } else {
        f32::NAN
    };
    println!(
        "  Training done: avg_loss={:.4}  ({:.1}s total)\n",
        train_avg,
        t_train.elapsed().as_secs_f32()
    );
    flush();

    // ── Evaluation: per-gap answer-token loss ─────────────────────────────────
    trainer.eval_mode = true;
    trainer.frozen_deq = true;
    trainer.frozen_emb = true;
    trainer.frozen_lm = true;

    println!("  Evaluation (eval_mode, no weight updates):");
    println!(
        "  {:>8}  {:>12}  {:>12}  {}",
        "gap", "answer_loss", "random_base", "retention"
    );
    println!("  {}", "─".repeat(56));
    flush();

    for &gap in &gaps {
        let mut rng_eval = StdRng::seed_from_u64(9999 + gap as u64);
        let mut loss_sum = 0.0f32;
        let mut n_valid = 0usize;

        for _ in 0..n_eval {
            let key = rng_eval.gen_range(0..N_KEYS);
            let val = rng_eval.gen_range(0..N_VALS);
            let seq = make_ar_sequence(key, val, gap, &mut rng_eval);
            let aloss = eval_answer_loss(&mut trainer, &seq, ctx_len, eps);
            if aloss.is_finite() {
                loss_sum += aloss;
                n_valid += 1;
            }
        }

        let avg = if n_valid > 0 {
            loss_sum / n_valid as f32
        } else {
            f32::NAN
        };
        let pct = avg / random_baseline * 100.0;
        let label = if pct < 40.0 {
            "MEMORY WORKS ✓"
        } else if pct < 80.0 {
            "partial"
        } else {
            "~ random"
        };

        println!(
            "  {:>8}  {:>12.4}  {:>12.4}  {:.1}% {}",
            gap, avg, random_baseline, pct, label
        );
        flush();
    }

    println!("\n  Done.");
    flush();
}
