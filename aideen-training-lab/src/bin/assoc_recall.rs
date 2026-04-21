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
//!   random_base  = log(VOCAB_SIZE) because the LM head predicts over the full vocab
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

struct ArSequence {
    seq: Vec<u32>,
    query_key: u32,
    query_val: u32,
}

// ── Builds one AR sequence ────────────────────────────────────────────────────
fn make_ar_sequence(
    key: u32,
    val: u32,
    gap: usize,
    pairs_per_seq: usize,
    rng: &mut StdRng,
) -> ArSequence {
    let pairs_per_seq = pairs_per_seq.clamp(1, N_KEYS as usize);
    // Format:
    //   pairs=1: KEY_i VAL_i FILLER×gap KEY_i QUERY VAL_i
    //   pairs>1: (KEY_j VAL_j)*N FILLER×gap KEY_q QUERY VAL_q
    // The queried pair is sampled from the stored pairs so multi-pair tests measure
    // arbitrary binding lookup, not only retention of the first pair.
    let mut seq = Vec::with_capacity(gap + 4 + 2 * pairs_per_seq);
    let mut pairs = Vec::with_capacity(pairs_per_seq);
    pairs.push((key, val));
    let mut used_keys = vec![key];
    for _ in 1..pairs_per_seq {
        let mut extra_key = rng.gen_range(0..N_KEYS);
        while used_keys.contains(&extra_key) {
            extra_key = rng.gen_range(0..N_KEYS);
        }
        used_keys.push(extra_key);
        let extra_val = rng.gen_range(0..N_VALS);
        pairs.push((extra_key, extra_val));
    }
    for &(pair_key, pair_val) in &pairs {
        seq.push(TOK_KEY_BASE + pair_key);
        seq.push(TOK_VAL_BASE + pair_val);
    }
    for _ in 0..gap {
        seq.push(TOK_FILLER_BASE + rng.gen_range(0..N_FILLER));
    }
    let (query_key, query_val) = pairs[rng.gen_range(0..pairs.len())];
    seq.push(TOK_KEY_BASE + query_key);
    seq.push(TOK_QUERY);
    seq.push(TOK_VAL_BASE + query_val); // answer token (last)
    ArSequence { seq, query_key, query_val }
}

// ── Chunked forward: carry M across context-window boundaries ─────────────────

fn train_ar_sequence(trainer: &mut Trainer, seq: &[u32], ctx_len: usize, eps: f32) -> f32 {
    if seq.len() < 2 {
        return f32::NAN;
    }
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];
    let mut first = true;
    let mut start = 0;
    let total_len = inputs.len();
    let mut last_loss = f32::NAN;
    let full_cap = trainer.gpu_sequence_capacity();

    if total_len <= full_cap {
        let mut full_targets =
            vec![aideen_backbone::gpu_lm_head::GpuLmHeadTrainer::IGNORE_TARGET; total_len];
        if let Some(last) = full_targets.last_mut() {
            *last = targets[total_len - 1];
        }
        return trainer.train_sequence(inputs, &full_targets, true, eps);
    }

    // Train continuously across the sequence. Only the final answer token is
    // supervised so this benchmark measures associative recall instead of
    // next-token modeling on the prompt itself.
    while start < total_len {
        let end = (start + ctx_len).min(total_len);
        let mut chunk_targets = vec![aideen_backbone::gpu_lm_head::GpuLmHeadTrainer::IGNORE_TARGET; end - start];
        if end == total_len {
            if let Some(last) = chunk_targets.last_mut() {
                *last = targets[end - 1];
            }
        }
        last_loss = trainer.train_sequence(&inputs[start..end], &chunk_targets, first, eps);
        first = false;
        start = end;
    }

    last_loss
}

#[cfg(feature = "wgpu")]
fn train_ar_sequence_with_assoc_trace(
    trainer: &mut Trainer,
    seq: &[u32],
    ctx_len: usize,
    eps: f32,
) -> (f32, Vec<String>) {
    if seq.len() < 2 {
        return (f32::NAN, Vec::new());
    }
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];
    let mut first = true;
    let mut start = 0;
    let total_len = inputs.len();
    let mut last_loss = f32::NAN;
    let mut trace = Vec::new();
    let full_cap = trainer.gpu_sequence_capacity();

    if total_len <= full_cap {
        let mut full_targets =
            vec![aideen_backbone::gpu_lm_head::GpuLmHeadTrainer::IGNORE_TARGET; total_len];
        if let Some(last) = full_targets.last_mut() {
            *last = targets[total_len - 1];
        }
        last_loss = trainer.train_sequence(inputs, &full_targets, true, eps);
        if let Some(gpu) = trainer.gpu_deq.as_ref() {
            let fw = gpu.read_debug_buffer();
            trace.push(format_assoc_chunk_summary(
                trainer, &fw, 0, 0, total_len, true,
            ));
        }
        return (last_loss, trace);
    }

    while start < total_len {
        let end = (start + ctx_len).min(total_len);
        let mut chunk_targets = vec![aideen_backbone::gpu_lm_head::GpuLmHeadTrainer::IGNORE_TARGET; end - start];
        if end == total_len {
            if let Some(last) = chunk_targets.last_mut() {
                *last = targets[end - 1];
            }
        }
        last_loss = trainer.train_sequence(&inputs[start..end], &chunk_targets, first, eps);
        if let Some(gpu) = trainer.gpu_deq.as_ref() {
            let fw = gpu.read_debug_buffer();
            trace.push(format_assoc_chunk_summary(
                trainer,
                &fw,
                trace.len(),
                start,
                end,
                end == total_len,
            ));
        }
        first = false;
        start = end;
    }

    (last_loss, trace)
}

fn eval_answer_loss(trainer: &mut Trainer, seq: &[u32], ctx_len: usize, eps: f32) -> f32 {
    if seq.len() < 2 {
        return f32::NAN;
    }
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];
    let total_len = inputs.len();
    let full_cap = trainer.gpu_sequence_capacity();

    if total_len <= full_cap {
        let _ = trainer.train_sequence(inputs, targets, true, eps);
        return trainer.eval_cached_hpooled_token_loss(total_len - 1, targets[total_len - 1]);
    }

    let mut first = true;
    let mut start = 0;
    let mut answer_loss = f32::NAN;

    // Evaluate across the full sequence by naturally advancing the memory state M.
    // The previous token's converged h* is retained in PrevHStarBuf automatically.
    while start < total_len {
        let end = (start + ctx_len).min(total_len);
        let _ = trainer.train_sequence(&inputs[start..end], &targets[start..end], first, eps);
        if end == total_len {
            answer_loss = trainer.eval_cached_hpooled_token_loss(end - start - 1, targets[end - 1]);
        }
        first = false;
        start = end;
    }

    answer_loss
}

#[cfg(feature = "wgpu")]
fn eval_answer_loss_with_assoc_trace(
    trainer: &mut Trainer,
    seq: &[u32],
    ctx_len: usize,
    eps: f32,
) -> (f32, Vec<String>) {
    if seq.len() < 2 {
        return (f32::NAN, Vec::new());
    }
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];
    let total_len = inputs.len();
    let full_cap = trainer.gpu_sequence_capacity();

    let mut first = true;
    let mut start = 0;
    let mut answer_loss = f32::NAN;
    let mut trace = Vec::new();

    if total_len <= full_cap {
        let _ = trainer.train_sequence(inputs, targets, true, eps);
        answer_loss = trainer.eval_cached_hpooled_token_loss(total_len - 1, targets[total_len - 1]);
        if let Some(gpu) = trainer.gpu_deq.as_ref() {
            let fw = gpu.read_debug_buffer();
            trace.push(format_assoc_chunk_summary(
                trainer, &fw, 0, 0, total_len, true,
            ));
        }
        return (answer_loss, trace);
    }

    while start < total_len {
        let end = (start + ctx_len).min(total_len);
        let _ = trainer.train_sequence(&inputs[start..end], &targets[start..end], first, eps);
        if end == total_len {
            answer_loss = trainer.eval_cached_hpooled_token_loss(end - start - 1, targets[end - 1]);
        }
        if let Some(gpu) = trainer.gpu_deq.as_ref() {
            let fw = gpu.read_debug_buffer();
            trace.push(format_assoc_chunk_summary(
                trainer,
                &fw,
                trace.len(),
                start,
                end,
                end == total_len,
            ));
        }
        first = false;
        start = end;
    }

    (answer_loss, trace)
}

fn flush() {
    let _ = std::io::stdout().flush();
}

#[cfg(feature = "wgpu")]
fn format_lm_top_debug(trainer: &Trainer, final_token_index: usize, target: u32) -> String {
    let Some(gpu) = trainer.gpu_deq.as_ref() else {
        return "lm_top=NA".to_string();
    };
    let Some(gpu_lm) = trainer.gpu_lm.as_ref() else {
        return "lm_top=NA".to_string();
    };
    let Ok((w, b, g)) = gpu_lm.read_weights(&gpu.device, &gpu.queue) else {
        return "lm_top=NA".to_string();
    };
    let h = gpu.read_hpooled_at(final_token_index);
    let d = trainer.config.d_r;
    let vocab = trainer.config.vocab_size;
    if h.len() < d || w.len() < vocab * d || b.len() < vocab || g.len() < d {
        return "lm_top=bad-shape".to_string();
    }
    let h_rms = (h.iter().map(|x| x * x).sum::<f32>() / d.max(1) as f32 + 1e-5).sqrt();
    let mut logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        let mut acc = b[v];
        let row = v * d;
        for j in 0..d {
            acc += w[row + j] * (h[j] / h_rms) * g[j];
        }
        logits[v] = acc;
    }
    let mut top = (0usize, f32::NEG_INFINITY);
    let mut second = (0usize, f32::NEG_INFINITY);
    for (idx, &logit) in logits.iter().enumerate() {
        if logit > top.1 {
            second = top;
            top = (idx, logit);
        } else if logit > second.1 {
            second = (idx, logit);
        }
    }
    let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom = logits.iter().map(|x| (*x - mx).exp()).sum::<f32>().max(1e-20);
    let target_idx = target as usize;
    let target_prob = if target_idx < logits.len() {
        (logits[target_idx] - mx).exp() / denom
    } else {
        0.0
    };
    format!(
        "lm_top={} top_logit={:.3} second={} margin={:.3} target={} target_logit={:.3} target_p={:.3e}",
        top.0,
        top.1,
        second.0,
        top.1 - second.1,
        target,
        logits.get(target_idx).copied().unwrap_or(f32::NAN),
        target_prob,
    )
}

#[cfg(feature = "wgpu")]
fn format_solve_case_debug(trainer: &Trainer) -> String {
    let Some(gpu) = trainer.gpu_deq.as_ref() else {
        return "solve=NA".to_string();
    };
    let fw = gpu.read_debug_buffer();
    let h = trainer.config.h_slots;
    let mut max_h = 0.0f32;
    let mut max_delta = 0.0f32;
    let mut max_err = 0.0f32;
    let mut exit_err = 0.0f32;
    let mut fail_sum = 0.0f32;
    for slot in 0..h {
        let base = 32 + slot * 5;
        let diag = 400 + slot * 12;
        let exits = 688 + slot * 4;
        max_delta = max_delta.max(fw.get(base).copied().unwrap_or(0.0));
        max_h = max_h.max(fw.get(base + 4).copied().unwrap_or(0.0));
        max_err = max_err.max(fw.get(diag).copied().unwrap_or(0.0));
        exit_err = exit_err.max(fw.get(diag + 8).copied().unwrap_or(0.0));
        fail_sum += fw.get(exits + 2).copied().unwrap_or(0.0);
    }
    format!(
        "solve_h={:.3e} solve_delta={:.3e} err_h={:.3e} exit_err={:.3e} fail_sum={:.0} marker={:.0}",
        max_h,
        max_delta,
        max_err,
        exit_err,
        fail_sum,
        fw.get(8).copied().unwrap_or(0.0),
    )
}

fn rms(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let ss: f32 = v.iter().map(|x| x * x).sum();
    (ss / v.len() as f32).sqrt()
}

#[cfg(feature = "wgpu")]
fn format_assoc_chunk_summary(
    trainer: &Trainer,
    fw: &[f32],
    chunk_idx: usize,
    start: usize,
    end: usize,
    final_chunk: bool,
) -> String {
    let h = trainer.config.h_slots;
    let mut qmax = 0.0f32;
    let mut qctx = 0.0f32;
    let mut qusage = 0.0f32;
    let mut spread = 0.0f32;
    let mut bank_cos = 0.0f32;
    let mut qrms = 0.0f32;
    let mut wallow = 0.0f32;
    let mut wgate = 0.0f32;
    let mut hcos = 0.0f32;
    let mut n = 0.0f32;
    for slot in 0..h {
        let base = 760 + slot * 10;
        let qbase = 820 + slot * 8;
        let hbase = 860 + slot * 4;
        if fw.len() > qbase + 7 && fw.len() > base + 9 && fw.len() > hbase + 3 {
            qmax += fw[qbase + 1];
            qctx += fw[qbase + 2];
            qusage += fw[qbase + 3];
            spread += fw[qbase + 4];
            bank_cos += fw[qbase + 5];
            qrms += fw[qbase + 7] / fw[qbase + 6].max(1.0);
            wallow += fw[base + 4];
            wgate += fw[base + 5];
            hcos += fw[hbase] / fw[hbase + 3].max(1.0);
            n += 1.0;
        }
    }
    let den = n.max(1.0);
    format!(
        "chunk={} [{}..{}) final={} qmax={:.3e} qctx={:.3e} usage={:.3e} spread={:.3e} bank_cos={:.3e} q_rms={:.3e} w_allow={:.3e} w_gate={:.3e} hcos={:.3e}",
        chunk_idx,
        start,
        end,
        final_chunk,
        qmax / den,
        qctx / den,
        qusage / den,
        spread / den,
        bank_cos / den,
        qrms / den,
        wallow / den,
        wgate / den,
        hcos / den,
    )
}

#[cfg(feature = "wgpu")]
fn format_assoc_forward_debug(
    trainer: &Trainer,
    fw: &[f32],
    label: &str,
    loss: f32,
) -> String {
    let h = trainer.config.h_slots;
    let mut parts = Vec::with_capacity(h);
    let mut query_parts = Vec::with_capacity(h);
    let mut h_parts = Vec::with_capacity(h);
    for slot in 0..h {
        let base = 760 + slot * 10;
        if fw.len() > base + 9 {
            parts.push(format!(
                "s{}:r_ent={:.3e},r_max={:.3e},ctx={:.3e},r_usage={:.3e},w_allow={:.3e},w_gate={:.3e},cos={:.3e},min_usage={:.3e},r_n={:.0},w_n={:.0}",
                slot,
                fw[base],
                fw[base + 1],
                fw[base + 2],
                fw[base + 3],
                fw[base + 4],
                fw[base + 5],
                fw[base + 6],
                fw[base + 7],
                fw[base + 8],
                fw[base + 9],
            ));
        }
        let qbase = 820 + slot * 8;
        if fw.len() > qbase + 7 {
            query_parts.push(format!(
                "s{}:q_ent={:.3e},q_max={:.3e},q_ctx={:.3e},q_usage={:.3e},q_spread={:.3e},bank_cos={:.3e},q_rms={:.3e},q_n={:.0}",
                slot,
                fw[qbase],
                fw[qbase + 1],
                fw[qbase + 2],
                fw[qbase + 3],
                fw[qbase + 4],
                fw[qbase + 5],
                fw[qbase + 7] / fw[qbase + 6].max(1.0),
                fw[qbase + 6],
            ));
        }
        let hbase = 860 + slot * 4;
        if fw.len() > hbase + 3 {
            let n = fw[hbase + 3].max(1.0);
            h_parts.push(format!(
                "s{}:hcos_avg={:.3e},hcos_max={:.3e},hcos_min={:.3e},h_n={:.0}",
                slot,
                fw[hbase] / n,
                fw[hbase + 1],
                fw[hbase + 2],
                fw[hbase + 3],
            ));
        }
    }
    format!(
        "{} loss={:.4} all=[{}] query=[{}] hsep=[{}]",
        label,
        loss,
        parts.join(" | "),
        query_parts.join(" | "),
        h_parts.join(" | ")
    )
}

fn main() {
    let assoc_only = std::env::var("AR_ASSOC_ONLY").ok().as_deref() == Some("1");
    let assoc_audit = std::env::var("AR_AUDIT").ok().as_deref() == Some("1");
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
    let pairs_per_seq: usize = std::env::var("AR_PAIRS_PER_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .clamp(1, N_KEYS as usize);
    let required_seq_cap = gaps
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        + 2 * pairs_per_seq
        + 2;
    if std::env::var("AIDEEN_SEQ_CAP").is_err() {
        unsafe {
            std::env::set_var("AIDEEN_SEQ_CAP", required_seq_cap.max(ctx_len).to_string());
        }
    }

    let mode_label = if assoc_only {
        "Assoc-only (no slot-attn, no continuous FPM)"
    } else if std::env::var("AIDEEN_DEQ_ONLY").ok().as_deref() == Some("1") {
        "DEQ-only (no memory)"
    } else if std::env::var("AIDEEN_NO_SLOT_ATTN").ok().as_deref() == Some("1") {
        "FPM (no slot-attn)"
    } else {
        "FPM+SlotAttn"
    };

    let random_baseline = (VOCAB_SIZE as f32).ln(); // full-vocab LM baseline

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              AIDEEN  Associative Recall Benchmark                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("  vocab={}  keys={}  vals={}", VOCAB_SIZE, N_KEYS, N_VALS);
    println!("  ctx_len={}  mode={}", ctx_len, mode_label);
    println!(
        "  n_train={}  n_eval={}  gaps={:?}  pairs_per_seq={}",
        n_train, n_eval, gaps, pairs_per_seq
    );
    println!("  random_baseline = log(vocab={}) = {:.4}", VOCAB_SIZE, random_baseline);
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
    arch.max_deq_iters = std::env::var("AIDEEN_MAX_DEQ_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
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
    let lr: f64 = std::env::var("AR_LR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-4);
    let mut trainer = Trainer::from_tokenizer_seeded(tok, lr as f32, 42);
    // Experimental only: tying the decoder to embeddings changes what the
    // benchmark measures, so it must not be the default memory baseline.
    if std::env::var("AR_TIE_LM_TO_EMB")
        .ok()
        .as_deref()
        .unwrap_or("0")
        != "0"
    {
        trainer.lm_head.w = trainer.tokenizer.embeddings.clone();
        trainer.frozen_emb = true;
    }
    if let Ok(v) = std::env::var("AIDEEN_DAMPING").and_then(|s| {
        s.parse::<f32>()
            .map_err(|_| std::env::VarError::NotPresent)
    }) {
        trainer.adaptive_damping = v;
        trainer.reasoning.damping = v;
    }
    // Associative memory needs write-enabled stages; stage 3 reads but does not write.
    // Force the benchmark onto the first coherent stage for memory learning unless
    // the user explicitly disabled the memory path.
    #[cfg(feature = "wgpu")]
    if let Some(gpu) = trainer.gpu_deq.as_mut() {
        if gpu.cached_residual_alpha > -1.5 {
            gpu.cached_fpm_stage = gpu.cached_fpm_stage.max(4);
        }
    }
    if let Some(alpha_assoc_override) = std::env::var("AR_ASSOC_ALPHA")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
    {
        trainer.reasoning.alpha_assoc.fill(alpha_assoc_override);
    } else if assoc_only {
        trainer.reasoning.alpha_assoc.fill(0.3);
    }
    if assoc_only {
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = trainer.gpu_deq.as_mut() {
            gpu.cached_fpm_alpha_m = 0.0;
            gpu.cached_fpm_stage = gpu.cached_fpm_stage.max(4);
        }
    }
    // Always read GPU loss after each train_sequence call so avg_loss is meaningful.
    trainer.cfg_loss_readback_every = 1;
    let eps = trainer.config.deq_epsilon;
    let flatten_mats = |mats: &[nalgebra::DMatrix<f32>]| -> Vec<f32> {
        mats.iter()
            .flat_map(|m| m.as_slice().iter().copied())
            .collect()
    };
    let init_alpha_assoc = trainer.reasoning.alpha_assoc.as_slice().to_vec();
    let init_wk_write: Vec<f32> = trainer.reasoning.w_k_write.as_slice().to_vec();
    let init_wk_assoc = flatten_mats(&trainer.reasoning.w_k_assoc);
    let init_wq_assoc = flatten_mats(&trainer.reasoning.w_q_assoc);

    println!("  GPU ready.");
    flush();

    #[cfg(feature = "wgpu")]
    if assoc_audit {
        trainer.enable_temporary_assoc_debug_sampling(1);
    }

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
    let mut last_train_case: Option<(usize, usize, ArSequence, f32, Vec<String>)> = None;

    for i in 0..n_train {
        // Cycle through gaps: each n_gaps sequences covers every gap once.
        let gap = gaps[i % n_gaps];
        let key = rng_train.gen_range(0..N_KEYS);
        let val = rng_train.gen_range(0..N_VALS);
        let ar = make_ar_sequence(key, val, gap, pairs_per_seq, &mut rng_train);
        #[cfg(feature = "wgpu")]
        let (loss, train_trace) = if assoc_audit {
            train_ar_sequence_with_assoc_trace(&mut trainer, &ar.seq, ctx_len, eps)
        } else {
            (train_ar_sequence(&mut trainer, &ar.seq, ctx_len, eps), Vec::new())
        };
        #[cfg(not(feature = "wgpu"))]
        let (loss, train_trace) = (train_ar_sequence(&mut trainer, &ar.seq, ctx_len, eps), Vec::new());
        if loss.is_finite() {
            train_loss_sum += loss;
            train_valid += 1;
        }
        if assoc_audit {
            last_train_case = Some((i, gap, ar, loss, train_trace));
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

    if assoc_audit {
        #[cfg(feature = "wgpu")]
        if let Some((train_idx, train_gap, train_ar, train_loss, train_trace)) =
            last_train_case.as_ref()
        {
            println!(
                "  [assoc-train-last] idx={} gap={} key={} val={} loss={:.4} trace=[{}]",
                train_idx,
                train_gap,
                train_ar.query_key,
                train_ar.query_val,
                train_loss,
                train_trace.join(" || ")
            );
        }
        #[cfg(feature = "wgpu")]
        trainer.sync_inference_weights();
        let alpha_now = trainer.reasoning.alpha_assoc.as_slice().to_vec();
        let wk_write_now: Vec<f32> = trainer.reasoning.w_k_write.as_slice().to_vec();
        let wk_assoc_now = flatten_mats(&trainer.reasoning.w_k_assoc);
        let wq_assoc_now = flatten_mats(&trainer.reasoning.w_q_assoc);
        let delta_rms = |a: &[f32], b: &[f32]| -> f32 {
            if a.len() != b.len() || a.is_empty() {
                return f32::NAN;
            }
            let ss: f32 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum();
            (ss / a.len() as f32).sqrt()
        };
        println!(
            "  [assoc-audit] decoupled=1 alpha(mean={:.4}, delta_rms={:.4e}) gate_key=W_k_write(rms={:.4e}, delta_rms={:.4e}) Wk_assoc(delta={:.4e}) Wq_assoc(delta={:.4e}) value=direct_signal",
            alpha_now.iter().copied().sum::<f32>() / alpha_now.len().max(1) as f32,
            delta_rms(&init_alpha_assoc, &alpha_now),
            rms(&wk_write_now),
            delta_rms(&init_wk_write, &wk_write_now),
            delta_rms(&init_wk_assoc, &wk_assoc_now),
            delta_rms(&init_wq_assoc, &wq_assoc_now),
        );
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = trainer.gpu_deq.as_ref() {
            let assoc = gpu.read_assoc_state();
            let d = trainer.config.d_r;
            let h = trainer.config.h_slots;
            let rank = 32usize;
            let banks = std::env::var("AR_ASSOC_BANKS_AUDIT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1usize);
            let bank_stride = rank + d + 1;
            let slot_stride = banks * bank_stride;
            let mut key_ss = 0.0f32;
            let mut key_n = 0usize;
            let mut val_ss = 0.0f32;
            let mut val_n = 0usize;
            let mut usage_sum = 0.0f32;
            let mut usage_max = 0.0f32;
            let mut usage_n = 0usize;
            for slot in 0..h {
                let slot_base = slot * slot_stride;
                for bank in 0..banks {
                    let base = slot_base + bank * bank_stride;
                    for &x in assoc.get(base..base + rank).unwrap_or(&[]) {
                        key_ss += x * x;
                        key_n += 1;
                    }
                    for &x in assoc.get(base + rank..base + rank + d).unwrap_or(&[]) {
                        val_ss += x * x;
                        val_n += 1;
                    }
                    if let Some(&usage) = assoc.get(base + rank + d) {
                        usage_sum += usage;
                        usage_max = usage_max.max(usage);
                        usage_n += 1;
                    }
                }
            }
            println!(
                "  [assoc-state] key_rms={:.4e} value_rms={:.4e} usage_mean={:.4e} usage_max={:.4e}",
                (key_ss / key_n.max(1) as f32).sqrt(),
                (val_ss / val_n.max(1) as f32).sqrt(),
                usage_sum / usage_n.max(1) as f32,
                usage_max,
            );
            // TEMPORARY ASSOCIATIVE DIAGNOSTIC: remove after backward learning path is localized.
            let bwd = gpu.read_assoc_bwd_debug();
            let mut bwd_parts = Vec::with_capacity(h);
            for slot in 0..h {
                let base = slot * 16;
                if bwd.len() <= base + 14 {
                    continue;
                }
                let v_den = bwd[base + 1].max(1.0);
                bwd_parts.push(format!(
                    "s{}:v_rms_avg={:.3e},score_abs_sum={:.3e},score_abs_max={:.3e},wq_step_sum={:.3e},wq_step_max={:.3e},wk_step_sum={:.3e},wk_step_max={:.3e},alpha_step_sum={:.3e},alpha_step_max={:.3e},key_grad_sum={:.3e},gprev_sum={:.3e},bind_sum={:.3e}",
                    slot,
                    bwd[base] / v_den,
                    bwd[base + 4],
                    bwd[base + 5],
                    bwd[base + 6],
                    bwd[base + 7],
                    bwd[base + 8],
                    bwd[base + 9],
                    bwd[base + 10],
                    bwd[base + 11],
                    bwd[base + 12],
                    bwd[base + 13],
                    bwd[base + 14],
                ));
            }
            println!("  [assoc-bwd-debug] {}", bwd_parts.join(" | "));
        }
        // TEMPORARY ASSOCIATIVE DIAGNOSTIC: remove this forced debug probe after root cause is closed.
        #[cfg(feature = "wgpu")]
        {
            trainer.enable_temporary_assoc_debug_sampling(1);
            let mut rng_diag = StdRng::seed_from_u64(777);
            let diag_ar = make_ar_sequence(0, 0, 0, pairs_per_seq, &mut rng_diag);
            trainer.eval_mode = true;
            trainer.frozen_deq = true;
            let diag_loss = eval_answer_loss(&mut trainer, &diag_ar.seq, ctx_len, eps);
            if let Some(gpu) = trainer.gpu_deq.as_ref() {
                let fw = gpu.read_debug_buffer();
                println!(
                    "  [assoc-forward-debug] {}",
                    format_assoc_forward_debug(&trainer, &fw, "diag_gap0", diag_loss)
                );
            }
            trainer.eval_mode = false;
            trainer.frozen_deq = false;
        }
        flush();
    }

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
        let mut gap_debug_line: Option<String> = None;

        for eval_idx in 0..n_eval {
            let key = rng_eval.gen_range(0..N_KEYS);
            let val = rng_eval.gen_range(0..N_VALS);
            let ar = make_ar_sequence(key, val, gap, pairs_per_seq, &mut rng_eval);
            let seq = &ar.seq;
            let mut case_trace: Option<Vec<String>> = None;
            #[cfg(feature = "wgpu")]
            let aloss = if assoc_audit {
                let (loss, trace) = eval_answer_loss_with_assoc_trace(&mut trainer, &seq, ctx_len, eps);
                case_trace = Some(trace);
                if let Some(gpu) = trainer.gpu_deq.as_ref() {
                    let fw = gpu.read_debug_buffer();
                    if eval_idx == 0 {
                        gap_debug_line = Some(format!(
                            "{} trace=[{}]",
                            format_assoc_forward_debug(
                                &trainer,
                                &fw,
                                &format!(
                                    "gap={} key={} val={} len={}",
                                    gap,
                                    ar.query_key,
                                    ar.query_val,
                                    seq.len()
                                ),
                                loss,
                            ),
                            case_trace.as_ref().map(|t| t.join(" || ")).unwrap_or_default()
                        ));
                    }
                }
                loss
            } else {
                eval_answer_loss(&mut trainer, &seq, ctx_len, eps)
            };
            #[cfg(not(feature = "wgpu"))]
            let aloss = eval_answer_loss(&mut trainer, &seq, ctx_len, eps);
            if aloss.is_finite() {
                loss_sum += aloss;
                n_valid += 1;
            }
            // TEMPORARY ASSOCIATIVE DIAGNOSTIC: remove once per-key/per-value failure distribution is closed.
            if assoc_audit {
                #[cfg(feature = "wgpu")]
                let lm_top =
                    format_lm_top_debug(&trainer, (seq.len() - 2) % ctx_len, TOK_VAL_BASE + ar.query_val);
                #[cfg(not(feature = "wgpu"))]
                let lm_top = "lm_top=NA".to_string();
                #[cfg(feature = "wgpu")]
                let solve_dbg = format_solve_case_debug(&trainer);
                #[cfg(not(feature = "wgpu"))]
                let solve_dbg = "solve=NA".to_string();
                println!(
                    "  [assoc-eval-case] gap={} idx={} key={} val={} loss={:.4} {} {} trace=[{}]",
                    gap,
                    eval_idx,
                    ar.query_key,
                    ar.query_val,
                    aloss,
                    lm_top,
                    solve_dbg,
                    case_trace.as_ref().map(|t| t.join(" || ")).unwrap_or_default()
                );
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
        if let Some(line) = gap_debug_line {
            println!("  [assoc-gap-debug] {}", line);
        }
        flush();
    }

    println!("\n  Done.");
    flush();
}
