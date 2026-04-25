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

// ── Train/eval full sequence when GPU capacity fits; chunk only as fallback ───

fn run_sequence_chunks<F>(total_len: usize, ctx_len: usize, full_cap: usize, mut f: F)
where
    F: FnMut(bool, usize, usize, bool),
{
    if total_len <= full_cap {
        f(true, 0, total_len, true);
        return;
    }

    let mut first = true;
    let mut start = 0;
    while start < total_len {
        let end = (start + ctx_len).min(total_len);
        let final_chunk = end == total_len;
        f(first, start, end, final_chunk);
        first = false;
        start = end;
    }
}

fn assoc_local_segment_mode() -> bool {
    std::env::var("AR_ASSOC_LOCAL_SEGMENTS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn internal_segment_recurrence_mode() -> bool {
    assoc_local_segment_mode()
        && std::env::var("AIDEEN_SEGMENT_MEMORY_TOKEN")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
}

fn assoc_segment_len(ctx_len: usize) -> Option<usize> {
    if !assoc_local_segment_mode() {
        return None;
    }
    let seg_len = std::env::var("AR_SEGMENT_LEN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(ctx_len.max(1));
    Some(seg_len.max(1))
}

fn segment_dense_train_mode() -> bool {
    std::env::var("AR_SEGMENT_DENSE_TRAIN")
        .ok()
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or_else(|| assoc_local_segment_mode())
}

fn effective_full_cap(trainer: &Trainer, ctx_len: usize) -> usize {
    if internal_segment_recurrence_mode() {
        trainer.gpu_sequence_capacity()
    } else {
        assoc_segment_len(ctx_len).unwrap_or_else(|| trainer.gpu_sequence_capacity())
    }
}

fn final_cached_token_index(total_len: usize, ctx_len: usize, full_cap: usize) -> usize {
    if total_len <= full_cap {
        return total_len.saturating_sub(1);
    }
    total_len.saturating_sub(1) % ctx_len
}

fn train_ar_sequence(trainer: &mut Trainer, seq: &[u32], ctx_len: usize, eps: f32) -> f32 {
    if seq.len() < 2 {
        return f32::NAN;
    }
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];
    let total_len = inputs.len();
    let full_cap = effective_full_cap(trainer, ctx_len);
    let mut last_loss = f32::NAN;
    run_sequence_chunks(total_len, ctx_len, full_cap, |first, start, end, _final_chunk| {
        if !first && assoc_local_segment_mode() {
            trainer.reset_local_segment_state();
        }
        let mut chunk_targets = if segment_dense_train_mode() {
            targets[start..end].to_vec()
        } else {
            vec![aideen_backbone::gpu_lm_head::GpuLmHeadTrainer::IGNORE_TARGET; end - start]
        };
        if !segment_dense_train_mode() && end == total_len {
            if let Some(last) = chunk_targets.last_mut() {
                *last = targets[end - 1];
            }
        }
        last_loss = trainer.train_sequence(&inputs[start..end], &chunk_targets, first, eps);
    });

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
    let total_len = inputs.len();
    let full_cap = effective_full_cap(trainer, ctx_len);
    let mut last_loss = f32::NAN;
    let mut trace = Vec::new();
    run_sequence_chunks(total_len, ctx_len, full_cap, |first, start, end, final_chunk| {
        if !first && assoc_local_segment_mode() {
            trainer.reset_local_segment_state();
        }
        let mut chunk_targets = if segment_dense_train_mode() {
            targets[start..end].to_vec()
        } else {
            vec![aideen_backbone::gpu_lm_head::GpuLmHeadTrainer::IGNORE_TARGET; end - start]
        };
        if !segment_dense_train_mode() && final_chunk {
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
                final_chunk,
            ));
        }
    });

    (last_loss, trace)
}

fn eval_answer_loss(trainer: &mut Trainer, seq: &[u32], ctx_len: usize, eps: f32) -> f32 {
    if seq.len() < 2 {
        return f32::NAN;
    }
    let inputs = &seq[..seq.len() - 1];
    let targets = &seq[1..];
    let total_len = inputs.len();
    let full_cap = effective_full_cap(trainer, ctx_len);
    let mut answer_loss = f32::NAN;
    run_sequence_chunks(total_len, ctx_len, full_cap, |first, start, end, final_chunk| {
        if !first && assoc_local_segment_mode() {
            trainer.reset_local_segment_state();
        }
        let _ = trainer.train_sequence(&inputs[start..end], &targets[start..end], first, eps);
        if final_chunk {
            answer_loss = trainer.eval_cached_hpooled_token_loss(end - start - 1, targets[end - 1]);
        }
    });

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
    let full_cap = effective_full_cap(trainer, ctx_len);
    let mut answer_loss = f32::NAN;
    let mut trace = Vec::new();
    run_sequence_chunks(total_len, ctx_len, full_cap, |first, start, end, final_chunk| {
        if !first && assoc_local_segment_mode() {
            trainer.reset_local_segment_state();
        }
        let _ = trainer.train_sequence(&inputs[start..end], &targets[start..end], first, eps);
        if final_chunk {
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
                final_chunk,
            ));
        }
    });

    (answer_loss, trace)
}

fn flush() {
    let _ = std::io::stdout().flush();
}

#[cfg(feature = "wgpu")]
fn format_lm_top_debug(
    trainer: &Trainer,
    final_token_index: usize,
    query_key_tok: u32,
    target: u32,
) -> String {
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
    let h_slots_flat = gpu.read_hnext_seq((final_token_index + 1) as u32);
    let assoc_state = gpu.read_assoc_state();
    let debug = gpu.read_debug_buffer();
    let d = trainer.config.d_r;
    let h_slots = trainer.config.h_slots;
    let vocab = trainer.config.vocab_size;
    let assoc_banks = std::env::var("AIDEEN_ASSOC_BANKS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    const ASSOC_RANK: usize = 32;
    let assoc_bank_stride = ASSOC_RANK + d + 1;
    let assoc_slot_stride = assoc_banks * assoc_bank_stride;
    if h.len() < d
        || h_slots_flat.len() < (final_token_index + 1) * h_slots * d
        || assoc_state.len() < h_slots * assoc_slot_stride
        || w.len() < vocab * d
        || b.len() < vocab
        || g.len() < d
    {
        return "lm_top=bad-shape".to_string();
    }
    let compute_logits = |h_vec: &[f32]| -> Vec<f32> {
        let h_rms = (h_vec.iter().map(|x| x * x).sum::<f32>() / d.max(1) as f32 + 1e-5).sqrt();
        let mut logits = vec![0.0f32; vocab];
        for v in 0..vocab {
            let mut acc = b[v];
            let row = v * d;
            for j in 0..d {
                acc += w[row + j] * (h_vec[j] / h_rms) * g[j];
            }
            logits[v] = acc;
        }
        logits
    };
    let compute_embed_logits = |h_vec: &[f32]| -> Vec<f32> {
        let h_rms = (h_vec.iter().map(|x| x * x).sum::<f32>() / d.max(1) as f32 + 1e-5).sqrt();
        let mut logits = vec![0.0f32; trainer.tokenizer.vocab_size()];
        for v in 0..trainer.tokenizer.vocab_size() {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += trainer.tokenizer.embeddings[(v, j)] * (h_vec[j] / h_rms);
            }
            logits[v] = acc;
        }
        logits
    };
    let emb_cos = |h_vec: &[f32], tok: u32| -> f32 {
        let tok_idx = tok as usize;
        if tok_idx >= trainer.tokenizer.embeddings.nrows() {
            return f32::NAN;
        }
        let mut dot = 0.0f32;
        let mut h_norm = 0.0f32;
        let mut e_norm = 0.0f32;
        for j in 0..d {
            let hv = h_vec[j];
            let ev = trainer.tokenizer.embeddings[(tok_idx, j)];
            dot += hv * ev;
            h_norm += hv * hv;
            e_norm += ev * ev;
        }
        dot / (h_norm * e_norm).max(1e-12).sqrt()
    };
    let logits = compute_logits(&h[..d]);
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
    let token_base = final_token_index * h_slots * d;
    let mut best_slot = 0usize;
    let mut best_slot_target_prob = f32::NEG_INFINITY;
    let mut best_slot_target_logit = f32::NEG_INFINITY;
    for slot in 0..h_slots {
        let slot_base = token_base + slot * d;
        let slot_logits = compute_logits(&h_slots_flat[slot_base..slot_base + d]);
        let slot_mx = slot_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let slot_denom = slot_logits
            .iter()
            .map(|x| (*x - slot_mx).exp())
            .sum::<f32>()
            .max(1e-20);
        let slot_target_prob = if target_idx < slot_logits.len() {
            (slot_logits[target_idx] - slot_mx).exp() / slot_denom
        } else {
            0.0
        };
        if slot_target_prob > best_slot_target_prob {
            best_slot = slot;
            best_slot_target_prob = slot_target_prob;
            best_slot_target_logit = slot_logits.get(target_idx).copied().unwrap_or(f32::NAN);
        }
    }
    let mut best_bank_slot = 0usize;
    let mut best_bank = 0usize;
    let mut best_bank_target_prob = f32::NEG_INFINITY;
    let mut best_bank_target_logit = f32::NEG_INFINITY;
    let mut best_bank_embed_target_prob = f32::NEG_INFINITY;
    let mut best_bank_embed_target_logit = f32::NEG_INFINITY;
    let mut best_bank_target_cos = f32::NEG_INFINITY;
    let mut best_bank_key_cos = f32::NEG_INFINITY;
    let mut best_bank_query_cos = f32::NEG_INFINITY;
    let mut best_bank_late_fuse_target_prob = f32::NEG_INFINITY;
    let mut best_bank_late_fuse_target_logit = f32::NEG_INFINITY;
    let mut chosen_read_slot = 0usize;
    let mut chosen_read_bank = 0usize;
    let mut chosen_read_prob = f32::NEG_INFINITY;
    let mut chosen_read_target_prob = f32::NEG_INFINITY;
    let mut chosen_read_target_logit = f32::NEG_INFINITY;
    for slot in 0..h_slots {
        let query_pick_base = 940 + slot * 2;
        let slot_read_bank = debug
            .get(query_pick_base)
            .copied()
            .unwrap_or(0.0)
            .max(0.0) as usize;
        let slot_read_prob = debug
            .get(query_pick_base + 1)
            .copied()
            .unwrap_or(f32::NEG_INFINITY);
        let slot_base = slot * assoc_slot_stride;
        for bank in 0..assoc_banks {
            let bank_base = slot_base + bank * assoc_bank_stride;
            let value_base = bank_base + ASSOC_RANK;
            let usage = assoc_state[bank_base + ASSOC_RANK + d];
            if usage <= 0.0 {
                continue;
            }
            let bank_value = &assoc_state[value_base..value_base + d];
            let bank_logits = compute_logits(bank_value);
            let bank_mx = bank_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let bank_denom = bank_logits
                .iter()
                .map(|x| (*x - bank_mx).exp())
                .sum::<f32>()
                .max(1e-20);
            let bank_target_prob = if target_idx < bank_logits.len() {
                (bank_logits[target_idx] - bank_mx).exp() / bank_denom
            } else {
                0.0
            };
            let bank_embed_logits = compute_embed_logits(bank_value);
            let bank_embed_mx = bank_embed_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let bank_embed_denom = bank_embed_logits
                .iter()
                .map(|x| (*x - bank_embed_mx).exp())
                .sum::<f32>()
                .max(1e-20);
            let bank_embed_target_prob = if target_idx < bank_embed_logits.len() {
                (bank_embed_logits[target_idx] - bank_embed_mx).exp() / bank_embed_denom
            } else {
                0.0
            };
            if bank == slot_read_bank && slot_read_prob > chosen_read_prob {
                chosen_read_slot = slot;
                chosen_read_bank = bank;
                chosen_read_prob = slot_read_prob;
                chosen_read_target_prob = bank_target_prob;
                chosen_read_target_logit =
                    bank_logits.get(target_idx).copied().unwrap_or(f32::NAN);
            }
            if bank_target_prob > best_bank_target_prob {
                best_bank_slot = slot;
                best_bank = bank;
                best_bank_target_prob = bank_target_prob;
                best_bank_target_logit = bank_logits.get(target_idx).copied().unwrap_or(f32::NAN);
                best_bank_embed_target_prob = bank_embed_target_prob;
                best_bank_embed_target_logit = bank_embed_logits
                    .get(target_idx)
                    .copied()
                    .unwrap_or(f32::NAN);
                best_bank_target_cos = emb_cos(bank_value, target);
                best_bank_key_cos = emb_cos(bank_value, query_key_tok);
                best_bank_query_cos = emb_cos(bank_value, TOK_QUERY);
                let mut fused = h[..d].to_vec();
                for j in 0..d {
                    fused[j] += bank_value[j];
                }
                let fused_logits = compute_logits(&fused);
                let fused_mx = fused_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let fused_denom = fused_logits
                    .iter()
                    .map(|x| (*x - fused_mx).exp())
                    .sum::<f32>()
                    .max(1e-20);
                best_bank_late_fuse_target_prob = if target_idx < fused_logits.len() {
                    (fused_logits[target_idx] - fused_mx).exp() / fused_denom
                } else {
                    0.0
                };
                best_bank_late_fuse_target_logit =
                    fused_logits.get(target_idx).copied().unwrap_or(f32::NAN);
            }
        }
    }
    format!(
        "lm_top={} top_logit={:.3} second={} margin={:.3} target={} target_logit={:.3} target_p={:.3e} best_slot={} best_slot_target_logit={:.3} best_slot_target_p={:.3e} chosen_read_slot={} chosen_read_bank={} chosen_read_prob={:.3e} chosen_read_target_logit={:.3} chosen_read_target_p={:.3e} best_bank_slot={} best_bank={} best_bank_target_logit={:.3} best_bank_target_p={:.3e} best_bank_embed_target_logit={:.3} best_bank_embed_target_p={:.3e} best_bank_target_cos={:.3e} best_bank_key_cos={:.3e} best_bank_query_cos={:.3e} late_fuse_target_logit={:.3} late_fuse_target_p={:.3e}",
        top.0,
        top.1,
        second.0,
        top.1 - second.1,
        target,
        logits.get(target_idx).copied().unwrap_or(f32::NAN),
        target_prob,
        best_slot,
        best_slot_target_logit,
        best_slot_target_prob,
        chosen_read_slot,
        chosen_read_bank,
        chosen_read_prob,
        chosen_read_target_logit,
        chosen_read_target_prob,
        best_bank_slot,
        best_bank,
        best_bank_target_logit,
        best_bank_target_prob,
        best_bank_embed_target_logit,
        best_bank_embed_target_prob,
        best_bank_target_cos,
        best_bank_key_cos,
        best_bank_query_cos,
        best_bank_late_fuse_target_logit,
        best_bank_late_fuse_target_prob,
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
    let mut prev_rms = 0.0f32;
    let mut kpre_rms = 0.0f32;
    let mut wkcand = 0.0f32;
    let mut wvcand = 0.0f32;
    let mut overwrite = 0.0f32;
    let mut n = 0.0f32;
    for slot in 0..h {
        let base = 760 + slot * 10;
        let qbase = 820 + slot * 8;
        let wbase = 900 + slot * 6;
        if fw.len() > qbase + 7 && fw.len() > base + 9 && fw.len() > wbase + 5 {
            qmax += fw[qbase + 1];
            qctx += fw[qbase + 2];
            qusage += fw[qbase + 3];
            spread += fw[qbase + 4];
            bank_cos += fw[qbase + 5];
            qrms += fw[qbase + 7] / fw[qbase + 6].max(1.0);
            wallow += fw[base + 4];
            wgate += fw[base + 5];
            prev_rms += fw[wbase + 3];
            kpre_rms += fw[wbase + 4];
            wkcand += fw[wbase + 0];
            wvcand += fw[wbase + 1];
            overwrite += fw[wbase + 5];
            n += 1.0;
        }
    }
    let den = n.max(1.0);
    format!(
        "chunk={} [{}..{}) final={} qmax={:.3e} qctx={:.3e} usage={:.3e} spread={:.3e} bank_cos={:.3e} q_rms={:.3e} w_allow={:.3e} w_gate={:.3e} prev_rms={:.3e} k_pre={:.3e} k_cand={:.3e} v_cand={:.3e} overwrite={:.3e}",
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
        prev_rms / den,
        kpre_rms / den,
        wkcand / den,
        wvcand / den,
        overwrite / den,
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
        let wbase = 900 + slot * 6;
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
        if fw.len() > wbase + 4 {
            parts.push(format!(
                "s{}:prev={:.3e},k_pre={:.3e},k_cand={:.3e},v_cand={:.3e},overwrite={:.3e},w_n={:.0}",
                slot,
                fw[wbase + 3],
                fw[wbase + 4],
                fw[wbase + 0],
                fw[wbase + 1],
                fw[wbase + 5],
                fw[wbase + 2],
            ));
        }
    }
    format!(
        "{} loss={:.4} all=[{}] query=[{}]",
        label,
        loss,
        parts.join(" | "),
        query_parts.join(" | "),
    )
}

fn main() {
    if std::env::var("AR_SEGMENT_MEMORY_TOKEN").is_ok()
        && std::env::var("AIDEEN_SEGMENT_MEMORY_TOKEN").is_err()
    {
        unsafe {
            std::env::set_var(
                "AIDEEN_SEGMENT_MEMORY_TOKEN",
                std::env::var("AR_SEGMENT_MEMORY_TOKEN").unwrap(),
            );
        }
    }
    if std::env::var("AR_SEGMENT_LEN").is_ok()
        && std::env::var("AIDEEN_INTERNAL_SEGMENT_LEN").is_err()
        && std::env::var("AIDEEN_SEGMENT_MEMORY_TOKEN")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    {
        unsafe {
            std::env::set_var(
                "AIDEEN_INTERNAL_SEGMENT_LEN",
                std::env::var("AR_SEGMENT_LEN").unwrap(),
            );
        }
    }
    let assoc_only = std::env::var("AR_ASSOC_ONLY").ok().as_deref() == Some("1");
    let assoc_audit = std::env::var("AR_AUDIT").ok().as_deref() == Some("1");
    let assoc_transition_gate_enabled = std::env::var("AIDEEN_ASSOC_TRANSITION_GATE")
        .ok()
        .map(|v| {
            let vl = v.trim().to_ascii_lowercase();
            vl == "1" || vl == "true" || vl == "yes"
        })
        .unwrap_or(false);
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
    // AIDEEN_SEQ_CAP override removed to enforce strict local learning chunking.
    // The sequence will now be properly split into ctx_len chunks by run_sequence_chunks.

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

    #[cfg(feature = "wgpu")]
    {
        // This benchmark is only meaningful on the DEQ GPU path. If adapter init
        // fails and we silently continue through CPU/fallback paths, the run emits
        // avg_loss=0, empty traces, and NaN evals that look like model regressions.
        // Fail fast instead of producing invalid metrics.
        if trainer.gpu_deq.is_none() {
            eprintln!("ERROR: assoc_recall requiere un backend GPU activo; la corrida actual quedó en fallback inválido.");
            std::process::exit(1);
        }
    }
    // Experimental only: tying the decoder to embeddings changes what the
    // benchmark measures, so it must not be the default memory baseline.
    if std::env::var("AR_TIE_LM_TO_EMB")
        .ok()
        .as_deref()
        .unwrap_or("0")
        != "0"
    {
        trainer.lm_head.w = trainer.tokenizer.embeddings.clone();
        trainer.frozen_emb = std::env::var("AR_TIE_LM_FREEZE_EMB")
            .ok()
            .as_deref()
            .unwrap_or("1")
            != "0";
    }
    if let Ok(v) = std::env::var("AIDEEN_DAMPING").and_then(|s| {
        s.parse::<f32>()
            .map_err(|_| std::env::VarError::NotPresent)
    }) {
        trainer.adaptive_damping = v;
        trainer.reasoning.damping = v;
    }
    // Associative memory needs a write-enabled stage. Stage 3 can inject/read
    // but cannot create bindings, so the benchmark forces the first coherent
    // training stage unless memory was explicitly disabled by configuration.
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
    let init_wv_assoc = flatten_mats(&trainer.reasoning.w_v_assoc);
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
        let wv_assoc_now = flatten_mats(&trainer.reasoning.w_v_assoc);
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
        let wv_delta = delta_rms(&init_wv_assoc, &wv_assoc_now);
        let wv_mode = if assoc_transition_gate_enabled {
            "transition_gate_active"
        } else {
            "transition_gate_inactive"
        };
        println!(
            "  [assoc-audit] decoupled=1 alpha(mean={:.4}, delta_rms={:.4e}) gate_key=W_k_write(rms={:.4e}, delta_rms={:.4e}) Wk_assoc(delta={:.4e}) Wv_assoc(mode={}, delta={:.4e}) Wq_assoc(delta={:.4e}) value=raw_token_identity",
            alpha_now.iter().copied().sum::<f32>() / alpha_now.len().max(1) as f32,
            delta_rms(&init_alpha_assoc, &alpha_now),
            rms(&wk_write_now),
            delta_rms(&init_wk_write, &wk_write_now),
            delta_rms(&init_wk_assoc, &wk_assoc_now),
            wv_mode,
            wv_delta,
            delta_rms(&init_wq_assoc, &wq_assoc_now),
        );
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = trainer.gpu_deq.as_ref() {
            let hist = gpu.read_hist_params_full();
            let assoc = gpu.read_assoc_state();
            let d = trainer.config.d_r;
            let h = trainer.config.h_slots;
            let rank = 32usize;
            let w_k_assoc_base = {
                let hist_mat = d * d;
                let hist_slot_scale = h * d;
                let hist_slot_bias = h * d;
                let hist_gate = h;
                let slot_anchor = h * d;
                let w_kv_write = 2 * h * d * rank;
                let b_delta = d;
                let flags = 21usize;
                let w_gate_hist = h * d;
                let w_write_gate = h * d;
                let b_write_mem = h;
                let gamma = h;
                let w_retain_up = h * d * rank;
                let w_retain_down = h * d * rank;
                let b_retain = h * d;
                let w_q_mem = h * d * rank;
                let w_k_mem = h * d * rank;
                let b_read_mem = h;
                hist_mat
                    + hist_slot_scale
                    + hist_slot_bias
                    + hist_gate
                    + slot_anchor
                    + w_kv_write
                    + b_delta
                    + flags
                    + w_gate_hist
                    + w_write_gate
                    + b_write_mem
                    + gamma
                    + w_retain_up
                    + w_retain_down
                    + b_retain
                    + w_q_mem
                    + w_k_mem
                    + b_read_mem
            };
            let w_v_assoc_base = w_k_assoc_base + h * d * rank;
            let w_q_assoc_base = w_v_assoc_base + h * d * rank;
            let wk_assoc_gpu_rms = rms(hist.get(w_k_assoc_base..w_v_assoc_base).unwrap_or(&[]));
            let wv_assoc_gpu_rms = rms(hist.get(w_v_assoc_base..w_q_assoc_base).unwrap_or(&[]));
            let wq_assoc_gpu_rms =
                rms(hist.get(w_q_assoc_base..w_q_assoc_base + h * d * rank).unwrap_or(&[]));
            let banks = std::env::var("AR_ASSOC_BANKS_AUDIT")
                .ok()
                .and_then(|s| s.parse().ok())
                .or_else(|| {
                    std::env::var("AIDEEN_ASSOC_BANKS")
                        .ok()
                        .and_then(|s| s.parse().ok())
                })
                .unwrap_or(1usize);
            let bank_stride = rank + d + 1;
            let slot_stride = banks * bank_stride;
            let mut key_ss = 0.0f32;
            let mut key_n = 0usize;
            let mut key_abs_max = 0.0f32;
            let mut val_ss = 0.0f32;
            let mut val_n = 0usize;
            let mut val_abs_max = 0.0f32;
            let mut usage_sum = 0.0f32;
            let mut usage_max = 0.0f32;
            let mut usage_n = 0usize;
            let mut usage_head: Vec<f32> = Vec::new();
            for slot in 0..h {
                let slot_base = slot * slot_stride;
                for bank in 0..banks {
                    let base = slot_base + bank * bank_stride;
                    for &x in assoc.get(base..base + rank).unwrap_or(&[]) {
                        key_ss += x * x;
                        key_abs_max = key_abs_max.max(x.abs());
                        key_n += 1;
                    }
                    for &x in assoc.get(base + rank..base + rank + d).unwrap_or(&[]) {
                        val_ss += x * x;
                        val_abs_max = val_abs_max.max(x.abs());
                        val_n += 1;
                    }
                    if let Some(&usage) = assoc.get(base + rank + d) {
                        usage_sum += usage;
                        usage_max = usage_max.max(usage);
                        usage_n += 1;
                        if usage_head.len() < 8 {
                            usage_head.push(usage);
                        }
                    }
                }
            }
            println!(
                "  [assoc-state] wk_gpu_rms={:.4e} wv_transition_gpu_rms={:.4e} wq_gpu_rms={:.4e} key_rms={:.4e} key_abs_max={:.4e} value_rms={:.4e} value_abs_max={:.4e} usage_mean={:.4e} usage_max={:.4e}",
                wk_assoc_gpu_rms,
                wv_assoc_gpu_rms,
                wq_assoc_gpu_rms,
                (key_ss / key_n.max(1) as f32).sqrt(),
                key_abs_max,
                (val_ss / val_n.max(1) as f32).sqrt(),
                val_abs_max,
                usage_sum / usage_n.max(1) as f32,
                usage_max,
            );
            // TEMPORARY ASSOCIATIVE DIAGNOSTIC: raw bank head for persistence inspection.
            if assoc.len() >= bank_stride {
                let key_head = assoc[0..rank.min(8)].to_vec();
                let value_head = assoc[rank..(rank + d).min(rank + 8)].to_vec();
                let usage0 = assoc.get(rank + d).copied().unwrap_or(0.0);
                println!(
                    "  [assoc-state-head] key={:?} value={:?} usage0={:.4e} usage_head={:?}",
                    key_head, value_head, usage0, usage_head
                );
            }
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
                    format_lm_top_debug(
                        &trainer,
                        final_cached_token_index(seq.len() - 1, ctx_len, trainer.gpu_sequence_capacity()),
                        TOK_KEY_BASE + ar.query_key,
                        TOK_VAL_BASE + ar.query_val,
                    );
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
