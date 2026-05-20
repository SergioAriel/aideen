#![cfg(feature = "wgpu")]

use aideen_backbone::{gpu_deq::GpuDeqBackend, FixedPointMemoryReasoning};
use aideen_core::state::ArchitectureConfig;
use std::sync::{Mutex, OnceLock};

fn env_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn rms(v: &[f32]) -> f32 {
    let sumsq = v.iter().map(|x| x * x).sum::<f32>();
    (sumsq / v.len().max(1) as f32 + 1e-6).sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn slice_token(seq: &[f32], token_idx: usize, h_slots: usize, d: usize) -> &[f32] {
    let start = token_idx * h_slots * d;
    let end = start + h_slots * d;
    &seq[start..end]
}

fn slice_slot(token: &[f32], slot_idx: usize, d: usize) -> &[f32] {
    let start = slot_idx * d;
    let end = start + d;
    &token[start..end]
}

fn cpu_hist_gated_forward_no_attn(
    s_in: &[f32],
    w_in: &[f32],
    w_hist: &[f32],
    hist_slot_scale: &[f32],
    hist_slot_bias: &[f32],
    hist_gate_logit: &[f32],
    slot_anchor: &[f32],
    w_x: &[f32],
    w_out: &[f32],
    a_log: &[f32],
    norm: &[f32],
    damping: f32,
    h_slots: usize,
    d: usize,
    use_prev_hstar: bool,
) -> Vec<f32> {
    let _ = hist_slot_bias;
    let seq_len = s_in.len() / d;
    let mut h_curr = vec![0.0f32; h_slots * d];
    let mut m_prev = vec![0.0f32; h_slots * d];
    let mut h_out = vec![0.0f32; seq_len * h_slots * d];

    for t in 0..seq_len {
        let inj = cpu_signal(&s_in[t * d..(t + 1) * d], w_in, d);
        let inj_rms = rms(&inj);

        let mut hist_ctx = vec![0.0f32; h_slots * d];
        for s in 0..h_slots {
            let off = s * d;
            let carrier = if t == 0 {
                vec![0.0f32; d]
            } else if use_prev_hstar {
                h_out[(t - 1) * h_slots * d + off..(t - 1) * h_slots * d + off + d].to_vec()
            } else {
                m_prev[off..off + d].to_vec()
            };
            let carrier_rms = rms(&carrier);
            let mut u = vec![0.0f32; d];
            for d_out in 0..d {
                let mut acc = hist_slot_scale[off + d_out] * (carrier[d_out] / carrier_rms);
                for j in 0..d {
                    acc += w_hist[d_out * d + j] * (carrier[j] / carrier_rms);
                }
                u[d_out] = acc;
            }
            let hist_rms = rms(&u);
            let tau = inj_rms;
            let hist_scale = (tau / hist_rms.max(1e-6)).min(1.0);
            let alpha = 0.08 + 0.20 / (1.0 + (-hist_gate_logit[s]).exp());
            for i in 0..d {
                hist_ctx[off + i] = alpha * u[i] * hist_scale;
            }
        }

        for s in 0..h_slots {
            let off = s * d;
            let mut combined = vec![0.0f32; d];
            for i in 0..d {
                combined[i] = inj[i] + hist_ctx[off + i] + slot_anchor[off + i];
            }
            let rms_combined = rms(&combined);
            for i in 0..d {
                let f_h = norm[i] * (combined[i] / rms_combined);
                h_out[t * h_slots * d + off + i] =
                    damping * f_h + (1.0 - damping) * h_curr[off + i];
            }
        }

        h_curr.copy_from_slice(&h_out[t * h_slots * d..(t + 1) * h_slots * d]);
        let mut next_m = vec![0.0f32; h_slots * d];
        for s in 0..h_slots {
            let off = s * d;
            let h_rms = rms(&h_curr[off..off + d]);
            let mut x_proj = vec![0.0f32; d];
            for i in 0..d {
                let a = 1.0 / (1.0 + a_log[i].exp());
                let mut acc = h_curr[off + i] / h_rms;
                for j in 0..d {
                    acc += w_x[j * d + i] * (h_curr[off + j] / h_rms);
                }
                x_proj[i] = a * m_prev[off + i] + (1.0 - a) * acc;
            }
            for i in 0..d {
                let mut out = x_proj[i];
                for j in 0..d {
                    out += w_out[j * d + i] * x_proj[j];
                }
                next_m[off + i] = out;
            }
        }
        m_prev = next_m;
    }

    h_out
}

fn cpu_signal(s_in: &[f32], w_in: &[f32], d: usize) -> Vec<f32> {
    let mut out = vec![0.0; d];
    for d_out in 0..d {
        let mut acc = 0.0;
        for j in 0..d {
            acc += w_in[j * d + d_out] * s_in[j];
        }
        out[d_out] = acc;
    }
    out
}

fn cpu_forward_map_no_fpm(
    h: &[f32],
    s_in: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    w_in: &[f32],
    norm: &[f32],
    damping: f32,
    h_slots: usize,
    d: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt().max(1.0);
    let signal = cpu_signal(s_in, w_in, d);

    let mut q = vec![0.0; h_slots * d];
    let mut k = vec![0.0; h_slots * d];
    let mut v = vec![0.0; h_slots * d];
    for s in 0..h_slots {
        let off = s * d;
        for d_out in 0..d {
            let mut q_acc = 0.0;
            let mut k_acc = 0.0;
            let mut v_acc = 0.0;
            for j in 0..d {
                let h_val = h[off + j];
                let w_idx = j * d + d_out;
                q_acc += w_q[w_idx] * h_val;
                k_acc += w_k[w_idx] * h_val;
                v_acc += w_v[w_idx] * h_val;
            }
            q[off + d_out] = q_acc;
            k[off + d_out] = k_acc;
            v[off + d_out] = v_acc;
        }
    }

    let mut attn_w = vec![0.0; h_slots * h_slots];
    for qs in 0..h_slots {
        let q_off = qs * d;
        let mut scores = vec![0.0; h_slots];
        for ks in 0..h_slots {
            let k_off = ks * d;
            let mut score = 0.0;
            for j in 0..d {
                score += q[q_off + j] * k[k_off + j];
            }
            scores[ks] = (score * scale).clamp(-4.0, 4.0);
        }
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0;
        for ks in 0..h_slots {
            let e = (scores[ks] - max_s).exp();
            attn_w[qs * h_slots + ks] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-12);
        for ks in 0..h_slots {
            attn_w[qs * h_slots + ks] *= inv_sum;
        }
    }

    let mut out = vec![0.0; h_slots * d];
    for qs in 0..h_slots {
        let q_off = qs * d;
        let mut mix = vec![0.0; d];
        for j in 0..d {
            let mut acc = 0.0;
            for ks in 0..h_slots {
                acc += attn_w[qs * h_slots + ks] * v[ks * d + j];
            }
            mix[j] = acc;
        }

        let mut combined = vec![0.0; d];
        let mut sumsq = 0.0;
        for d_out in 0..d {
            let mut attn_out = 0.0;
            for j in 0..d {
                attn_out += w_o[j * d + d_out] * mix[j];
            }
            let c = attn_out + signal[d_out];
            combined[d_out] = c;
            sumsq += c * c;
        }
        let rms = (sumsq / d as f32 + 1e-6).sqrt();
        for d_out in 0..d {
            let f_h = norm[d_out] * (combined[d_out] / rms);
            out[q_off + d_out] = damping * f_h + (1.0 - damping) * h[q_off + d_out];
        }
    }
    out
}

fn run_two_token_sequence_with_seed(
    config: &ArchitectureConfig,
    token_carry: bool,
    prev_dim: usize,
    curr_dim: usize,
    seed: u64,
) -> Vec<f32> {
    run_two_token_sequence_with_mode(config, token_carry, prev_dim, curr_dim, seed, false)
}

fn run_two_token_sequence_with_mode(
    config: &ArchitectureConfig,
    token_carry: bool,
    prev_dim: usize,
    curr_dim: usize,
    seed: u64,
    hist_gated: bool,
) -> Vec<f32> {
    if token_carry {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "1");
    } else {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "0");
    }
    std::env::set_var("AIDEEN_DEQ_NO_FPM", "1");
    if hist_gated {
        std::env::set_var("AIDEEN_DEQ_HIST_GATED", "1");
    } else {
        std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    }

    let reasoning = FixedPointMemoryReasoning::new_with_seed(config.clone(), seed);
    let weights = reasoning.export_weights();
    let w_q = weights.get("reasoning.w_q").expect("w_q missing");
    let w_k = weights.get("reasoning.w_k").expect("w_k missing");
    let w_v_flat = reasoning.w_v_gpu_flat();
    let w_v = &w_v_flat;
    let w_o_flat = reasoning.w_o_gpu_flat();
    let w_o = &w_o_flat;
    let w_in = weights.get("reasoning.w_in").expect("w_in missing");
    let w_x = weights.get("reasoning.w_x").expect("w_x missing");
    let w_out = weights.get("reasoning.w_out").expect("w_out missing");
    let a_log = weights.get("reasoning.a_log").expect("a_log missing");
    let norm = weights
        .get("reasoning.norm_scale")
        .expect("norm_scale missing");
    let w_hist = weights
        .get("reasoning.w_hist_shared")
        .expect("w_hist_shared missing");
    let hist_slot_scale = weights
        .get("reasoning.hist_slot_scale")
        .expect("hist_slot_scale missing");
    let hist_slot_bias = weights
        .get("reasoning.hist_slot_bias")
        .expect("hist_slot_bias missing");
    let hist_gate_logit = weights
        .get("reasoning.hist_gate_logit")
        .expect("hist_gate_logit missing");
    let slot_anchor = weights
        .get("reasoning.slot_anchor")
        .expect("slot_anchor missing");
    let d = config.d_r;
    let h = config.h_slots;
    let w_delta_zeros = vec![0.0f32; h * d];
    let b_delta_zeros = vec![0.0f32; h];
    let w_delta = weights.get("reasoning.w_delta").unwrap_or(&w_delta_zeros);
    let b_delta = weights.get("reasoning.b_delta").unwrap_or(&b_delta_zeros);
    let w_gate_hist_zeros = vec![0.0f32; h * d];
    let w_forget_zeros = vec![0.0f32; h * d];
    let b_forget_init = vec![3.0f32; h];
    let w_retain_up_zeros = vec![0.0f32; h * d * 32];
    let w_retain_down_zeros = vec![0.0f32; h * 32 * d];
    let b_retain_zeros = vec![0.0f32; h * d];
    let w_q_mem_zeros = vec![0.0f32; h * d * 32];
    let w_k_mem_zeros = vec![0.0f32; h * d * 32];
    let b_read_mem_zeros = vec![0.0f32; h];
    let w_k_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_v_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_q_assoc_zeros = vec![0.0f32; h * d * 32];
    let alpha_assoc_zeros = vec![0.0f32; h];

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");
    gpu.upload_weights(
        &gpu.queue,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        w_hist,
        hist_slot_scale,
        hist_slot_bias,
        hist_gate_logit,
        slot_anchor,
        w_delta,
        b_delta,
        &w_gate_hist_zeros,
        &w_forget_zeros,
        &b_forget_init,
        &w_retain_up_zeros,
        &w_retain_down_zeros,
        &b_retain_zeros,
        &w_q_mem_zeros,
        &w_k_mem_zeros,
        &b_read_mem_zeros,
        &w_k_assoc_zeros,
        &w_v_assoc_zeros,
        &w_q_assoc_zeros,
        &alpha_assoc_zeros,
    );

    let seq_len = 2u32;
    let damping = 0.9f32;
    let mut s_in = vec![0.0f32; seq_len as usize * d];
    s_in[prev_dim] = 1.0;
    s_in[d + curr_dim] = 1.0;

    gpu.reset_state();
    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        false,
    )
    .expect("GPU forward failed");
    gpu.read_hnext_seq(seq_len)
}

fn run_single_token_sequence_with_seed(
    config: &ArchitectureConfig,
    token_carry: bool,
    curr_dim: usize,
    seed: u64,
) -> Vec<f32> {
    run_single_token_sequence_with_mode(config, token_carry, curr_dim, seed, false)
}

fn run_single_token_sequence_with_mode(
    config: &ArchitectureConfig,
    token_carry: bool,
    curr_dim: usize,
    seed: u64,
    hist_gated: bool,
) -> Vec<f32> {
    if token_carry {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "1");
    } else {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "0");
    }
    std::env::set_var("AIDEEN_DEQ_NO_FPM", "1");
    if hist_gated {
        std::env::set_var("AIDEEN_DEQ_HIST_GATED", "1");
    } else {
        std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    }

    let reasoning = FixedPointMemoryReasoning::new_with_seed(config.clone(), seed);
    let weights = reasoning.export_weights();
    let w_q = weights.get("reasoning.w_q").expect("w_q missing");
    let w_k = weights.get("reasoning.w_k").expect("w_k missing");
    let w_v_flat = reasoning.w_v_gpu_flat();
    let w_v = &w_v_flat;
    let w_o_flat = reasoning.w_o_gpu_flat();
    let w_o = &w_o_flat;
    let w_in = weights.get("reasoning.w_in").expect("w_in missing");
    let w_x = weights.get("reasoning.w_x").expect("w_x missing");
    let w_out = weights.get("reasoning.w_out").expect("w_out missing");
    let a_log = weights.get("reasoning.a_log").expect("a_log missing");
    let norm = weights
        .get("reasoning.norm_scale")
        .expect("norm_scale missing");
    let w_hist = weights
        .get("reasoning.w_hist_shared")
        .expect("w_hist_shared missing");
    let hist_slot_scale = weights
        .get("reasoning.hist_slot_scale")
        .expect("hist_slot_scale missing");
    let hist_slot_bias = weights
        .get("reasoning.hist_slot_bias")
        .expect("hist_slot_bias missing");
    let hist_gate_logit = weights
        .get("reasoning.hist_gate_logit")
        .expect("hist_gate_logit missing");
    let slot_anchor = weights
        .get("reasoning.slot_anchor")
        .expect("slot_anchor missing");
    let d = config.d_r;
    let h = config.h_slots;
    let w_delta_zeros = vec![0.0f32; h * d];
    let b_delta_zeros = vec![0.0f32; h];
    let w_delta = weights.get("reasoning.w_delta").unwrap_or(&w_delta_zeros);
    let b_delta = weights.get("reasoning.b_delta").unwrap_or(&b_delta_zeros);
    let w_gate_hist_zeros = vec![0.0f32; h * d];
    let w_forget_zeros = vec![0.0f32; h * d];
    let b_forget_init = vec![3.0f32; h];
    let w_retain_up_zeros = vec![0.0f32; h * d * 32];
    let w_retain_down_zeros = vec![0.0f32; h * 32 * d];
    let b_retain_zeros = vec![0.0f32; h * d];
    let w_q_mem_zeros = vec![0.0f32; h * d * 32];
    let w_k_mem_zeros = vec![0.0f32; h * d * 32];
    let b_read_mem_zeros = vec![0.0f32; h];
    let w_k_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_v_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_q_assoc_zeros = vec![0.0f32; h * d * 32];
    let alpha_assoc_zeros = vec![0.0f32; h];

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");
    gpu.upload_weights(
        &gpu.queue,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        w_hist,
        hist_slot_scale,
        hist_slot_bias,
        hist_gate_logit,
        slot_anchor,
        w_delta,
        b_delta,
        &w_gate_hist_zeros,
        &w_forget_zeros,
        &b_forget_init,
        &w_retain_up_zeros,
        &w_retain_down_zeros,
        &b_retain_zeros,
        &w_q_mem_zeros,
        &w_k_mem_zeros,
        &b_read_mem_zeros,
        &w_k_assoc_zeros,
        &w_v_assoc_zeros,
        &w_q_assoc_zeros,
        &alpha_assoc_zeros,
    );

    let seq_len = 1u32;
    let damping = 0.9f32;
    let mut s_in = vec![0.0f32; d];
    s_in[curr_dim] = 1.0;

    gpu.reset_state();
    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        false,
    )
    .expect("GPU forward failed");
    gpu.read_hnext_seq(seq_len)
}

fn run_two_token_sequence_with_assoc_enabled(
    config: &ArchitectureConfig,
    token_carry: bool,
    prev_dim: usize,
    curr_dim: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    if token_carry {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "1");
    } else {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "0");
    }
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    std::env::set_var("AIDEEN_FPM_STAGE", "4");
    std::env::set_var("AIDEEN_ASSOC_BANKS", "1");
    std::env::set_var("AIDEEN_ASSOC_READ", "1");
    let assoc_rel_value_mix = std::env::var("AIDEEN_ASSOC_REL_VALUE_MIX")
        .unwrap_or_else(|_| "1".to_string());
    std::env::set_var("AIDEEN_ASSOC_REL_VALUE_MIX", assoc_rel_value_mix);

    let reasoning = FixedPointMemoryReasoning::new_with_seed(config.clone(), seed);
    let weights = reasoning.export_weights();
    let w_q = weights.get("reasoning.w_q").expect("w_q missing");
    let w_k = weights.get("reasoning.w_k").expect("w_k missing");
    let w_v_flat = reasoning.w_v_gpu_flat();
    let w_v = &w_v_flat;
    let w_o_flat = reasoning.w_o_gpu_flat();
    let w_o = &w_o_flat;
    let w_in = weights.get("reasoning.w_in").expect("w_in missing");
    let w_x = weights.get("reasoning.w_x").expect("w_x missing");
    let w_out = weights.get("reasoning.w_out").expect("w_out missing");
    let a_log = weights.get("reasoning.a_log").expect("a_log missing");
    let norm = weights
        .get("reasoning.norm_scale")
        .expect("norm_scale missing");
    let w_hist = weights
        .get("reasoning.w_hist_shared")
        .expect("w_hist_shared missing");
    let hist_slot_scale = weights
        .get("reasoning.hist_slot_scale")
        .expect("hist_slot_scale missing");
    let hist_slot_bias = weights
        .get("reasoning.hist_slot_bias")
        .expect("hist_slot_bias missing");
    let hist_gate_logit = weights
        .get("reasoning.hist_gate_logit")
        .expect("hist_gate_logit missing");
    let slot_anchor = weights
        .get("reasoning.slot_anchor")
        .expect("slot_anchor missing");
    let d = config.d_r;
    let h = config.h_slots;
    let w_delta_zeros = vec![0.0f32; h * d];
    let b_delta_zeros = vec![0.0f32; h];
    let w_delta = weights.get("reasoning.w_delta").unwrap_or(&w_delta_zeros);
    let b_delta = weights.get("reasoning.b_delta").unwrap_or(&b_delta_zeros);
    let w_gate_hist_zeros = vec![0.0f32; h * d];
    let w_forget_zeros = vec![0.0f32; h * d];
    let b_forget_init = vec![3.0f32; h];
    let w_retain_up_zeros = vec![0.0f32; h * d * 32];
    let w_retain_down_zeros = vec![0.0f32; h * 32 * d];
    let b_retain_zeros = vec![0.0f32; h * d];
    let w_q_mem_zeros = vec![0.0f32; h * d * 32];
    let w_k_mem_zeros = vec![0.0f32; h * d * 32];
    let b_read_mem_zeros = vec![0.0f32; h];
    let w_k_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_v_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_q_assoc_zeros = vec![0.0f32; h * d * 32];
    let alpha_assoc_zeros = vec![0.0f32; h];

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");
    gpu.upload_weights(
        &gpu.queue,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        w_hist,
        hist_slot_scale,
        hist_slot_bias,
        hist_gate_logit,
        slot_anchor,
        w_delta,
        b_delta,
        &w_gate_hist_zeros,
        &w_forget_zeros,
        &b_forget_init,
        &w_retain_up_zeros,
        &w_retain_down_zeros,
        &b_retain_zeros,
        &w_q_mem_zeros,
        &w_k_mem_zeros,
        &b_read_mem_zeros,
        &w_k_assoc_zeros,
        &w_v_assoc_zeros,
        &w_q_assoc_zeros,
        &alpha_assoc_zeros,
    );

    let seq_len = 2u32;
    let damping = 0.9f32;
    let mut s_in = vec![0.0f32; seq_len as usize * d];
    s_in[prev_dim] = 1.0;
    s_in[d + curr_dim] = 1.0;

    gpu.reset_state();
    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        false,
    )
    .expect("GPU forward failed");
    let hnext = gpu.read_hnext_seq(seq_len);
    let assoc = gpu.read_assoc_state();
    let assoc_read = gpu.read_assoc_read_seq(seq_len);
    let assoc_pooled = gpu.read_assoc_pooled_seq(seq_len);
    (hnext, assoc, assoc_read, assoc_pooled)
}

fn run_four_token_sequence_with_assoc_enabled(
    config: &ArchitectureConfig,
    token_carry: bool,
    key_dim: usize,
    mid_dim: usize,
    final_dim: usize,
    seed: u64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    if token_carry {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "1");
    } else {
        std::env::set_var("AIDEEN_DEQ_TOKEN_CARRY", "0");
    }
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    std::env::set_var("AIDEEN_FPM_STAGE", "4");
    std::env::set_var("AIDEEN_ASSOC_BANKS", "1");
    std::env::set_var("AIDEEN_ASSOC_READ", "1");
    let assoc_rel_value_mix = std::env::var("AIDEEN_ASSOC_REL_VALUE_MIX")
        .unwrap_or_else(|_| "1".to_string());
    std::env::set_var("AIDEEN_ASSOC_REL_VALUE_MIX", assoc_rel_value_mix);

    let reasoning = FixedPointMemoryReasoning::new_with_seed(config.clone(), seed);
    let weights = reasoning.export_weights();
    let w_q = weights.get("reasoning.w_q").expect("w_q missing");
    let w_k = weights.get("reasoning.w_k").expect("w_k missing");
    let w_v_flat = reasoning.w_v_gpu_flat();
    let w_v = &w_v_flat;
    let w_o_flat = reasoning.w_o_gpu_flat();
    let w_o = &w_o_flat;
    let w_in = weights.get("reasoning.w_in").expect("w_in missing");
    let w_x = weights.get("reasoning.w_x").expect("w_x missing");
    let w_out = weights.get("reasoning.w_out").expect("w_out missing");
    let a_log = weights.get("reasoning.a_log").expect("a_log missing");
    let norm = weights
        .get("reasoning.norm_scale")
        .expect("norm_scale missing");
    let w_hist = weights
        .get("reasoning.w_hist_shared")
        .expect("w_hist_shared missing");
    let hist_slot_scale = weights
        .get("reasoning.hist_slot_scale")
        .expect("hist_slot_scale missing");
    let hist_slot_bias = weights
        .get("reasoning.hist_slot_bias")
        .expect("hist_slot_bias missing");
    let hist_gate_logit = weights
        .get("reasoning.hist_gate_logit")
        .expect("hist_gate_logit missing");
    let slot_anchor = weights
        .get("reasoning.slot_anchor")
        .expect("slot_anchor missing");
    let d = config.d_r;
    let h = config.h_slots;
    let w_delta_zeros = vec![0.0f32; h * d];
    let b_delta_zeros = vec![0.0f32; h];
    let w_delta = weights.get("reasoning.w_delta").unwrap_or(&w_delta_zeros);
    let b_delta = weights.get("reasoning.b_delta").unwrap_or(&b_delta_zeros);
    let w_gate_hist_zeros = vec![0.0f32; h * d];
    let w_forget_zeros = vec![0.0f32; h * d];
    let b_forget_init = vec![3.0f32; h];
    let w_retain_up_zeros = vec![0.0f32; h * d * 32];
    let w_retain_down_zeros = vec![0.0f32; h * 32 * d];
    let b_retain_zeros = vec![0.0f32; h * d];
    let w_q_mem_zeros = vec![0.0f32; h * d * 32];
    let w_k_mem_zeros = vec![0.0f32; h * d * 32];
    let b_read_mem_zeros = vec![0.0f32; h];
    let w_k_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_v_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_q_assoc_zeros = vec![0.0f32; h * d * 32];
    let alpha_assoc_zeros = vec![0.0f32; h];

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");
    gpu.upload_weights(
        &gpu.queue,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        w_hist,
        hist_slot_scale,
        hist_slot_bias,
        hist_gate_logit,
        slot_anchor,
        w_delta,
        b_delta,
        &w_gate_hist_zeros,
        &w_forget_zeros,
        &b_forget_init,
        &w_retain_up_zeros,
        &w_retain_down_zeros,
        &b_retain_zeros,
        &w_q_mem_zeros,
        &w_k_mem_zeros,
        &b_read_mem_zeros,
        &w_k_assoc_zeros,
        &w_v_assoc_zeros,
        &w_q_assoc_zeros,
        &alpha_assoc_zeros,
    );

    let seq_len = 4u32;
    let damping = 0.9f32;
    let mut s_in = vec![0.0f32; seq_len as usize * d];
    s_in[key_dim] = 1.0;
    s_in[d + mid_dim] = 1.0;
    s_in[2 * d + key_dim] = 1.0;
    s_in[3 * d + final_dim] = 1.0;

    gpu.reset_state();
    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        false,
    )
    .expect("GPU forward failed");
    let hnext = gpu.read_hnext_seq(seq_len);
    let assoc = gpu.read_assoc_state();
    let assoc_read = gpu.read_assoc_read_seq(seq_len);
    let assoc_pooled = gpu.read_assoc_pooled_seq(seq_len);
    (hnext, assoc, assoc_read, assoc_pooled)
}

fn numeric_jt_v_no_fpm(
    h_star: &[f32],
    v: &[f32],
    s_in: &[f32],
    w_q: &[f32],
    w_k: &[f32],
    w_v: &[f32],
    w_o: &[f32],
    w_in: &[f32],
    norm: &[f32],
    damping: f32,
    h_slots: usize,
    d: usize,
) -> Vec<f32> {
    let eps = 1e-3f32;
    let mut out = vec![0.0; h_star.len()];
    for idx in 0..h_star.len() {
        let mut h_plus = h_star.to_vec();
        let mut h_minus = h_star.to_vec();
        h_plus[idx] += eps;
        h_minus[idx] -= eps;
        let f_plus = cpu_forward_map_no_fpm(
            &h_plus, s_in, w_q, w_k, w_v, w_o, w_in, norm, damping, h_slots, d,
        );
        let f_minus = cpu_forward_map_no_fpm(
            &h_minus, s_in, w_q, w_k, w_v, w_o, w_in, norm, damping, h_slots, d,
        );
        let mut phi_plus = 0.0;
        let mut phi_minus = 0.0;
        for (fp, vv) in f_plus.iter().zip(v.iter()) {
            phi_plus += fp * vv;
        }
        for (fm, vv) in f_minus.iter().zip(v.iter()) {
            phi_minus += fm * vv;
        }
        out[idx] = (phi_plus - phi_minus) / (2.0 * eps);
    }
    out
}

#[test]
fn test_hist_gated_forward_uses_prev_fpm_carrier() {
    let _guard = env_test_lock();
    std::env::set_var("AIDEEN_DEQ_HIST_GATED", "1");

    let mut config = ArchitectureConfig::default();
    config.d_r = 16;
    config.h_slots = 2;
    config.ctx_len = 3;
    config.max_deq_iters = 1;

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");
    gpu.reset_state();

    let seq_len = 3u32;
    let d = config.d_r;
    let h = config.h_slots;
    let len_sq = d * d;
    let damping = 0.9f32;

    let mut s_in = vec![0.0f32; seq_len as usize * d];
    s_in[0] = 1.0;
    s_in[d + 1] = 1.0;
    s_in[2 * d + 2] = 1.0;

    let w_q = vec![0.0f32; len_sq];
    let w_k = vec![0.0f32; len_sq];
    let w_v = vec![0.0f32; h * len_sq]; // per-slot: h_slots * d*d
    let w_o = vec![0.0f32; len_sq];
    let mut w_in = vec![0.0f32; len_sq];
    let w_x = vec![0.0f32; len_sq];
    let w_out = vec![0.0f32; len_sq];
    let a_log = vec![-0.5f32; d];
    let norm = vec![1.0f32; d];
    let mut w_hist = vec![0.0f32; len_sq];
    let hist_slot_scale = vec![0.0f32; h * d];
    let hist_slot_bias = vec![0.0f32; h * d];
    let hist_gate_logit = vec![-1.098_612_3_f32; h];
    let slot_anchor = vec![0.0f32; h * d];
    let w_delta = vec![0.0f32; len_sq];
    let b_delta = vec![0.0f32; d];
    let w_gate_hist = vec![0.0f32; h * d];
    let w_forget = vec![0.0f32; h * d];
    let b_forget = vec![3.0f32; h];
    let w_retain_up = vec![0.0f32; h * d * 32];
    let w_retain_down = vec![0.0f32; h * 32 * d];
    let b_retain_full = vec![0.0f32; h * d];
    let w_q_mem = vec![0.0f32; h * d * 32];
    let w_k_mem = vec![0.0f32; h * d * 32];
    let b_read_mem = vec![0.0f32; h];
    let w_k_assoc = vec![0.0f32; h * d * 32];
    let w_v_assoc = vec![0.0f32; h * d * 32];
    let w_q_assoc = vec![0.0f32; h * d * 32];
    let alpha_assoc = vec![0.0f32; h];

    for i in 0..d {
        w_in[i * d + i] = 1.0;
        w_hist[i * d + i] = 1.0;
    }

    gpu.upload_weights(
        &gpu.queue,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &w_in,
        &w_x,
        &w_out,
        &a_log,
        &norm,
        &w_hist,
        &hist_slot_scale,
        &hist_slot_bias,
        &hist_gate_logit,
        &slot_anchor,
        &w_delta,
        &b_delta,
        &w_gate_hist,
        &w_forget,
        &b_forget,
        &w_retain_up,
        &w_retain_down,
        &b_retain_full,
        &w_q_mem,
        &w_k_mem,
        &b_read_mem,
        &w_k_assoc,
        &w_v_assoc,
        &w_q_assoc,
        &alpha_assoc,
    );

    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &w_in,
        &w_x,
        &w_out,
        &a_log,
        &norm,
        false,
    )
    .expect("GPU DEQ forward failed");

    let gpu_h = gpu.read_hnext_seq(seq_len);
    let expected_h = cpu_hist_gated_forward_no_attn(
        &s_in,
        &w_in,
        &w_hist,
        &hist_slot_scale,
        &hist_slot_bias,
        &hist_gate_logit,
        &slot_anchor,
        &w_x,
        &w_out,
        &a_log,
        &norm,
        damping,
        h,
        d,
        false,
    );
    let alt_hstar = cpu_hist_gated_forward_no_attn(
        &s_in,
        &w_in,
        &w_hist,
        &hist_slot_scale,
        &hist_slot_bias,
        &hist_gate_logit,
        &slot_anchor,
        &w_x,
        &w_out,
        &a_log,
        &norm,
        damping,
        h,
        d,
        true,
    );

    let gpu_token2 = &gpu_h[2 * h * d..3 * h * d];
    let exp_token2 = &expected_h[2 * h * d..3 * h * d];
    let alt_token2 = &alt_hstar[2 * h * d..3 * h * d];

    let diff_expected = max_abs_diff(gpu_token2, exp_token2);
    let diff_alt = max_abs_diff(gpu_token2, alt_token2);

    assert!(
        diff_expected < 1e-3,
        "hist_gated token2 mismatch against prev-M reference: diff={diff_expected}"
    );
    assert!(
        diff_alt > 5e-2,
        "hist_gated token2 unexpectedly matches prev-H* carrier too closely: diff_alt={diff_alt}"
    );

    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
}

#[test]
fn test_hist_gated_starts_near_no_fpm_with_real_initialized_weights() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 32;
    config.h_slots = 2;
    config.ctx_len = 3;
    config.max_deq_iters = 2;

    let reasoning = FixedPointMemoryReasoning::new_with_seed(config.clone(), 42);
    let weights = reasoning.export_weights();
    let w_q = weights.get("reasoning.w_q").expect("w_q missing");
    let w_k = weights.get("reasoning.w_k").expect("w_k missing");
    let w_v_flat = reasoning.w_v_gpu_flat();
    let w_v = &w_v_flat;
    let w_o = weights.get("reasoning.w_o").expect("w_o missing");
    let w_in = weights.get("reasoning.w_in").expect("w_in missing");
    let w_x = weights.get("reasoning.w_x").expect("w_x missing");
    let w_out = weights.get("reasoning.w_out").expect("w_out missing");
    let a_log = weights.get("reasoning.a_log").expect("a_log missing");
    let norm = weights
        .get("reasoning.norm_scale")
        .expect("norm_scale missing");
    let w_hist = weights
        .get("reasoning.w_hist_shared")
        .expect("w_hist_shared missing");
    let hist_slot_scale = weights
        .get("reasoning.hist_slot_scale")
        .expect("hist_slot_scale missing");
    let hist_slot_bias = weights
        .get("reasoning.hist_slot_bias")
        .expect("hist_slot_bias missing");
    let hist_gate_logit = weights
        .get("reasoning.hist_gate_logit")
        .expect("hist_gate_logit missing");
    let slot_anchor = weights
        .get("reasoning.slot_anchor")
        .expect("slot_anchor missing");
    let w_delta = weights.get("reasoning.w_delta").expect("w_delta missing");
    let b_delta = weights.get("reasoning.b_delta").expect("b_delta missing");
    let d = config.d_r;
    let h = config.h_slots;
    let w_gate_hist_zeros = vec![0.0f32; h * d];
    let w_gate_hist = weights
        .get("reasoning.w_gate_hist")
        .unwrap_or(&w_gate_hist_zeros);
    let w_forget_zeros = vec![0.0f32; h * d];
    let b_forget_init = vec![3.0f32; h];
    let w_retain_up_zeros = vec![0.0f32; h * d * 32];
    let w_retain_down_zeros = vec![0.0f32; h * 32 * d];
    let b_retain_zeros = vec![0.0f32; h * d];
    let w_q_mem_zeros = vec![0.0f32; h * d * 32];
    let w_k_mem_zeros = vec![0.0f32; h * d * 32];
    let b_read_mem_zeros = vec![0.0f32; h];
    let w_k_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_v_assoc_zeros = vec![0.0f32; h * d * 32];
    let w_q_assoc_zeros = vec![0.0f32; h * d * 32];
    let alpha_assoc_zeros = vec![0.0f32; h];
    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");

    let seq_len = 3u32;
    let damping = 0.9f32;
    let mut s_in = vec![0.0f32; seq_len as usize * d];
    s_in[0] = 1.0;
    s_in[d + 1] = 1.0;
    s_in[2 * d + 2] = 1.0;

    gpu.upload_weights(
        &gpu.queue,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        w_hist,
        hist_slot_scale,
        hist_slot_bias,
        hist_gate_logit,
        slot_anchor,
        w_delta,
        b_delta,
        w_gate_hist,
        &w_forget_zeros,
        &b_forget_init,
        &w_retain_up_zeros,
        &w_retain_down_zeros,
        &b_retain_zeros,
        &w_q_mem_zeros,
        &w_k_mem_zeros,
        &b_read_mem_zeros,
        &w_k_assoc_zeros,
        &w_v_assoc_zeros,
        &w_q_assoc_zeros,
        &alpha_assoc_zeros,
    );

    std::env::set_var("AIDEEN_DEQ_NO_FPM", "1");
    gpu.reset_state();
    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        false,
    )
    .expect("GPU no-fpm forward failed");
    let no_fpm = gpu.read_hnext_seq(seq_len);
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");

    std::env::set_var("AIDEEN_DEQ_HIST_GATED", "1");
    gpu.reset_state();
    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        w_q,
        w_k,
        w_v,
        w_o,
        w_in,
        w_x,
        w_out,
        a_log,
        norm,
        false,
    )
    .expect("GPU hist-gated forward failed");
    let hist_gated = gpu.read_hnext_seq(seq_len);
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");

    let token2_no_fpm = &no_fpm[2 * h * d..3 * h * d];
    let token2_hist = &hist_gated[2 * h * d..3 * h * d];
    let diff = max_abs_diff(token2_no_fpm, token2_hist);

    assert!(
        diff < 1e-4,
        "hist_gated should start near no-fpm under neutral interface init: diff={diff}"
    );
}

#[test]
#[ignore = "Diagnostic geometry probe for token-carry attractor drag."]
fn test_token_carry_attractor_drag_geometry_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev_a = 0usize;
    let prev_b = 1usize;
    let curr = 2usize;
    let seed = 42u64;

    let carry_on_a = run_two_token_sequence_with_seed(&config, true, prev_a, curr, seed);
    let carry_on_b = run_two_token_sequence_with_seed(&config, true, prev_b, curr, seed);
    let carry_off_a = run_two_token_sequence_with_seed(&config, false, prev_a, curr, seed);
    let carry_off_b = run_two_token_sequence_with_seed(&config, false, prev_b, curr, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");

    let h = config.h_slots;
    let d = config.d_r;
    let prev_on_a = slice_token(&carry_on_a, 0, h, d);
    let curr_on_a = slice_token(&carry_on_a, 1, h, d);
    let prev_on_b = slice_token(&carry_on_b, 0, h, d);
    let curr_on_b = slice_token(&carry_on_b, 1, h, d);
    let prev_off_a = slice_token(&carry_off_a, 0, h, d);
    let curr_off_a = slice_token(&carry_off_a, 1, h, d);
    let prev_off_b = slice_token(&carry_off_b, 0, h, d);
    let curr_off_b = slice_token(&carry_off_b, 1, h, d);

    let same_current_diff_prev_cos_on = cosine(curr_on_a, curr_on_b);
    let same_current_diff_prev_cos_off = cosine(curr_off_a, curr_off_b);
    let same_current_diff_prev_l2_on = l2_distance(curr_on_a, curr_on_b);
    let same_current_diff_prev_l2_off = l2_distance(curr_off_a, curr_off_b);

    let curr_prev_cos_on_a = cosine(curr_on_a, prev_on_a);
    let curr_prev_cos_on_b = cosine(curr_on_b, prev_on_b);
    let curr_prev_cos_off_a = cosine(curr_off_a, prev_off_a);
    let curr_prev_cos_off_b = cosine(curr_off_b, prev_off_b);

    println!("[carry-drag] same-current diff-prev cosine: on={same_current_diff_prev_cos_on:.6} off={same_current_diff_prev_cos_off:.6}");
    println!("[carry-drag] same-current diff-prev l2: on={same_current_diff_prev_l2_on:.6} off={same_current_diff_prev_l2_off:.6}");
    println!("[carry-drag] curr-vs-prev cosine A: on={curr_prev_cos_on_a:.6} off={curr_prev_cos_off_a:.6}");
    println!("[carry-drag] curr-vs-prev cosine B: on={curr_prev_cos_on_b:.6} off={curr_prev_cos_off_b:.6}");
    println!(
        "[carry-drag] norms curr_on_a={:.6} curr_on_b={:.6} curr_off_a={:.6} curr_off_b={:.6}",
        rms(curr_on_a),
        rms(curr_on_b),
        rms(curr_off_a),
        rms(curr_off_b)
    );

    for value in [
        same_current_diff_prev_cos_on,
        same_current_diff_prev_cos_off,
        same_current_diff_prev_l2_on,
        same_current_diff_prev_l2_off,
        curr_prev_cos_on_a,
        curr_prev_cos_on_b,
        curr_prev_cos_off_a,
        curr_prev_cos_off_b,
    ] {
        assert!(value.is_finite(), "non-finite geometry metric: {value}");
    }
}

#[test]
#[ignore = "Diagnostic geometry probe: same token alone vs after previous token."]
fn test_token_context_reposition_geometry_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev = 0usize;
    let curr = 2usize;
    let seed = 42u64;

    let single_on = run_single_token_sequence_with_seed(&config, true, curr, seed);
    let single_off = run_single_token_sequence_with_seed(&config, false, curr, seed);
    let two_on = run_two_token_sequence_with_seed(&config, true, prev, curr, seed);
    let two_off = run_two_token_sequence_with_seed(&config, false, prev, curr, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");

    let h = config.h_slots;
    let d = config.d_r;
    let single_on_curr = slice_token(&single_on, 0, h, d);
    let single_off_curr = slice_token(&single_off, 0, h, d);
    let two_on_curr = slice_token(&two_on, 1, h, d);
    let two_off_curr = slice_token(&two_off, 1, h, d);
    let two_on_prev = slice_token(&two_on, 0, h, d);
    let two_off_prev = slice_token(&two_off, 0, h, d);

    let same_token_single_vs_two_cos_on = cosine(single_on_curr, two_on_curr);
    let same_token_single_vs_two_cos_off = cosine(single_off_curr, two_off_curr);
    let same_token_single_vs_two_l2_on = l2_distance(single_on_curr, two_on_curr);
    let same_token_single_vs_two_l2_off = l2_distance(single_off_curr, two_off_curr);
    let two_curr_vs_prev_cos_on = cosine(two_on_curr, two_on_prev);
    let two_curr_vs_prev_cos_off = cosine(two_off_curr, two_off_prev);

    println!("[carry-reposition] same-token single-vs-two cosine: on={same_token_single_vs_two_cos_on:.6} off={same_token_single_vs_two_cos_off:.6}");
    println!("[carry-reposition] same-token single-vs-two l2: on={same_token_single_vs_two_l2_on:.6} off={same_token_single_vs_two_l2_off:.6}");
    println!("[carry-reposition] two-token curr-vs-prev cosine: on={two_curr_vs_prev_cos_on:.6} off={two_curr_vs_prev_cos_off:.6}");
    println!(
        "[carry-reposition] norms single_on={:.6} two_on={:.6} single_off={:.6} two_off={:.6}",
        rms(single_on_curr),
        rms(two_on_curr),
        rms(single_off_curr),
        rms(two_off_curr)
    );

    for value in [
        same_token_single_vs_two_cos_on,
        same_token_single_vs_two_cos_off,
        same_token_single_vs_two_l2_on,
        same_token_single_vs_two_l2_off,
        two_curr_vs_prev_cos_on,
        two_curr_vs_prev_cos_off,
    ] {
        assert!(value.is_finite(), "non-finite geometry metric: {value}");
    }
}

#[test]
#[ignore = "Diagnostic geometry probe: per-slot contextual repositioning."]
fn test_token_context_reposition_per_slot_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev = 0usize;
    let curr = 2usize;
    let seed = 42u64;

    let single_on = run_single_token_sequence_with_seed(&config, true, curr, seed);
    let two_on = run_two_token_sequence_with_seed(&config, true, prev, curr, seed);
    let single_off = run_single_token_sequence_with_seed(&config, false, curr, seed);
    let two_off = run_two_token_sequence_with_seed(&config, false, prev, curr, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");

    let h = config.h_slots;
    let d = config.d_r;
    let single_on_curr = slice_token(&single_on, 0, h, d);
    let two_on_curr = slice_token(&two_on, 1, h, d);
    let two_on_prev = slice_token(&two_on, 0, h, d);
    let single_off_curr = slice_token(&single_off, 0, h, d);
    let two_off_curr = slice_token(&two_off, 1, h, d);
    let two_off_prev = slice_token(&two_off, 0, h, d);

    for slot in 0..h {
        let s_on = slice_slot(single_on_curr, slot, d);
        let t_on = slice_slot(two_on_curr, slot, d);
        let p_on = slice_slot(two_on_prev, slot, d);
        let s_off = slice_slot(single_off_curr, slot, d);
        let t_off = slice_slot(two_off_curr, slot, d);
        let p_off = slice_slot(two_off_prev, slot, d);

        let slot_single_vs_two_cos_on = cosine(s_on, t_on);
        let slot_single_vs_two_cos_off = cosine(s_off, t_off);
        let slot_curr_vs_prev_cos_on = cosine(t_on, p_on);
        let slot_curr_vs_prev_cos_off = cosine(t_off, p_off);
        let slot_single_vs_two_l2_on = l2_distance(s_on, t_on);
        let slot_single_vs_two_l2_off = l2_distance(s_off, t_off);

        println!(
            "[carry-reposition-slot] slot={} single-vs-two cos on={:.6} off={:.6} | curr-vs-prev cos on={:.6} off={:.6} | single-vs-two l2 on={:.6} off={:.6}",
            slot,
            slot_single_vs_two_cos_on,
            slot_single_vs_two_cos_off,
            slot_curr_vs_prev_cos_on,
            slot_curr_vs_prev_cos_off,
            slot_single_vs_two_l2_on,
            slot_single_vs_two_l2_off
        );

        for value in [
            slot_single_vs_two_cos_on,
            slot_single_vs_two_cos_off,
            slot_curr_vs_prev_cos_on,
            slot_curr_vs_prev_cos_off,
            slot_single_vs_two_l2_on,
            slot_single_vs_two_l2_off,
        ] {
            assert!(value.is_finite(), "slot {slot} non-finite geometry metric: {value}");
        }
    }
}

#[test]
#[ignore = "Diagnostic geometry probe: triangle between single current and two previous-token contexts."]
fn test_token_context_triangle_geometry_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev_a = 0usize;
    let prev_b = 1usize;
    let curr = 2usize;
    let seed = 42u64;

    let single_on = run_single_token_sequence_with_seed(&config, true, curr, seed);
    let two_on_a = run_two_token_sequence_with_seed(&config, true, prev_a, curr, seed);
    let two_on_b = run_two_token_sequence_with_seed(&config, true, prev_b, curr, seed);
    let single_off = run_single_token_sequence_with_seed(&config, false, curr, seed);
    let two_off_a = run_two_token_sequence_with_seed(&config, false, prev_a, curr, seed);
    let two_off_b = run_two_token_sequence_with_seed(&config, false, prev_b, curr, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");

    let h = config.h_slots;
    let d = config.d_r;
    let single_on_curr = slice_token(&single_on, 0, h, d);
    let two_on_a_curr = slice_token(&two_on_a, 1, h, d);
    let two_on_b_curr = slice_token(&two_on_b, 1, h, d);
    let single_off_curr = slice_token(&single_off, 0, h, d);
    let two_off_a_curr = slice_token(&two_off_a, 1, h, d);
    let two_off_b_curr = slice_token(&two_off_b, 1, h, d);

    let on_single_to_a_cos = cosine(single_on_curr, two_on_a_curr);
    let on_single_to_b_cos = cosine(single_on_curr, two_on_b_curr);
    let on_a_to_b_cos = cosine(two_on_a_curr, two_on_b_curr);
    let on_single_to_a_l2 = l2_distance(single_on_curr, two_on_a_curr);
    let on_single_to_b_l2 = l2_distance(single_on_curr, two_on_b_curr);
    let on_a_to_b_l2 = l2_distance(two_on_a_curr, two_on_b_curr);

    let off_single_to_a_cos = cosine(single_off_curr, two_off_a_curr);
    let off_single_to_b_cos = cosine(single_off_curr, two_off_b_curr);
    let off_a_to_b_cos = cosine(two_off_a_curr, two_off_b_curr);
    let off_single_to_a_l2 = l2_distance(single_off_curr, two_off_a_curr);
    let off_single_to_b_l2 = l2_distance(single_off_curr, two_off_b_curr);
    let off_a_to_b_l2 = l2_distance(two_off_a_curr, two_off_b_curr);

    println!(
        "[carry-triangle] on cos single->A={:.6} single->B={:.6} A->B={:.6}",
        on_single_to_a_cos, on_single_to_b_cos, on_a_to_b_cos
    );
    println!(
        "[carry-triangle] on l2  single->A={:.6} single->B={:.6} A->B={:.6}",
        on_single_to_a_l2, on_single_to_b_l2, on_a_to_b_l2
    );
    println!(
        "[carry-triangle] off cos single->A={:.6} single->B={:.6} A->B={:.6}",
        off_single_to_a_cos, off_single_to_b_cos, off_a_to_b_cos
    );
    println!(
        "[carry-triangle] off l2  single->A={:.6} single->B={:.6} A->B={:.6}",
        off_single_to_a_l2, off_single_to_b_l2, off_a_to_b_l2
    );

    for value in [
        on_single_to_a_cos,
        on_single_to_b_cos,
        on_a_to_b_cos,
        on_single_to_a_l2,
        on_single_to_b_l2,
        on_a_to_b_l2,
        off_single_to_a_cos,
        off_single_to_b_cos,
        off_a_to_b_cos,
        off_single_to_a_l2,
        off_single_to_b_l2,
        off_a_to_b_l2,
    ] {
        assert!(value.is_finite(), "non-finite triangle metric: {value}");
    }
}

#[test]
#[ignore = "Diagnostic geometry probe: triangle under hist_gated memory path."]
fn test_hist_gated_context_triangle_geometry_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev_a = 0usize;
    let prev_b = 1usize;
    let curr = 2usize;
    let seed = 42u64;

    let single_on = run_single_token_sequence_with_mode(&config, true, curr, seed, true);
    let two_on_a = run_two_token_sequence_with_mode(&config, true, prev_a, curr, seed, true);
    let two_on_b = run_two_token_sequence_with_mode(&config, true, prev_b, curr, seed, true);
    let single_off = run_single_token_sequence_with_mode(&config, false, curr, seed, true);
    let two_off_a = run_two_token_sequence_with_mode(&config, false, prev_a, curr, seed, true);
    let two_off_b = run_two_token_sequence_with_mode(&config, false, prev_b, curr, seed, true);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");

    let h = config.h_slots;
    let d = config.d_r;
    let single_on_curr = slice_token(&single_on, 0, h, d);
    let two_on_a_curr = slice_token(&two_on_a, 1, h, d);
    let two_on_b_curr = slice_token(&two_on_b, 1, h, d);
    let single_off_curr = slice_token(&single_off, 0, h, d);
    let two_off_a_curr = slice_token(&two_off_a, 1, h, d);
    let two_off_b_curr = slice_token(&two_off_b, 1, h, d);

    let on_single_to_a_cos = cosine(single_on_curr, two_on_a_curr);
    let on_single_to_b_cos = cosine(single_on_curr, two_on_b_curr);
    let on_a_to_b_cos = cosine(two_on_a_curr, two_on_b_curr);
    let on_single_to_a_l2 = l2_distance(single_on_curr, two_on_a_curr);
    let on_single_to_b_l2 = l2_distance(single_on_curr, two_on_b_curr);
    let on_a_to_b_l2 = l2_distance(two_on_a_curr, two_on_b_curr);

    let off_single_to_a_cos = cosine(single_off_curr, two_off_a_curr);
    let off_single_to_b_cos = cosine(single_off_curr, two_off_b_curr);
    let off_a_to_b_cos = cosine(two_off_a_curr, two_off_b_curr);
    let off_single_to_a_l2 = l2_distance(single_off_curr, two_off_a_curr);
    let off_single_to_b_l2 = l2_distance(single_off_curr, two_off_b_curr);
    let off_a_to_b_l2 = l2_distance(two_off_a_curr, two_off_b_curr);

    println!(
        "[hist-gated-triangle] on  cos single->A={:.6} single->B={:.6} A->B={:.6}",
        on_single_to_a_cos, on_single_to_b_cos, on_a_to_b_cos
    );
    println!(
        "[hist-gated-triangle] on  l2  single->A={:.6} single->B={:.6} A->B={:.6}",
        on_single_to_a_l2, on_single_to_b_l2, on_a_to_b_l2
    );
    println!(
        "[hist-gated-triangle] off cos single->A={:.6} single->B={:.6} A->B={:.6}",
        off_single_to_a_cos, off_single_to_b_cos, off_a_to_b_cos
    );
    println!(
        "[hist-gated-triangle] off l2  single->A={:.6} single->B={:.6} A->B={:.6}",
        off_single_to_a_l2, off_single_to_b_l2, off_a_to_b_l2
    );

    for value in [
        on_single_to_a_cos,
        on_single_to_b_cos,
        on_a_to_b_cos,
        on_single_to_a_l2,
        on_single_to_b_l2,
        on_a_to_b_l2,
        off_single_to_a_cos,
        off_single_to_b_cos,
        off_a_to_b_cos,
        off_single_to_a_l2,
        off_single_to_b_l2,
        off_a_to_b_l2,
    ] {
        assert!(value.is_finite(), "non-finite hist-gated triangle metric: {value}");
    }
}

#[test]
#[ignore = "Diagnostic probe: does Assoc differentiate previous contexts when enabled?"]
fn test_assoc_state_differentiates_previous_context_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev_a = 0usize;
    let prev_b = 1usize;
    let curr = 2usize;
    let seed = 42u64;

    let (_h_a_on, assoc_a_on, _assoc_read_a_on, _assoc_pooled_a_on) =
        run_two_token_sequence_with_assoc_enabled(&config, true, prev_a, curr, seed);
    let (_h_b_on, assoc_b_on, _assoc_read_b_on, _assoc_pooled_b_on) =
        run_two_token_sequence_with_assoc_enabled(&config, true, prev_b, curr, seed);
    let (_h_a_off, assoc_a_off, _assoc_read_a_off, _assoc_pooled_a_off) =
        run_two_token_sequence_with_assoc_enabled(&config, false, prev_a, curr, seed);
    let (_h_b_off, assoc_b_off, _assoc_read_b_off, _assoc_pooled_b_off) =
        run_two_token_sequence_with_assoc_enabled(&config, false, prev_b, curr, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    std::env::remove_var("AIDEEN_FPM_STAGE");
    std::env::remove_var("AIDEEN_ASSOC_BANKS");
    std::env::remove_var("AIDEEN_ASSOC_READ");
    std::env::remove_var("AIDEEN_ASSOC_REL_VALUE_MIX");

    let d = config.d_r;
    let h = config.h_slots;
    let assoc_banks = 1usize;
    let assoc_rank = 32usize;
    let assoc_bank_stride = assoc_rank + d + 1;
    let assoc_slot_stride = assoc_banks * assoc_bank_stride;
    let assoc_slots = h / 2;

    let mut keys_a_on = Vec::new();
    let mut keys_b_on = Vec::new();
    let mut values_a_on = Vec::new();
    let mut values_b_on = Vec::new();
    let mut usages_a_on = Vec::new();
    let mut usages_b_on = Vec::new();
    let mut keys_a_off = Vec::new();
    let mut keys_b_off = Vec::new();
    let mut values_a_off = Vec::new();
    let mut values_b_off = Vec::new();
    let mut usages_a_off = Vec::new();
    let mut usages_b_off = Vec::new();

    for assoc_slot in 0..assoc_slots {
        let slot = assoc_slot + h / 2;
        let slot_base = slot * assoc_slot_stride;
        let bank_base = slot_base;
        let key_base = bank_base;
        let value_base = bank_base + assoc_rank;
        let usage_idx = value_base + d;

        keys_a_on.extend_from_slice(&assoc_a_on[key_base..key_base + assoc_rank]);
        keys_b_on.extend_from_slice(&assoc_b_on[key_base..key_base + assoc_rank]);
        values_a_on.extend_from_slice(&assoc_a_on[value_base..value_base + d]);
        values_b_on.extend_from_slice(&assoc_b_on[value_base..value_base + d]);
        usages_a_on.push(assoc_a_on[usage_idx]);
        usages_b_on.push(assoc_b_on[usage_idx]);

        keys_a_off.extend_from_slice(&assoc_a_off[key_base..key_base + assoc_rank]);
        keys_b_off.extend_from_slice(&assoc_b_off[key_base..key_base + assoc_rank]);
        values_a_off.extend_from_slice(&assoc_a_off[value_base..value_base + d]);
        values_b_off.extend_from_slice(&assoc_b_off[value_base..value_base + d]);
        usages_a_off.push(assoc_a_off[usage_idx]);
        usages_b_off.push(assoc_b_off[usage_idx]);
    }

    let key_cos_on = cosine(&keys_a_on, &keys_b_on);
    let key_l2_on = l2_distance(&keys_a_on, &keys_b_on);
    let value_cos_on = cosine(&values_a_on, &values_b_on);
    let value_l2_on = l2_distance(&values_a_on, &values_b_on);
    let usage_l2_on = l2_distance(&usages_a_on, &usages_b_on);

    let key_cos_off = cosine(&keys_a_off, &keys_b_off);
    let key_l2_off = l2_distance(&keys_a_off, &keys_b_off);
    let value_cos_off = cosine(&values_a_off, &values_b_off);
    let value_l2_off = l2_distance(&values_a_off, &values_b_off);
    let usage_l2_off = l2_distance(&usages_a_off, &usages_b_off);

    println!(
        "[assoc-diff] on  key cos={:.6} l2={:.6} | value cos={:.6} l2={:.6} | usage l2={:.6}",
        key_cos_on, key_l2_on, value_cos_on, value_l2_on, usage_l2_on
    );
    println!(
        "[assoc-diff] off key cos={:.6} l2={:.6} | value cos={:.6} l2={:.6} | usage l2={:.6}",
        key_cos_off, key_l2_off, value_cos_off, value_l2_off, usage_l2_off
    );

    for value in [
        key_cos_on,
        key_l2_on,
        value_cos_on,
        value_l2_on,
        usage_l2_on,
        key_cos_off,
        key_l2_off,
        value_cos_off,
        value_l2_off,
        usage_l2_off,
    ] {
        assert!(value.is_finite(), "non-finite assoc metric: {value}");
    }
}

#[test]
#[ignore = "Diagnostic probe: does pooled associative readout differentiate previous contexts?"]
fn test_assoc_pooled_differentiates_previous_context_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev_a = 0usize;
    let prev_b = 1usize;
    let mid = 2usize;
    let final_tok = 3usize;
    let seed = 42u64;

    let (_h_a_on, _assoc_a_on, _assoc_read_a_on, assoc_pooled_a_on) =
        run_four_token_sequence_with_assoc_enabled(&config, true, prev_a, mid, final_tok, seed);
    let (_h_b_on, _assoc_b_on, _assoc_read_b_on, assoc_pooled_b_on) =
        run_four_token_sequence_with_assoc_enabled(&config, true, prev_b, mid, final_tok, seed);
    let (_h_a_off, _assoc_a_off, _assoc_read_a_off, assoc_pooled_a_off) =
        run_four_token_sequence_with_assoc_enabled(&config, false, prev_a, mid, final_tok, seed);
    let (_h_b_off, _assoc_b_off, _assoc_read_b_off, assoc_pooled_b_off) =
        run_four_token_sequence_with_assoc_enabled(&config, false, prev_b, mid, final_tok, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    std::env::remove_var("AIDEEN_FPM_STAGE");
    std::env::remove_var("AIDEEN_ASSOC_BANKS");
    std::env::remove_var("AIDEEN_ASSOC_READ");
    std::env::remove_var("AIDEEN_ASSOC_REL_VALUE_MIX");

    let d = config.d_r;
    let pooled_a_on_curr = &assoc_pooled_a_on[3 * d..4 * d];
    let pooled_b_on_curr = &assoc_pooled_b_on[3 * d..4 * d];
    let pooled_a_off_curr = &assoc_pooled_a_off[3 * d..4 * d];
    let pooled_b_off_curr = &assoc_pooled_b_off[3 * d..4 * d];

    let cos_on = cosine(pooled_a_on_curr, pooled_b_on_curr);
    let l2_on = l2_distance(pooled_a_on_curr, pooled_b_on_curr);
    let cos_off = cosine(pooled_a_off_curr, pooled_b_off_curr);
    let l2_off = l2_distance(pooled_a_off_curr, pooled_b_off_curr);

    println!(
        "[assoc-pooled-diff] on  cos={:.6} l2={:.6}",
        cos_on, l2_on
    );
    println!(
        "[assoc-pooled-diff] off cos={:.6} l2={:.6}",
        cos_off, l2_off
    );

    for value in [cos_on, l2_on, cos_off, l2_off] {
        assert!(value.is_finite(), "non-finite assoc pooled metric: {value}");
    }
    assert!(
        l2_on > 1.0e-4,
        "assoc pooled read still failed to differentiate previous context"
    );
    assert!(
        l2_off > 1.0e-4,
        "assoc pooled read still failed to differentiate previous context without carry"
    );
}

#[test]
#[ignore = "Diagnostic probe: does direct associative read differentiate previous contexts?"]
fn test_assoc_read_differentiates_previous_context_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev_a = 0usize;
    let prev_b = 1usize;
    let mid = 2usize;
    let final_tok = 3usize;
    let seed = 42u64;

    let (_h_a_on, _assoc_a_on, assoc_read_a_on, _assoc_pooled_a_on) =
        run_four_token_sequence_with_assoc_enabled(&config, true, prev_a, mid, final_tok, seed);
    let (_h_b_on, _assoc_b_on, assoc_read_b_on, _assoc_pooled_b_on) =
        run_four_token_sequence_with_assoc_enabled(&config, true, prev_b, mid, final_tok, seed);
    let (_h_a_off, _assoc_a_off, assoc_read_a_off, _assoc_pooled_a_off) =
        run_four_token_sequence_with_assoc_enabled(&config, false, prev_a, mid, final_tok, seed);
    let (_h_b_off, _assoc_b_off, assoc_read_b_off, _assoc_pooled_b_off) =
        run_four_token_sequence_with_assoc_enabled(&config, false, prev_b, mid, final_tok, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    std::env::remove_var("AIDEEN_FPM_STAGE");
    std::env::remove_var("AIDEEN_ASSOC_BANKS");
    std::env::remove_var("AIDEEN_ASSOC_READ");
    std::env::remove_var("AIDEEN_ASSOC_REL_VALUE_MIX");

    let h = config.h_slots;
    let d = config.d_r;
    let read_a_on_curr = slice_token(&assoc_read_a_on, 3, h, d);
    let read_b_on_curr = slice_token(&assoc_read_b_on, 3, h, d);
    let read_a_off_curr = slice_token(&assoc_read_a_off, 3, h, d);
    let read_b_off_curr = slice_token(&assoc_read_b_off, 3, h, d);

    let cos_on = cosine(read_a_on_curr, read_b_on_curr);
    let l2_on = l2_distance(read_a_on_curr, read_b_on_curr);
    let cos_off = cosine(read_a_off_curr, read_b_off_curr);
    let l2_off = l2_distance(read_a_off_curr, read_b_off_curr);

    println!("[assoc-read-diff] on  cos={:.6} l2={:.6}", cos_on, l2_on);
    println!("[assoc-read-diff] off cos={:.6} l2={:.6}", cos_off, l2_off);

    for value in [cos_on, l2_on, cos_off, l2_off] {
        assert!(value.is_finite(), "non-finite assoc read metric: {value}");
    }
    assert!(
        l2_on > 1.0e-4,
        "assoc read still failed to differentiate previous context"
    );
    assert!(
        l2_off > 1.0e-4,
        "assoc read still failed to differentiate previous context without carry"
    );
}

#[test]
#[ignore = "Diagnostic probe: does assoc-enabled H_next differentiate previous contexts?"]
fn test_assoc_enabled_hnext_differentiates_previous_context_probe() {
    let _guard = env_test_lock();
    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 4;
    config.max_deq_iters = 4;

    let prev_a = 0usize;
    let prev_b = 1usize;
    let mid = 2usize;
    let final_tok = 3usize;
    let seed = 42u64;

    let (h_a_on, _assoc_a_on, _assoc_read_a_on, _assoc_pooled_a_on) =
        run_four_token_sequence_with_assoc_enabled(&config, true, prev_a, mid, final_tok, seed);
    let (h_b_on, _assoc_b_on, _assoc_read_b_on, _assoc_pooled_b_on) =
        run_four_token_sequence_with_assoc_enabled(&config, true, prev_b, mid, final_tok, seed);
    let (h_a_off, _assoc_a_off, _assoc_read_a_off, _assoc_pooled_a_off) =
        run_four_token_sequence_with_assoc_enabled(&config, false, prev_a, mid, final_tok, seed);
    let (h_b_off, _assoc_b_off, _assoc_read_b_off, _assoc_pooled_b_off) =
        run_four_token_sequence_with_assoc_enabled(&config, false, prev_b, mid, final_tok, seed);

    std::env::remove_var("AIDEEN_DEQ_TOKEN_CARRY");
    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    std::env::remove_var("AIDEEN_DEQ_HIST_GATED");
    std::env::remove_var("AIDEEN_FPM_STAGE");
    std::env::remove_var("AIDEEN_ASSOC_BANKS");
    std::env::remove_var("AIDEEN_ASSOC_READ");
    std::env::remove_var("AIDEEN_ASSOC_REL_VALUE_MIX");

    let h = config.h_slots;
    let d = config.d_r;
    let curr_a_on = slice_token(&h_a_on, 3, h, d);
    let curr_b_on = slice_token(&h_b_on, 3, h, d);
    let curr_a_off = slice_token(&h_a_off, 3, h, d);
    let curr_b_off = slice_token(&h_b_off, 3, h, d);

    let cos_on = cosine(curr_a_on, curr_b_on);
    let l2_on = l2_distance(curr_a_on, curr_b_on);
    let cos_off = cosine(curr_a_off, curr_b_off);
    let l2_off = l2_distance(curr_a_off, curr_b_off);

    println!("[assoc-hnext-diff] on  cos={:.6} l2={:.6}", cos_on, l2_on);
    println!("[assoc-hnext-diff] off cos={:.6} l2={:.6}", cos_off, l2_off);

    for value in [cos_on, l2_on, cos_off, l2_off] {
        assert!(value.is_finite(), "non-finite assoc hnext metric: {value}");
    }
}

#[test]
#[ignore = "Numeric reference for staged Picard no-fpm attention adjoint."]
fn test_staged_picard_matches_numeric_reference_no_fpm_small() {
    std::env::set_var("AIDEEN_DEQ_NO_FPM", "1");

    let mut config = ArchitectureConfig::default();
    config.d_r = 16;
    config.h_slots = 2;
    config.ctx_len = 1;
    config.max_deq_iters = 4;

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");

    let seq_len = 1u32;
    let d = config.d_r;
    let h = config.h_slots;
    let len_sq = d * d;
    let damping = 0.9f32;

    let s_in = vec![0.1f32; d];
    let w_q = vec![0.02f32; len_sq];
    let w_k = vec![0.015f32; len_sq];
    let w_v = vec![0.01f32; len_sq];
    let w_o = vec![0.012f32; len_sq];
    let w_in = vec![0.01f32; len_sq];
    let w_x = vec![0.0f32; len_sq];
    let w_out = vec![0.0f32; len_sq];
    let a_log = vec![-0.5f32; d];
    let norm = vec![1.0f32; d];

    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &w_in,
        &w_x,
        &w_out,
        &a_log,
        &norm,
        true,
    )
    .expect("GPU DEQ forward failed");

    let dl = vec![0.05f32; d];
    gpu.queue
        .write_buffer(&gpu.adj_bufs.b_dl, 0, bytemuck::cast_slice(&dl));
    gpu.run_staged_adjoint_picard_no_readback(seq_len, damping, 8, None, None, true, 1)
        .expect("Staged Picard adjoint dispatch failed");
    let v_picard = gpu.read_adj_v_out(seq_len);
    let h_star = gpu.read_hnext();

    let mut v_ref = vec![0.05f32 / h as f32; h * d];
    let b_rep = v_ref.clone();
    for _ in 0..8 {
        let jt_v = numeric_jt_v_no_fpm(
            &h_star, &v_ref, &s_in, &w_q, &w_k, &w_v, &w_o, &w_in, &norm, damping, h, d,
        );
        for i in 0..v_ref.len() {
            v_ref[i] = b_rep[i] + jt_v[i];
        }
    }

    let cos = cosine(&v_picard, &v_ref);
    assert!(
        cos > 0.80,
        "Staged Picard/numeric cosine too low for no-fpm small case: {cos}"
    );

    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
}

#[test]
#[ignore = "Performance smoke for staged Picard only."]
fn test_staged_picard_only_perf_smoke_no_fpm() {
    std::env::set_var("AIDEEN_DEQ_NO_FPM", "1");

    let mut config = ArchitectureConfig::default();
    config.d_r = 512;
    config.h_slots = 8;
    config.ctx_len = 8;
    config.max_deq_iters = 4;

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");

    let seq_len = 8u32;
    let d = config.d_r;
    let len_sq = d * d;
    let damping = 0.9f32;

    let s_in = vec![0.1f32; seq_len as usize * d];
    let w_q = vec![0.02f32; len_sq];
    let w_k = vec![0.015f32; len_sq];
    let w_v = vec![0.01f32; len_sq];
    let w_o = vec![0.012f32; len_sq];
    let w_in = vec![0.01f32; len_sq];
    let w_x = vec![0.0f32; len_sq];
    let w_out = vec![0.0f32; len_sq];
    let a_log = vec![-0.5f32; d];
    let norm = vec![1.0f32; d];

    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &w_in,
        &w_x,
        &w_out,
        &a_log,
        &norm,
        true,
    )
    .expect("GPU DEQ forward failed");

    let dl = vec![0.05f32; seq_len as usize * d];
    gpu.queue
        .write_buffer(&gpu.adj_bufs.b_dl, 0, bytemuck::cast_slice(&dl));

    let t0 = std::time::Instant::now();
    gpu.run_staged_adjoint_picard_no_readback(seq_len, damping, 8, None, None, true, 1)
        .expect("Staged Picard failed");
    let staged_ms = t0.elapsed().as_millis();

    eprintln!(
        "[PICARD-PERF] seq_len={} d={} h_slots={} staged_ms={}",
        seq_len, d, config.h_slots, staged_ms
    );

    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
}

fn run_staged_picard_perf_case(seq_len: u32, d: usize, h_slots: usize, iters: u32) -> u128 {
    std::env::set_var("AIDEEN_DEQ_NO_FPM", "1");

    let mut config = ArchitectureConfig::default();
    config.d_r = d;
    config.h_slots = h_slots;
    config.ctx_len = seq_len as usize;
    config.max_deq_iters = 4;

    let gpu = GpuDeqBackend::new_blocking(config.clone()).expect("GpuDeqBackend init failed");

    let len_sq = d * d;
    let damping = 0.9f32;

    let s_in = vec![0.1f32; seq_len as usize * d];
    let w_q = vec![0.02f32; len_sq];
    let w_k = vec![0.015f32; len_sq];
    let w_v = vec![0.01f32; len_sq];
    let w_o = vec![0.012f32; len_sq];
    let w_in = vec![0.01f32; len_sq];
    let w_x = vec![0.0f32; len_sq];
    let w_out = vec![0.0f32; len_sq];
    let a_log = vec![-0.5f32; d];
    let norm = vec![1.0f32; d];

    gpu.run_forward_deq_no_readback(
        1,
        seq_len,
        config.max_deq_iters as u32,
        config.deq_epsilon,
        damping,
        &s_in,
        &w_q,
        &w_k,
        &w_v,
        &w_o,
        &w_in,
        &w_x,
        &w_out,
        &a_log,
        &norm,
        true,
    )
    .expect("GPU DEQ forward failed");

    let dl = vec![0.05f32; seq_len as usize * d];
    gpu.queue
        .write_buffer(&gpu.adj_bufs.b_dl, 0, bytemuck::cast_slice(&dl));

    let t0 = std::time::Instant::now();
    gpu.run_staged_adjoint_picard_no_readback(seq_len, damping, iters, None, None, true, 1)
        .expect("Staged Picard failed");
    let elapsed = t0.elapsed().as_millis();

    std::env::remove_var("AIDEEN_DEQ_NO_FPM");
    elapsed
}

#[test]
#[ignore = "Sweep staged Picard runtime against sequence length."]
fn test_staged_picard_seq_len_sweep_no_fpm() {
    let d = 512usize;
    let h_slots = 8usize;
    let iters = 8u32;
    let seqs = [8u32, 16, 32, 64, 98];
    for seq_len in seqs {
        let ms = run_staged_picard_perf_case(seq_len, d, h_slots, iters);
        eprintln!(
            "[PICARD-SWEEP] seq_len={} d={} h_slots={} iters={} staged_ms={}",
            seq_len, d, h_slots, iters, ms
        );
    }
}

#[test]
#[ignore = "Exact staged Picard perf case matching trainer fused profile."]
fn test_staged_picard_exact_trainer_case_no_fpm() {
    let ms = run_staged_picard_perf_case(98, 512, 8, 6);
    eprintln!(
        "[PICARD-TRAINER-CASE] seq_len=98 d=512 h_slots=8 iters=6 staged_ms={}",
        ms
    );
}

#[test]
#[ignore = "Single-iteration staged Picard perf case matching trainer geometry."]
fn test_staged_picard_exact_trainer_case_one_iter_no_fpm() {
    let ms = run_staged_picard_perf_case(98, 512, 8, 1);
    eprintln!(
        "[PICARD-TRAINER-CASE-1] seq_len=98 d=512 h_slots=8 iters=1 staged_ms={}",
        ms
    );
}
