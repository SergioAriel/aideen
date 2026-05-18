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
    config.d_r = 16;
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
    gpu.run_staged_adjoint_picard_no_readback(seq_len, damping, 8, None, true, 1)
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
    gpu.run_staged_adjoint_picard_no_readback(seq_len, damping, 8, None, true, 1)
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
    gpu.run_staged_adjoint_picard_no_readback(seq_len, damping, iters, None, true, 1)
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
