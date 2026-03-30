use aideen_backbone::tokenizer::Tokenizer;
extern crate bytemuck;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use aideen_training::trainer::Trainer;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
    Gpu,
}

#[derive(Debug)]
struct TestResult {
    name: &'static str,
    status: &'static str,
    detail: String,
}

fn make_config() -> ArchitectureConfig {
    let mut cfg = ArchitectureConfig::default();
    cfg.vocab_size = 64;
    cfg.ctx_len = 32;
    cfg.d_r = 96;
    cfg.h_slots = 8;
    cfg.max_deq_iters = 12;
    cfg.adj_iters = 8;
    cfg.train_deq = true;
    cfg.deq_epsilon = 1e-4;
    cfg.deq_grad_scale = 0.01;
    cfg
}

fn make_trainer(seed: u64, backend: Backend, cfg: &ArchitectureConfig) -> Trainer {
    let mut tok = Tokenizer::new_empty(cfg.vocab_size, cfg.clone());
    tok.vocab = (0..cfg.vocab_size)
        .map(|i| (32u8 + (i as u8 % 90)) as char)
        .collect();

    let mut rng = StdRng::seed_from_u64(seed);
    tok.embeddings = DMatrix::from_fn(cfg.vocab_size, cfg.d_r, |_, _| {
        (rng.gen::<f32>() - 0.5) * 0.02
    });

    let mut t = Trainer::from_tokenizer_seeded(tok, 3e-4, seed);
    t.config = cfg.clone();
    t.training_config.lr = 3e-4;
    if backend == Backend::Cpu {
        t.gpu_deq = None;
        t.gpu_lm = None;
        t.gpu_emb = None;
    }
    t
}

fn copy_weights(dst: &mut Trainer, src: &Trainer) {
    dst.tokenizer.embeddings = src.tokenizer.embeddings.clone();
    dst.reasoning.w_q = src.reasoning.w_q.clone();
    dst.reasoning.w_k = src.reasoning.w_k.clone();
    dst.reasoning.w_v = src.reasoning.w_v.clone();
    dst.reasoning.w_o = src.reasoning.w_o.clone();
    dst.reasoning.w_in = src.reasoning.w_in.clone();
    dst.reasoning.w_x = src.reasoning.w_x.clone();
    dst.reasoning.w_out = src.reasoning.w_out.clone();
    dst.reasoning.a_log = src.reasoning.a_log.clone();
    dst.reasoning.norm_scale = src.reasoning.norm_scale.clone();
    dst.reasoning.damping = src.reasoning.damping;

    dst.lm_head.w = src.lm_head.w.clone();
    dst.lm_head.b = src.lm_head.b.clone();
    dst.lm_head.g = src.lm_head.g.clone();
}

fn synthetic_stream(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((i * 7 + i / 3 + 11) % vocab) as u32)
        .collect()
}

fn vec_cos(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = *x as f64;
        let yf = *y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn min_max(v: &[f32]) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in v {
        mn = mn.min(x);
        mx = mx.max(x);
    }
    if !mn.is_finite() || !mx.is_finite() {
        (0.0, 0.0)
    } else {
        (mn, mx)
    }
}

fn delta_vec(after: &[f32], before: &[f32]) -> Vec<f32> {
    after
        .iter()
        .zip(before.iter())
        .map(|(a, b)| a - b)
        .collect::<Vec<f32>>()
}

fn test_forward_parity(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let base = make_trainer(7, Backend::Gpu, cfg);
    if base.gpu_deq.is_none() {
        return TestResult {
            name: "forward_parity_cpu_vs_gpu",
            status: "SKIP",
            detail: "gpu_deq=None (no backend available)".to_string(),
        };
    }

    let mut cpu = make_trainer(8, Backend::Cpu, cfg);
    let mut gpu = make_trainer(9, Backend::Gpu, cfg);
    copy_weights(&mut cpu, &base);
    copy_weights(&mut gpu, &base);

    cpu.frozen_deq = true;
    cpu.frozen_emb = true;
    cpu.frozen_lm = true;
    gpu.frozen_deq = true;
    gpu.frozen_emb = true;
    gpu.frozen_lm = true;

    let ctx = &tokens[0..cfg.ctx_len];
    let tgt = tokens[cfg.ctx_len];
    let l_cpu = cpu.train_step(ctx, tgt, true);
    let l_gpu = gpu.train_step(ctx, tgt, true);
    let rel = ((l_gpu - l_cpu).abs()) / l_cpu.abs().max(1e-6);
    let pass = rel < 0.10 && l_cpu.is_finite() && l_gpu.is_finite();

    TestResult {
        name: "forward_parity_cpu_vs_gpu",
        status: if pass { "PASS" } else { "FAIL" },
        detail: format!(
            "loss_cpu={:.6} loss_gpu={:.6} rel_err={:.4}",
            l_cpu, l_gpu, rel
        ),
    }
}

fn test_cpu_residual_curve(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let t = make_trainer(13, Backend::Cpu, cfg);
    let q = t
        .tokenizer
        .embed_context(&tokens[0..cfg.ctx_len], cfg.ctx_len);
    let mut h = t.reasoning.init(&q);
    let mut deltas = Vec::new();

    for _ in 0..cfg.max_deq_iters.max(2) {
        let h_next = t.reasoning.step(&h, &q, None);
        let mut dmax = 0.0f32;
        for (a, b) in h_next.to_flat().iter().zip(h.to_flat().iter()) {
            dmax = dmax.max((a - b).abs());
        }
        deltas.push(dmax);
        h = h_next;
    }

    let first = deltas.first().copied().unwrap_or(0.0);
    let last = deltas.last().copied().unwrap_or(0.0);
    let non_inc = deltas.windows(2).filter(|w| w[1] <= w[0] * 1.05).count();
    let ratio_non_inc = non_inc as f32 / (deltas.len().saturating_sub(1).max(1) as f32);
    let pass = last < first && ratio_non_inc >= 0.6;

    TestResult {
        name: "cpu_deq_residual_curve",
        status: if pass { "PASS" } else { "FAIL" },
        detail: format!(
            "first={:.3e} last={:.3e} non_inc_ratio={:.2}",
            first, last, ratio_non_inc
        ),
    }
}

fn tiny_overfit_run(
    trainer: &mut Trainer,
    seq: &[u32],
    targets: &[u32],
    steps: usize,
    eps: f32,
) -> (f32, f32) {
    let ctx = seq;
    let tgt = *targets.last().unwrap_or(&0);

    // Forward-only loss without updating.
    let (f_deq, f_emb, f_lm) = (trainer.frozen_deq, trainer.frozen_emb, trainer.frozen_lm);
    trainer.frozen_deq = true;
    trainer.frozen_emb = true;
    trainer.frozen_lm = true;
    let l0 = trainer.train_step(ctx, tgt, true);
    trainer.frozen_deq = f_deq;
    trainer.frozen_emb = f_emb;
    trainer.frozen_lm = f_lm;

    for _ in 0..steps {
        let _ = trainer.train_step(ctx, tgt, true);
    }
    trainer.sync_inference_weights();

    trainer.frozen_deq = true;
    trainer.frozen_emb = true;
    trainer.frozen_lm = true;
    let l1 = trainer.train_step(ctx, tgt, true);
    trainer.frozen_deq = f_deq;
    trainer.frozen_emb = f_emb;
    trainer.frozen_lm = f_lm;

    let _ = eps; // keep stable signature so as not to break callers.
    (l0, l1)
}

fn test_tiny_overfit(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let seq = &tokens[0..cfg.ctx_len];
    let targets = &tokens[1..=cfg.ctx_len];
    let mut cpu = make_trainer(21, Backend::Cpu, cfg);
    let mut gpu = make_trainer(22, Backend::Gpu, cfg);
    copy_weights(&mut gpu, &cpu);

    let (c0, c1) = tiny_overfit_run(&mut cpu, seq, targets, 60, 1e-4);
    let cpu_ok = c1.is_finite() && c0.is_finite() && c1 < c0 * 0.90;

    let gpu_ok_and_metrics: (bool, Option<(f32, f32)>) = if gpu.gpu_deq.is_some() {
        let (g0, g1) = tiny_overfit_run(&mut gpu, seq, targets, 60, 1e-4);
        (
            g1 < g0 * 0.90 && g0.is_finite() && g1.is_finite(),
            Some((g0, g1)),
        )
    } else {
        (true, None)
    };

    let pass = cpu_ok && gpu_ok_and_metrics.0;
    let gpu_msg = match gpu_ok_and_metrics.1 {
        Some((g0, g1)) => format!(" gpu:{:.4}->{:.4}", g0, g1),
        None => " gpu:SKIP".to_string(),
    };
    TestResult {
        name: "tiny_overfit",
        status: if pass { "PASS" } else { "FAIL" },
        detail: format!("cpu:{:.4}->{:.4}{}", c0, c1, gpu_msg),
    }
}

fn test_update_parity(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let base = make_trainer(31, Backend::Gpu, cfg);
    if base.gpu_deq.is_none() {
        return TestResult {
            name: "update_parity_cpu_vs_gpu",
            status: "SKIP",
            detail: "gpu_deq=None".to_string(),
        };
    }
    let mut cpu = make_trainer(32, Backend::Cpu, cfg);
    let mut gpu = make_trainer(33, Backend::Gpu, cfg);
    copy_weights(&mut cpu, &base);
    copy_weights(&mut gpu, &base);
    cpu.frozen_lm = true;
    cpu.frozen_emb = true;
    gpu.frozen_lm = true;
    gpu.frozen_emb = true;

    let seq = &tokens[0..cfg.ctx_len];
    let targets = &tokens[1..=cfg.ctx_len];
    let tgt = *targets.last().unwrap_or(&0);

    let pre_wx = cpu.reasoning.w_x.as_slice().to_vec();
    let pre_alog = cpu.reasoning.a_log.as_slice().to_vec();

    let _ = cpu.train_step(seq, tgt, true);
    let _ = gpu.train_step(seq, tgt, true);
    gpu.sync_inference_weights();

    let dc_wx_cpu = delta_vec(cpu.reasoning.w_x.as_slice(), &pre_wx);
    let dc_alog_cpu = delta_vec(cpu.reasoning.a_log.as_slice(), &pre_alog);

    let dc_wx_gpu = delta_vec(gpu.reasoning.w_x.as_slice(), &pre_wx);
    let dc_alog_gpu = delta_vec(gpu.reasoning.a_log.as_slice(), &pre_alog);

    let cos_wx = vec_cos(&dc_wx_cpu, &dc_wx_gpu);
    let cos_alog = vec_cos(&dc_alog_cpu, &dc_alog_gpu);
    let nr_wx = l2_norm(&dc_wx_gpu) / l2_norm(&dc_wx_cpu).max(1e-9);
    let nr_alog = l2_norm(&dc_alog_gpu) / l2_norm(&dc_alog_cpu).max(1e-9);

    let pass = cos_wx > 0.70
        && cos_alog > 0.70
        && (0.2..=5.0).contains(&nr_wx)
        && (0.2..=5.0).contains(&nr_alog);

    TestResult {
        name: "update_parity_cpu_vs_gpu",
        status: if pass { "PASS" } else { "FAIL" },
        detail: format!(
            "frozen_lm_emb cos(wx/alog)={:.3}/{:.3} norm_ratio={:.2}/{:.2}",
            cos_wx, cos_alog, nr_wx, nr_alog
        ),
    }
}

fn test_ablation_iters(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let mut normal = make_trainer(41, Backend::Cpu, cfg);
    let mut one_iter = make_trainer(42, Backend::Cpu, cfg);
    copy_weights(&mut one_iter, &normal);
    one_iter.adaptive_max_iters = 1;

    let seq = &tokens[0..cfg.ctx_len];
    let targets = &tokens[1..=cfg.ctx_len];
    let (_, ln) = tiny_overfit_run(&mut normal, seq, targets, 80, 1e-4);
    let (_, la) = tiny_overfit_run(&mut one_iter, seq, targets, 80, 1e-4);

    let pass = ln <= la * 1.05;
    let note = if la + 1e-6 < ln {
        "solver_suspect=YES"
    } else {
        "solver_suspect=NO"
    };
    TestResult {
        name: "ablation_one_iter_vs_normal",
        status: if pass { "PASS" } else { "WARN" },
        detail: format!("loss_normal={:.4} loss_1iter={:.4} {}", ln, la, note),
    }
}

/// Traces intermediate vectors (h_star, dl_dh, v, grad_wx, grad_alog) for one update step
/// and reports cosine similarity at each stage to locate where CPU vs GPU directions first diverge.
fn test_update_direction(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let base = make_trainer(51, Backend::Gpu, cfg);
    if base.gpu_deq.is_none() {
        return TestResult {
            name: "update_direction_stages",
            status: "SKIP",
            detail: "gpu_deq=None".to_string(),
        };
    }

    let mut cpu = make_trainer(52, Backend::Cpu, cfg);
    let mut gpu = make_trainer(53, Backend::Gpu, cfg);
    copy_weights(&mut cpu, &base);
    copy_weights(&mut gpu, &base);
    cpu.frozen_lm = true;
    cpu.frozen_emb = true;
    cpu.frozen_deq = true;
    gpu.frozen_lm = true;
    gpu.frozen_emb = true;
    gpu.frozen_deq = true;

    let seq = &tokens[0..cfg.ctx_len];
    let tgt = tokens[cfg.ctx_len];

    // ── CPU intermediates ──────────────────────────────────────────────────────
    let query = cpu.tokenizer.embed_context(seq, cfg.ctx_len);
    // Match trainer dynamics used by GPU path (adaptive defaults at step 0).
    cpu.reasoning.damping = cpu.adaptive_damping;
    let mut h = cpu.reasoning.init(&query);
    for _ in 0..cpu.adaptive_max_iters.max(1) {
        h = cpu.reasoning.step(&h, &query, None);
    }
    let h_star_cpu_flat = h.to_flat();

    let d_r = cfg.d_r;
    let h_slots = cfg.h_slots;
    let mut h_pooled_cpu = vec![0.0f32; d_r];
    for k in 0..h_slots {
        for d in 0..d_r {
            h_pooled_cpu[d] += h_star_cpu_flat[k * d_r + d];
        }
    }
    for v in h_pooled_cpu.iter_mut() {
        *v /= h_slots as f32;
    }

    let logits_cpu = cpu.lm_head.forward_on_flat(&h_pooled_cpu);
    let dl_dlogits_cpu = aideen_training::loss::cross_entropy_grad(&logits_cpu, tgt);
    let h_pooled_cpu_dv = nalgebra::DVector::from_vec(h_pooled_cpu.clone());
    let (_, dl_dh_cpu) = aideen_training::gradients::lmhead_backward(
        &dl_dlogits_cpu,
        &h_pooled_cpu_dv,
        &cpu.lm_head.w,
        &cpu.lm_head.g,
    );

    cpu.frozen_deq = false; // re-enable for deq_implicit_grad (read-only call)
    let v_cpu = aideen_training::gradients::deq_implicit_grad(
        &cpu.reasoning,
        &h,
        &query,
        &dl_dh_cpu,
        cfg.adj_iters,
    );
    cpu.frozen_deq = true;

    let gs = cfg.deq_grad_scale;
    let grad_wx_cpu: Vec<f32> = (v_cpu.clone() * query.transpose() * gs).as_slice().to_vec();
    let grad_alog_cpu: Vec<f32> = (v_cpu.clone() * gs).as_slice().to_vec();

    // ── GPU intermediates ──────────────────────────────────────────────────────
    // Run the same DEQ setup as CPU (same pooled query, same init state) to
    // isolate math parity from trainer pipeline differences.
    let h_init_gpu = gpu.reasoning.init(&query);
    let h_init_gpu_flat = h_init_gpu.to_flat();
    let gpu_query = gpu.tokenizer.embed_context(seq, cfg.ctx_len);
    let s_in_gpu = gpu_query.as_slice().to_vec();
    let gpu_deq = gpu.gpu_deq.as_ref().unwrap();
    gpu_deq.queue.write_buffer(
        &gpu_deq.bridge.hcurr_buf,
        0,
        bytemuck::cast_slice(&h_init_gpu_flat),
    );
    let _ = gpu_deq.run_forward_deq_no_readback(
        1,
        1,
        gpu.adaptive_max_iters.max(1) as u32,
        cfg.deq_epsilon,
        gpu.adaptive_damping,
        &s_in_gpu,
        &gpu.reasoning.w_q_gpu_flat(),
        &gpu.reasoning.w_k_gpu_flat(),
        &gpu.reasoning.w_v_gpu_flat(),
        &gpu.reasoning.w_o_gpu_flat(),
        &gpu.reasoning.w_in_gpu_flat(),
        gpu.reasoning.w_x.as_slice(),
        gpu.reasoning.w_out.as_slice(),
        gpu.reasoning.a_log.as_slice(),
        gpu.reasoning.norm_scale.as_slice(),
        true,
    );
    let h_star_gpu_flat = gpu_deq.read_hnext();
    let h_star_gpu = HSlots::from_flat(&h_star_gpu_flat, cfg);

    let mut h_pooled_gpu = vec![0.0f32; d_r];
    for k in 0..h_slots {
        for d in 0..d_r {
            h_pooled_gpu[d] += h_star_gpu_flat[k * d_r + d];
        }
    }
    for v in h_pooled_gpu.iter_mut() {
        *v /= h_slots as f32;
    }

    let logits_gpu = gpu.lm_head.forward_on_flat(&h_pooled_gpu);
    let dl_dlogits_gpu = aideen_training::loss::cross_entropy_grad(&logits_gpu, tgt);
    let h_pooled_gpu_dv = nalgebra::DVector::from_vec(h_pooled_gpu.clone());
    let (_, dl_dh_gpu) = aideen_training::gradients::lmhead_backward(
        &dl_dlogits_gpu,
        &h_pooled_gpu_dv,
        &gpu.lm_head.w,
        &gpu.lm_head.g,
    );

    let v_gpu = aideen_training::gradients::deq_implicit_grad(
        &gpu.reasoning,
        &h_star_gpu,
        &gpu_query,
        &dl_dh_gpu,
        cfg.adj_iters,
    );

    let grad_wx_gpu: Vec<f32> = (v_gpu.clone() * gpu_query.transpose() * gs)
        .as_slice()
        .to_vec();
    let grad_alog_gpu: Vec<f32> = (v_gpu.clone() * gs).as_slice().to_vec();

    // ── Cosines at each stage ──────────────────────────────────────────────────
    let cos_hstar = vec_cos(&h_star_cpu_flat, &h_star_gpu_flat);
    let cos_dlh = vec_cos(dl_dh_cpu.as_slice(), dl_dh_gpu.as_slice());
    let cos_v = vec_cos(v_cpu.as_slice(), v_gpu.as_slice());
    let cos_wx = vec_cos(&grad_wx_cpu, &grad_wx_gpu);
    let cos_alog = vec_cos(&grad_alog_cpu, &grad_alog_gpu);

    // Identify where direction first breaks (cosine drops below 0.65).
    let first_break = if cos_hstar < 0.65 {
        "h_star"
    } else if cos_dlh < 0.65 {
        "dl_dh"
    } else if cos_v < 0.65 {
        "v_cg"
    } else if cos_wx < 0.65 {
        "grad_wx"
    } else {
        "none"
    };

    let pass = cos_wx > 0.65 && cos_alog > 0.65;
    TestResult {
        name: "update_direction_stages",
        status: if pass { "PASS" } else { "FAIL" },
        detail: format!(
            "h*={:.3} dlh={:.3} v={:.3} wx={:.3} al={:.3} break={}",
            cos_hstar, cos_dlh, cos_v, cos_wx, cos_alog, first_break
        ),
    }
}

/// Compares a single DEQ step CPU vs GPU with identical h_init, weights, and s_in.
/// This isolates the step formula from training pipeline conventions (injection, pooling, etc.).
/// A high cosine (>0.95) means the shader math matches the CPU implementation.
fn test_deq_step_parity(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let t = make_trainer(61, Backend::Gpu, cfg);
    let gpu_deq = match t.gpu_deq.as_ref() {
        Some(g) => g,
        None => {
            return TestResult {
                name: "deq_step_parity_cpu_vs_gpu",
                status: "SKIP",
                detail: "gpu_deq=None".to_string(),
            }
        }
    };

    let seq = &tokens[0..cfg.ctx_len];
    let query = t.tokenizer.embed_context(seq, cfg.ctx_len);
    let s_in: Vec<f32> = query.as_slice().to_vec();

    // ── CPU: 1 step ──────────────────────────────────────────────────────────
    let h_init = t.reasoning.init(&query); // broadcast query → all slots
    let h_new_cpu = t.reasoning.step(&h_init, &query, None);
    let h_new_cpu_flat = h_new_cpu.to_flat();

    // ── GPU: same h_init, same s_in, 1 iteration ─────────────────────────────
    // Write h_init into hcurr_buf so GPU starts from the identical state as CPU.
    let h_init_flat = h_init.to_flat();
    gpu_deq.queue.write_buffer(
        &gpu_deq.bridge.hcurr_buf,
        0,
        bytemuck::cast_slice(&h_init_flat),
    );

    let damping = t.reasoning.damping; // same β as CPU damped_update
    let _ = gpu_deq.run_forward_deq_no_readback(
        1, // batch_size
        1, // seq_len
        1, // max_iters = 1 (single step)
        cfg.deq_epsilon,
        damping,
        &s_in,
        &t.reasoning.w_q_gpu_flat(),
        &t.reasoning.w_k_gpu_flat(),
        &t.reasoning.w_v_gpu_flat(),
        &t.reasoning.w_o_gpu_flat(),
        &t.reasoning.w_in_gpu_flat(),
        t.reasoning.w_x.as_slice(),
        t.reasoning.w_out.as_slice(),
        t.reasoning.a_log.as_slice(),
        t.reasoning.norm_scale.as_slice(),
        true, // upload weights
    );
    let h_new_gpu_flat = gpu_deq.read_hnext();

    let cos = vec_cos(&h_new_cpu_flat, &h_new_gpu_flat);
    let nr = l2_norm(&h_new_gpu_flat) / l2_norm(&h_new_cpu_flat).max(1e-9);
    let pass = cos > 0.95;

    TestResult {
        name: "deq_step_parity_cpu_vs_gpu",
        status: if pass { "PASS" } else { "FAIL" },
        detail: format!("1-step cos={:.4} norm_ratio={:.3}", cos, nr),
    }
}

/// End-to-end H trace (CPU vs GPU) with identical init state and identical injection.
/// Runs 1-step DEQ repeatedly and reports where trajectories diverge.
fn test_h_trace(cfg: &ArchitectureConfig, tokens: &[u32]) -> TestResult {
    let t = make_trainer(71, Backend::Gpu, cfg);
    let gpu_deq = match t.gpu_deq.as_ref() {
        Some(g) => g,
        None => {
            return TestResult {
                name: "h_trace_cpu_vs_gpu",
                status: "SKIP",
                detail: "gpu_deq=None".to_string(),
            }
        }
    };

    let seq = &tokens[0..cfg.ctx_len];
    let query = t.tokenizer.embed_context(seq, cfg.ctx_len);
    let s_in: Vec<f32> = query.as_slice().to_vec();
    let damping = t.reasoning.damping;

    let trace_iters = cfg.max_deq_iters.min(8).max(4);
    let mut h_cpu = t.reasoning.init(&query);
    let h0 = h_cpu.to_flat();

    // Initialize GPU with the same exact state as CPU.
    gpu_deq
        .queue
        .write_buffer(&gpu_deq.bridge.hcurr_buf, 0, bytemuck::cast_slice(&h0));

    let mut logs = Vec::new();
    let mut first_break: Option<usize> = None;

    for it in 0..trace_iters {
        let h_cpu_next = t.reasoning.step(&h_cpu, &query, None);
        let h_cpu_flat = h_cpu_next.to_flat();

        let _ = gpu_deq.run_forward_deq_no_readback(
            1, // batch_size
            1, // seq_len
            1, // max_iters
            cfg.deq_epsilon,
            damping,
            &s_in,
            &t.reasoning.w_q_gpu_flat(),
            &t.reasoning.w_k_gpu_flat(),
            &t.reasoning.w_v_gpu_flat(),
            &t.reasoning.w_o_gpu_flat(),
            &t.reasoning.w_in_gpu_flat(),
            t.reasoning.w_x.as_slice(),
            t.reasoning.w_out.as_slice(),
            t.reasoning.a_log.as_slice(),
            t.reasoning.norm_scale.as_slice(),
            true,
        );
        let h_gpu_flat = gpu_deq.read_hnext();

        let cos = vec_cos(&h_cpu_flat, &h_gpu_flat);
        let nr = l2_norm(&h_gpu_flat) / l2_norm(&h_cpu_flat).max(1e-9);
        let mad = max_abs_diff(&h_cpu_flat, &h_gpu_flat);
        let (cpu_min, cpu_max) = min_max(&h_cpu_flat);
        let (gpu_min, gpu_max) = min_max(&h_gpu_flat);

        if first_break.is_none() && cos < 0.95 {
            first_break = Some(it + 1);
        }

        logs.push(format!(
            "i{} cos={:.3} nr={:.3} max|d|={:.3e} cpu[{:.3},{:.3}] gpu[{:.3},{:.3}]",
            it + 1,
            cos,
            nr,
            mad,
            cpu_min,
            cpu_max,
            gpu_min,
            gpu_max
        ));

        // Feed GPU output back as next input state.
        gpu_deq.queue.write_buffer(
            &gpu_deq.bridge.hcurr_buf,
            0,
            bytemuck::cast_slice(&h_gpu_flat),
        );
        h_cpu = h_cpu_next;
    }

    let pass = first_break.is_none();
    let break_msg = match first_break {
        Some(i) => format!("break_at_iter={}", i),
        None => "break_at_iter=none".to_string(),
    };
    let head = logs.iter().take(4).cloned().collect::<Vec<_>>().join(" | ");

    TestResult {
        name: "h_trace_cpu_vs_gpu",
        status: if pass { "PASS" } else { "FAIL" },
        detail: format!("{} {}", break_msg, head),
    }
}

fn main() {
    let cfg = make_config();
    let tokens = synthetic_stream(cfg.ctx_len * 4, cfg.vocab_size);

    let tests = vec![
        test_forward_parity(&cfg, &tokens),
        test_cpu_residual_curve(&cfg, &tokens),
        test_h_trace(&cfg, &tokens),
        test_tiny_overfit(&cfg, &tokens),
        test_update_parity(&cfg, &tokens),
        test_ablation_iters(&cfg, &tokens),
        test_update_direction(&cfg, &tokens),
        test_deq_step_parity(&cfg, &tokens),
    ];

    println!();
    println!("AIDEEN DEQ Diagnostics");
    println!(
        "config: d_r={} h_slots={} ctx_len={}",
        cfg.d_r, cfg.h_slots, cfg.ctx_len
    );
    println!("{}", "-".repeat(92));
    println!("{:<34} {:<8} {}", "test", "status", "detail");
    println!("{}", "-".repeat(92));
    for t in &tests {
        println!("{:<34} {:<8} {}", t.name, t.status, t.detail);
    }
    println!("{}", "-".repeat(92));

    let fails = tests.iter().filter(|t| t.status == "FAIL").count();
    if fails > 0 {
        std::process::exit(2);
    }
}
