use aideen_backbone::spectral_norm;
use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use aideen_training::loss::cross_entropy;
use aideen_training::trainer::Trainer;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone, Copy)]
struct TokenMetrics {
    iters: usize,
    max_delta: f32,
    contractivity: f32,
    conv_ok: bool,
    loss: f32,
}

#[derive(Debug)]
struct RunSummary {
    name: &'static str,
    mean_loss: f32,
    mean_iters: f32,
    mean_delta: f32,
    mean_contr: f32,
    conv_ratio: f32,
}

fn make_config() -> ArchitectureConfig {
    let mut cfg = ArchitectureConfig::default();
    cfg.vocab_size = 96;
    cfg.ctx_len = 32;
    cfg.d_r = 96;
    cfg.h_slots = 8;
    cfg.max_deq_iters = 16;
    cfg.deq_epsilon = 1e-4;
    cfg.train_deq = true;
    cfg
}

fn make_cpu_trainer(seed: u64, cfg: &ArchitectureConfig) -> Trainer {
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
    t.gpu_deq = None;
    t.gpu_lm = None;
    t.gpu_emb = None;
    t
}

fn synth_tokens(n: usize, vocab: usize) -> Vec<u32> {
    (0..n).map(|i| ((i * 11 + i / 3 + 17) % vocab) as u32).collect()
}

fn rms_norm(x: &DVector<f32>, scale: &DVector<f32>) -> DVector<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / (x.len().max(1) as f32);
    let rms = (mean_sq + 1e-6).sqrt();
    x.map(|v| v / rms).component_mul(scale)
}

fn cross_slot_attn(trainer: &Trainer, h: &HSlots) -> HSlots {
    let d_r = trainer.config.d_r;
    let h_slots = trainer.config.h_slots;
    let scale = (d_r as f32).sqrt().recip();

    let qs: Vec<DVector<f32>> = (0..h_slots).map(|k| &trainer.reasoning.w_q * h.slot(k)).collect();
    let ks: Vec<DVector<f32>> = (0..h_slots).map(|k| &trainer.reasoning.w_k * h.slot(k)).collect();
    let vs: Vec<DVector<f32>> = (0..h_slots).map(|k| &trainer.reasoning.w_v * h.slot(k)).collect();

    let mut next = HSlots::zeros(&trainer.config);
    for q_idx in 0..h_slots {
        let raw_scores: Vec<f32> = ks.iter().map(|k| qs[q_idx].dot(k) * scale).collect();
        let max_s = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw_scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum::<f32>().max(1e-12);
        let attn_w: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        let mixed = attn_w
            .iter()
            .zip(vs.iter())
            .map(|(a, v)| v * *a)
            .fold(DVector::zeros(d_r), |acc, v| acc + v);
        next.set_slot(q_idx, &(&trainer.reasoning.w_o * mixed));
    }
    next
}

fn deq_step_pure(trainer: &Trainer, h: &HSlots, query: &DVector<f32>) -> HSlots {
    let d_r = trainer.config.d_r;
    let h_slots = trainer.config.h_slots;
    let input_signal = &trainer.reasoning.w_in * query;
    let h_attn = cross_slot_attn(trainer, h);

    let mut out = HSlots::zeros(&trainer.config);
    for k in 0..h_slots {
        let h_prev = h.slot(k);
        let attn = h_attn.slot(k);
        let combined = attn + &input_signal + (&h_prev * trainer.reasoning.residual_alpha);
        let f_h = rms_norm(&combined, &trainer.reasoning.norm_scale);
        let beta = trainer.reasoning.damping;
        let h_new = f_h * beta + h_prev * (1.0 - beta);
        let mut clipped = h_new.clone();
        for d in 0..d_r {
            clipped[d] = clipped[d].clamp(-10.0, 10.0);
        }
        out.set_slot(k, &clipped);
    }
    out
}

fn pooled(h: &HSlots, cfg: &ArchitectureConfig) -> DVector<f32> {
    let mut acc = DVector::zeros(cfg.d_r);
    for k in 0..cfg.h_slots {
        acc += h.slot(k);
    }
    acc / cfg.h_slots as f32
}

fn run_picard_current(trainer: &Trainer, h0: &HSlots, q: &DVector<f32>) -> (HSlots, usize, f32, f32) {
    let mut h = h0.clone();
    let mut prev_delta = 0.0f32;
    let mut contr = 0.0f32;
    let mut final_delta = 0.0f32;
    let mut used = trainer.config.max_deq_iters;
    for i in 0..trainer.config.max_deq_iters {
        let h_next = trainer.reasoning.step(&h, q, None);
        let mut dmax = 0.0f32;
        for (a, b) in h_next.to_flat().iter().zip(h.to_flat().iter()) {
            dmax = dmax.max((a - b).abs());
        }
        if i > 0 && prev_delta > 1e-12 {
            contr = dmax / prev_delta;
        }
        prev_delta = dmax;
        final_delta = dmax;
        h = h_next;
        if dmax < trainer.config.deq_epsilon {
            used = i + 1;
            break;
        }
    }
    (h, used, final_delta, contr)
}

fn run_picard_pure(trainer: &Trainer, h0: &HSlots, q: &DVector<f32>) -> (HSlots, usize, f32, f32) {
    let mut h = h0.clone();
    let mut prev_delta = 0.0f32;
    let mut contr = 0.0f32;
    let mut final_delta = 0.0f32;
    let mut used = trainer.config.max_deq_iters;
    for i in 0..trainer.config.max_deq_iters {
        let h_next = deq_step_pure(trainer, &h, q);
        let mut dmax = 0.0f32;
        for (a, b) in h_next.to_flat().iter().zip(h.to_flat().iter()) {
            dmax = dmax.max((a - b).abs());
        }
        if i > 0 && prev_delta > 1e-12 {
            contr = dmax / prev_delta;
        }
        prev_delta = dmax;
        final_delta = dmax;
        h = h_next;
        if dmax < trainer.config.deq_epsilon {
            used = i + 1;
            break;
        }
    }
    (h, used, final_delta, contr)
}

fn run_current(trainer: &Trainer, tokens: &[u32]) -> RunSummary {
    let mut h_state = HSlots::zeros(&trainer.config);
    let mut rows = Vec::<TokenMetrics>::new();

    for t in trainer.config.ctx_len..(tokens.len() - 1) {
        let ctx = &tokens[t - trainer.config.ctx_len..t];
        let target = tokens[t + 1];
        let q = trainer.tokenizer.embed_context(ctx, trainer.config.ctx_len);

        let (h_star, iters, max_delta, contr) = run_picard_current(trainer, &h_state, &q);
        let logits = trainer.lm_head.forward(&h_star);
        let loss = cross_entropy(&logits, target);
        rows.push(TokenMetrics {
            iters,
            max_delta,
            contractivity: contr,
            conv_ok: max_delta < trainer.config.deq_epsilon,
            loss,
        });
        h_state = h_star;
    }

    summarize("current_deq", &rows)
}

fn run_pure_plus_external_mamba(trainer: &Trainer, tokens: &[u32]) -> RunSummary {
    let mut m_state = DVector::zeros(trainer.config.d_r);
    let mut rows = Vec::<TokenMetrics>::new();

    for t in trainer.config.ctx_len..(tokens.len() - 1) {
        let ctx = &tokens[t - trainer.config.ctx_len..t];
        let target = tokens[t + 1];
        let q = trainer.tokenizer.embed_context(ctx, trainer.config.ctx_len);

        // h0 initialized from external Mamba memory state (broadcast to slots)
        let h0 = HSlots::from_broadcast(&m_state, &trainer.config);
        let (h_star, iters, max_delta, contr) = run_picard_pure(trainer, &h0, &q);

        let h_pool = pooled(&h_star, &trainer.config);
        let logits = trainer.lm_head.forward_on_flat(h_pool.as_slice());
        let loss = cross_entropy(&logits, target);
        rows.push(TokenMetrics {
            iters,
            max_delta,
            contractivity: contr,
            conv_ok: max_delta < trainer.config.deq_epsilon,
            loss,
        });

        // External Mamba state update across tokens: m_t = a*m_{t-1} + (1-a)*W_x*h_pool
        // Use slot-0 decay (single shared state, legacy diagnostic mode).
        let a = nalgebra::DVector::from_fn(trainer.config.d_r, |d, _| {
            1.0 / (1.0 + trainer.reasoning.a_log[(0, d)].exp())
        });
        let x = &trainer.reasoning.w_x * &h_pool;
        m_state = a.zip_map(&m_state, |aa, mm| aa * mm)
            + a.map(|aa| 1.0 - aa).zip_map(&x, |bb, xx| bb * xx);
        m_state = &trainer.reasoning.w_out * m_state;
    }

    summarize("pure_deq_ext_mamba", &rows)
}

fn run_pure_per_slot(trainer: &Trainer, tokens: &[u32]) -> RunSummary {
    let h_slots = trainer.config.h_slots;
    let mut m_states: Vec<DVector<f32>> = vec![DVector::zeros(trainer.config.d_r); h_slots];
    let mut rows = Vec::<TokenMetrics>::new();

    for t in trainer.config.ctx_len..(tokens.len() - 1) {
        let ctx = &tokens[t - trainer.config.ctx_len..t];
        let target = tokens[t + 1];
        let q = trainer.tokenizer.embed_context(ctx, trainer.config.ctx_len);

        // h0: cada slot arranca desde su propio estado Mamba cross-token
        let mut h0 = HSlots::zeros(&trainer.config);
        for k in 0..h_slots {
            h0.set_slot(k, &m_states[k]);
        }
        let (h_star, iters, max_delta, contr) = run_picard_pure(trainer, &h0, &q);

        let h_pool = pooled(&h_star, &trainer.config);
        let logits = trainer.lm_head.forward_on_flat(h_pool.as_slice());
        let loss = cross_entropy(&logits, target);
        rows.push(TokenMetrics {
            iters,
            max_delta,
            contractivity: contr,
            conv_ok: max_delta < trainer.config.deq_epsilon,
            loss,
        });

        // Actualizar cada estado Mamba por slot independientemente
        for k in 0..h_slots {
            let a = nalgebra::DVector::from_fn(trainer.config.d_r, |d, _| {
                1.0 / (1.0 + trainer.reasoning.a_log[(k, d)].exp())
            });
            let h_k = h_star.slot(k);
            let x = &trainer.reasoning.w_x * &h_k;
            let new_state = a.zip_map(&m_states[k], |aa, mm| aa * mm)
                + a.map(|aa| 1.0 - aa).zip_map(&x, |bb, xx| bb * xx);
            m_states[k] = &trainer.reasoning.w_out * new_state;
        }
    }

    summarize("pure_per_slot", &rows)
}

fn run_sweep(trainer: &mut Trainer, tokens: &[u32]) {
    let alphas: &[f32] = &[1.0, 0.5, 0.2, 0.0];
    let dampings: &[f32] = &[0.70, 0.80, 0.90, 0.95];
    let iters_list: &[usize] = &[12, 16, 20];

    println!();
    println!("SWEEP: residual_alpha × damping × max_iters  (pure_deq + ext_mamba per_slot)");
    println!(
        "{:<8} {:<6} {:<6}  {:<8} {:<10} {:<8} {:<8}",
        "alpha", "damp", "iters", "loss", "maxΔ", "contr", "conv%"
    );
    println!("{}", "-".repeat(66));

    for &alpha in alphas {
        for &damp in dampings {
            for &max_iters in iters_list {
                trainer.reasoning.residual_alpha = alpha;
                trainer.reasoning.damping = damp;
                trainer.config.max_deq_iters = max_iters;

                let s = run_pure_per_slot(trainer, tokens);
                println!(
                    "{:<8.2} {:<6.2} {:<6}  {:<8.4} {:<10.3e} {:<8.3} {:<7.1}%",
                    alpha, damp, max_iters,
                    s.mean_loss, s.mean_delta, s.mean_contr, s.conv_ratio * 100.0
                );
            }
        }
        println!();
    }
}

/// Aplica spectral norm a todas las matrices de atención con un threshold dado.
/// Crea un trainer fresco (misma semilla) para que cada punto del sweep sea independiente.
fn renorm_for_deq(trainer: &mut Trainer, threshold: f32) {
    let n = 20;
    spectral_norm::normalize_if_needed(&mut trainer.reasoning.w_q, threshold, n);
    spectral_norm::normalize_if_needed(&mut trainer.reasoning.w_k, threshold, n);
    spectral_norm::normalize_if_needed(&mut trainer.reasoning.w_v, threshold, n);
    spectral_norm::normalize_if_needed(&mut trainer.reasoning.w_o, threshold, n);
    spectral_norm::normalize_if_needed(&mut trainer.reasoning.w_in, threshold, n);
}

fn run_init_sweep(cfg: &ArchitectureConfig, tokens: &[u32]) {
    let thresholds: &[f32] = &[0.70, 0.30, 0.10, 0.05];
    let alphas: &[f32] = &[0.0, 0.2, 1.0];
    let damping = 0.70_f32;
    let max_iters = 20_usize;

    println!();
    println!(
        "INIT SWEEP: spectral_threshold × residual_alpha  (damping={damping:.2}, max_iters={max_iters})"
    );
    println!(
        "{:<8} {:<8}  {:<8} {:<10} {:<8} {:<8}",
        "thresh", "alpha", "loss", "maxΔ", "contr", "conv%"
    );
    println!("{}", "-".repeat(60));

    for &threshold in thresholds {
        for &alpha in alphas {
            // Trainer fresco con la misma semilla para cada punto del sweep
            let mut t = make_cpu_trainer(2026, cfg);
            t.reasoning.damping = damping;
            t.reasoning.residual_alpha = alpha;
            t.config.max_deq_iters = max_iters;
            renorm_for_deq(&mut t, threshold);

            let s = run_pure_per_slot(&t, tokens);
            println!(
                "{:<8.2} {:<8.2}  {:<8.4} {:<10.3e} {:<8.3} {:<7.1}%",
                threshold, alpha, s.mean_loss, s.mean_delta, s.mean_contr,
                s.conv_ratio * 100.0
            );
        }
        println!();
    }
}

fn summarize(name: &'static str, rows: &[TokenMetrics]) -> RunSummary {
    let n = rows.len().max(1) as f32;
    RunSummary {
        name,
        mean_loss: rows.iter().map(|r| r.loss).sum::<f32>() / n,
        mean_iters: rows.iter().map(|r| r.iters as f32).sum::<f32>() / n,
        mean_delta: rows.iter().map(|r| r.max_delta).sum::<f32>() / n,
        mean_contr: rows.iter().map(|r| r.contractivity).sum::<f32>() / n,
        conv_ratio: rows.iter().filter(|r| r.conv_ok).count() as f32 / n,
    }
}

fn print_summary(s: &RunSummary) {
    println!(
        "{:<20} loss={:.4} iters={:.2} maxΔ={:.3e} contr={:.3} conv_ok={:.1}%",
        s.name,
        s.mean_loss,
        s.mean_iters,
        s.mean_delta,
        s.mean_contr,
        s.conv_ratio * 100.0
    );
}

fn main() {
    let cfg = make_config();
    let mut trainer = make_cpu_trainer(2026, &cfg);
    // v14 defaults: sin residual dentro del DEQ, damping moderado
    trainer.reasoning.damping = 0.70;
    trainer.reasoning.residual_alpha = 0.0;

    let tokens = synth_tokens(512, cfg.vocab_size);

    let cur = run_current(&trainer, &tokens);
    let pure = run_pure_plus_external_mamba(&trainer, &tokens);
    let per_slot = run_pure_per_slot(&trainer, &tokens);

    println!();
    println!("AIDEEN DEQ Architecture Sandbox (CPU only)");
    println!(
        "config: d_r={} h_slots={} ctx={} max_iters={} eps={} damping={} residual_alpha={}",
        cfg.d_r,
        cfg.h_slots,
        cfg.ctx_len,
        cfg.max_deq_iters,
        cfg.deq_epsilon,
        trainer.reasoning.damping,
        trainer.reasoning.residual_alpha
    );
    println!("{}", "-".repeat(100));
    print_summary(&cur);
    print_summary(&pure);
    print_summary(&per_slot);
    println!("{}", "-".repeat(100));
    println!("Interpretación: menor maxΔ/contr y mayor conv_ok = dinámica DEQ más sana.");
    println!("per_slot vs broadcast: ver si slots independientes mejoran contractividad.");

    run_sweep(&mut trainer, &tokens);
    run_init_sweep(&cfg, &tokens);
}
