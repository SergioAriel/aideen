use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use aideen_training::trainer::Trainer;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Clone, Copy, Debug)]
enum AblationStage {
    InputOnly,
    InputMamba,
    InputAttn,
    Full,
}

#[derive(Debug)]
struct StageResult {
    stage: &'static str,
    sens_cos: f32,
    sens_l2: f32,
    first_delta: f32,
    last_delta: f32,
    non_inc_ratio: f32,
    jac_gain: f32,
    attn_norm: f32,
    mamba_norm: f32,
    input_norm: f32,
    resid_norm: f32,
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

fn zero_mat(m: &mut DMatrix<f32>) {
    m.fill(0.0);
}

fn apply_stage(trainer: &mut Trainer, stage: AblationStage) {
    match stage {
        AblationStage::InputOnly => {
            zero_mat(&mut trainer.reasoning.w_q);
            zero_mat(&mut trainer.reasoning.w_k);
            zero_mat(&mut trainer.reasoning.w_v);
            zero_mat(&mut trainer.reasoning.w_o);
            zero_mat(&mut trainer.reasoning.w_x);
            zero_mat(&mut trainer.reasoning.w_out);
        }
        AblationStage::InputMamba => {
            zero_mat(&mut trainer.reasoning.w_q);
            zero_mat(&mut trainer.reasoning.w_k);
            zero_mat(&mut trainer.reasoning.w_v);
            zero_mat(&mut trainer.reasoning.w_o);
        }
        AblationStage::InputAttn => {
            zero_mat(&mut trainer.reasoning.w_x);
            zero_mat(&mut trainer.reasoning.w_out);
        }
        AblationStage::Full => {}
    }
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

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn diff_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f32>()
        .sqrt()
}

fn flatten_step(trainer: &Trainer, h: &HSlots, s: &DVector<f32>) -> Vec<f32> {
    trainer.reasoning.step(h, s, None).to_flat()
}

fn estimate_local_jac_gain(
    trainer: &Trainer,
    h: &HSlots,
    s: &DVector<f32>,
    probes: usize,
    eps: f32,
    seed: u64,
) -> f32 {
    let base = flatten_step(trainer, h, s);
    let n = base.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut best = 0.0f32;

    for _ in 0..probes {
        let mut v = vec![0.0f32; n];
        for x in &mut v {
            *x = rng.gen_range(-1.0f32..1.0f32);
        }
        let vnorm = l2(&v).max(1e-12);
        for x in &mut v {
            *x /= vnorm;
        }

        let h0 = h.to_flat();
        let h_pert: Vec<f32> = h0
            .iter()
            .zip(v.iter())
            .map(|(a, b)| a + eps * b)
            .collect();
        let h_pert_slots = HSlots::from_flat(&h_pert, &trainer.config);
        let y_pert = flatten_step(trainer, &h_pert_slots, s);
        let gain = diff_l2(&y_pert, &base) / eps.max(1e-12);
        best = best.max(gain);
    }
    best
}

fn run_picard(
    trainer: &Trainer,
    s: &DVector<f32>,
    iters: usize,
) -> (Vec<f32>, f32, f32, f32, HSlots) {
    let mut h = trainer.reasoning.init(s);
    let mut deltas = Vec::with_capacity(iters.max(1));
    for _ in 0..iters.max(1) {
        let h_next = trainer.reasoning.step(&h, s, None);
        let mut dmax = 0.0f32;
        for (a, b) in h_next.to_flat().iter().zip(h.to_flat().iter()) {
            dmax = dmax.max((a - b).abs());
        }
        deltas.push(dmax);
        h = h_next;
    }
    let first = *deltas.first().unwrap_or(&0.0);
    let last = *deltas.last().unwrap_or(&0.0);
    let non_inc = deltas
        .windows(2)
        .filter(|w| w[1] <= w[0] * 1.02)
        .count();
    let ratio = non_inc as f32 / (deltas.len().saturating_sub(1).max(1) as f32);
    (h.to_flat(), first, last, ratio, h)
}

fn component_norms(trainer: &Trainer, h: &HSlots, s: &DVector<f32>) -> (f32, f32, f32, f32) {
    let d_r = trainer.config.d_r;
    let h_slots = trainer.config.h_slots;
    let scale = (d_r as f32).sqrt().recip();

    let qs: Vec<DVector<f32>> = (0..h_slots).map(|k| &trainer.reasoning.w_q * h.slot(k)).collect();
    let ks: Vec<DVector<f32>> = (0..h_slots).map(|k| &trainer.reasoning.w_k * h.slot(k)).collect();
    let vs: Vec<DVector<f32>> = (0..h_slots).map(|k| &trainer.reasoning.w_v * h.slot(k)).collect();

    let mut attn_total = 0.0f32;
    for q_idx in 0..h_slots {
        let raw_scores: Vec<f32> = ks.iter().map(|k| qs[q_idx].dot(k) * scale).collect();
        let max_s = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw_scores.iter().map(|v| (v - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum::<f32>().max(1e-12);
        let attn: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
        let mixed = attn
            .iter()
            .zip(vs.iter())
            .map(|(a, v)| v * *a)
            .fold(DVector::zeros(d_r), |acc, v| acc + v);
        let out = &trainer.reasoning.w_o * mixed;
        attn_total += out.dot(&out);
    }

    let mut mamba_total = 0.0f32;
    for k in 0..h_slots {
        let a_bar = nalgebra::DVector::from_fn(trainer.config.d_r, |d, _| {
            1.0 / (1.0 + trainer.reasoning.a_log[(k, d)].exp())
        });
        let b_bar = a_bar.map(|a| 1.0 - a);
        let hs = h.slot(k);
        let x_proj = &trainer.reasoning.w_x * &hs;
        let y = a_bar.zip_map(&hs, |a, hv| a * hv) + b_bar.zip_map(&x_proj, |b, xv| b * xv);
        let out = &trainer.reasoning.w_out * y;
        mamba_total += out.dot(&out);
    }

    let input = &trainer.reasoning.w_in * s;
    let input_total = input.dot(&input) * h_slots as f32;
    let mut resid_total = 0.0f32;
    for k in 0..h_slots {
        let hs = h.slot(k);
        resid_total += hs.dot(&hs) * trainer.reasoning.residual_alpha * trainer.reasoning.residual_alpha;
    }

    (
        attn_total.sqrt(),
        mamba_total.sqrt(),
        input_total.sqrt(),
        resid_total.sqrt(),
    )
}

fn stage_label(stage: AblationStage) -> &'static str {
    match stage {
        AblationStage::InputOnly => "input_only",
        AblationStage::InputMamba => "input_mamba",
        AblationStage::InputAttn => "input_attn",
        AblationStage::Full => "full",
    }
}

fn run_stage(cfg: &ArchitectureConfig, stage: AblationStage) -> StageResult {
    let mut trainer = make_cpu_trainer(123, cfg);
    apply_stage(&mut trainer, stage);

    let tokens1: Vec<u32> = (0..cfg.ctx_len).map(|i| ((i * 7 + 3) % cfg.vocab_size) as u32).collect();
    let tokens2: Vec<u32> = (0..cfg.ctx_len).map(|i| ((i * 13 + 19) % cfg.vocab_size) as u32).collect();
    let s1 = trainer.tokenizer.embed_context(&tokens1, cfg.ctx_len);
    let s2 = trainer.tokenizer.embed_context(&tokens2, cfg.ctx_len);

    let (h1, first, last, non_inc_ratio, hslots_last) =
        run_picard(&trainer, &s1, cfg.max_deq_iters.max(2));
    let (h2, _, _, _, _) = run_picard(&trainer, &s2, cfg.max_deq_iters.max(2));

    let jac_gain = estimate_local_jac_gain(&trainer, &hslots_last, &s1, 8, 1e-3, 777);
    let (attn_n, mamba_n, input_n, resid_n) = component_norms(&trainer, &hslots_last, &s1);

    StageResult {
        stage: stage_label(stage),
        sens_cos: vec_cos(&h1, &h2),
        sens_l2: diff_l2(&h1, &h2),
        first_delta: first,
        last_delta: last,
        non_inc_ratio,
        jac_gain,
        attn_norm: attn_n,
        mamba_norm: mamba_n,
        input_norm: input_n,
        resid_norm: resid_n,
    }
}

fn main() {
    let cfg = make_config();
    let stages = [
        AblationStage::InputOnly,
        AblationStage::InputMamba,
        AblationStage::InputAttn,
        AblationStage::Full,
    ];
    let results: Vec<StageResult> = stages.into_iter().map(|s| run_stage(&cfg, s)).collect();

    println!();
    println!("AIDEEN DEQ Ablation Diagnostics (CPU)");
    println!(
        "config: d_r={} h_slots={} ctx_len={} max_deq_iters={} residual_alpha={:.3} damping={:.3}",
        cfg.d_r,
        cfg.h_slots,
        cfg.ctx_len,
        cfg.max_deq_iters,
        make_cpu_trainer(123, &cfg).reasoning.residual_alpha,
        make_cpu_trainer(123, &cfg).reasoning.damping
    );
    println!("{}", "-".repeat(148));
    println!(
        "{:<12} {:>8} {:>10} {:>11} {:>11} {:>10} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "stage",
        "cos(h1,h2)",
        "l2(h1-h2)",
        "delta_1",
        "delta_last",
        "non_inc",
        "jac_gain",
        "attn_norm",
        "mamba_norm",
        "input_norm",
        "resid_norm"
    );
    println!("{}", "-".repeat(148));
    for r in results {
        println!(
            "{:<12} {:>8.3} {:>10.3e} {:>11.3e} {:>11.3e} {:>10.2} {:>9.3} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e}",
            r.stage,
            r.sens_cos,
            r.sens_l2,
            r.first_delta,
            r.last_delta,
            r.non_inc_ratio,
            r.jac_gain,
            r.attn_norm,
            r.mamba_norm,
            r.input_norm,
            r.resid_norm
        );
    }
    println!("{}", "-".repeat(148));
    println!("note: jac_gain is empirical local ||J_f|| via finite-difference probes (higher => less contractive).");
}

