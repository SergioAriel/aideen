use aideen_backbone::mamba_slot_reasoning::MambaSlotReasoning;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::{DMatrix, DVector};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

fn cross_slot_attn_with(
    w_q: &DMatrix<f32>,
    w_k: &DMatrix<f32>,
    w_v: &DMatrix<f32>,
    w_o: &DMatrix<f32>,
    h: &HSlots,
    cfg: &ArchitectureConfig,
) -> HSlots {
    let d_r = cfg.d_r;
    let h_slots = cfg.h_slots;
    let scale = (d_r as f32).sqrt().recip();

    let qs: Vec<DVector<f32>> = (0..h_slots).map(|k| w_q * h.slot(k)).collect();
    let ks: Vec<DVector<f32>> = (0..h_slots).map(|k| w_k * h.slot(k)).collect();
    let vs: Vec<DVector<f32>> = (0..h_slots).map(|k| w_v * h.slot(k)).collect();

    let mut next = HSlots::zeros(cfg);
    for q_idx in 0..h_slots {
        let raw_scores: Vec<f32> = ks.iter().map(|k| qs[q_idx].dot(k) * scale).collect();
        let max_s = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw_scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let attn: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        let mixed: DVector<f32> = attn
            .iter()
            .zip(vs.iter())
            .map(|(a, v)| v * *a)
            .fold(DVector::zeros(d_r), |acc, v| acc + v);
        next.set_slot(q_idx, &(w_o * mixed));
    }
    next
}

fn loss_from_attn(
    w_q: &DMatrix<f32>,
    w_k: &DMatrix<f32>,
    w_v: &DMatrix<f32>,
    w_o: &DMatrix<f32>,
    h: &HSlots,
    upstream: &HSlots,
    cfg: &ArchitectureConfig,
) -> f32 {
    let attn = cross_slot_attn_with(w_q, w_k, w_v, w_o, h, cfg);
    let mut acc = 0.0;
    for s in 0..cfg.h_slots {
        acc += upstream.slot(s).dot(&attn.slot(s));
    }
    acc
}

fn analytical_grads(
    r: &MambaSlotReasoning,
    h: &HSlots,
    upstream: &HSlots,
) -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {
    let cfg = r.config();
    let d = cfg.d_r;
    let hs = cfg.h_slots;
    let scale = (d as f32).sqrt().recip();

    let qs: Vec<DVector<f32>> = (0..hs).map(|k| &r.w_q * h.slot(k)).collect();
    let ks: Vec<DVector<f32>> = (0..hs).map(|k| &r.w_k * h.slot(k)).collect();
    let vs: Vec<DVector<f32>> = (0..hs).map(|k| &r.w_v * h.slot(k)).collect();

    let mut g_wo = DMatrix::<f32>::zeros(d, d);
    let mut g_wv = DMatrix::<f32>::zeros(d, d);
    let mut g_wq = DMatrix::<f32>::zeros(d, d);
    let mut g_wk = DMatrix::<f32>::zeros(d, d);

    for q_idx in 0..hs {
        let raw_scores: Vec<f32> = ks.iter().map(|k| qs[q_idx].dot(k) * scale).collect();
        let max_s = raw_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw_scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let attn: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        let mixed: DVector<f32> = attn
            .iter()
            .zip(vs.iter())
            .map(|(a, v)| v * *a)
            .fold(DVector::zeros(d), |acc, v| acc + v);
        let g_attn = upstream.slot(q_idx);
        g_wo += &g_attn * mixed.transpose();

        let g_mix = r.w_o.transpose() * &g_attn;
        let g_alpha: Vec<f32> = vs.iter().map(|v| g_mix.dot(v)).collect();
        let alpha_dot_g: f32 = attn.iter().zip(g_alpha.iter()).map(|(a, g)| a * g).sum();
        let g_score: Vec<f32> = attn
            .iter()
            .zip(g_alpha.iter())
            .map(|(a, g)| a * (g - alpha_dot_g))
            .collect();

        let mut g_q = DVector::<f32>::zeros(d);
        for (k_idx, a) in attn.iter().enumerate() {
            g_wv += (g_mix.clone() * h.slot(k_idx).transpose()) * *a;
            let gk = &qs[q_idx] * (scale * g_score[k_idx]);
            g_wk += gk * h.slot(k_idx).transpose();
            g_q += &ks[k_idx] * (scale * g_score[k_idx]);
        }
        g_wq += g_q * h.slot(q_idx).transpose();
    }

    (g_wo, g_wv, g_wq, g_wk)
}

fn finite_diff_grad(
    w_q: &DMatrix<f32>,
    w_k: &DMatrix<f32>,
    w_v: &DMatrix<f32>,
    w_o: &DMatrix<f32>,
    h: &HSlots,
    upstream: &HSlots,
    cfg: &ArchitectureConfig,
    wrt: &str,
    eps: f32,
) -> DMatrix<f32> {
    let d = cfg.d_r;
    let mut grad = DMatrix::<f32>::zeros(d, d);
    for row in 0..d {
        for col in 0..d {
            let mut wq_p = w_q.clone();
            let mut wq_m = w_q.clone();
            let mut wk_p = w_k.clone();
            let mut wk_m = w_k.clone();
            let mut wv_p = w_v.clone();
            let mut wv_m = w_v.clone();
            let mut wo_p = w_o.clone();
            let mut wo_m = w_o.clone();
            match wrt {
                "wq" => {
                    wq_p[(row, col)] += eps;
                    wq_m[(row, col)] -= eps;
                }
                "wk" => {
                    wk_p[(row, col)] += eps;
                    wk_m[(row, col)] -= eps;
                }
                "wv" => {
                    wv_p[(row, col)] += eps;
                    wv_m[(row, col)] -= eps;
                }
                "wo" => {
                    wo_p[(row, col)] += eps;
                    wo_m[(row, col)] -= eps;
                }
                _ => panic!("unsupported wrt={wrt}"),
            }
            let lp = loss_from_attn(&wq_p, &wk_p, &wv_p, &wo_p, h, upstream, cfg);
            let lm = loss_from_attn(&wq_m, &wk_m, &wv_m, &wo_m, h, upstream, cfg);
            grad[(row, col)] = (lp - lm) / (2.0 * eps);
        }
    }
    grad
}

fn cosine(a: &DMatrix<f32>, b: &DMatrix<f32>) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let av = a[(i, j)];
            let bv = b[(i, j)];
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
    }
    dot / ((na.sqrt() * nb.sqrt()).max(1e-12))
}

fn rel_err(a: &DMatrix<f32>, b: &DMatrix<f32>) -> f32 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let dv = a[(i, j)] - b[(i, j)];
            num += dv * dv;
            den += b[(i, j)] * b[(i, j)];
        }
    }
    num.sqrt() / den.sqrt().max(1e-12)
}

fn main() {
    let mut cfg = ArchitectureConfig::default();
    cfg.d_r = env_usize("AIDEEN_GRADCHECK_DR", 16);
    cfg.h_slots = env_usize("AIDEEN_GRADCHECK_HSLOTS", 4);

    let seed = env_u64("AIDEEN_GRADCHECK_SEED", 42);
    let eps = env_f32("AIDEEN_GRADCHECK_EPS", 1e-3);
    let r = MambaSlotReasoning::new_with_seed(cfg.clone(), seed);

    let mut h = HSlots::zeros(&cfg);
    let mut upstream = HSlots::zeros(&cfg);
    for s in 0..cfg.h_slots {
        let mut hv = DVector::zeros(cfg.d_r);
        let mut gv = DVector::zeros(cfg.d_r);
        for d in 0..cfg.d_r {
            hv[d] = (((s * cfg.d_r + d + 1) as f32) * 0.137).sin() * 0.25;
            gv[d] = (((s * cfg.d_r + d + 3) as f32) * 0.173).cos() * 0.25;
        }
        h.set_slot(s, &hv);
        upstream.set_slot(s, &gv);
    }

    let (g_wo, g_wv, g_wq, g_wk) = analytical_grads(&r, &h, &upstream);
    let fd_wo = finite_diff_grad(&r.w_q, &r.w_k, &r.w_v, &r.w_o, &h, &upstream, &cfg, "wo", eps);
    let fd_wv = finite_diff_grad(&r.w_q, &r.w_k, &r.w_v, &r.w_o, &h, &upstream, &cfg, "wv", eps);
    let fd_wq = finite_diff_grad(&r.w_q, &r.w_k, &r.w_v, &r.w_o, &h, &upstream, &cfg, "wq", eps);
    let fd_wk = finite_diff_grad(&r.w_q, &r.w_k, &r.w_v, &r.w_o, &h, &upstream, &cfg, "wk", eps);

    println!("DEQ attention grad check");
    println!(
        "config: d_r={} h_slots={} seed={} eps={}",
        cfg.d_r, cfg.h_slots, seed, eps
    );
    println!(
        "W_o: cos={:.6} rel_err={:.6}",
        cosine(&g_wo, &fd_wo),
        rel_err(&g_wo, &fd_wo)
    );
    println!(
        "W_v: cos={:.6} rel_err={:.6}",
        cosine(&g_wv, &fd_wv),
        rel_err(&g_wv, &fd_wv)
    );
    println!(
        "W_q: cos={:.6} rel_err={:.6}",
        cosine(&g_wq, &fd_wq),
        rel_err(&g_wq, &fd_wq)
    );
    println!(
        "W_k: cos={:.6} rel_err={:.6}",
        cosine(&g_wk, &fd_wk),
        rel_err(&g_wk, &fd_wk)
    );
}
