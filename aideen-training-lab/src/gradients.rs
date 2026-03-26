use aideen_core::{reasoning::Reasoning, state::HSlots};
use nalgebra::{DMatrix, DVector};

// ── LmHead gradientes ────────────────────────────────────────────────────────

/// LmHead backprop results.
pub struct LmHeadGrads {
    pub dw: DMatrix<f32>,
    pub db: DVector<f32>,
    pub dg: DVector<f32>,
}

pub fn lmhead_backward(
    dl_dlogits: &DVector<f32>,
    h_pooled: &DVector<f32>,
    w_lm: &DMatrix<f32>,
    g: &DVector<f32>,
) -> (LmHeadGrads, DVector<f32>) {
    let eps = 1e-5;
    let d = h_pooled.len() as f32;
    let mean_sq = h_pooled.map(|v| v * v).mean();
    let rms = (mean_sq + eps).sqrt();
    let h_norm = h_pooled.map(|v| v / rms);
    let h_rms = h_norm.component_mul(g);

    let dw = dl_dlogits * h_rms.transpose();
    let db = dl_dlogits.clone();
    let dl_dh_rms = w_lm.transpose() * dl_dlogits;

    let dg = dl_dh_rms.component_mul(&h_norm);

    let dx = dl_dh_rms.component_mul(g) / rms;
    let sum_dx_h = dx.dot(h_pooled);
    let dl_dh = dx - h_pooled.map(|v| v * sum_dx_h / (d * rms * rms));

    (LmHeadGrads { dw, db, dg }, dl_dh)
}

// ── DEQ Implicit Differentiation ─────────────────────────────────────────────

pub fn deq_implicit_grad<R: Reasoning>(
    reasoning: &R,
    h_star: &HSlots,
    query: &DVector<f32>,
    dl_dh_pooled: &DVector<f32>,
    adj_iters: usize,
) -> DVector<f32> {
    let config = reasoning.config();
    let d_r = config.d_r;
    let h_slots = config.h_slots;
    let eps = 1e-3_f32;

    let dl_per_slot = dl_dh_pooled / h_slots as f32;
    let mut rhs = vec![0.0f32; h_slots * d_r];
    for k in 0..h_slots {
        for d in 0..d_r {
            rhs[k * d_r + d] = dl_per_slot[d];
        }
    }
    let b = DVector::from_vec(rhs);
    let n = b.len();

    let apply_i_minus_jt = |v: &DVector<f32>| -> DVector<f32> {
        let mut h_plus = h_star.clone();
        for k in 0..h_slots {
            let mut slot = h_plus.slot(k);
            for d in 0..d_r {
                slot[d] += eps * v[k * d_r + d];
            }
            h_plus.set_slot(k, &slot);
        }

        let f_plus = reasoning.step(&h_plus, query, None);
        let f_star = reasoning.step(h_star, query, None);

        let flat_plus = DVector::from_vec(f_plus.to_flat());
        let flat_star = DVector::from_vec(f_star.to_flat());
        let jv = (flat_plus - flat_star) / eps;

        v - jv
    };

    let mut v = DVector::zeros(n);
    let mut r = &b - &apply_i_minus_jt(&v);
    let mut p = r.clone();
    let mut rs_old = r.dot(&r);

    for _ in 0..adj_iters {
        let ap = apply_i_minus_jt(&p);
        let p_dot_ap = p.dot(&ap);
        if p_dot_ap.abs() < 1e-12 {
            break;
        }
        let alpha = rs_old / p_dot_ap;

        v += alpha * &p;
        r -= alpha * &ap;

        let rs_new = r.dot(&r);
        if rs_new.sqrt() < 1e-6 {
            break;
        }

        let beta = rs_new / rs_old;
        p = &r + beta * &p;
        rs_old = rs_new;
    }

    // Pool the implicit adjoint across slots (same symmetry used in forward pooling).
    // Using only slot 0 is fragile and can desynchronize CPU/GPU directions.
    let mut pooled = DVector::zeros(d_r);
    for k in 0..h_slots {
        for d in 0..d_r {
            pooled[d] += v[k * d_r + d];
        }
    }
    pooled / h_slots as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use aideen_core::state::ArchitectureConfig;

    #[test]
    fn lmhead_grad_has_correct_shapes() {
        let d_r = 512;
        let vocab = 64;
        let dl = DVector::from_element(vocab, 0.1f32);
        let h = DVector::from_element(d_r, 0.5);
        let w = DMatrix::from_element(vocab, d_r, 0.01);
        let g = DVector::from_element(d_r, 1.0);

        let (grads, dl_dh) = lmhead_backward(&dl, &h, &w, &g);
        assert_eq!(grads.dw.nrows(), vocab);
        assert_eq!(grads.dw.ncols(), d_r);
        assert_eq!(grads.db.len(), vocab);
        assert_eq!(grads.dg.len(), d_r);
        assert_eq!(dl_dh.len(), d_r);
    }

    #[test]
    fn lmhead_grad_reduces_loss() {
        use crate::loss;
        use aideen_backbone::lm_head::LmHead;

        let mut config = ArchitectureConfig::default();
        config.vocab_size = 32;
        let d_r = config.d_r;
        let mut head = LmHead::new(config.clone());
        let h_star = HSlots::from_broadcast(&DVector::from_element(d_r, 0.3), &config);
        let target = 5u32;

        let logits_before = head.forward(&h_star);
        let loss_before = loss::cross_entropy(&logits_before, target);

        let dl = loss::cross_entropy_grad(&logits_before, target);
        let h_pooled = head.pool_h_star(&h_star);
        let (grads, _) = lmhead_backward(&dl, &h_pooled, &head.w, &head.g);

        let lr = 0.1;
        head.w -= lr * &grads.dw;
        head.b -= lr * &grads.db;
        head.g -= lr * &grads.dg;

        let logits_after = head.forward(&h_star);
        let loss_after = loss::cross_entropy(&logits_after, target);

        assert!(loss_after < loss_before);
    }
}
