//! Cross-entropy loss y su gradiente analítico.

use nalgebra::DVector;

/// Cross-entropy loss entre logits y un target one-hot.
/// `logits`: output crudo del LmHead [vocab_size].
/// `target`: índice del token correcto.
pub fn cross_entropy(logits: &DVector<f32>, target: u32) -> f32 {
    let probs = softmax(logits);
    let p = probs[target as usize].max(1e-10); // clamp para evitar log(0)
    -p.ln()
}

/// Versión slice-based exacta. Es la misma CE estable que arriba,
/// pero evita construir temporales adicionales cuando el caller
/// ya trabaja con buffers planos reutilizables.
pub fn cross_entropy_slice(logits: &[f32], target: u32) -> f32 {
    let target_idx = target as usize;
    if logits.is_empty() || target_idx >= logits.len() {
        return 0.0;
    }
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut target_exp = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max_l).exp();
        sum_exp += e;
        if i == target_idx {
            target_exp = e;
        }
    }
    let p = (target_exp / sum_exp.max(1e-10)).max(1e-10);
    -p.ln()
}

/// Gradiente de cross-entropy respecto a logits.
/// dL/d_logits = softmax(logits) - one_hot(target)
/// Esta es la fórmula analítica exacta — no requiere autograd.
pub fn cross_entropy_grad(logits: &DVector<f32>, target: u32) -> DVector<f32> {
    let mut grad = softmax(logits);
    grad[target as usize] -= 1.0;
    grad
}

/// Llena `grad_out` con `softmax(logits) - one_hot(target)` y devuelve la CE exacta.
pub fn cross_entropy_and_grad_slice(logits: &[f32], target: u32, grad_out: &mut [f32]) -> f32 {
    let target_idx = target as usize;
    if logits.is_empty() || target_idx >= logits.len() || grad_out.len() < logits.len() {
        return 0.0;
    }
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    let mut target_exp = 0.0f32;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max_l).exp();
        grad_out[i] = e;
        sum_exp += e;
        if i == target_idx {
            target_exp = e;
        }
    }
    let denom = sum_exp.max(1e-10);
    for g in grad_out.iter_mut().take(logits.len()) {
        *g /= denom;
    }
    grad_out[target_idx] -= 1.0;
    let p = (target_exp / denom).max(1e-10);
    -p.ln()
}

/// Softmax estable (restar max para evitar overflow).
fn softmax(logits: &DVector<f32>) -> DVector<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: DVector<f32> = logits.map(|l| (l - max_l).exp());
    let sum_exp: f32 = exps.sum();
    exps / sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_zero_for_perfect_prediction() {
        // Logits con un valor muy alto en la posición correcta
        let mut logits = DVector::zeros(10);
        logits[3] = 100.0;
        let loss = cross_entropy(&logits, 3);
        assert!(
            loss < 0.001,
            "loss debería ser ~0 con predicción perfecta, got {loss}"
        );
    }

    #[test]
    fn cross_entropy_positive_for_wrong_prediction() {
        let logits = DVector::from_element(10, 0.0); // uniform
        let loss = cross_entropy(&logits, 5);
        assert!(
            loss > 1.0,
            "loss debería ser alto con predicción uniform, got {loss}"
        );
    }

    #[test]
    fn grad_sums_to_zero() {
        // Propiedad: sum(softmax - one_hot) = 1 - 1 = 0
        let logits = DVector::from_fn(8, |i, _| (i as f32) * 0.5);
        let grad = cross_entropy_grad(&logits, 3);
        let sum: f32 = grad.sum();
        assert!(sum.abs() < 1e-5, "gradiente debe sumar ~0, got {sum}");
    }

    #[test]
    fn grad_is_negative_at_target() {
        let logits = DVector::from_element(8, 0.0); // uniform
        let grad = cross_entropy_grad(&logits, 2);
        // softmax(uniform) = 1/8 = 0.125, grad[2] = 0.125 - 1.0 = -0.875
        assert!(grad[2] < 0.0, "gradiente en target debe ser negativo");
    }

    #[test]
    fn slice_ce_matches_dvector_ce() {
        let logits = DVector::from_fn(11, |i, _| (i as f32) * 0.13 - 0.4);
        let ce_vec = cross_entropy(&logits, 4);
        let ce_slice = cross_entropy_slice(logits.as_slice(), 4);
        assert!((ce_vec - ce_slice).abs() < 1e-6);
    }

    #[test]
    fn slice_grad_matches_dvector_grad() {
        let logits = DVector::from_fn(9, |i, _| (i as f32) * 0.27 - 1.1);
        let grad_vec = cross_entropy_grad(&logits, 3);
        let mut grad_slice = vec![0.0f32; logits.len()];
        let ce_slice = cross_entropy_and_grad_slice(logits.as_slice(), 3, &mut grad_slice);
        let ce_vec = cross_entropy(&logits, 3);
        assert!((ce_vec - ce_slice).abs() < 1e-6);
        for i in 0..logits.len() {
            assert!((grad_vec[i] - grad_slice[i]).abs() < 1e-6);
        }
    }
}
