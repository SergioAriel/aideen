//! Cross-entropy loss and its analytic gradient.

use nalgebra::DVector;

/// Cross-entropy loss between logits and a one-hot target.
/// `logits`: raw output of the LmHead [vocab_size].
/// `target`: index of the correct token.
pub fn cross_entropy(logits: &DVector<f32>, target: u32) -> f32 {
    let probs = softmax(logits);
    let p = probs[target as usize].max(1e-10); // clamp to avoid log(0)
    -p.ln()
}

/// Exact slice-based version. It is the same stable CE as above,
/// but avoids building additional temporaries when the caller
/// already works with reusable flat buffers.
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

/// Gradient of cross-entropy with respect to logits.
/// dL/d_logits = softmax(logits) - one_hot(target)
/// This is the exact analytic formula — no autograd required.
pub fn cross_entropy_grad(logits: &DVector<f32>, target: u32) -> DVector<f32> {
    let mut grad = softmax(logits);
    grad[target as usize] -= 1.0;
    grad
}

/// Fills `grad_out` with `softmax(logits) - one_hot(target)` and returns the exact CE.
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

/// Stable softmax (subtract max to avoid overflow).
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
        // Logits with a very high value at the correct position
        let mut logits = DVector::zeros(10);
        logits[3] = 100.0;
        let loss = cross_entropy(&logits, 3);
        assert!(
            loss < 0.001,
            "loss should be ~0 with a perfect prediction, got {loss}"
        );
    }

    #[test]
    fn cross_entropy_positive_for_wrong_prediction() {
        let logits = DVector::from_element(10, 0.0); // uniform
        let loss = cross_entropy(&logits, 5);
        assert!(
            loss > 1.0,
            "loss should be high with a uniform prediction, got {loss}"
        );
    }

    #[test]
    fn grad_sums_to_zero() {
        // Property: sum(softmax - one_hot) = 1 - 1 = 0
        let logits = DVector::from_fn(8, |i, _| (i as f32) * 0.5);
        let grad = cross_entropy_grad(&logits, 3);
        let sum: f32 = grad.sum();
        assert!(sum.abs() < 1e-5, "gradient must sum to ~0, got {sum}");
    }

    #[test]
    fn grad_is_negative_at_target() {
        let logits = DVector::from_element(8, 0.0); // uniform
        let grad = cross_entropy_grad(&logits, 2);
        // softmax(uniform) = 1/8 = 0.125, grad[2] = 0.125 - 1.0 = -0.875
        assert!(grad[2] < 0.0, "gradient at target must be negative");
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
