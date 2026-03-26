//! Spectral Normalization to guarantee DEQ convergence.
//!
//! ## Mathematical foundation
//! For the DEQ H_{t+1} = f(H_t) to converge by Banach's Theorem,
//! f must be a contraction: ||f(x) - f(y)|| <= L ||x - y|| with L < 1.
//!
//! The spectral norm of a matrix W is its largest singular value sigma_max(W).
//! If sigma_max(W) <= 1, multiplication by W is non-expansive.
//!
//! ## Efficient estimation: Power Iteration
//! Computes sigma_max(W) without full SVD -- O(n) iterations x O(n^2) x O(n).
//! Converges in 5-20 iterations for well-conditioned matrices.
//!
//! ## Damping (beta-relaxation)
//! Alternative/complement to SN: H_{t+1} = beta*f(H_t) + (1-beta)*H_t
//! Guarantees convergence even if f is not contractive, at the cost of
//! requiring more iterations. beta=0.5 is conservative, beta=0.9 is aggressive.

use nalgebra::{DMatrix, DVector};

/// Estimates sigma_max(W) via power iteration.
///
/// `n_iter` = 20 is sufficient for D_R x D_R matrices.
/// Returns 1.0 if the matrix is zero or near-zero.
pub fn spectral_norm(w: &DMatrix<f32>, n_iter: usize) -> f32 {
    let rows = w.nrows();
    let cols = w.ncols();
    if rows == 0 || cols == 0 {
        return 1.0;
    }

    // Initialize uniform vector u
    let norm_u = (rows as f32).sqrt();
    let mut u: DVector<f32> = DVector::from_element(rows, 1.0 / norm_u);

    let mut sigma = 1.0_f32;

    for _ in 0..n_iter {
        // v = W^T u / ||W^T u||
        let wtu = w.transpose() * &u;
        let wtu_norm = wtu.norm();
        if wtu_norm < 1e-12 {
            return 0.0;
        }
        let v = wtu / wtu_norm;

        // u_new = W v / ||W v||
        let wv = w * &v;
        let wv_norm = wv.norm();
        if wv_norm < 1e-12 {
            return 0.0;
        }
        u = wv / wv_norm;

        // σ = u^T W v
        sigma = (u.transpose() * w * &v)[0];
    }

    sigma.abs().max(1e-12)
}

/// Normalizes W so that sigma_max(W) = target_sigma (default = 1.0).
///
/// After normalizing, multiplication by W is non-expansive.
/// For strict contractivity, use target_sigma < 1.
pub fn normalize(w: &DMatrix<f32>, target_sigma: f32, n_iter: usize) -> DMatrix<f32> {
    let sigma = spectral_norm(w, n_iter);
    if sigma < 1e-12 {
        return w.clone();
    }
    w * (target_sigma / sigma)
}

/// Normalizes in-place if sigma_max > threshold.
/// Useful to call after each training step.
pub fn normalize_if_needed(w: &mut DMatrix<f32>, threshold: f32, n_iter: usize) {
    let sigma = spectral_norm(w, n_iter);
    if sigma > threshold {
        *w *= threshold / sigma;
    }
}

/// Picard mixture with damping beta in (0, 1).
///
/// h_next = beta * f(h) + (1-beta) * h
///
/// Guarantees convergence for any f if beta is sufficiently small.
/// Useful as a fallback when contraction cannot be guaranteed via SN.
///
/// - beta = 1.0 -> vanilla DEQ (may diverge)
/// - beta = 0.5 -> conservative, always converges if f is bounded
/// - beta = 0.9 -> aggressive, converges only if f is nearly contractive
pub fn damped_update(h_curr: &DVector<f32>, f_h: &DVector<f32>, beta: f32) -> DVector<f32> {
    f_h * beta + h_curr * (1.0 - beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn spectral_norm_identity_is_one() {
        let w = DMatrix::<f32>::identity(64, 64);
        let sigma = spectral_norm(&w, 20);
        assert!((sigma - 1.0).abs() < 1e-3, "σ(I) = 1.0, got {sigma}");
    }

    #[test]
    fn spectral_norm_scaled_identity() {
        let w = DMatrix::<f32>::identity(64, 64) * 3.7_f32;
        let sigma = spectral_norm(&w, 20);
        assert!((sigma - 3.7).abs() < 0.05, "σ(3.7·I) ≈ 3.7, got {sigma}");
    }

    #[test]
    fn normalize_reduces_sigma_to_target() {
        let w = DMatrix::<f32>::identity(64, 64) * 5.0_f32;
        let w_norm = normalize(&w, 1.0, 20);
        let sigma = spectral_norm(&w_norm, 20);
        assert!(
            (sigma - 1.0).abs() < 0.05,
            "sigma after normalize should be approx 1.0, got {sigma}"
        );
    }

    #[test]
    fn normalize_if_needed_only_triggers_above_threshold() {
        let mut w = DMatrix::<f32>::identity(32, 32) * 0.5_f32;
        normalize_if_needed(&mut w, 1.0, 20);
        // σ = 0.5 < threshold=1.0, no debería cambiar
        let sigma = spectral_norm(&w, 20);
        assert!(
            (sigma - 0.5).abs() < 0.05,
            "should remain at 0.5, got {sigma}"
        );
    }

    #[test]
    fn damped_update_interpolates_correctly() {
        let h: DVector<f32> = DVector::from_element(4, 0.0);
        let fh: DVector<f32> = DVector::from_element(4, 1.0);
        let result = damped_update(&h, &fh, 0.5);
        // β=0.5 → resultado = 0.5*1 + 0.5*0 = 0.5
        assert!(
            (result[0] - 0.5).abs() < 1e-6,
            "damped_update with beta=0.5 should yield 0.5"
        );
    }
}
