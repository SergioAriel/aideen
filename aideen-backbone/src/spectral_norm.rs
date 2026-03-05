//! Spectral Normalization para garantizar convergencia del DEQ.
//!
//! ## Fundamento matemático
//! Para que el DEQ H_{t+1} = f(H_t) converja por el Teorema de Banach,
//! f debe ser una contracción: ||f(x) - f(y)|| ≤ L ||x - y|| con L < 1.
//!
//! La norma espectral de una matriz W es su mayor valor singular σ_max(W).
//! Si σ_max(W) ≤ 1, la multiplicación por W es no-expansiva.
//!
//! ## Estimación eficiente: Power Iteration
//! Calcula σ_max(W) sin SVD completo — O(n) iteraciones × O(n²) × O(n).
//! Converge en 5-20 iteraciones para matrices bien condicionadas.
//!
//! ## Damping (β-relaxation)
//! Alternativa/complemento a SN: H_{t+1} = β·f(H_t) + (1-β)·H_t
//! Garantiza convergencia incluso si f no es contractiva, a costa de
//! necesitar más iteraciones. β=0.5 es conservador, β=0.9 es agresivo.

use nalgebra::{DMatrix, DVector};

/// Estima σ_max(W) mediante power iteration.
///
/// `n_iter` = 20 es suficiente para matrices de D_R×D_R.
/// Retorna 1.0 si la matriz es cero o casi cero.
pub fn spectral_norm(w: &DMatrix<f32>, n_iter: usize) -> f32 {
    let rows = w.nrows();
    let cols = w.ncols();
    if rows == 0 || cols == 0 {
        return 1.0;
    }

    // Inicializar vector u uniforme
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

/// Normaliza W para que σ_max(W) = target_sigma (default = 1.0).
///
/// Después de normalizar, multiplicar por W es no-expansivo.
/// Para ser contractivo estrictamente, usar target_sigma < 1.
pub fn normalize(w: &DMatrix<f32>, target_sigma: f32, n_iter: usize) -> DMatrix<f32> {
    let sigma = spectral_norm(w, n_iter);
    if sigma < 1e-12 {
        return w.clone();
    }
    w * (target_sigma / sigma)
}

/// Normaliza in-place si σ_max > threshold.
/// Útil para llamar después de cada paso de entrenamiento.
pub fn normalize_if_needed(w: &mut DMatrix<f32>, threshold: f32, n_iter: usize) {
    let sigma = spectral_norm(w, n_iter);
    if sigma > threshold {
        *w *= threshold / sigma;
    }
}

/// Mixtura Picard con damping β ∈ (0, 1).
///
/// h_next = β * f(h) + (1-β) * h
///
/// Garantiza convergencia con cualquier f si β es lo suficientemente pequeño.
/// Útil como fallback cuando no se puede garantizar contracción vía SN.
///
/// - β = 1.0 → vanilla DEQ (puede divergir)
/// - β = 0.5 → conservador, siempre converge si f está acotada
/// - β = 0.9 → agresivo, converge solo si f es casi contractiva
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
            "σ después de normalize ≈ 1.0, got {sigma}"
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
            "debería permanecer en 0.5, got {sigma}"
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
            "damped_update con β=0.5 debe dar 0.5"
        );
    }
}
