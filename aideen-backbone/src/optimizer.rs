//! Optimizador Adam para pesos nalgebra DMatrix/DVector.

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Adam optimizer con first/second moment tracking.
pub struct Adam {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    // Momentos por nombre de parámetro
    m_mat: HashMap<String, DMatrix<f32>>,
    v_mat: HashMap<String, DMatrix<f32>>,
    m_vec: HashMap<String, DVector<f32>>,
    v_vec: HashMap<String, DVector<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m_mat: HashMap::new(),
            v_mat: HashMap::new(),
            m_vec: HashMap::new(),
            v_vec: HashMap::new(),
            t: 0,
        }
    }

    /// Incrementar el step counter (llamar una vez por train_step).
    pub fn tick(&mut self) {
        self.t += 1;
    }

    pub fn step_count(&self) -> usize {
        self.t
    }

    /// Actualiza una DMatrix in-place con Adam.
    pub fn step_matrix(&mut self, name: &str, w: &mut DMatrix<f32>, grad: &DMatrix<f32>) {
        let key = name.to_string();
        let (nrows, ncols) = (w.nrows(), w.ncols());

        let m = self
            .m_mat
            .entry(key.clone())
            .or_insert_with(|| DMatrix::zeros(nrows, ncols));
        let v = self
            .v_mat
            .entry(key)
            .or_insert_with(|| DMatrix::zeros(nrows, ncols));

        // m = β₁·m + (1-β₁)·grad
        *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
        // v = β₂·v + (1-β₂)·grad²
        *v = &*v * self.beta2 + grad.map(|g| g * g) * (1.0 - self.beta2);

        // Bias correction
        let t = self.t.max(1) as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        // w -= lr · m_hat / (sqrt(v_hat) + eps)
        for (w_i, (m_i, v_i)) in w.iter_mut().zip(m.iter().zip(v.iter())) {
            let m_hat = m_i / bc1;
            let v_hat = v_i / bc2;
            *w_i -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    /// Actualiza un DVector in-place con Adam.
    pub fn step_vector(&mut self, name: &str, w: &mut DVector<f32>, grad: &DVector<f32>) {
        let key = name.to_string();
        let len = w.len();

        let m = self
            .m_vec
            .entry(key.clone())
            .or_insert_with(|| DVector::zeros(len));
        let v = self.v_vec.entry(key).or_insert_with(|| DVector::zeros(len));

        *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
        *v = &*v * self.beta2 + grad.map(|g| g * g) * (1.0 - self.beta2);

        let t = self.t.max(1) as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (w_i, (m_i, v_i)) in w.iter_mut().zip(m.iter().zip(v.iter())) {
            let m_hat = m_i / bc1;
            let v_hat = v_i / bc2;
            *w_i -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_reduces_simple_quadratic() {
        // Minimizar f(w) = ||w - target||² con Adam
        let target = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut w = DVector::zeros(3);
        let mut opt = Adam::new(0.1);

        for _ in 0..100 {
            opt.tick();
            let grad = &w - &target; // df/dw = 2(w - target), ignoramos el 2
            opt.step_vector("w", &mut w, &grad);
        }

        let dist = (&w - &target).norm();
        assert!(dist < 0.1, "Adam debería converger, dist={dist}");
    }
}
