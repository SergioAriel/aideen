use nalgebra::DVector;

/// Stable integrator for the global state
pub struct Integrator {
    pub alpha: f32,
    pub epsilon: f32,
}

impl Integrator {
    pub fn new(alpha: f32, epsilon: f32) -> Self {
        Self { alpha, epsilon }
    }

    /// Applies S ← S + tanh(α · Δ)
    /// Returns true if the change was significant
    pub fn apply(&self, s: &mut DVector<f32>, delta: &DVector<f32>) -> bool {
        let mut norm = 0.0;

        for (si, di) in s.iter_mut().zip(delta.iter()) {
            let c = (self.alpha * di).tanh();
            *si += c;
            norm += c * c;
        }

        norm.sqrt() > self.epsilon
    }
}
