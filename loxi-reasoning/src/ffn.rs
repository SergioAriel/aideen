use loxi_core::reasoning::Reasoning;
use nalgebra::DVector;

/// Implementación concreta de un modelo experto FFN
pub struct ExpertFFN {
    pub w1: nalgebra::DMatrix<f32>,
    pub w2: nalgebra::DMatrix<f32>,
}

impl Reasoning for ExpertFFN {
    fn init(&self, s: &DVector<f32>) -> DVector<f32> {
        // Inicialización básica (puede ser s o ceros)
        s.clone()
    }

    fn step(&self, h: &DVector<f32>, _s: &DVector<f32>) -> DVector<f32> {
        // h_next = W2 * act(W1 * h)
        let hidden = (&self.w1 * h).map(|x| x.max(0.0)); // ReLU
        &self.w2 * hidden
    }
}
