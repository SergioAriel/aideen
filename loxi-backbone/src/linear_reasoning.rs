use loxi_core::reasoning::Reasoning;
use loxi_core::state::D_R;
use nalgebra::{DMatrix, DVector};

/// Backbone mínimo: razonamiento lineal + tanh
pub struct LinearReasoning {
    /// Pesos W ∈ R^{D_R × D_R}
    weights: DMatrix<f32>,
}

impl LinearReasoning {
    pub fn new() -> Self {
        // Inicialización simple y estable
        let weights = DMatrix::identity(D_R, D_R) * 0.9;
        Self { weights }
    }
}

impl Reasoning for LinearReasoning {
    fn init(&self, s: &DVector<f32>) -> DVector<f32> {
        // Inicializa H como una copia del estado completo
        s.clone()
    }

    fn step(&self, h: &DVector<f32>, _s: &DVector<f32>) -> DVector<f32> {
        let mut next = h.clone();

        // Extraemos solo la porción de razonamiento de H
        let h_r = h.rows(512, D_R);

        // Calculamos la proyección lineal paso
        let next_r_proj = &self.weights * h_r;

        // Asignamos con tanh a la porción correspondiente en next
        let mut next_r = next.rows_mut(512, D_R);
        for (v, n) in next_r.iter_mut().zip(next_r_proj.iter()) {
            *v = n.tanh();
        }

        next
    }
}
