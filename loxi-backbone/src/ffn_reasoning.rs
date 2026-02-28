use loxi_core::reasoning::Reasoning;
use loxi_core::state::D_R;
use nalgebra::{DMatrix, DVector};

/// Backbone: Red Feed-Forward de 2 capas con expansión.
pub struct FfnReasoning {
    w1: DMatrix<f32>,
    b1: DVector<f32>,
    w2: DMatrix<f32>,
    b2: DVector<f32>,
}

impl FfnReasoning {
    pub fn new(hidden_dim: usize) -> Self {
        // Inicialización determinista simple para evitar depender de `rand` y
        // mantener la reproducibilidad de los tests y demostraciones.
        // Distribuye pesos en un rango pequeño [-0.05, 0.05].
        let w1 = DMatrix::from_fn(hidden_dim, D_R, |r, c| {
            let val = ((r * 31 + c * 17) % 256) as f32 / 256.0;
            val * 0.1 - 0.05
        });
        let b1 = DVector::zeros(hidden_dim);

        let w2 = DMatrix::from_fn(D_R, hidden_dim, |r, c| {
            let val = ((r * 19 + c * 23) % 256) as f32 / 256.0;
            val * 0.1 - 0.05
        });
        let b2 = DVector::zeros(D_R);

        Self { w1, b1, w2, b2 }
    }
}

impl Reasoning for FfnReasoning {
    fn init(&self, s: &DVector<f32>) -> DVector<f32> {
        s.clone()
    }

    fn step(&self, h: &DVector<f32>, _s: &DVector<f32>) -> DVector<f32> {
        let h_r = h.rows(0, D_R);

        // Capa Oculta: h_mid = tanh(W1 * h_r + b1)
        let mut h_mid = &self.w1 * h_r + &self.b1;
        for v in h_mid.iter_mut() {
            *v = v.tanh();
        }

        // Capa de Salida: h_out = tanh(W2 * h_mid + b2)
        let mut h_out_r = &self.w2 * h_mid + &self.b2;
        for v in h_out_r.iter_mut() {
            *v = v.tanh();
        }

        let mut next = h.clone();
        next.rows_mut(0, D_R).copy_from(&h_out_r);

        next
    }
}

impl loxi_core::reasoning::MutableReasoning for FfnReasoning {
    fn perturb_weight(&mut self, eps: f32) -> usize {
        // Para simplificar Nivel 3, tratamos todos los pesos como un solo vector.
        // Simulamos la perturbación en w1 por ahora.
        let seed = (self.w1[0] * 1000.0).abs() as usize;
        let idx = (seed + 1) % self.w1.len();
        self.w1[idx] += eps;
        idx
    }

    fn revert_weight(&mut self, index: usize, eps: f32) {
        self.w1[index] -= eps;
    }

    fn apply_update(&mut self, jacobian: &loxi_core::reasoning::JacobianEstimate, step: f32) {
        // θ = θ + η * sign(ΔQ) * (delta_h / eps)
        // Pero en LOXI simplificamos: el Jacobiano ya es delta_h.
        // Aplicamos la corrección al mismo peso que perturbamos.
        let delta = step * (jacobian.delta_h.norm() / (jacobian.eps + 1e-8));
        self.w1[jacobian.weight_index] += delta;
    }
}
