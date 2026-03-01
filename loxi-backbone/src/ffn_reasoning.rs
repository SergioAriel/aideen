use loxi_core::compute::{ComputeBackend, TensorId};
use loxi_core::reasoning::Reasoning;
use loxi_core::state::D_R;
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;

/// Backbone: Red Feed-Forward de 2 capas con expansión.
pub struct FfnReasoning {
    w1: DMatrix<f32>,
    b1: DVector<f32>,
    w2: DMatrix<f32>,
    b2: DVector<f32>,
    // ⚠️ Constitutional Note:
    // Interior mutability is ONLY allowed for GPU tensor handles (TensorId).
    // No learning, no weight mutation, no semantic state is stored here.
    w1_id: RefCell<Option<TensorId>>,
    w2_id: RefCell<Option<TensorId>>,
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

        Self {
            w1,
            b1,
            w2,
            b2,
            w1_id: RefCell::new(None),
            w2_id: RefCell::new(None),
        }
    }
}

impl Reasoning for FfnReasoning {
    fn init(&self, s: &DVector<f32>) -> DVector<f32> {
        s.clone()
    }

    fn step(
        &self,
        h: &DVector<f32>,
        _s: &DVector<f32>,
        mut exec: Option<&mut dyn ComputeBackend>,
    ) -> DVector<f32> {
        let h_r = h.rows(0, D_R).into_owned();

        if let Some(be) = exec.as_mut() {
            if self.w1_id.borrow().is_none() {
                *self.w1_id.borrow_mut() = be.load_tensor(self.w1.as_slice()).ok();
            }
            if self.w2_id.borrow().is_none() {
                *self.w2_id.borrow_mut() = be.load_tensor(self.w2.as_slice()).ok();
            }

            if let (Some(w1_id), Some(w2_id)) =
                (self.w1_id.borrow().as_ref(), self.w2_id.borrow().as_ref())
            {
                if let Ok(i_id) = be.load_tensor(h_r.as_slice()) {
                    let out_dim_1 = self.w1.nrows();
                    if let Ok(mut h_mid) = be.ffn_forward(w1_id, &i_id, out_dim_1) {
                        h_mid += &self.b1;
                        for v in h_mid.iter_mut() {
                            *v = v.tanh();
                        }

                        if let Ok(mid_id) = be.load_tensor(h_mid.as_slice()) {
                            let out_dim_2 = self.w2.nrows();
                            if let Ok(mut h_out_r) = be.ffn_forward(w2_id, &mid_id, out_dim_2) {
                                h_out_r += &self.b2;
                                for v in h_out_r.iter_mut() {
                                    *v = v.tanh();
                                }

                                let mut next = h.clone();
                                next.rows_mut(0, D_R).copy_from(&h_out_r);
                                return next;
                            }
                        }
                    }
                }
            }
        }

        // Fallback or CPU exactly as before
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
