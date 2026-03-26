use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::{DMatrix, DVector};

/// Minimal backbone: linear reasoning + tanh
pub struct LinearReasoning {
    pub config: ArchitectureConfig,
    /// Weights W ∈ R^{D_R × D_R}
    weights: DMatrix<f32>,
}

impl LinearReasoning {
    pub fn new(config: ArchitectureConfig) -> Self {
        let d_r = config.d_r;
        // Inicialización simple y estable
        let weights = DMatrix::identity(d_r, d_r) * 0.9;
        Self { weights, config }
    }
}

impl Reasoning for LinearReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.config
    }

    fn init(&self, s: &DVector<f32>) -> HSlots {
        let d_r = self.config.d_r;
        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };
        HSlots::from_broadcast(&s_r, &self.config)
    }

    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn aideen_core::compute::ComputeBackend>,
    ) -> HSlots {
        let mut next = HSlots::zeros(&self.config);
        let h_slots = self.config.h_slots;
        for k in 0..h_slots {
            let h_r = h.slot(k);
            // Proyección lineal + tanh por slot
            let mut next_r = &self.weights * &h_r;
            for v in next_r.iter_mut() {
                *v = v.tanh();
            }
            next.set_slot(k, &next_r);
        }
        next
    }
}
