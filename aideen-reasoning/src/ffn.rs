use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::DVector;

/// Concrete implementation of an expert FFN model
pub struct ExpertFFN {
    pub config: ArchitectureConfig,
    pub w1: nalgebra::DMatrix<f32>,
    pub w2: nalgebra::DMatrix<f32>,
}

impl Reasoning for ExpertFFN {
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
            let h_slot = h.slot(k);
            // h_next = W2 * ReLU(W1 * h_slot)
            let hidden = (&self.w1 * &h_slot).map(|x| x.max(0.0));
            let out = &self.w2 * hidden;
            next.set_slot(k, &out);
        }
        next
    }
}
