use aideen_core::reasoning::Reasoning;
use aideen_core::state::HSlots;
use nalgebra::DVector;

/// Generic Deep Equilibrium Model (DEQ)
pub struct GeneralDEQ<R: Reasoning> {
    pub inner_model: R,
}

impl<R: Reasoning> GeneralDEQ<R> {
    pub fn new(model: R) -> Self {
        Self { inner_model: model }
    }
}

// Un DEQ es simplemente un wrapper que expone la lógica iterativa
impl<R: Reasoning> Reasoning for GeneralDEQ<R> {
    fn config(&self) -> &aideen_core::state::ArchitectureConfig {
        self.inner_model.config()
    }

    fn init(&self, s: &DVector<f32>) -> HSlots {
        self.inner_model.init(s)
    }

    fn step(
        &self,
        h: &HSlots,
        s: &DVector<f32>,
        exec: Option<&mut dyn aideen_core::compute::ComputeBackend>,
    ) -> HSlots {
        self.inner_model.step(h, s, exec)
    }
}
