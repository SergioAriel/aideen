use loxi_core::reasoning::Reasoning;
use nalgebra::DVector;

/// Deep Equilibrium Model (DEQ) genérico
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
    fn init(&self, s: &DVector<f32>) -> DVector<f32> {
        self.inner_model.init(s)
    }

    fn step(
        &self,
        h: &DVector<f32>,
        s: &DVector<f32>,
        exec: Option<&mut dyn loxi_core::compute::ComputeBackend>,
    ) -> DVector<f32> {
        self.inner_model.step(h, s, exec)
    }
}
