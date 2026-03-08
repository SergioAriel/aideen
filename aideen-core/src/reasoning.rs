use crate::state::HSlots;
use nalgebra::DVector;

/// Razonamiento iterativo (DEQ-like) - La propuesta de Δ
pub trait Reasoning {
    /// Retorna la configuración de la arquitectura.
    fn config(&self) -> &crate::state::ArchitectureConfig;

    /// Inicializa el estado interno H₀ a partir del contexto global S.
    /// Por defecto, emite el subespacio S_R "broadcast" a los K slots.
    fn init(&self, s: &DVector<f32>) -> HSlots;

    /// Paso iterativo Hₜ₊₁ = f(Hₜ, S)
    /// Cada slot se actualiza independientemente; una capa de mezcla opcional
    /// puede propagar información entre slots.
    fn step(
        &self,
        h: &HSlots,
        s: &DVector<f32>,
        exec: Option<&mut dyn crate::compute::ComputeBackend>,
    ) -> HSlots;
}
