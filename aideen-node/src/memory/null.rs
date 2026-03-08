use aideen_core::memory::Memory;
use nalgebra::DVector;

/// Implementación de producción "sin memoria".
/// Operaciones no-op. Permite usar AideenNode sin memoria activada.
pub struct NullMemory;

impl Memory for NullMemory {
    fn write(&mut self, _h: DVector<f32>) {}
    fn query(&self, _query: &DVector<f32>, _k: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}
