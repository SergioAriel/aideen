use aideen_core::memory::Memory;
use nalgebra::DVector;

/// Production "no memory" implementation.
/// No-op operations. Allows using AideenNode without active memory.
pub struct NullMemory;

impl Memory for NullMemory {
    fn write(&mut self, _h: DVector<f32>) {}
    fn query(&self, _query: &DVector<f32>, _k: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}
