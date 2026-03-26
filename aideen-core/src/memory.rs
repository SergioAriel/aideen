use nalgebra::DVector;

/// Minimal contract for the agent memory
pub trait Memory {
    fn write(&mut self, invariant: DVector<f32>);

    /// Geometric search on the manifold
    fn query(&self, query: &DVector<f32>, k: usize) -> Vec<DVector<f32>>;
}
