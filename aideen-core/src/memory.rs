use nalgebra::DVector;

/// Contrato mínimo de memoria del agente
pub trait Memory {
    fn write(&mut self, invariant: DVector<f32>);

    /// Búsqueda geométrica en el manifold
    fn query(&self, query: &DVector<f32>, k: usize) -> Vec<DVector<f32>>;
}
