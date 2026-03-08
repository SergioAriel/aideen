//! Contratos constitucionales de cómputo.

use nalgebra::DVector;

/// Identificador de un tensor cargado en el backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorId(pub usize);

/// Trait constitucional para backend de cómputo.
/// Debe implementarse por infraestructura no compartida pero responsable de operaciones aceleradas.
pub trait ComputeBackend {
    /// Inicializa o registra un tensor y devuelve su ID en el dispositivo.
    fn load_tensor(&mut self, data: &[f32]) -> Result<TensorId, String>;

    /// Ejecuta el forward pass de FFN (multiplicación matriz-vector) puro.
    /// `weights` y `input` son IDs de tensores previamente cargados.
    fn ffn_forward(
        &mut self,
        weights: &TensorId,
        input: &TensorId,
        out_dim: usize,
    ) -> Result<DVector<f32>, String>;
}
