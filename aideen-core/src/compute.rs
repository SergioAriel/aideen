//! Constitutional compute contracts.

use nalgebra::DVector;

/// Identifier for a tensor loaded in the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorId(pub usize);

/// Constitutional trait for compute backend.
/// Must be implemented by non-shared infrastructure responsible for accelerated operations.
pub trait ComputeBackend {
    /// Initializes or registers a tensor and returns its device ID.
    fn load_tensor(&mut self, data: &[f32]) -> Result<TensorId, String>;

    /// Executes a pure FFN forward pass (matrix-vector multiplication).
    /// `weights` and `input` are IDs of previously loaded tensors.
    fn ffn_forward(
        &mut self,
        weights: &TensorId,
        input: &TensorId,
        out_dim: usize,
    ) -> Result<DVector<f32>, String>;
}
