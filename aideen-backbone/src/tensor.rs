use serde::{Deserialize, Serialize};

/// Pure Tensor definition for P2P transport or WGSL Buffer loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    // Note: in the future, Vec<f32> may be replaced by bytes (Vec<u8>)
    // and generic types (f16, bf16) for quantization.
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let expected_size: usize = shape.iter().product();
        assert_eq!(
            expected_size,
            data.len(),
            "Shape {:?} does not match data length {}",
            shape,
            data.len()
        );
        Self { shape, data }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}
