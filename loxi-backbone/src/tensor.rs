use serde::{Deserialize, Serialize};

/// Definición pura de un Tensor para transporte P2P o carga en Buffer WGSL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    // Nota: en el futuro, Vec<f32> puede reemplazarse por bytes (Vec<u8>)
    // y tipos genéricos (f16, bf16) para cuantización.
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
