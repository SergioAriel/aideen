use bytemuck::{Pod, Zeroable};

/// WebGPU Uniform Buffer defining the shape and properties of the current forward pass.
/// Must be #[repr(C)] and 16-byte aligned to comply with standard shader memory layouts (std140).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ComputeShape {
    pub batch_size: u32,
    pub seq_len: u32,
    pub d_model: u32,
    pub num_experts: u32,
}

/// Represents a raw chunk of contiguous memory (FP32/FP16) guaranteed to be safe
/// for direct `memcpy` into the GPU VRAM via WebGPU Buffers.
/// The data is passed directly from PyTorch or the P2P socket without deserialization.
pub struct AlignedTensor<'a> {
    pub data: &'a [f32],
    pub shape: ComputeShape,
}

impl<'a> AlignedTensor<'a> {
    /// Creates a new tensor wrapper. Validates the size matches the expected dimensions.
    pub fn new(data: &'a [f32], shape: ComputeShape) -> Option<Self> {
        let expected_size = (shape.batch_size * shape.seq_len * shape.d_model) as usize;
        if data.len() == expected_size {
            Some(Self { data, shape })
        } else {
            None
        }
    }

    /// Converts the floating point slice directly into an array of bytes `[u8]`
    /// This is strictly safe (Zero-Copy) because f32 is inherently a `Pod` type.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.data)
    }
}
