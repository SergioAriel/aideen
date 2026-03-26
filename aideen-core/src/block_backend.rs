/// `BlockBackend` — contract for a compute backend capable of executing
/// a complete Mamba SSM step on f32 slices.
///
/// Separated from `ComputeBackend` to avoid contaminating the generic contract
/// with Mamba-specific types. Implemented by:
///   - `CpuBlockBackend`   (nalgebra, always available, fallback)
///   - `WgpuBlockBackend` (wgpu + WGSL shaders from aideen-block, feature "wgpu")
///
/// ## Buffer conventions for `mamba_batch_step`
/// ```text
/// Input :  x   [d_model]          — input state/activation
///          dt  [d_model]          — per-channel timescale (delta)
///          a   [d_model]          — per-channel decay (log-domain)
///          b   [d_model]          — per-channel input gate
///          c   [d_model]          — per-channel output gate
/// Output:  y   [d_model]          — SSM output
/// ```
/// The backend is responsible for:
///   1. Uploading buffers to GPU (or keeping them on CPU)
///   2. Executing the kernel (single sequence step, seq_len=1)
///   3. Returning y as Vec<f32>
pub trait BlockBackend: Send {
    fn mamba_batch_step(
        &mut self,
        x: &[f32],  // [d_model]
        dt: &[f32], // [d_model]
        a: &[f32],  // [d_model]
        b: &[f32],  // [d_model]
        c: &[f32],  // [d_model]
    ) -> Result<Vec<f32>, String>;
}
