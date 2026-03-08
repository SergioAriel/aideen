/// `BlockBackend` — contrato para un backend de cómputo capaz de ejecutar
/// un paso completo del SSM de Mamba sobre slices de f32.
///
/// Separado de `ComputeBackend` para no contaminar el contrato genérico
/// con tipos específicos de Mamba. Implementado por:
///   - `CpuBlockBackend`   (nalgebra, siempre disponible, fallback)
///   - `WgpuBlockBackend` (wgpu + WGSL shaders de aideen-block, feature "wgpu")
///
/// ## Convención de buffers para `mamba_batch_step`
/// ```text
/// Input :  x   [d_model]          — estado/activación de entrada
///          dt  [d_model]          — timescale por canal (delta)
///          a   [d_model]          — decay log-domain por canal
///          b   [d_model]          — input gate por canal
///          c   [d_model]          — output gate por canal
/// Output:  y   [d_model]          — salida del SSM
/// ```
/// El backend es responsable de:
///   1. Subir los buffers a GPU (o mantenerlos en CPU)
///   2. Ejecutar el kernel (único paso de secuencia, seq_len=1)
///   3. Retornar y como Vec<f32>
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
