// Motor de cómputo GPU (wgpu / WGSL)
// Regla: NO puede importar `system`

use loxi_backbone::tensor::Tensor;

/// Contrato abstracto para el backend de ejecución.
/// Evita que el `system` se acople a detalles específicos de `wgpu`.
pub trait ComputeBackend {
    /// Toma un tensor del backbone y lo carga en la memoria del dispositivo (VRAM).
    fn load_tensor(&mut self, tensor: &Tensor) -> Result<(), String>;

    /// Sugerencia inicial: Ejecuta el pipeline de WGSL utilizando los tensores cargados.
    fn execute_shader(&mut self, shader_id: &str) -> Result<Tensor, String>;
}

pub mod wgpu_backend;
