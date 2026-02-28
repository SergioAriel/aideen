use crate::engine::ComputeBackend;
use loxi_backbone::tensor::Tensor;

/// Backend de ejecución basado en WebGPU `wgpu`.
/// Responsable exclusivo de cargar tensores en VRAM y ejecutar shaders WGSL.
/// Aislado completamente de control, ética y estado global (Constitución LOXI).
pub struct WgpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl WgpuBackend {
    /// Inicializa la conexión con el hardware de video (GPU).
    pub fn new() -> Result<Self, String> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Result<Self, String> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find an appropriate GPU adapter: {}", e))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("Failed to create WebGPU device: {}", e))?;

        Ok(Self { device, queue })
    }
}

impl ComputeBackend for WgpuBackend {
    fn load_tensor(&mut self, tensor: &Tensor) -> Result<(), String> {
        // En una iteración real, esto crearía o actualizaría un wgpu::Buffer.
        // Aquí simulamos que cargamos el Tensor en VRAM aislando al `loxi-core`
        // de saber cómo ocurrió esta transferencia.
        println!(
            "[WgpuBackend] Tensor de tamaño {} cargado en VRAM.",
            tensor.numel()
        );
        Ok(())
    }

    fn execute_shader(&mut self, shader_id: &str) -> Result<Tensor, String> {
        // En una iteración real, esto crearía un CommandEncoder, haría un Dispatch
        // del Compute Shader y leería el resultado mapeándolo de vuelta a RAM.
        println!("[WgpuBackend] Ejecutando shader '{}' en GPU...", shader_id);

        // Retornamos un tensor dummy como prueba
        Ok(Tensor::new(vec![1], vec![0.0]))
    }
}
