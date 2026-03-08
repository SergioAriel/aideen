pub mod async_bridge;
pub mod cg_bridge;
pub mod deq_bridge;
pub mod mamba;
pub mod model;
pub mod moe;
pub mod router;
pub mod tensor;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

use wgpu;

/// Represents the connection to the physical GPU
/// (Apple Metal on macOS/iOS, Vulkan on Android/Linux, WebGPU in browsers)
pub struct ComputeState {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl ComputeState {
    /// Request access to the GPU and initialize the compute queues.
    /// Returns `None` if no suitable adapter is found.
    pub async fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Aideen-Block-V8"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        Some(Self { device, queue })
    }
}
