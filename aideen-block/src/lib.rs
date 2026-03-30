pub mod async_bridge;
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

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
        {
            Some(adapter) => adapter,
            None => {
                eprintln!("[ComputeState] No compatible GPU adapter found.");
                return None;
            }
        };

        let subgroup_supported = adapter.features().contains(wgpu::Features::SUBGROUP);
        if subgroup_supported {
            eprintln!("[ComputeState] SUBGROUP supported.");
        } else {
            eprintln!("[ComputeState] SUBGROUP not supported; falling back to portable path.");
        }

        let (device, queue) = match adapter
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
        {
            Ok((device, queue)) => (device, queue),
            Err(err) => {
                eprintln!("[ComputeState] request_device failed: {err:?}");
                return None;
            }
        };

        Some(Self { device, queue })
    }
}
