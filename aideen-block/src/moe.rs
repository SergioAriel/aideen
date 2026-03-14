// Rust bridge to the `moe.wgsl` compute shader.
// Orchestrates the gating multiplication of expert weights directly on VRAM.

use wgpu;

pub struct RustMoEBridge {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl RustMoEBridge {
    pub fn new(device: &wgpu::Device) -> Self {
        // Load the WGSL shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Loxi MoE Gating Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/moe.wgsl").into()),
        });

        // Define the memory bindings (Shape, MoE_Probs, Expert_Outputs, Residual_Stream)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MoE Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // binding(0) shape
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(1) MoE Probs (from Router)
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(2) Expert Outputs
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // binding(3) Residual Stream
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MoE Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MoE Gating Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("moe_residual_gating"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}
