use aideen_core::compute::{ComputeBackend, TensorId};
use bytemuck::{Pod, Zeroable};
use nalgebra::DVector;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct PushConstants {
    in_dim: u32,
    out_dim: u32,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    buffers: HashMap<usize, wgpu::Buffer>,
    next_id: usize,
}

impl WgpuBackend {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or("No WGPU adapter found")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Loxi Engine Device"),
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 8,
                        ..Default::default()
                    },
                },
                None,
            )
            .await
            .map_err(|e| e.to_string())?;

        let shader_src = include_str!("shaders/ffn_forward.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ffn_forward_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ffn_forward_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
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
                    binding: 2,
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
            label: Some("ffn_forward_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..8,
            }],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ffn_forward_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            buffers: HashMap::new(),
            next_id: 1,
        })
    }
}

impl ComputeBackend for WgpuBackend {
    fn load_tensor(&mut self, data: &[f32]) -> Result<TensorId, String> {
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("tensor_{}", self.next_id)),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let id = self.next_id;
        self.buffers.insert(id, buffer);
        self.next_id += 1;
        Ok(TensorId(id))
    }

    fn ffn_forward(
        &mut self,
        weights: &TensorId,
        input: &TensorId,
        out_dim: usize,
    ) -> Result<DVector<f32>, String> {
        let w_buf = self.buffers.get(&weights.0).ok_or("Weights not found")?;
        let i_buf = self.buffers.get(&input.0).ok_or("Input not found")?;

        // Asumiendo w_buf size = in_dim * out_dim * 4 bytes
        let in_dim = (w_buf.size() as usize / 4) / out_dim;

        let out_size = (out_dim * 4) as wgpu::BufferAddress;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output_buffer"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ffn_forward_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: i_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ffn_forward_encoder"),
            });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ffn_forward_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let pc = PushConstants {
                in_dim: in_dim as u32,
                out_dim: out_dim as u32,
            };
            cpass.set_push_constants(0, bytemuck::cast_slice(&[pc]));

            let wg_count = (out_dim as f32 / 64.0).ceil() as u32;
            cpass.dispatch_workgroups(wg_count, 1, 1);
        }

        // Read result
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_size);
        self.queue.submit(Some(encoder.finish()));

        let buf_slice = staging_buf.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = pollster::block_on(receiver.receive()) {
            let data = buf_slice.get_mapped_range();
            let result_slice: &[f32] = bytemuck::cast_slice(&data);
            let result_vec = DVector::from_column_slice(result_slice);
            drop(data);
            staging_buf.unmap();
            Ok(result_vec)
        } else {
            Err("Fallo al leer buffer de GPU".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wgpu_ffn_forward() {
        let mut backend = WgpuBackend::new().await.unwrap();

        let out_dim = 2;

        // Weights: 2 rows x 4 cols (Column-major layout matching nalgebra)
        // Row 0: [ 1.0, 2.0, 3.0, 4.0 ]
        // Row 1: [-1.0,-2.0,-3.0,-4.0 ]
        // Flattened by column:
        let w_data = vec![
            1.0, -1.0, // Col 0
            2.0, -2.0, // Col 1
            3.0, -3.0, // Col 2
            4.0, -4.0, // Col 3
        ];
        let i_data = vec![1.0, 1.0, 1.0, 1.0];

        let w_id = backend.load_tensor(&w_data).unwrap();
        let i_id = backend.load_tensor(&i_data).unwrap();

        let gpu_result = backend.ffn_forward(&w_id, &i_id, out_dim).unwrap();

        assert_eq!(gpu_result.len(), 2);
        assert!((gpu_result[0] - 10.0).abs() < 1e-4);
        assert!((gpu_result[1] + 10.0).abs() < 1e-4);

        println!("GPU FFN RESULT: {:?}", gpu_result);
    }
}
