use aideen_core::state::ArchitectureConfig;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct TrainParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    step_t: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
}

pub struct GpuLmHeadTrainer {
    pub config: ArchitectureConfig,
    pub vocab_size: usize,
    h_buf: wgpu::Buffer,
    w_buf: wgpu::Buffer,
    b_buf: wgpu::Buffer,
    pub dl_dh_buf: wgpu::Buffer,
    loss_buf: wgpu::Buffer,
    probs_buf: wgpu::Buffer,
    m_w_buf: wgpu::Buffer,
    v_w_buf: wgpu::Buffer,
    m_b_buf: wgpu::Buffer,
    v_b_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    target_indices_buf: wgpu::Buffer,

    probs_pipeline: wgpu::ComputePipeline,
    probs_bg: wgpu::BindGroup,

    update_pipeline: wgpu::ComputePipeline,
    update_bg: wgpu::BindGroup,
}

impl GpuLmHeadTrainer {
    pub fn new(device: &wgpu::Device, vocab_size: usize, config: ArchitectureConfig) -> Self {
        let d_r = config.d_r;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Train Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_train.wgsl").into()),
        });

        // --- PIPELINE 1: Probs & Loss (Softmax) ---
        let bgl_probs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LM Probs BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let probs_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LM Probs Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LM Probs PL"),
                    bind_group_layouts: &[&bgl_probs],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("lm_probs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- PIPELINE 2: Update & dl_dh ---
        let bgl_update = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LM Update BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
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

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LM Update Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LM Update PL"),
                    bind_group_layouts: &[&bgl_update],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("lm_update_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Params"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let h_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM h_pooled"),
            size: (d_r * 256 * 4) as u64, // Max seq_len = 256
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM W"),
            size: (vocab_size * d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM B"),
            size: (vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dl_dh_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dL/dh"),
            size: (d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Loss"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let probs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Probs"),
            size: (256 * vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let m_w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM m_w"),
            size: (vocab_size * d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM v_w"),
            size: (vocab_size * d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let m_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM m_b"),
            size: (vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let v_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM v_b"),
            size: (vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let target_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Target Indices"),
            size: (256 * 4) as u64, // Max seq_len = 256
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let probs_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Probs BG"),
            layout: &bgl_probs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: target_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: probs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: loss_buf.as_entire_binding(),
                },
            ],
        });

        let update_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Update BG"),
            layout: &bgl_update,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dl_dh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: m_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: v_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: m_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: v_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: target_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: probs_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            config,
            vocab_size,
            h_buf,
            w_buf,
            b_buf,
            dl_dh_buf,
            loss_buf,
            probs_buf,
            m_w_buf,
            v_w_buf,
            m_b_buf,
            v_b_buf,
            params_buf,
            target_indices_buf,
            probs_pipeline,
            probs_bg,
            update_pipeline,
            update_bg,
        }
    }

    pub fn train_step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_pooled: &[f32],
        targets: &[u32],
        lr: f32,
        step_t: u32,
        w_cpu: &[f32],
        b_cpu: &[f32],
        upload_weights: bool,
    ) -> Result<(f32, Vec<f32>), String> {
        let d_r = self.config.d_r;
        let seq_len = targets.len();
        if h_pooled.len() != d_r * seq_len {
            // MVP: We only expect `seq_len * d_r` when processing an array of sequences, but wait
            // The signature of `train_step` is mostly a fallback. `train_step_from_buffer` is the primary GPU path.
            // Let's ensure length validation matches sequences.
            return Err(format!(
                "h_pooled dimension mismatch: expected {}, got {}",
                d_r * seq_len,
                h_pooled.len()
            ));
        }

        let params = TrainParams {
            d_model: d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: seq_len as u32,
            step_t,
            lr_bits: lr.to_bits(),
            beta1_bits: 0.9f32.to_bits(),
            beta2_bits: 0.999f32.to_bits(),
            eps_bits: 1e-8f32.to_bits(),
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.target_indices_buf, 0, bytemuck::cast_slice(targets));
        queue.write_buffer(&self.h_buf, 0, bytemuck::cast_slice(h_pooled));
        if upload_weights {
            queue.write_buffer(&self.w_buf, 0, bytemuck::cast_slice(w_cpu));
            queue.write_buffer(&self.b_buf, 0, bytemuck::cast_slice(b_cpu));
            let zeros_w = vec![0.0f32; self.vocab_size * d_r];
            let zeros_b = vec![0.0f32; self.vocab_size];
            queue.write_buffer(&self.m_w_buf, 0, bytemuck::cast_slice(&zeros_w));
            queue.write_buffer(&self.v_w_buf, 0, bytemuck::cast_slice(&zeros_w));
            queue.write_buffer(&self.m_b_buf, 0, bytemuck::cast_slice(&zeros_b));
            queue.write_buffer(&self.v_b_buf, 0, bytemuck::cast_slice(&zeros_b));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Train Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Probs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probs_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.update_bg, &[]);
            let v_parts = (self.vocab_size as u32 + 15) / 16;
            let d_parts = (d_r as u32 + 15) / 16;
            pass.dispatch_workgroups(v_parts, d_parts, 1);
        }

        let dl_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dL/dh Staging"),
            size: (d_r * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let loss_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM loss staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.dl_dh_buf, 0, &dl_staging, 0, (d_r * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.loss_buf, 0, &loss_staging, 0, 4);
        queue.submit(Some(encoder.finish()));

        let dl_slice = dl_staging.slice(..);
        let loss_slice = loss_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dl_slice.map_async(wgpu::MapMode::Read, |_| {});
        loss_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let dl = bytemuck::cast_slice(&dl_slice.get_mapped_range()).to_vec();
            let loss_int = bytemuck::cast_slice::<u8, i32>(&loss_slice.get_mapped_range())[0];
            let loss = loss_int as f32 / 10000.0;
            dl_staging.unmap();
            loss_staging.unmap();
            Ok((loss, dl))
        } else {
            Err("LM train map failed".to_string())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_step_from_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_src: &wgpu::Buffer,
        h_offset: u64,
        targets: &[u32],
        lr: f32,
        step_t: u32,
        w_cpu: &[f32],
        b_cpu: &[f32],
        upload_weights: bool,
    ) -> Result<(f32, Vec<u8>), String> {
        let seq_len = targets.len();
        let params = TrainParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: seq_len as u32,
            step_t,
            lr_bits: lr.to_bits(),
            beta1_bits: 0.9f32.to_bits(),
            beta2_bits: 0.999f32.to_bits(),
            eps_bits: 1e-8f32.to_bits(),
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.target_indices_buf, 0, bytemuck::cast_slice(targets));
        if upload_weights {
            queue.write_buffer(&self.w_buf, 0, bytemuck::cast_slice(w_cpu));
            queue.write_buffer(&self.b_buf, 0, bytemuck::cast_slice(b_cpu));
            let zeros_w = vec![0.0f32; self.vocab_size * self.config.d_r];
            let zeros_b = vec![0.0f32; self.vocab_size];
            queue.write_buffer(&self.m_w_buf, 0, bytemuck::cast_slice(&zeros_w));
            queue.write_buffer(&self.v_w_buf, 0, bytemuck::cast_slice(&zeros_w));
            queue.write_buffer(&self.m_b_buf, 0, bytemuck::cast_slice(&zeros_b));
            queue.write_buffer(&self.v_b_buf, 0, bytemuck::cast_slice(&zeros_b));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Train Encoder (buffer input)"),
        });

        let copy_size = (self.config.d_r * seq_len * 4) as u64;
        encoder.copy_buffer_to_buffer(h_src, h_offset, &self.h_buf, 0, copy_size);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Probs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probs_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.update_bg, &[]);
            let v_parts = (self.vocab_size as u32 + 15) / 16;
            let d_parts = (self.config.d_r as u32 + 15) / 16;
            pass.dispatch_workgroups(v_parts, d_parts, 1);
        }

        let dl_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dL/dh Staging"),
            size: (self.config.d_r * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let loss_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM loss staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &self.dl_dh_buf,
            0,
            &dl_staging,
            0,
            (self.config.d_r * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&self.loss_buf, 0, &loss_staging, 0, 4);
        queue.submit(Some(encoder.finish()));

        let dl_slice = dl_staging.slice(..);
        let loss_slice = loss_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        dl_slice.map_async(wgpu::MapMode::Read, |_| {});
        loss_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let dl = dl_slice.get_mapped_range().to_vec();
            let loss_int = bytemuck::cast_slice::<u8, i32>(&loss_slice.get_mapped_range())[0];
            let loss = loss_int as f32 / 10000.0;
            dl_staging.unmap();
            loss_staging.unmap();
            Ok((loss, dl))
        } else {
            Err("LM train map failed".to_string())
        }
    }

    /// Executa el entrenamiento del LM Head sin leer resultados de vuelta a la CPU.
    /// Ideal para encadenar con el retro-propagación de la DEQ en la GPU.
    pub fn train_step_no_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_src: &wgpu::Buffer,
        h_offset: u64,
        targets: &[u32],
        lr: f32,
        step_t: u32,
    ) -> Result<(), String> {
        let seq_len = targets.len();
        let params = TrainParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: seq_len as u32,
            step_t,
            lr_bits: lr.to_bits(),
            beta1_bits: 0.9f32.to_bits(),
            beta2_bits: 0.999f32.to_bits(),
            eps_bits: 1e-8f32.to_bits(),
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.target_indices_buf, 0, bytemuck::cast_slice(targets));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Train Encoder (NO READBACK)"),
        });

        let copy_size = (self.config.d_r * seq_len * 4) as u64;
        encoder.copy_buffer_to_buffer(h_src, h_offset, &self.h_buf, 0, copy_size);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Probs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probs_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.update_bg, &[]);
            let v_parts = (self.vocab_size as u32 + 15) / 16;
            let d_parts = (self.config.d_r as u32 + 15) / 16;
            pass.dispatch_workgroups(v_parts, d_parts, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn read_weights(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let w_bytes = (self.vocab_size * self.config.d_r * 4) as u64;
        let b_bytes = (self.vocab_size * 4) as u64;
        let w_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM W staging"),
            size: w_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let b_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM B staging"),
            size: b_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.w_buf, 0, &w_staging, 0, w_bytes);
        encoder.copy_buffer_to_buffer(&self.b_buf, 0, &b_staging, 0, b_bytes);
        queue.submit(Some(encoder.finish()));

        let w_slice = w_staging.slice(..);
        let b_slice = b_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        w_slice.map_async(wgpu::MapMode::Read, |_| {});
        b_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = rx.recv() {
            let w = bytemuck::cast_slice(&w_slice.get_mapped_range()).to_vec();
            let b = bytemuck::cast_slice(&b_slice.get_mapped_range()).to_vec();
            w_staging.unmap();
            b_staging.unmap();
            Ok((w, b))
        } else {
            Err("LM weights map failed".to_string())
        }
    }
}
