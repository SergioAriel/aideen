use aideen_core::state::ArchitectureConfig;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct EmbeddingParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    ctx_len: u32,
    lr_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct GpuEmbeddingTrainer {
    pub config: ArchitectureConfig,
    vocab_size: usize,
    max_seq_len: usize,
    gather_pipeline: wgpu::ComputePipeline,
    query_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    token_ids_buf: wgpu::Buffer,
    emb_buf: wgpu::Buffer,
    seq_buf: wgpu::Buffer,
    query_buf: wgpu::Buffer,
    dl_dh_buf: wgpu::Buffer,
}

impl GpuEmbeddingTrainer {
    pub fn new(
        device: &wgpu::Device,
        vocab_size: usize,
        max_seq_len: usize,
        config: ArchitectureConfig,
    ) -> Self {
        let d_r = config.d_r;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AIDEEN Embedding Train Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/embedding_train.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Embedding Train BGL"),
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
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Embedding Train Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let gather_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Embedding Gather Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("embedding_gather_seq"),
            compilation_options: Default::default(),
            cache: None,
        });
        let query_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Embedding Query Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("embedding_build_query"),
            compilation_options: Default::default(),
            cache: None,
        });
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Embedding Update Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("embedding_sgd_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Params"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let token_ids_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Token IDs"),
            size: (max_seq_len * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let emb_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Weights"),
            size: (vocab_size * d_r * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let seq_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Sequence Out"),
            size: (max_seq_len * d_r * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let query_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Query Out"),
            size: (d_r * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dl_dh_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding dL/dh"),
            size: (d_r * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Embedding Train Bind Group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: token_ids_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: emb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: seq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: query_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dl_dh_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            vocab_size,
            max_seq_len,
            gather_pipeline,
            query_pipeline,
            update_pipeline,
            bind_group,
            params_buf,
            token_ids_buf,
            emb_buf,
            seq_buf,
            query_buf,
            dl_dh_buf,
            config,
        }
    }

    pub fn prepare_sequence_and_query(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tokens: &[u32],
        ctx_len: usize,
        emb_cpu: &[f32],
        upload_weights: bool,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if tokens.is_empty() || tokens.len() > self.max_seq_len {
            return Err("token sequence length invalid for GPU embedding".to_string());
        }

        let d_r = self.config.d_r;
        let params = EmbeddingParams {
            d_model: d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: tokens.len() as u32,
            ctx_len: ctx_len.min(tokens.len()) as u32,
            lr_bits: 0f32.to_bits(),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.token_ids_buf, 0, bytemuck::cast_slice(tokens));
        if upload_weights {
            queue.write_buffer(&self.emb_buf, 0, bytemuck::cast_slice(emb_cpu));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Embedding Prepare Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Embedding Gather Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.gather_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(tokens.len() as u32, (d_r as u32).div_ceil(64), 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Embedding Query Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.query_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        let seq_bytes = (tokens.len() * d_r * std::mem::size_of::<f32>()) as u64;
        let seq_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Seq Staging"),
            size: seq_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let query_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Query Staging"),
            size: (d_r * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.seq_buf, 0, &seq_staging, 0, seq_bytes);
        encoder.copy_buffer_to_buffer(
            &self.query_buf,
            0,
            &query_staging,
            0,
            (d_r * std::mem::size_of::<f32>()) as u64,
        );
        queue.submit(Some(encoder.finish()));

        let seq_slice = seq_staging.slice(..);
        let query_slice = query_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        seq_slice.map_async(wgpu::MapMode::Read, |_| {});
        query_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let s_sequence =
                bytemuck::cast_slice::<u8, f32>(&seq_slice.get_mapped_range()).to_vec();
            let query = bytemuck::cast_slice::<u8, f32>(&query_slice.get_mapped_range()).to_vec();
            seq_staging.unmap();
            query_staging.unmap();
            Ok((s_sequence, query))
        } else {
            Err("embedding prepare map failed".to_string())
        }
    }

    pub fn apply_embedding_update(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tokens: &[u32],
        dl_dh: &[f32],
        lr: f32,
    ) -> Result<(), String> {
        if tokens.is_empty() || tokens.len() > self.max_seq_len {
            return Err("token sequence length invalid for GPU embedding update".to_string());
        }
        if dl_dh.len() != self.config.d_r {
            return Err("dl_dh dimension mismatch".to_string());
        }

        let params = EmbeddingParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: tokens.len() as u32,
            ctx_len: tokens.len() as u32,
            lr_bits: lr.to_bits(),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.token_ids_buf, 0, bytemuck::cast_slice(tokens));
        queue.write_buffer(&self.dl_dh_buf, 0, bytemuck::cast_slice(dl_dh));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Embedding Update Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Embedding Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Actualiza embeddings usando el gradiente dL/dh ya presente en la GPU.
    pub fn apply_embedding_update_from_buffer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tokens: &[u32],
        dl_dh_src: &wgpu::Buffer,
        lr: f32,
    ) -> Result<(), String> {
        let params = EmbeddingParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: tokens.len() as u32,
            ctx_len: tokens.len() as u32,
            lr_bits: lr.to_bits(),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.token_ids_buf, 0, bytemuck::cast_slice(tokens));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Embedding Update Encoder (GPU buffer)"),
        });

        // Copiar dL/dh desde el buffer de origen (LM Head) al buffer local de dL/dh
        let dl_size = (self.config.d_r * 4) as u64;
        encoder.copy_buffer_to_buffer(dl_dh_src, 0, &self.dl_dh_buf, 0, dl_size);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Embedding Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn read_weights(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>, &'static str> {
        let size = (self.vocab_size * self.config.d_r * std::mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Embedding Readback Staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Embedding Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.emb_buf, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            Ok(data)
        } else {
            Err("embedding read weights map failed")
        }
    }
}
