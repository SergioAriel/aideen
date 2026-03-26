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
    ternary_flag: u32,
    num_samples: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct GpuLmHeadTrainer {
    pub config: ArchitectureConfig,
    pub vocab_size: usize,
    h_buf: wgpu::Buffer,
    w_buf: wgpu::Buffer,
    b_buf: wgpu::Buffer,
    pub dl_dh_buf: wgpu::Buffer,
    pub dl_dh_temp_buf: wgpu::Buffer,
    _dl_dh_rms_red_buf: wgpu::Buffer,

    loss_buf: wgpu::Buffer,
    g_buf: wgpu::Buffer,
    moments_w_buf: wgpu::Buffer, // Combined m_w, v_w (Vocab * D_model * vec2)
    moments_b_buf: wgpu::Buffer, // Combined m_b, v_b (Vocab * vec2)
    moments_g_buf: wgpu::Buffer, // Combined m_g, v_g (D_model * vec2)
    params_buf: wgpu::Buffer,
    target_indices_buf: wgpu::Buffer,
    sampled_indices_buf: wgpu::Buffer,
    dl_staging_buf: wgpu::Buffer,
    loss_staging_buf: wgpu::Buffer,

    probs_pipeline: wgpu::ComputePipeline,
    probs_bg: wgpu::BindGroup,

    update_pipeline: wgpu::ComputePipeline,

    backprop_pipeline: wgpu::ComputePipeline,
    backprop_rms_reduce_pipeline: wgpu::ComputePipeline,
    backprop_rms_apply_pipeline: wgpu::ComputePipeline,
    backprop_reduce_pipeline: wgpu::ComputePipeline,
    _ternary_pipeline: wgpu::ComputePipeline,
    _s_h_rms_buf: wgpu::Buffer,
    last_sampled_indices: Vec<u32>,
    last_num_samples: u32,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
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

        // --- PIPELINE 2: Update & AdamW ---
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LM Update Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LM Update PL"),
                    bind_group_layouts: &[&bgl_probs],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("lm_update_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- PIPELINE 3: Backprop (dl_dh_t) ---
        let backprop_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LM Backprop T Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LM Backprop T PL"),
                    bind_group_layouts: &[&bgl_probs],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("lm_backprop_h_t_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- PIPELINE 3.5: Backprop RMS Reduce ---
        let backprop_rms_reduce_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Backprop RMS Reduce Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Backprop RMS Reduce PL"),
                        bind_group_layouts: &[&bgl_probs],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("lm_backprop_rms_reduce_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- PIPELINE 3.6: Backprop RMS Apply ---
        let backprop_rms_apply_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Backprop RMS Apply Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Backprop RMS Apply PL"),
                        bind_group_layouts: &[&bgl_probs],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("lm_backprop_rms_apply_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- PIPELINE 4: Backprop Reduce ---
        let backprop_reduce_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Backprop Reduce Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Backprop Reduce PL"),
                        bind_group_layouts: &[&bgl_probs],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &shader,
                entry_point: Some("lm_backprop_reduce_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- PIPELINE 5: Ternary Project ---
        let ternary_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LM Ternary Pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LM Ternary PL"),
                    bind_group_layouts: &[&bgl_probs],
                    push_constant_ranges: &[],
                }),
            ),
            module: &shader,
            entry_point: Some("lm_project_ternary_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Params"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lm_batch_size = std::env::var("AIDEEN_BATCH_SIZE")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let safe_ctx = config.ctx_len.max(1024) * lm_batch_size;
        let h_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM h_pooled"),
            size: (d_r * safe_ctx * 4) as u64,
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
            size: (safe_ctx * d_r * 4) as u64,
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

        // Consolidated Moments — COPY_SRC needed for checkpoint readback
        let moments_w_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Moments W"),
            size: (vocab_size * d_r * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let moments_b_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Moments B"),
            size: (vocab_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let g_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM g_lm"),
            size: (d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let moments_g_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Moments G"),
            size: (d_r * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let target_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Target Indices"),
            size: (safe_ctx * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let probs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Probs Temp"),
            size: (safe_ctx * config.num_samples * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let rms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM RMS"),
            size: (safe_ctx * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let dl_dh_temp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dL/dh Temp"),
            size: (safe_ctx * d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dl_dh_rms_red_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dL/dh RMS Reduce"),
            size: (safe_ctx * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampled_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Sampled Indices"),
            size: (config.num_samples * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let s_h_rms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Head S_h_rms"),
            size: (safe_ctx * config.d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dl_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dL/dh Staging"),
            size: (safe_ctx * d_r * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let loss_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM loss staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let probs_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Unified BG"),
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
                    binding: 4,
                    resource: dl_dh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: loss_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: moments_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: moments_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: target_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: probs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: moments_g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: rms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: dl_dh_temp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: sampled_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: s_h_rms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: dl_dh_rms_red_buf.as_entire_binding(),
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
            _dl_dh_rms_red_buf: dl_dh_rms_red_buf,
            loss_buf,
            g_buf,
            moments_w_buf,
            moments_b_buf,
            moments_g_buf,
            params_buf,
            target_indices_buf,
            dl_dh_temp_buf,
            sampled_indices_buf,
            dl_staging_buf,
            loss_staging_buf,
            probs_pipeline,
            probs_bg,
            update_pipeline,
            backprop_pipeline,
            backprop_rms_reduce_pipeline,
            backprop_rms_apply_pipeline,
            backprop_reduce_pipeline,
            _ternary_pipeline: ternary_pipeline,
            _s_h_rms_buf: s_h_rms_buf,
            last_sampled_indices: Vec::new(),
            last_num_samples: 0,
        }
    }

    pub fn last_sampled_indices(&self) -> &[u32] {
        &self.last_sampled_indices
    }

    pub fn last_num_samples(&self) -> u32 {
        self.last_num_samples
    }

    pub fn train_step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_pooled: &[f32],
        targets: &[u32],
        lr: f32,
        step_t: u32,
        w_cpu: &[f32],
        b_cpu: &[f32],
        g_cpu: &[f32],
        upload_weights: bool,
        ternary: bool,
    ) -> Result<(f32, Vec<f32>), String> {
        let d_r = self.config.d_r;
        let seq_len = targets.len();
        if h_pooled.len() != d_r * seq_len {
            // Log warning but don't return error during dev/optimization
            // println!("Warning: h_pooled dimension mismatch (expected {}, got {})", d_r * seq_len, h_pooled.len());
        }

        // Sampled Softmax: targets + random negatives — zero allocs
        let max_samples = self.config.num_samples;
        let mut rng = rand::thread_rng();
        let mut sampled_indices = Vec::with_capacity(max_samples);
        sampled_indices.extend_from_slice(targets);
        if sampled_indices.len() > max_samples {
            sampled_indices.truncate(max_samples);
        } else {
            let needed = max_samples - sampled_indices.len();
            for _ in 0..needed {
                use rand::Rng;
                sampled_indices.push(rng.gen_range(0..self.vocab_size as u32));
            }
        }
        sampled_indices.sort_unstable();
        sampled_indices.dedup();
        // Final guard to ensure we don't exceed the buffer if dedup didn't shrink it enough
        if sampled_indices.len() > max_samples {
            sampled_indices.truncate(max_samples);
        }
        let actual_num_samples = sampled_indices.len() as u32;
        assert!(actual_num_samples <= 512, "num_samples ({actual_num_samples}) exceeds lm_train.wgsl shared memory limit of 512");
        self.last_sampled_indices = sampled_indices.clone();
        self.last_num_samples = actual_num_samples;

        let lm_batch_cap = std::env::var("AIDEEN_BATCH_SIZE")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        if seq_len > self.config.ctx_len * lm_batch_cap {
            return Err(format!(
                "seq_len {} exceeds architectural limit of {}",
                seq_len,
                self.config.ctx_len * lm_batch_cap
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
            ternary_flag: if ternary { 1 } else { 0 },
            num_samples: actual_num_samples,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.target_indices_buf, 0, bytemuck::cast_slice(targets));
        queue.write_buffer(
            &self.sampled_indices_buf,
            0,
            bytemuck::cast_slice(&sampled_indices),
        );
        queue.write_buffer(&self.h_buf, 0, bytemuck::cast_slice(h_pooled));
        if upload_weights {
            queue.write_buffer(&self.w_buf, 0, bytemuck::cast_slice(w_cpu));
            queue.write_buffer(&self.b_buf, 0, bytemuck::cast_slice(b_cpu));
            queue.write_buffer(&self.g_buf, 0, bytemuck::cast_slice(g_cpu));
            // Consolidated Moments
            let zero_w = vec![0u8; self.vocab_size * d_r * 8];
            let zero_b = vec![0u8; self.vocab_size * 8];
            let zero_g = vec![0u8; d_r * 8];
            queue.write_buffer(&self.moments_w_buf, 0, &zero_w);
            queue.write_buffer(&self.moments_b_buf, 0, &zero_b);
            queue.write_buffer(&self.moments_g_buf, 0, &zero_g);
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
            pass.set_bind_group(0, &self.probs_bg, &[]);
            let v_parts = (actual_num_samples + 15) / 16;
            let d_parts = (d_r as u32 + 15) / 16;
            pass.dispatch_workgroups(d_parts, v_parts, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop T Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, d_r as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop RMS Reduce Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_rms_reduce_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop RMS Apply Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_rms_apply_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, d_r as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop Reduce Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_reduce_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            let d_parts = (d_r as u32 + 255) / 256;
            pass.dispatch_workgroups(d_parts, 1, 1);
        }

        // Ternary project is now fused into update_pipeline

        encoder.copy_buffer_to_buffer(
            &self.dl_dh_buf,
            0,
            &self.dl_staging_buf,
            0,
            (d_r * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&self.loss_buf, 0, &self.loss_staging_buf, 0, 4);
        queue.submit(Some(encoder.finish()));

        let dl_slice = self.dl_staging_buf.slice(..);
        let loss_slice = self.loss_staging_buf.slice(..);
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
            self.dl_staging_buf.unmap();
            self.loss_staging_buf.unmap();
            Ok((loss, dl))
        } else {
            Err("LM train map failed".to_string())
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn train_step_from_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_src: &wgpu::Buffer,
        h_offset: u64,
        targets: &[u32],
        lr: f32,
        step_t: u32,
        w_cpu: &[f32],
        b_cpu: &[f32],
        g_cpu: &[f32],
        upload_weights: bool,
        ternary: bool,
    ) -> Result<(f32, Vec<u8>), String> {
        let seq_len = targets.len();

        let max_samples = self.config.num_samples;
        let mut rng = rand::thread_rng();
        let mut sampled_indices = Vec::with_capacity(max_samples + targets.len());
        sampled_indices.extend_from_slice(targets);
        let needed = max_samples.saturating_sub(targets.len());
        for _ in 0..needed {
            use rand::Rng;
            sampled_indices.push(rng.gen_range(0..self.vocab_size as u32));
        }
        sampled_indices.sort_unstable();
        sampled_indices.dedup();
        // Truncar a max_samples para no desbordar sampled_indices_buf.
        sampled_indices.truncate(max_samples);
        let actual_num_samples = sampled_indices.len() as u32;
        assert!(actual_num_samples <= 512, "num_samples ({actual_num_samples}) exceeds lm_train.wgsl shared memory limit of 512");
        self.last_sampled_indices = sampled_indices.clone();
        self.last_num_samples = actual_num_samples;

        let params = TrainParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: seq_len as u32,
            step_t,
            lr_bits: lr.to_bits(),
            beta1_bits: 0.9f32.to_bits(),
            beta2_bits: 0.999f32.to_bits(),
            eps_bits: 1e-8f32.to_bits(),
            ternary_flag: if ternary { 1 } else { 0 },
            num_samples: actual_num_samples,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.target_indices_buf, 0, bytemuck::cast_slice(targets));
        queue.write_buffer(
            &self.sampled_indices_buf,
            0,
            bytemuck::cast_slice(&sampled_indices),
        );
        if upload_weights {
            queue.write_buffer(&self.w_buf, 0, bytemuck::cast_slice(w_cpu));
            queue.write_buffer(&self.b_buf, 0, bytemuck::cast_slice(b_cpu));
            queue.write_buffer(&self.g_buf, 0, bytemuck::cast_slice(g_cpu));
            // Consolidated Moments
            let zero_w = vec![0u8; self.vocab_size * self.config.d_r * 8];
            let zero_b = vec![0u8; self.vocab_size * 8];
            let zero_g = vec![0u8; self.config.d_r * 8];
            queue.write_buffer(&self.moments_w_buf, 0, &zero_w);
            queue.write_buffer(&self.moments_b_buf, 0, &zero_b);
            queue.write_buffer(&self.moments_g_buf, 0, &zero_g);
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
            pass.set_bind_group(0, &self.probs_bg, &[]);
            let v_parts = (actual_num_samples + 15) / 16;
            let d_parts = (self.config.d_r as u32 + 15) / 16;
            pass.dispatch_workgroups(d_parts, v_parts, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop T Pass (buffer input)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, self.config.d_r as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop RMS Reduce Pass (buffer input)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_rms_reduce_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop RMS Apply Pass (buffer input)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_rms_apply_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, self.config.d_r as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop Reduce Pass (buffer input)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_reduce_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            let d_parts = (self.config.d_r as u32 + 255) / 256;
            pass.dispatch_workgroups(d_parts, 1, 1);
        }

        // Ternary project is now fused into update_pipeline

        encoder.copy_buffer_to_buffer(
            &self.dl_dh_buf,
            0,
            &self.dl_staging_buf,
            0,
            (self.config.d_r * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&self.loss_buf, 0, &self.loss_staging_buf, 0, 4);
        queue.submit(Some(encoder.finish()));

        let dl_slice = self.dl_staging_buf.slice(..);
        let loss_slice = self.loss_staging_buf.slice(..);
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
            self.dl_staging_buf.unmap();
            self.loss_staging_buf.unmap();
            Ok((loss, dl))
        } else {
            Err("LM train map failed".to_string())
        }
    }

    /// Executes LM Head training without reading results back to the CPU.
    /// Ideal for chaining with DEQ backpropagation on the GPU.
    pub fn train_step_no_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_src: &wgpu::Buffer,
        h_offset: u64,
        targets: &[u32],
        lr: f32,
        step_t: u32,
        ternary: bool,
        read_loss: bool,
    ) -> Result<f32, String> {
        let seq_len = targets.len();

        let max_samples = self.config.num_samples;
        let mut rng = rand::thread_rng();
        let mut sampled_indices = Vec::with_capacity(max_samples + targets.len());
        sampled_indices.extend_from_slice(targets);
        let needed = max_samples.saturating_sub(targets.len());
        for _ in 0..needed {
            use rand::Rng;
            sampled_indices.push(rng.gen_range(0..self.vocab_size as u32));
        }
        sampled_indices.sort_unstable();
        sampled_indices.dedup();
        // Truncar a max_samples para no desbordar sampled_indices_buf
        sampled_indices.truncate(max_samples);
        let actual_num_samples = sampled_indices.len() as u32;
        assert!(actual_num_samples <= 512, "num_samples ({actual_num_samples}) exceeds lm_train.wgsl shared memory limit of 512");
        self.last_sampled_indices = sampled_indices.clone();
        self.last_num_samples = actual_num_samples;

        let params = TrainParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            seq_len: seq_len as u32,
            step_t,
            lr_bits: lr.to_bits(),
            beta1_bits: 0.9f32.to_bits(),
            beta2_bits: 0.999f32.to_bits(),
            eps_bits: 1e-8f32.to_bits(),
            ternary_flag: if ternary { 1 } else { 0 },
            num_samples: actual_num_samples,
            _pad1: 0,
            _pad2: 0,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.target_indices_buf, 0, bytemuck::cast_slice(targets));
        queue.write_buffer(
            &self.sampled_indices_buf,
            0,
            bytemuck::cast_slice(&sampled_indices),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Train Encoder (NO READBACK)"),
        });

        let copy_size = (self.config.d_r * seq_len * 4) as u64;
        encoder.copy_buffer_to_buffer(h_src, h_offset, &self.h_buf, 0, copy_size);
        encoder.clear_buffer(&self.loss_buf, 0, None);
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
            pass.set_bind_group(0, &self.probs_bg, &[]);
            let v_parts = (actual_num_samples + 15) / 16;
            let d_parts = (self.config.d_r as u32 + 15) / 16;
            pass.dispatch_workgroups(d_parts, v_parts, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop T Pass (NO READBACK)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, self.config.d_r as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop RMS Reduce Pass (NO READBACK)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_rms_reduce_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop RMS Apply Pass (NO READBACK)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_rms_apply_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(seq_len as u32, self.config.d_r as u32, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Backprop Reduce Pass (NO READBACK)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.backprop_reduce_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            let d_parts = (self.config.d_r as u32 + 255) / 256;
            pass.dispatch_workgroups(d_parts, 1, 1);
        }

        // Ternary project is now fused into update_pipeline

        // Always copy loss to staging so read_cached_loss() can retrieve it later.
        encoder.copy_buffer_to_buffer(&self.loss_buf, 0, &self.loss_staging_buf, 0, 4);
        queue.submit(Some(encoder.finish()));

        if !read_loss {
            return Ok(0.0); // Don't block — caller reads loss via read_cached_loss()
        }

        // Leer pérdida (solo si se solicita)
        let slice = self.loss_staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let loss_int = i32::from_le_bytes(data[0..4].try_into().unwrap());
            drop(data);
            self.loss_staging_buf.unmap();
            Ok(loss_int as f32 / 10000.0)
        } else {
            Err("Failed to read loss from GPU".to_string())
        }
    }

    /// Reads the loss value cached in `loss_staging_buf` from the last completed step.
    /// Call after device.poll(Wait) — GPU is already idle so Poll suffices here.
    pub fn read_cached_loss(&self, device: &wgpu::Device) -> f32 {
        let slice = self.loss_staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait); // <--- CAMBIADO de Poll a Wait para evitar Deadlock
        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let loss_int = i32::from_le_bytes(data[0..4].try_into().unwrap());
            drop(data);
            self.loss_staging_buf.unmap();
            loss_int as f32 / 10000.0
        } else {
            0.0
        }
    }

    /// Read back the reduced dL/dh vector from GPU memory.
    pub fn read_dl_dh_temp(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        d_model: usize,
    ) -> Result<Vec<f32>, String> {
        let bytes = (d_model * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dl_dh_temp staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM dl_dh_temp readback"),
        });
        encoder.copy_buffer_to_buffer(&self.dl_dh_temp_buf, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            Ok(data)
        } else {
            Err("LM dl_dh_temp map failed".to_string())
        }
    }

    /// Read back the final dL/dh vector (post-RMSNorm) from GPU memory.
    pub fn read_dl_dh(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        d_model: usize,
    ) -> Result<Vec<f32>, String> {
        let bytes = (d_model * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dl_dh staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM dl_dh readback"),
        });
        encoder.copy_buffer_to_buffer(&self.dl_dh_buf, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let vec = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
            drop(data);
            staging.unmap();
            Ok(vec)
        } else {
            Err("LM dl_dh map failed".to_string())
        }
    }

    pub fn read_weights(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let w_bytes = (self.vocab_size * self.config.d_r * 4) as u64;
        let b_bytes = (self.vocab_size * 4) as u64;
        let g_bytes = (self.config.d_r * 4) as u64;
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
        let g_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM G staging"),
            size: g_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.w_buf, 0, &w_staging, 0, w_bytes);
        encoder.copy_buffer_to_buffer(&self.b_buf, 0, &b_staging, 0, b_bytes);
        encoder.copy_buffer_to_buffer(&self.g_buf, 0, &g_staging, 0, g_bytes);
        queue.submit(Some(encoder.finish()));

        let w_slice = w_staging.slice(..);
        let b_slice = b_staging.slice(..);
        let g_slice = g_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        w_slice.map_async(wgpu::MapMode::Read, |_| {});
        b_slice.map_async(wgpu::MapMode::Read, |_| {});
        g_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = rx.recv() {
            let w = bytemuck::cast_slice(&w_slice.get_mapped_range()).to_vec();
            let b = bytemuck::cast_slice(&b_slice.get_mapped_range()).to_vec();
            let g = bytemuck::cast_slice(&g_slice.get_mapped_range()).to_vec();
            w_staging.unmap();
            b_staging.unmap();
            g_staging.unmap();
            Ok((w, b, g))
        } else {
            Err("LM weights map failed".to_string())
        }
    }

    /// Reads Adam moments (m, v) from the GPU for checkpoint.
    /// Returns (m_w, v_w, m_b, v_b, m_g, v_g) -- all as Vec<f32>.
    /// The GPU format is interleaved float2: [m0, v0, m1, v1, ...].
    pub fn read_moments(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>), String> {
        let n_w = self.vocab_size * self.config.d_r;
        let n_b = self.vocab_size;
        let n_g = self.config.d_r;

        let mw_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM mw stage"),
            size: (n_w * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mb_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM mb stage"),
            size: (n_b * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mg_stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM mg stage"),
            size: (n_g * 8) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Moments Readback"),
        });
        enc.copy_buffer_to_buffer(&self.moments_w_buf, 0, &mw_stage, 0, (n_w * 8) as u64);
        enc.copy_buffer_to_buffer(&self.moments_b_buf, 0, &mb_stage, 0, (n_b * 8) as u64);
        enc.copy_buffer_to_buffer(&self.moments_g_buf, 0, &mg_stage, 0, (n_g * 8) as u64);
        queue.submit(Some(enc.finish()));

        let mw_sl = mw_stage.slice(..);
        let mb_sl = mb_stage.slice(..);
        let mg_sl = mg_stage.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        mw_sl.map_async(wgpu::MapMode::Read, |_| {});
        mb_sl.map_async(wgpu::MapMode::Read, |_| {});
        mg_sl.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);

        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            return Err("LM moments map failed".to_string());
        }
        let deinterleave = |buf: &[f32]| -> (Vec<f32>, Vec<f32>) {
            let n = buf.len() / 2;
            (
                (0..n).map(|i| buf[2 * i]).collect(),
                (0..n).map(|i| buf[2 * i + 1]).collect(),
            )
        };
        let raw_w: Vec<f32> = bytemuck::cast_slice(&mw_sl.get_mapped_range()).to_vec();
        let raw_b: Vec<f32> = bytemuck::cast_slice(&mb_sl.get_mapped_range()).to_vec();
        let raw_g: Vec<f32> = bytemuck::cast_slice(&mg_sl.get_mapped_range()).to_vec();
        mw_stage.unmap();
        mb_stage.unmap();
        mg_stage.unmap();

        let (m_w, v_w) = deinterleave(&raw_w);
        let (m_b, v_b) = deinterleave(&raw_b);
        let (m_g, v_g) = deinterleave(&raw_g);
        Ok((m_w, v_w, m_b, v_b, m_g, v_g))
    }

    /// Uploads previously saved Adam moments back to the GPU.
    /// Expects the same 6 vecs returned by `read_moments`.
    pub fn write_moments(
        &self,
        queue: &wgpu::Queue,
        m_w: &[f32],
        v_w: &[f32],
        m_b: &[f32],
        v_b: &[f32],
        m_g: &[f32],
        v_g: &[f32],
    ) {
        let interleave = |m: &[f32], v: &[f32]| -> Vec<f32> {
            m.iter().zip(v.iter()).flat_map(|(&a, &b)| [a, b]).collect()
        };
        queue.write_buffer(
            &self.moments_w_buf,
            0,
            bytemuck::cast_slice(&interleave(m_w, v_w)),
        );
        queue.write_buffer(
            &self.moments_b_buf,
            0,
            bytemuck::cast_slice(&interleave(m_b, v_b)),
        );
        queue.write_buffer(
            &self.moments_g_buf,
            0,
            bytemuck::cast_slice(&interleave(m_g, v_g)),
        );
    }

    /// Uploads w, b, g to the GPU without dispatching any kernel.
    /// Use to initialize weights the first time before `train_step_no_readback`.
    pub fn upload_weights_only(
        &self,
        queue: &wgpu::Queue,
        w_cpu: &[f32],
        b_cpu: &[f32],
        g_cpu: &[f32],
    ) {
        queue.write_buffer(&self.w_buf, 0, bytemuck::cast_slice(w_cpu));
        queue.write_buffer(&self.b_buf, 0, bytemuck::cast_slice(b_cpu));
        queue.write_buffer(&self.g_buf, 0, bytemuck::cast_slice(g_cpu));
        // Consolidated Moments
        let zero_w = vec![0u8; self.vocab_size * self.config.d_r * 8];
        let zero_b = vec![0u8; self.vocab_size * 8];
        let zero_g = vec![0u8; self.config.d_r * 8];
        queue.write_buffer(&self.moments_w_buf, 0, &zero_w);
        queue.write_buffer(&self.moments_b_buf, 0, &zero_b);
        queue.write_buffer(&self.moments_g_buf, 0, &zero_g);
    }
}
