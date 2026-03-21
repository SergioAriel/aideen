use bytemuck::{Pod, Zeroable};
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct DeqComputeShape {
    pub batch_size: u32,
    pub d_model: u32,
    pub h_slots: u32,
    pub max_iters: u32,
    pub epsilon: f32,
    pub damping: f32,
    pub seq_len: u32,
    pub residual_alpha: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DeqUpdateParams {
    mat_len: u32,
    vec_len: u32,
    _pad0: u32,
    _pad1: u32,
    lr: f32,
    _pad2: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct SpectralParams {
    d_model: u32,
    n_iters: u32,
    attn_threshold: f32,
    win_threshold: f32,
    mamba_threshold: f32,
    wv_threshold: f32,
    wo_threshold: f32,
    h_slots: u32,
    mat_idx: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct RustDeqBridge {
    pub h_slots: u32,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub update_pipeline: wgpu::ComputePipeline,
    pub update_bind_group: wgpu::BindGroup,
    update_params_buf: wgpu::Buffer,
    update_grad_mat_buf: wgpu::Buffer,
    update_grad_vec_buf: wgpu::Buffer,

    pub spectral_renorm_pipeline: wgpu::ComputePipeline,
    pub spectral_renorm_bg: wgpu::BindGroup,
    spectral_renorm_params_buf: wgpu::Buffer,

    pub uniform_buf: wgpu::Buffer,
    pub s_buf: wgpu::Buffer,
    pub wq_buf: wgpu::Buffer,
    pub wk_buf: wgpu::Buffer,
    pub wv_buf: wgpu::Buffer,
    pub wo_buf: wgpu::Buffer,
    pub win_buf: wgpu::Buffer,
    pub hist_params_buf: wgpu::Buffer,
    pub wx_buf: wgpu::Buffer,
    pub wout_buf: wgpu::Buffer,
    pub a_buf: wgpu::Buffer,
    pub n_buf: wgpu::Buffer,
    pub hcurr_buf: wgpu::Buffer,
    pub hnext_buf: wgpu::Buffer,
    pub conv_buf: wgpu::Buffer,
    pub scratch_buf: wgpu::Buffer,
    pub hpooled_buf: wgpu::Buffer,
    pub debug_buf: wgpu::Buffer,

    pub bind_group: wgpu::BindGroup,
}

impl RustDeqBridge {
    pub fn new(
        device: &wgpu::Device,
        d_model: u32,
        h_slots: u32,
        max_batch_size: u32,
        max_seq_len: u32,
    ) -> Self {
        let use_exact_forward = std::env::var("AIDEEN_DEQ_FORWARD_EXACT")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        let shader_src = if use_exact_forward {
            include_str!("shaders/deq_forward_exact.wgsl")
        } else {
            include_str!("shaders/deq_forward.wgsl")
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AIDEEN Full DEQ Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let mut entries = Vec::new();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        for i in 1..=10 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        for i in 11..=15 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 16,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DEQ Core Bind Group Layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DEQ Core Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEQ Forward Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("deq_forward_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AIDEEN DEQ SGD Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/deq_sgd_update.wgsl").into()),
        });
        let update_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DEQ Update BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });
        let update_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DEQ Update Pipeline Layout"),
            bind_group_layouts: &[&update_bgl],
            push_constant_ranges: &[],
        });
        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEQ SGD Update Pipeline"),
            layout: Some(&update_pl),
            module: &update_shader,
            entry_point: Some("deq_sgd_update"),
            compilation_options: Default::default(),
            cache: None,
        });

        let uniform_size = std::mem::size_of::<DeqComputeShape>() as u64;
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Shape Uniform"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let s_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("S_in"),
            size: (max_batch_size * max_seq_len * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let wq_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("W_q"),
            size: (d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let wk_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("W_k"),
            size: (d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let wv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("W_v"),
            size: (d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let wo_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("W_o"),
            size: (d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let win_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("W_in"),
            size: (h_slots * d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let hist_params_len =
            2u32 * d_model * d_model + 3u32 * h_slots * d_model + h_slots + d_model + 21u32;
        let hist_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hist Params"),
            size: (hist_params_len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let wx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("W_x"),
            size: (d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let wout_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("W_out"),
            size: (d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let a_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("A_log"),
            size: (d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let n_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NormScale"),
            size: (d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Update Params"),
            size: std::mem::size_of::<DeqUpdateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let update_grad_mat_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Update Grad Mat"),
            size: (d_model * d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let update_grad_vec_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Update Grad Vec"),
            size: (d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let h_bytes =
            (max_batch_size as u64) * (h_slots as u64) * (d_model as u64) * 4u64;
        let hcurr_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_curr"),
            size: h_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // H_next stores exact H*_t for each token:
        // deq_forward.wgsl indexes it as (batch_idx * seq_len + t) * (h_slots * d_model).
        let hnext_bytes = (max_batch_size as u64)
            * (max_seq_len as u64)
            * (h_slots as u64)
            * (d_model as u64)
            * 4u64;
        let hnext_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_next"),
            size: hnext_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let conv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Converged"),
            size: (max_batch_size as u64) * 4u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Stride por (batch, token):
        //   q [h*d] + k [h*d] + v [h*d] + attn_out [h*d] + mamba [h*d] + signal [h*d]
        //   + m_inner [h*d] + attn_weights [h*h]
        //   = d*(7h) + h²
        // deq_forward.wgsl indexa Scratch como:
        //   (batch_idx * seq_len + t) * scratch_stride
        // por lo que el buffer debe reservar batch * seq_len * stride.
        //
        // BUG (viejo): scratch_stride = d_model * (h_slots * 6 + 1) + h_slots * h_slots;
        // Faltaba la región m_inner [h*d] y "signal [d]" era incorrecto (es [h*d] con W_in per-slot).
        // Para d=512, h=8: viejo=25152 floats/token, correcto=28736. Escrituras de attn_weights OOB.
        let scratch_stride = d_model * h_slots * 7 + h_slots * h_slots;
        let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scratchpad"),
            size: (max_batch_size as u64)
                * (max_seq_len as u64)
                * (scratch_stride as u64)
                * 4u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hpooled_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_pooled"),
            size: (max_batch_size as u64)
                * (max_seq_len as u64)
                * (d_model as u64)
                * 4u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debug_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Debug Log"),
            size: 2048, // 512 floats (extra space for per-token debug)
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DEQ Forward Persistent Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: win_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: n_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: hist_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: hcurr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: hnext_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: hpooled_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: debug_buf.as_entire_binding(),
                },
            ],
        });
        let update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DEQ Update BindGroup"),
            layout: &update_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: update_grad_mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: update_grad_vec_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: win_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: n_buf.as_entire_binding(),
                },
            ],
        });

        // --- Spectral Renorm pipeline ---
        let spectral_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AIDEEN Spectral Renorm Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spectral_renorm.wgsl").into()),
        });
        let mk_storage_rw = |b: u32| wgpu::BindGroupLayoutEntry {
            binding: b,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Create intermediate buffers for spectral renorm (power iteration vectors u and v)
        let s_u_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectral Renorm Vector U"),
            size: (d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let s_v_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectral Renorm Vector V"),
            size: (d_model * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spectral_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spectral Renorm BGL"),
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
                mk_storage_rw(1),
                mk_storage_rw(2),
                mk_storage_rw(3),
                mk_storage_rw(4),
                mk_storage_rw(5),
                mk_storage_rw(6),
                mk_storage_rw(7),
                mk_storage_rw(8), // s_u
                mk_storage_rw(9), // s_v
            ],
        });
        let spectral_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spectral Renorm Pipeline Layout"),
            bind_group_layouts: &[&spectral_bgl],
            push_constant_ranges: &[],
        });
        let spectral_renorm_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Spectral Renorm Pipeline"),
                layout: Some(&spectral_pl),
                module: &spectral_shader,
                entry_point: Some("spectral_renorm_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let spectral_renorm_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spectral Renorm Params"),
            size: std::mem::size_of::<SpectralParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spectral_renorm_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spectral Renorm BindGroup"),
            layout: &spectral_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spectral_renorm_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: win_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: s_u_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: s_v_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            h_slots,
            pipeline,
            bind_group_layout,
            update_pipeline,
            update_bind_group,
            update_params_buf,
            update_grad_mat_buf,
            update_grad_vec_buf,
            spectral_renorm_pipeline,
            spectral_renorm_bg,
            spectral_renorm_params_buf,
            uniform_buf,
            s_buf,
            wq_buf,
            wk_buf,
            wv_buf,
            wo_buf,
            win_buf,
            hist_params_buf,
            wx_buf,
            wout_buf,
            a_buf,
            n_buf,
            hcurr_buf,
            hnext_buf,
            conv_buf,
            scratch_buf,
            hpooled_buf,
            debug_buf,
            bind_group,
        }
    }

    pub fn apply_sgd_update(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lr: f32,
        grad_mat: &[f32],
        grad_vec: &[f32],
    ) -> Result<(), &'static str> {
        let mat_len = grad_mat.len() as u32;
        let vec_len = grad_vec.len() as u32;
        let params = DeqUpdateParams {
            mat_len,
            vec_len,
            _pad0: 0,
            _pad1: 0,
            lr,
            _pad2: [0.0; 3],
        };
        queue.write_buffer(&self.update_params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.update_grad_mat_buf, 0, bytemuck::cast_slice(grad_mat));
        queue.write_buffer(&self.update_grad_vec_buf, 0, bytemuck::cast_slice(grad_vec));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ SGD Update Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEQ SGD Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.update_bind_group, &[]);
            let n = mat_len.max(vec_len);
            pass.dispatch_workgroups((n + 255) / 256, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Runs spectral renormalization on all 7 DEQ weight matrices (W_q..W_out) fully on GPU.
    /// Uses power iteration with `n_iters` steps; scales down any matrix whose spectral norm
    /// exceeds `threshold`.
    pub fn renormalize_spectral(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        d_model: u32,
        attn_threshold: f32,
        win_threshold: f32,
        mamba_threshold: f32,
        wv_threshold: f32,
        wo_threshold: f32,
        n_iters: u32,
    ) {
        let h_slots = self.h_slots;
        let mat_count = 6 + h_slots;
        for mat_idx in 0..mat_count {
            let params = SpectralParams {
                d_model,
                n_iters,
                attn_threshold,
                win_threshold,
                mamba_threshold,
                wv_threshold,
                wo_threshold,
                h_slots,
                mat_idx,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            queue.write_buffer(
                &self.spectral_renorm_params_buf,
                0,
                bytemuck::bytes_of(&params),
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Spectral Renorm Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Spectral Renorm Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.spectral_renorm_pipeline);
                pass.set_bind_group(0, &self.spectral_renorm_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
        }
    }

    pub fn read_weights(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        d_model: u32,
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
            Vec<f32>,
        ),
        &'static str,
    > {
        let mat_size = (d_model * d_model * 4) as u64;
        let win_size = (self.h_slots * d_model * d_model * 4) as u64;
        let vec_size = (d_model * 4) as u64;

        let create_staging = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DEQ Readback Staging"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };

        let st_q = create_staging(mat_size);
        let st_k = create_staging(mat_size);
        let st_v = create_staging(mat_size);
        let st_o = create_staging(mat_size);
        let st_in = create_staging(win_size);
        let st_x = create_staging(mat_size);
        let st_out = create_staging(mat_size);
        let st_a = create_staging(vec_size);
        let st_n = create_staging(vec_size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.wq_buf, 0, &st_q, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.wk_buf, 0, &st_k, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.wv_buf, 0, &st_v, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.wo_buf, 0, &st_o, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.win_buf, 0, &st_in, 0, win_size);
        encoder.copy_buffer_to_buffer(&self.wx_buf, 0, &st_x, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.wout_buf, 0, &st_out, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.a_buf, 0, &st_a, 0, vec_size);
        encoder.copy_buffer_to_buffer(&self.n_buf, 0, &st_n, 0, vec_size);

        queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        let s_q = st_q.slice(..);
        let s_k = st_k.slice(..);
        let s_v = st_v.slice(..);
        let s_o = st_o.slice(..);
        let s_in = st_in.slice(..);
        let s_x = st_x.slice(..);
        let s_out = st_out.slice(..);
        let s_a = st_a.slice(..);
        let s_n = st_n.slice(..);

        s_q.map_async(wgpu::MapMode::Read, |_| {});
        s_k.map_async(wgpu::MapMode::Read, |_| {});
        s_v.map_async(wgpu::MapMode::Read, |_| {});
        s_o.map_async(wgpu::MapMode::Read, |_| {});
        s_in.map_async(wgpu::MapMode::Read, |_| {});
        s_x.map_async(wgpu::MapMode::Read, |_| {});
        s_out.map_async(wgpu::MapMode::Read, |_| {});
        s_a.map_async(wgpu::MapMode::Read, |_| {});
        s_n.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let cast = |s: &wgpu::BufferSlice| -> Vec<f32> {
                bytemuck::cast_slice(&s.get_mapped_range()).to_vec()
            };
            let res = (
                cast(&s_q),
                cast(&s_k),
                cast(&s_v),
                cast(&s_o),
                cast(&s_in),
                cast(&s_x),
                cast(&s_out),
                cast(&s_a),
                cast(&s_n),
            );
            st_q.unmap();
            st_k.unmap();
            st_v.unmap();
            st_o.unmap();
            st_in.unmap();
            st_x.unmap();
            st_out.unmap();
            st_a.unmap();
            st_n.unmap();
            Ok(res)
        } else {
            Err("DEQ read weights map failed")
        }
    }

    pub fn run_forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm_scale: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<u32>), &'static str> {
        let d_model = shape.d_model as usize;
        let b = shape.batch_size as usize;
        let h = shape.h_slots as usize;
        let mut encoder = self.encode_forward(
            device,
            queue,
            shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm_scale,
            update_weights,
        );

        let h_size = b * h * d_model;
        let h_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_next Staging"),
            size: (h_size * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let c_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Converged Staging"),
            size: (b * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&self.hnext_buf, 0, &h_staging, 0, (h_size * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.conv_buf, 0, &c_staging, 0, (b * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let h_slice = h_staging.slice(..);
        let c_slice = c_staging.slice(..);

        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        // CORRECCIÓN: MAPEAMOS AMBOS BUFFERS!
        h_slice.map_async(wgpu::MapMode::Read, |_| {});
        c_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });

        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let out_h_data = bytemuck::cast_slice(&h_slice.get_mapped_range()).to_vec();
            let out_c_data = bytemuck::cast_slice(&c_slice.get_mapped_range()).to_vec();
            h_staging.unmap();
            c_staging.unmap();
            Ok((out_h_data, out_c_data))
        } else {
            Err("GPU Buffer Map Failed")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_forward_pooled(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm_scale: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<u32>), &'static str> {
        let d_model = shape.d_model as usize;
        let b = shape.batch_size as usize;
        let mut encoder = self.encode_forward(
            device,
            queue,
            shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm_scale,
            update_weights,
        );

        let pooled_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_pooled Staging"),
            size: (b * (shape.seq_len as usize) * d_model * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let c_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Converged Staging"),
            size: (b * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.hpooled_buf,
            0,
            &pooled_staging,
            0,
            (b * (shape.seq_len as usize) * d_model * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&self.conv_buf, 0, &c_staging, 0, (b * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let pooled_slice = pooled_staging.slice(..);
        let c_slice = c_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        pooled_slice.map_async(wgpu::MapMode::Read, |_| {});
        c_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let out_pooled = bytemuck::cast_slice(&pooled_slice.get_mapped_range()).to_vec();
            let out_c = bytemuck::cast_slice(&c_slice.get_mapped_range()).to_vec();
            pooled_staging.unmap();
            c_staging.unmap();
            Ok((out_pooled, out_c))
        } else {
            Err("GPU Buffer Map Failed")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_forward_gpu_only(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
    ) -> wgpu::CommandEncoder {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(shape));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ Forward GPU-Only Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEQ Forward GPU-Only Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(shape.batch_size.max(1), 1, 1);
        }
        encoder
    }

    pub fn run_forward_gpu_only(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
    ) {
        let encoder = self.encode_forward_gpu_only(device, queue, shape);
        queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_forward_pooled_with_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm_scale: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>), &'static str> {
        let d_model = shape.d_model as usize;
        let b = shape.batch_size as usize;
        let h = shape.h_slots as usize;
        let mut encoder = self.encode_forward(
            device,
            queue,
            shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm_scale,
            update_weights,
        );

        let pooled_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_pooled Staging"),
            size: (b * (shape.seq_len as usize) * d_model * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let h_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_star Staging"),
            size: (b * h * d_model * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let c_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Converged Staging"),
            size: (b * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &self.hpooled_buf,
            0,
            &pooled_staging,
            0,
            (b * (shape.seq_len as usize) * d_model * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.hnext_buf,
            0,
            &h_staging,
            0,
            (b * h * d_model * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&self.conv_buf, 0, &c_staging, 0, (b * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let pooled_slice = pooled_staging.slice(..);
        let h_slice = h_staging.slice(..);
        let c_slice = c_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        pooled_slice.map_async(wgpu::MapMode::Read, |_| {});
        h_slice.map_async(wgpu::MapMode::Read, |_| {});
        c_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let out_pooled = bytemuck::cast_slice(&pooled_slice.get_mapped_range()).to_vec();
            let out_h = bytemuck::cast_slice(&h_slice.get_mapped_range()).to_vec();
            let out_c = bytemuck::cast_slice(&c_slice.get_mapped_range()).to_vec();
            pooled_staging.unmap();
            h_staging.unmap();
            c_staging.unmap();
            Ok((out_pooled, out_h, out_c))
        } else {
            Err("GPU Buffer Map Failed")
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_forward_no_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm_scale: &[f32],
        update_weights: bool,
    ) -> Result<(), &'static str> {
        let encoder = self.encode_forward(
            device,
            queue,
            shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm_scale,
            update_weights,
        );
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm_scale: &[f32],
        update_weights: bool,
    ) -> wgpu::CommandEncoder {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(shape));
        queue.write_buffer(&self.s_buf, 0, bytemuck::cast_slice(s_in));

        if update_weights {
            queue.write_buffer(&self.wq_buf, 0, bytemuck::cast_slice(w_q));
            queue.write_buffer(&self.wk_buf, 0, bytemuck::cast_slice(w_k));
            queue.write_buffer(&self.wv_buf, 0, bytemuck::cast_slice(w_v));
            queue.write_buffer(&self.wo_buf, 0, bytemuck::cast_slice(w_o));
            queue.write_buffer(&self.win_buf, 0, bytemuck::cast_slice(w_in));
            queue.write_buffer(&self.wx_buf, 0, bytemuck::cast_slice(w_x));
            queue.write_buffer(&self.wout_buf, 0, bytemuck::cast_slice(w_out));
            queue.write_buffer(&self.a_buf, 0, bytemuck::cast_slice(a_log));
            queue.write_buffer(&self.n_buf, 0, bytemuck::cast_slice(norm_scale));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ Forward Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEQ Forward Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(shape.batch_size.max(1), 1, 1);
        }
        encoder
    }

    pub fn read_debug_buffer(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let size = self.debug_buf.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Debug Staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ Debug Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.debug_buf, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            data
        } else {
            vec![]
        }
    }

    pub fn read_scratch_buffer(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
        let size = self.scratch_buf.size();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Scratch Staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ Scratch Readback"),
        });
        encoder.copy_buffer_to_buffer(&self.scratch_buf, 0, &staging, 0, size);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            data
        } else {
            vec![]
        }
    }
}
