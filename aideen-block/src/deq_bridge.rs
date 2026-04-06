use bytemuck::{Pod, Zeroable};
#[repr(C)]
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
    pub debug_enable: u32,
    pub token_start: u32,
    pub token_count: u32,
    pub diag_zero_win: u32,
    pub diag_one_iter: u32,
    pub fpm_alpha_m: f32,
    pub fpm_tau: f32,
    pub fpm_persist_beta: f32,
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
    pub slot_attn_unified_pipeline: wgpu::ComputePipeline,
    pub hist_v2_signal_cache_pipeline: wgpu::ComputePipeline,
    pub hist_v2_project_pipeline: wgpu::ComputePipeline,
    pub hist_v2_temporal_pipeline: wgpu::ComputePipeline,
    pub pool_pipeline: wgpu::ComputePipeline,
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
    pub all_weights_buf: wgpu::Buffer,
    pub hcurr_buf: wgpu::Buffer,
    pub hnext_buf: wgpu::Buffer,
    pub scratch_buf: wgpu::Buffer,
    pub hpooled_buf: wgpu::Buffer,
    pub debug_buf: wgpu::Buffer,
    pub hist_ctx_buf: wgpu::Buffer,
    pub mstate_buf: wgpu::Buffer,
    pub signal_cache_buf: wgpu::Buffer,
    pub h_hist_buf: wgpu::Buffer,
    pub hist_ctx_enabled: bool,
    pub fpm_enabled: bool,

    pub bind_group: wgpu::BindGroup,
}

/// AllWeights flat-buffer byte offsets (verified 256-byte aligned for d=512, h=8).
/// Layout: W_q | W_k | W_v | W_o (per-slot) | W_in | W_x | W_out | A_log | NormScale | HistParams
pub fn aw_wqk_bytes(d: u64, h: u64) -> u64 { (h * d * d + h * d) * 4 }
pub fn aw_wk_byte_off(d: u64, h: u64) -> u64 { aw_wqk_bytes(d, h) }
pub fn aw_wv_byte_off(d: u64, h: u64) -> u64 { 2 * aw_wqk_bytes(d, h) }
pub fn aw_wo_byte_off(d: u64, h: u64) -> u64 { aw_wv_byte_off(d, h) + h * d * d * 4 }
pub fn aw_win_byte_off(d: u64, h: u64) -> u64 { aw_wo_byte_off(d, h) + h * d * d * 4 }
pub fn aw_wx_byte_off(d: u64, h: u64) -> u64 { aw_win_byte_off(d, h) + h * d * d * 4 }
pub fn aw_wout_byte_off(d: u64, h: u64) -> u64 { aw_wx_byte_off(d, h) + d * d * 4 }
pub fn aw_alog_byte_off(d: u64, h: u64) -> u64 { aw_wout_byte_off(d, h) + d * d * 4 }
pub fn aw_nscale_byte_off(d: u64, h: u64) -> u64 { aw_alog_byte_off(d, h) + h * d * 4 }
pub fn aw_hist_byte_off(d: u64, h: u64) -> u64 { aw_nscale_byte_off(d, h) + d * 4 }
pub fn aw_total_bytes(d: u64, h: u64, hist_len: u64) -> u64 { aw_hist_byte_off(d, h) + hist_len * 4 }

impl RustDeqBridge {
    fn clean_scratch_stride(d_model: u32, h_slots: u32) -> u32 {
        let signal = d_model * h_slots;
        signal * 2
    }

    fn pooled_elements(batch_size: u32, seq_len: u32, d_model: u32) -> u64 {
        let b = batch_size as u64;
        let t = seq_len as u64;
        let d = d_model as u64;
        b * t * d
    }

    fn shape_bytes(shape: &DeqComputeShape) -> [u8; 80] {
        let mut out = [0u8; 80];
        let raw = bytemuck::bytes_of(shape);
        out[..raw.len()].copy_from_slice(raw);
        out
    }

    pub fn new(
        device: &wgpu::Device,
        d_model: u32,
        h_slots: u32,
        max_batch_size: u32,
        max_seq_len: u32,
        _subgroup_supported: bool,
    ) -> Self {
        let token_carry_enabled = std::env::var("AIDEEN_DEQ_TOKEN_CARRY")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(true);
        let slot_attn_head_dim = std::env::var("AIDEEN_DEQ_SLOT_ATTN_HEAD_DIM")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|v| v.clamp(8, 32))
            .unwrap_or(32);

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

        // binding 1: S_in (read-only)
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        // binding 2: AllWeights (read-only)
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        // bindings 3-11: H_curr, H_next, Scratch, H_pooled, DebugLog, HistCtx, MState, SignalCache, H_hist
        for i in 3u32..=11 {
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
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DEQ Core Bind Group Layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DEQ Core Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pool_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AIDEEN DEQ Forward Pool Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/deq_forward_pool.wgsl").into()),
        });
        let pool_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DEQ Forward Pool Pipeline"),
            layout: Some(&pipeline_layout),
            module: &pool_shader,
            entry_point: Some("deq_forward_pool_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let unified_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AIDEEN DEQ SlotAttn Unified Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/deq_slot_attn_unified_clean.wgsl").into(),
            ),
        });
        let fpm_enabled = std::env::var("AIDEEN_FPM")
            .ok()
            .map(|v| { let vl = v.trim().to_ascii_lowercase(); vl == "1" || vl == "true" })
            .unwrap_or(false);
        // Sanity rule: FPM and H_hist are mutually exclusive experimental paths.
        // If FPM is enabled, disable H_hist so the shader measures one memory mechanism at a time.
        let h_hist_enabled = !fpm_enabled && std::env::var("AIDEEN_H_HIST")
            .ok()
            .map(|v| { let vl = v.trim().to_ascii_lowercase(); vl == "1" || vl == "true" })
            .unwrap_or(false);
        let hist_ctx_enabled = !fpm_enabled
            && !h_hist_enabled
            && std::env::var("AIDEEN_DEQ_HIST_GATED")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(false);
        let fpm_mem_iters = std::env::var("AIDEEN_FPM_MEM_ITERS")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(1)
            .clamp(1, 8);
        let mut slot_attn_constants = std::collections::HashMap::new();
        slot_attn_constants.insert(
            "SLOT_ATTN_HEAD_DIM".to_string(),
            slot_attn_head_dim as f64,
        );
        slot_attn_constants.insert(
            "ENABLE_TOKEN_CARRY".to_string(),
            if token_carry_enabled { 1.0 } else { 0.0 },
        );
        slot_attn_constants.insert(
            "ENABLE_H_HIST".to_string(),
            if h_hist_enabled { 1.0 } else { 0.0 },
        );
        slot_attn_constants.insert(
            "ENABLE_HIST_CTX".to_string(),
            if hist_ctx_enabled { 1.0 } else { 0.0 },
        );
        slot_attn_constants.insert(
            "ENABLE_FPM".to_string(),
            if fpm_enabled { 1.0 } else { 0.0 },
        );
        slot_attn_constants.insert(
            "FPM_MEM_ITERS".to_string(),
            fpm_mem_iters as f64,
        );
        let slot_attn_unified_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DEQ SlotAttn Unified Pipeline"),
                layout: Some(&pipeline_layout),
                module: &unified_shader,
                entry_point: Some("deq_slot_attn_unified_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &slot_attn_constants,
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            });
        let hist_signal_cache_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hist V2 Signal Cache Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/hist_v2_signal_cache.wgsl").into(),
            ),
        });
        let hist_v2_signal_cache_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Hist V2 Signal Cache Pipeline"),
                layout: Some(&pipeline_layout),
                module: &hist_signal_cache_shader,
                entry_point: Some("hist_v2_signal_cache_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let hist_project_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hist V2 Project Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/hist_v2_project.wgsl").into(),
            ),
        });
        let hist_v2_project_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Hist V2 Project Pipeline"),
                layout: Some(&pipeline_layout),
                module: &hist_project_shader,
                entry_point: Some("hist_v2_project_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let hist_temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Hist V2 Temporal Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/hist_v2_temporal.wgsl").into(),
            ),
        });
        let hist_v2_temporal_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Hist V2 Temporal Pipeline"),
                layout: Some(&pipeline_layout),
                module: &hist_temporal_shader,
                entry_point: Some("hist_v2_temporal_main"),
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

        let uniform_size = 80u64;
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
        // Single AllWeights buffer: W_q | W_k | W_v | W_o | W_in | W_x | W_out | A_log | NormScale | HistParams
        const RETAIN_RANK: u32 = 32;
        let hist_params_len = (h_slots + 1u32) * d_model * d_model
            + 5u32 * h_slots * d_model
            + 2u32 * h_slots
            + d_model
            + 21u32
            + h_slots  // γ per slot
            + 2u32 * h_slots * d_model * RETAIN_RANK  // W_retain_up + W_retain_down
            + h_slots * d_model;                        // b_retain
        let d64 = d_model as u64;
        let h64 = h_slots as u64;
        let all_weights_size = aw_total_bytes(d64, h64, hist_params_len as u64);
        let all_weights_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AllWeights"),
            size: all_weights_size,
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

        let h_bytes = (max_batch_size as u64) * (h_slots as u64) * (d_model as u64) * 4u64;
        // Clean DEQ core only needs the current hidden state. The old doubled allocation kept
        // extra carry/state from the historical path, which this solve no longer uses.
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
        // Clean DEQ core scratch layout per (batch, token):
        //   signal [h*d]
        // The old full-DEQ layout reserved q/k/v/attn/mamba/history regions that the clean
        // solve no longer touches.
        let scratch_stride = Self::clean_scratch_stride(
            d_model,
            h_slots,
        );
        let scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scratchpad"),
            size: (max_batch_size as u64) * (max_seq_len as u64) * (scratch_stride as u64) * 4u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hpooled_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_pooled"),
            size: Self::pooled_elements(max_batch_size, max_seq_len, d_model) * 4u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let debug_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DEQ Debug Log"),
            size: 4096, // 1024 floats for slot observability and FPM diagnostics
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hist_ctx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hist V2 Context"),
            size: hnext_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mstate_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hist V2 MState"),
            size: h_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let signal_cache_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Hist V2 Signal Cache"),
            size: (max_batch_size as u64) * (max_seq_len as u64) * (d_model as u64) * 4u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // H_hist: persistent hidden-state history, same shape as H_curr (no seq_len dimension)
        let h_hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_hist"),
            size: h_bytes,
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
                    resource: all_weights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hcurr_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hnext_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: hpooled_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: debug_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: hist_ctx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: mstate_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: signal_cache_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: h_hist_buf.as_entire_binding(),
                },
            ],
        });
        let mat_sz = std::num::NonZeroU64::new(d64 * d64 * 4);
        let wqk_sz = std::num::NonZeroU64::new(aw_wqk_bytes(d64, h64));
        let wv_sz = std::num::NonZeroU64::new(h64 * d64 * d64 * 4);
        let wo_sz = std::num::NonZeroU64::new(h64 * d64 * d64 * 4);
        let win_sz = std::num::NonZeroU64::new(h64 * d64 * d64 * 4);
        let alog_sz = std::num::NonZeroU64::new(h64 * d64 * 4);
        let nscale_sz = std::num::NonZeroU64::new(d64 * 4);
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: 0,
                        size: wqk_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_wk_byte_off(d64, h64),
                        size: wqk_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf, offset: aw_wv_byte_off(d64, h64), size: wv_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf, offset: aw_wo_byte_off(d64, h64), size: wo_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_win_byte_off(d64, h64),
                        size: win_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_wx_byte_off(d64, h64),
                        size: mat_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_wout_byte_off(d64, h64),
                        size: mat_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_alog_byte_off(d64, h64),
                        size: alog_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_nscale_byte_off(d64, h64),
                        size: nscale_sz,
                    }),
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: 0,
                        size: wqk_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_wk_byte_off(d64, h64),
                        size: wqk_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf, offset: aw_wv_byte_off(d64, h64), size: wv_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf, offset: aw_wo_byte_off(d64, h64), size: wo_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_win_byte_off(d64, h64),
                        size: win_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_wx_byte_off(d64, h64),
                        size: mat_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &all_weights_buf,
                        offset: aw_wout_byte_off(d64, h64),
                        size: mat_sz,
                    }),
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
            slot_attn_unified_pipeline,
            hist_v2_signal_cache_pipeline,
            hist_v2_project_pipeline,
            hist_v2_temporal_pipeline,
            pool_pipeline,
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
            all_weights_buf,
            hcurr_buf,
            hnext_buf,
            scratch_buf,
            hpooled_buf,
            debug_buf,
            hist_ctx_buf,
            mstate_buf,
            signal_cache_buf,
            h_hist_buf,
            hist_ctx_enabled,
            fpm_enabled,
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

    /// Runs spectral renormalization on all DEQ weight matrices (per-slot where applicable) on GPU.
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
        // Per-slot matrices: W_q/W_k/W_v/W_o/W_in (h each) + shared W_x/W_out.
        let mat_count = 5 * h_slots + 2;
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
        let d = d_model as u64;
        let h = self.h_slots as u64;
        let mat_size = d * d * 4;
        let wqk_size = aw_wqk_bytes(d, h);
        let win_size = h * d * d * 4;
        let vec_size = d * 4;
        let a_vec_size = h * d * 4;

        let create_staging = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("DEQ Readback Staging"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };

        let st_q = create_staging(wqk_size);
        let st_k = create_staging(wqk_size);
        let wv_size = h * mat_size;
        let st_v = create_staging(wv_size);
        let wo_size = h * mat_size;
        let st_o = create_staging(wo_size);
        let st_in = create_staging(win_size);
        let st_x = create_staging(mat_size);
        let st_out = create_staging(mat_size);
        let st_a = create_staging(a_vec_size);
        let st_n = create_staging(vec_size);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.all_weights_buf, 0, &st_q, 0, wqk_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_wk_byte_off(d, h), &st_k, 0, wqk_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_wv_byte_off(d, h), &st_v, 0, wv_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_wk_byte_off(d, h), &st_k, 0, wqk_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_wv_byte_off(d, h), &st_v, 0, wv_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_wo_byte_off(d, h), &st_o, 0, wo_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_win_byte_off(d, h), &st_in, 0, win_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_wx_byte_off(d, h), &st_x, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_wout_byte_off(d, h), &st_out, 0, mat_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_alog_byte_off(d, h), &st_a, 0, a_vec_size);
        encoder.copy_buffer_to_buffer(&self.all_weights_buf, aw_nscale_byte_off(d, h), &st_n, 0, vec_size);

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
        encoder.copy_buffer_to_buffer(&self.hnext_buf, 0, &h_staging, 0, (h_size * 4) as u64);
        queue.submit(Some(encoder.finish()));

        let h_slice = h_staging.slice(..);

        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        h_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });

        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let out_h_data = bytemuck::cast_slice(&h_slice.get_mapped_range()).to_vec();
            h_staging.unmap();
            Ok((out_h_data, vec![0u32; b]))
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
        let b = shape.batch_size as usize;
        let pooled_floats = Self::pooled_elements(
            shape.batch_size,
            shape.seq_len,
            shape.d_model,
        ) as usize;
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
            size: (pooled_floats * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &self.hpooled_buf,
            0,
            &pooled_staging,
            0,
            (pooled_floats * 4) as u64,
        );
        queue.submit(Some(encoder.finish()));

        let pooled_slice = pooled_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        pooled_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let out_pooled = bytemuck::cast_slice(&pooled_slice.get_mapped_range()).to_vec();
            pooled_staging.unmap();
            Ok((out_pooled, vec![0u32; b]))
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
        let shape_bytes = Self::shape_bytes(shape);
        queue.write_buffer(&self.uniform_buf, 0, &shape_bytes);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DEQ Forward GPU-Only Encoder"),
        });

        let fpm_slot_stride = if self.fpm_enabled {
            let slot_stride = (shape.h_slots as u64) * (shape.d_model as u64) * 4u64;
            // FPM pre-dispatch: seed MState from h_hist_buf on the first chunk.
            // For subsequent chunks MState is already set from the previous chunk's post-copy.
            if shape.token_start == 0 {
                for b in 0..shape.batch_size as u64 {
                    let src_off = b * slot_stride;
                    encoder.copy_buffer_to_buffer(&self.h_hist_buf, src_off, &self.mstate_buf, b * slot_stride, slot_stride);
                }
            }
            Some(slot_stride)
        } else {
            None
        };

        {
            if self.hist_ctx_enabled {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hist V2 Signal Cache Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.hist_v2_signal_cache_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(
                    shape.batch_size.max(1),
                    shape.token_count.max(1),
                    1,
                );
            }
        }
        {
            if self.hist_ctx_enabled {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hist V2 Project Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.hist_v2_project_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(
                    shape.batch_size.max(1),
                    shape.h_slots.max(1),
                    shape.token_count.max(1),
                );
            }
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEQ Forward Unified SlotAttn Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.slot_attn_unified_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(shape.batch_size.max(1), shape.h_slots.max(1), 1);
        }
        {
            if self.hist_ctx_enabled {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Hist V2 Temporal Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.hist_v2_temporal_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(shape.batch_size.max(1), shape.h_slots.max(1), 1);
            }
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEQ Forward Pool Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pool_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(
                shape.d_model.div_ceil(256).max(1),
                shape.batch_size.max(1),
                shape.token_count.max(1),
            );
        }
        // FPM post-dispatch: copy each batch element's LAST-token m_new → mstate_buf.
        // Executes after the compute pass (commands in one encoder are ordered).
        if let Some(slot_stride) = fpm_slot_stride {
            let last_token = (shape.token_start + shape.token_count - 1) as u64;
            for b in 0..shape.batch_size as u64 {
                let src_off = (b * shape.seq_len as u64 + last_token) * slot_stride;
                let dst_off = b * slot_stride;
                encoder.copy_buffer_to_buffer(&self.hist_ctx_buf, src_off, &self.mstate_buf, dst_off, slot_stride);
            }
        }
        encoder
    }

    pub fn run_forward_gpu_only(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &DeqComputeShape,
    ) {
        // Single submit for the entire chunk — fpm_m_cache carry between tokens
        // is handled inside the shader (intra-slot, workgroup memory).
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
        let pooled_floats = Self::pooled_elements(
            shape.batch_size,
            shape.seq_len,
            shape.d_model,
        ) as usize;
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
            size: (pooled_floats * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let h_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("H_star Staging"),
            size: (b * h * d_model * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(
            &self.hpooled_buf,
            0,
            &pooled_staging,
            0,
            (pooled_floats * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.hnext_buf,
            0,
            &h_staging,
            0,
            (b * h * d_model * 4) as u64,
        );
        queue.submit(Some(encoder.finish()));

        let pooled_slice = pooled_staging.slice(..);
        let h_slice = h_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        pooled_slice.map_async(wgpu::MapMode::Read, |_| {});
        h_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let out_pooled = bytemuck::cast_slice(&pooled_slice.get_mapped_range()).to_vec();
            let out_h = bytemuck::cast_slice(&h_slice.get_mapped_range()).to_vec();
            pooled_staging.unmap();
            h_staging.unmap();
            Ok((out_pooled, out_h, vec![0u32; b]))
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
        let shape_bytes = Self::shape_bytes(shape);
        queue.write_buffer(&self.uniform_buf, 0, &shape_bytes);
        queue.write_buffer(&self.s_buf, 0, bytemuck::cast_slice(s_in));

        if update_weights {
            let d = shape.d_model as u64;
            let h = shape.h_slots as u64;
            queue.write_buffer(&self.all_weights_buf, 0, bytemuck::cast_slice(w_q));
            queue.write_buffer(
                &self.all_weights_buf,
                aw_wk_byte_off(d, h),
                bytemuck::cast_slice(w_k),
            );
            queue.write_buffer(
                &self.all_weights_buf,
                aw_wv_byte_off(d, h),
                bytemuck::cast_slice(w_v),
            );
            queue.write_buffer(
                &self.all_weights_buf,
                aw_wo_byte_off(d, h),
                bytemuck::cast_slice(w_o),
            );
            queue.write_buffer(
                &self.all_weights_buf,
                aw_win_byte_off(d, h),
                bytemuck::cast_slice(w_in),
            );
            queue.write_buffer(
                &self.all_weights_buf,
                aw_wx_byte_off(d, h),
                bytemuck::cast_slice(w_x),
            );
            queue.write_buffer(
                &self.all_weights_buf,
                aw_wout_byte_off(d, h),
                bytemuck::cast_slice(w_out),
            );
            queue.write_buffer(
                &self.all_weights_buf,
                aw_alog_byte_off(d, h),
                bytemuck::cast_slice(a_log),
            );
            queue.write_buffer(
                &self.all_weights_buf,
                aw_nscale_byte_off(d, h),
                bytemuck::cast_slice(norm_scale),
            );
        }

        self.encode_forward_gpu_only(device, queue, shape)
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
