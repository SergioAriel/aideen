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
    active_targets: u32,
    tie_lambda_bits: u32,
    assoc_logit_lambda_bits: u32,
    full_vocab_estimate: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExactForwardParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    vocab_start: u32,
    vocab_chunk: u32,
    _pad0: [u32; 7],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExactWUpdateParams {
    d_model: u32,
    vocab_size: u32,
    step_t: u32,
    _pad0: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    grad_scale_bits: u32,
    // WGSL uniform layout rounds this struct up to 64 bytes.
    _pad1: [u32; 7],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExactWApplyParams {
    num_weights: u32,
    step_t: u32,
    groups_x: u32,
    _pad0: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    grad_scale_bits: u32,
    _pad1: [u32; 7],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExactBgApplyParams {
    vocab_size: u32,
    d_model: u32,
    step_t: u32,
    groups_x: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    grad_scale_bits: u32,
    _pad1: [u32; 7],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExactAccumParams {
    d_model: u32,
    vocab_size: u32,
    token_index: u32,
    _pad0: u32,
    rms_bits: u32,
    _pad1: [u32; 15],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ExactTokenParams {
    d_model: u32,
    vocab_size: u32,
    token_index: u32,
    target_index: u32,
    seq_scale_bits: u32,
    _pad1: [u32; 19],
}

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

pub struct GpuLmHeadTrainer {
    pub config: ArchitectureConfig,
    pub vocab_size: usize,
    sample_capacity: usize,
    sample_seed: u64,
    h_buf: wgpu::Buffer,
    w_buf: wgpu::Buffer,
    b_buf: wgpu::Buffer,
    emb_ref_buf: wgpu::Buffer,
    pub dl_dh_buf: wgpu::Buffer,
    pub dl_dh_temp_buf: wgpu::Buffer,

    loss_buf: wgpu::Buffer,
    rms_buf: wgpu::Buffer,
    g_buf: wgpu::Buffer,
    moments_w_buf: wgpu::Buffer, // Combined m_w, v_w (Vocab * D_model * vec2)
    moments_b_buf: wgpu::Buffer, // Combined m_b, v_b (Vocab * vec2)
    moments_g_buf: wgpu::Buffer, // Combined m_g, v_g (D_model * vec2)
    params_buf: wgpu::Buffer,
    target_indices_buf: wgpu::Buffer,
    sampled_indices_buf: wgpu::Buffer,
    s_h_rms_buf: wgpu::Buffer,
    dl_staging_buf: wgpu::Buffer,
    loss_staging_buf: wgpu::Buffer,

    probs_pipeline: wgpu::ComputePipeline,
    probs_bgl: wgpu::BindGroupLayout,
    probs_bg: wgpu::BindGroup,

    update_pipeline: wgpu::ComputePipeline,
    backprop_pipeline: wgpu::ComputePipeline,
    lm_scratch_buf: wgpu::Buffer,
    fused_b19: bool,
    cfg_lm_debug: bool,
    cfg_lm_emb_tie_lambda: f32,
    cfg_assoc_logit_lambda: f32,
    cfg_full_vocab_estimate: bool,
    cfg_coverage_sampler: bool,
    cfg_exact_forward_gpu: bool,
    exact_chunk_cap: u32,
    exact_seq_cap: u32,
    exact_forward_params_buf: wgpu::Buffer,
    exact_forward_logits_buf: wgpu::Buffer,
    exact_forward_bgl: wgpu::BindGroupLayout,
    exact_forward_pipeline: wgpu::ComputePipeline,
    exact_w_update_params_buf: wgpu::Buffer,
    exact_w_update_dl_buf: wgpu::Buffer,
    exact_w_update_h_rms_buf: wgpu::Buffer,
    exact_w_update_out_buf: wgpu::Buffer,
    exact_w_update_bgl: wgpu::BindGroupLayout,
    exact_w_update_pipeline: wgpu::ComputePipeline,
    exact_w_apply_params_buf: wgpu::Buffer,
    exact_w_grad_buf: wgpu::Buffer,
    exact_w_apply_bgl: wgpu::BindGroupLayout,
    exact_w_apply_pipeline: wgpu::ComputePipeline,
    exact_bg_apply_params_buf: wgpu::Buffer,
    exact_db_grad_buf: wgpu::Buffer,
    exact_dg_grad_buf: wgpu::Buffer,
    exact_bg_apply_bgl: wgpu::BindGroupLayout,
    exact_bg_apply_pipeline: wgpu::ComputePipeline,
    exact_accum_params_buf: wgpu::Buffer,
    exact_accum_token_bgl: wgpu::BindGroupLayout,
    exact_accum_bias_bgl: wgpu::BindGroupLayout,
    exact_accum_dldh_bgl: wgpu::BindGroupLayout,
    exact_accum_token_pipeline: wgpu::ComputePipeline,
    exact_accum_bias_pipeline: wgpu::ComputePipeline,
    exact_accum_dldh_pipeline: wgpu::ComputePipeline,
    exact_token_params_buf: wgpu::Buffer,
    exact_token_logits_buf: wgpu::Buffer,
    exact_token_loss_buf: wgpu::Buffer,
    exact_token_loss_staging_buf: wgpu::Buffer,
    exact_token_logits_bgl: wgpu::BindGroupLayout,
    exact_token_lossgrad_bgl: wgpu::BindGroupLayout,
    exact_token_logits_pipeline: wgpu::ComputePipeline,
    exact_token_lossgrad_pipeline: wgpu::ComputePipeline,
    last_sampled_indices: Vec<u32>,
    last_num_samples: u32,
    sampled_indices_reuse: Vec<u32>, // pre-alloc to avoid heap alloc per training step
}

impl GpuLmHeadTrainer {
    pub const IGNORE_TARGET: u32 = u32::MAX;

    fn active_target_count(&self, targets: &[u32]) -> u32 {
        targets
            .iter()
            .filter(|&&t| t < self.vocab_size as u32)
            .count()
            .max(1) as u32
    }

    fn parse_env_seed(name: &str) -> Option<u64> {
        let raw = std::env::var(name).ok()?;
        let s = raw.trim();
        s.strip_prefix("0x")
            .or_else(|| s.strip_prefix("0X"))
            .map_or_else(
                || s.parse::<u64>().ok(),
                |hex| u64::from_str_radix(hex, 16).ok(),
            )
    }

    fn deterministic_sample_candidate(&self, step_t: u32, draw: u64) -> u32 {
        let mut x = self.sample_seed
            ^ ((step_t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ draw.wrapping_mul(0xD1B5_4A32_D192_ED03);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        (x % self.vocab_size as u64) as u32
    }

    fn coverage_sample_candidate(&self, step_t: u32, draw: u64) -> u32 {
        let vocab = self.vocab_size as u64;
        if vocab <= 1 {
            return 0;
        }
        let mut stride = (self.sample_seed | 1) % vocab;
        if stride == 0 {
            stride = 1;
        }
        while gcd_u64(stride, vocab) != 1 {
            stride = (stride + 2) % vocab;
            if stride == 0 {
                stride = 1;
            }
        }
        let start = (u64::from(step_t).wrapping_mul(self.sample_capacity as u64)) % vocab;
        ((start + draw.wrapping_mul(stride)) % vocab) as u32
    }

    fn build_sampled_indices(&mut self, targets: &[u32], step_t: u32) -> u32 {
        // MAX_SHADER_SAMPLES must match s_indices_cache and s_logits array sizes in lm_train.wgsl.
        // Exceeding this causes OOB writes to workgroup memory, corrupting probs and gradients.
        const MAX_SHADER_SAMPLES: usize = 2048;
        let desired = self
            .config
            .num_samples
            .max(targets.len())
            .min(self.vocab_size)
            .min(self.sample_capacity)
            .min(MAX_SHADER_SAMPLES);
        self.sampled_indices_reuse.clear();
        self.sampled_indices_reuse.extend(
            targets
                .iter()
                .copied()
                .filter(|&t| t < self.vocab_size as u32),
        );
        self.sampled_indices_reuse.sort_unstable();
        self.sampled_indices_reuse.dedup();

        let mut seen: std::collections::HashSet<u32> =
            self.sampled_indices_reuse.iter().copied().collect();
        let mut draw = 0_u64;
        while self.sampled_indices_reuse.len() < desired {
            let candidate = if self.cfg_coverage_sampler {
                self.coverage_sample_candidate(step_t, draw)
            } else {
                self.deterministic_sample_candidate(step_t, draw)
            };
            draw = draw.wrapping_add(1);
            if seen.insert(candidate) {
                self.sampled_indices_reuse.push(candidate);
            }
        }
        self.sampled_indices_reuse.sort_unstable();
        self.sampled_indices_reuse.len() as u32
    }

    fn make_probs_bg_for_h(&self, device: &wgpu::Device, h_buf: &wgpu::Buffer) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Unified BG (external h)"),
            layout: &self.probs_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.dl_dh_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.loss_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.moments_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.moments_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.target_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.lm_scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.moments_g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.rms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.dl_dh_temp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: self.sampled_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: self.s_h_rms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: self.emb_ref_buf.as_entire_binding(),
                },
            ],
        })
    }

    fn make_exact_forward_bg_for_h(
        &self,
        device: &wgpu::Device,
        h_buf: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact Forward BG"),
            layout: &self.exact_forward_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_forward_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.exact_forward_logits_buf.as_entire_binding(),
                },
            ],
        })
    }

    pub fn new(device: &wgpu::Device, vocab_size: usize, config: ArchitectureConfig) -> Self {
        let d_r = config.d_r;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Train Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_train.wgsl").into()),
        });
        let exact_forward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_exact_forward.wgsl").into()),
        });
        let exact_w_update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact W Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_exact_w_update.wgsl").into()),
        });
        let exact_w_apply_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact W Apply Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_exact_w_apply.wgsl").into()),
        });
        let exact_bg_apply_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact BG Apply Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lm_exact_bg_apply.wgsl").into()),
        });
        let exact_accum_token_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact Accum Token Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/lm_exact_accum_token.wgsl").into(),
            ),
        });
        let exact_accum_bias_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact Accum Bias Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/lm_exact_accum_bias.wgsl").into(),
            ),
        });
        let exact_accum_dldh_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact Accum dL/dh Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/lm_exact_accum_dldh.wgsl").into(),
            ),
        });
        let exact_token_logits_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LM Exact Token Logits Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/lm_exact_token_logits.wgsl").into(),
            ),
        });
        let exact_token_lossgrad_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("LM Exact Token LossGrad Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shaders/lm_exact_token_lossgrad.wgsl").into(),
                ),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

        let exact_forward_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LM Exact Forward BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let exact_forward_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact Forward Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact Forward PL"),
                        bind_group_layouts: &[&exact_forward_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_forward_shader,
                entry_point: Some("lm_exact_forward_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_w_update_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Exact W Update BGL"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let exact_w_update_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact W Update Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact W Update PL"),
                        bind_group_layouts: &[&exact_w_update_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_w_update_shader,
                entry_point: Some("lm_exact_w_update_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_w_apply_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LM Exact W Apply BGL"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let exact_w_apply_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact W Apply Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact W Apply PL"),
                        bind_group_layouts: &[&exact_w_apply_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_w_apply_shader,
                entry_point: Some("lm_exact_w_apply_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_bg_apply_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Exact BG Apply BGL"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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
        let exact_bg_apply_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact BG Apply Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact BG Apply PL"),
                        bind_group_layouts: &[&exact_bg_apply_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_bg_apply_shader,
                entry_point: Some("lm_exact_bg_apply_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_accum_token_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Exact Accum Token BGL"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });
        let exact_accum_bias_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Exact Accum Bias BGL"),
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
                ],
            });
        let exact_accum_dldh_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Exact Accum dL/dh BGL"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });
        let exact_accum_token_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact Accum Token Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact Accum Token PL"),
                        bind_group_layouts: &[&exact_accum_token_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_accum_token_shader,
                entry_point: Some("lm_exact_accum_token_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_accum_bias_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact Accum Bias Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact Accum Bias PL"),
                        bind_group_layouts: &[&exact_accum_bias_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_accum_bias_shader,
                entry_point: Some("lm_exact_accum_bias_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_accum_dldh_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact Accum dL/dh Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact Accum dL/dh PL"),
                        bind_group_layouts: &[&exact_accum_dldh_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_accum_dldh_shader,
                entry_point: Some("lm_exact_accum_dldh_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_token_logits_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Exact Token Logits BGL"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        let exact_token_lossgrad_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("LM Exact Token LossGrad BGL"),
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
                ],
            });
        let exact_token_logits_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact Token Logits Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact Token Logits PL"),
                        bind_group_layouts: &[&exact_token_logits_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_token_logits_shader,
                entry_point: Some("lm_exact_token_logits_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let exact_token_lossgrad_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LM Exact Token LossGrad Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("LM Exact Token LossGrad PL"),
                        bind_group_layouts: &[&exact_token_lossgrad_bgl],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &exact_token_lossgrad_shader,
                entry_point: Some("lm_exact_token_lossgrad_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // --- PIPELINE 5: Ternary Project ---
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
        let exact_chunk_cap = std::env::var("AIDEEN_LM_EXACT_GPU_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(1024)
            .max(1)
            .min(vocab_size as u32);
        let sample_capacity = config.num_samples.max(safe_ctx);
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
        let emb_ref_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Embedding Reference"),
            size: (vocab_size * d_r * 4) as u64,
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
        let rms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM RMS"),
            size: (safe_ctx * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
        let scratch_floats = (safe_ctx * sample_capacity)
            + safe_ctx
            + (safe_ctx * config.d_r)
            + safe_ctx
            + (safe_ctx * config.d_r);
        let lm_scratch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Scratch"),
            size: (scratch_floats * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dl_dh_temp_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM dL/dh Temp"),
            size: (safe_ctx * d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampled_indices_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Sampled Indices"),
            size: (sample_capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sample_seed = Self::parse_env_seed("AIDEEN_LM_SAMPLE_SEED")
            .or_else(|| Self::parse_env_seed("AIDEEN_TRAIN_SEED"))
            .unwrap_or(0xA1DE_EE5A_4D5E_ED5E);
        let s_h_rms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM s_h_rms"),
            size: (safe_ctx * d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
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
        let exact_forward_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact Forward Params"),
            size: std::mem::size_of::<ExactForwardParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_forward_logits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact Forward Logits"),
            size: (safe_ctx as u64) * (exact_chunk_cap as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_w_update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact W Update Params"),
            size: std::mem::size_of::<ExactWUpdateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_w_update_dl_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact W Update dl"),
            size: (vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_w_update_h_rms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact W Update h_rms"),
            size: (d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_w_update_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact W Update out"),
            size: (d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_w_apply_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact W Apply Params"),
            size: std::mem::size_of::<ExactWApplyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_w_grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact W Grad"),
            size: (vocab_size * d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_bg_apply_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact BG Apply Params"),
            size: std::mem::size_of::<ExactBgApplyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_db_grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact dB Grad"),
            size: (vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_dg_grad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact dG Grad"),
            size: (d_r * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_accum_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact Accum Params"),
            size: std::mem::size_of::<ExactAccumParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_token_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact Token Params"),
            size: std::mem::size_of::<ExactTokenParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_token_logits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact Token Logits"),
            size: (vocab_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let exact_token_loss_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact Token Loss"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let exact_token_loss_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LM Exact Token Loss Staging"),
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
                    resource: lm_scratch_buf.as_entire_binding(),
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
                    resource: emb_ref_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            config,
            vocab_size,
            sample_capacity,
            sample_seed,
            h_buf,
            w_buf,
            b_buf,
            emb_ref_buf,
            dl_dh_buf,
            loss_buf,
            rms_buf,
            g_buf,
            moments_w_buf,
            moments_b_buf,
            moments_g_buf,
            params_buf,
            target_indices_buf,
            dl_dh_temp_buf,
            sampled_indices_buf,
            s_h_rms_buf,
            dl_staging_buf,
            loss_staging_buf,
            probs_pipeline,
            probs_bgl: bgl_probs,
            probs_bg,
            update_pipeline,
            backprop_pipeline,
            lm_scratch_buf,
            fused_b19: std::env::var("AIDEEN_LM_FUSED_B19")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(true),
            cfg_lm_debug: std::env::var("AIDEEN_LM_DEBUG")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(false),
            cfg_lm_emb_tie_lambda: std::env::var("AIDEEN_LM_EMB_TIE_LAMBDA")
                .ok()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0)
                .max(0.0),
            cfg_assoc_logit_lambda: std::env::var("AIDEEN_ASSOC_LOGIT_LAMBDA")
                .ok()
                .and_then(|s| s.trim().parse::<f32>().ok())
                .unwrap_or(0.0)
                .max(0.0),
            cfg_full_vocab_estimate: std::env::var("AIDEEN_LM_FULL_VOCAB_ESTIMATE")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(false),
            cfg_coverage_sampler: std::env::var("AIDEEN_LM_COVERAGE_SAMPLER")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(false),
            cfg_exact_forward_gpu: std::env::var("AIDEEN_LM_EXACT_GPU_FORWARD")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(false),
            exact_chunk_cap,
            exact_seq_cap: safe_ctx as u32,
            exact_forward_params_buf,
            exact_forward_logits_buf,
            exact_forward_bgl,
            exact_forward_pipeline,
            exact_w_update_params_buf,
            exact_w_update_dl_buf,
            exact_w_update_h_rms_buf,
            exact_w_update_out_buf,
            exact_w_update_bgl,
            exact_w_update_pipeline,
            exact_w_apply_params_buf,
            exact_w_grad_buf,
            exact_w_apply_bgl,
            exact_w_apply_pipeline,
            exact_bg_apply_params_buf,
            exact_db_grad_buf,
            exact_dg_grad_buf,
            exact_bg_apply_bgl,
            exact_bg_apply_pipeline,
            exact_accum_params_buf,
            exact_accum_token_bgl,
            exact_accum_bias_bgl,
            exact_accum_dldh_bgl,
            exact_accum_token_pipeline,
            exact_accum_bias_pipeline,
            exact_accum_dldh_pipeline,
            exact_token_params_buf,
            exact_token_logits_buf,
            exact_token_loss_buf,
            exact_token_loss_staging_buf,
            exact_token_logits_bgl,
            exact_token_lossgrad_bgl,
            exact_token_logits_pipeline,
            exact_token_lossgrad_pipeline,
            last_sampled_indices: Vec::new(),
            last_num_samples: 0,
            sampled_indices_reuse: Vec::new(),
        }
    }

    pub fn exact_forward_gpu_enabled(&self) -> bool {
        self.cfg_exact_forward_gpu
    }

    pub fn read_exact_logits_from_h(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_buf: &wgpu::Buffer,
        seq_len: u32,
    ) -> Result<Vec<f32>, String> {
        if !self.cfg_exact_forward_gpu {
            return Err("exact GPU forward disabled".to_string());
        }
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        if seq_len > self.exact_seq_cap {
            return Err(format!(
                "seq_len {} exceeds exact forward cap {}",
                seq_len, self.exact_seq_cap
            ));
        }

        let mut out = vec![0.0f32; seq_len as usize * self.vocab_size];
        let chunk_cap = self.exact_chunk_cap.max(1) as usize;
        for vocab_start in (0..self.vocab_size).step_by(chunk_cap) {
            let chunk = (self.vocab_size - vocab_start).min(chunk_cap);
            let params = ExactForwardParams {
                d_model: self.config.d_r as u32,
                vocab_size: self.vocab_size as u32,
                seq_len,
                vocab_start: vocab_start as u32,
                vocab_chunk: chunk as u32,
                _pad0: [0; 7],
            };
            queue.write_buffer(
                &self.exact_forward_params_buf,
                0,
                bytemuck::bytes_of(&params),
            );
            let bg = self.make_exact_forward_bg_for_h(device, h_buf);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("LM Exact Forward Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LM Exact Forward Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.exact_forward_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(((chunk as u32) + 7) / 8, (seq_len + 7) / 8, 1);
            }
            queue.submit(Some(encoder.finish()));
            let chunk_logits = self.read_buffer_prefix(
                device,
                queue,
                &self.exact_forward_logits_buf,
                seq_len as usize * chunk,
                "LM exact forward logits",
            )?;
            for t in 0..seq_len as usize {
                let src_base = t * chunk;
                let dst_base = t * self.vocab_size + vocab_start;
                out[dst_base..dst_base + chunk]
                    .copy_from_slice(&chunk_logits[src_base..src_base + chunk]);
            }
        }
        Ok(out)
    }

    pub fn read_exact_logits_from_slice(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_pooled: &[f32],
        seq_len: u32,
    ) -> Result<Vec<f32>, String> {
        let expected = seq_len as usize * self.config.d_r;
        if h_pooled.len() != expected {
            return Err(format!(
                "h_pooled len {} does not match expected {} for seq_len {}",
                h_pooled.len(),
                expected,
                seq_len
            ));
        }
        queue.write_buffer(&self.h_buf, 0, bytemuck::cast_slice(h_pooled));
        self.read_exact_logits_from_h(device, queue, &self.h_buf, seq_len)
    }

    pub fn exact_update_w_from_token(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dl_scaled: &[f32],
        h_rms: &[f32],
        step_t: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        grad_scale: f32,
    ) -> Result<Vec<f32>, String> {
        if dl_scaled.len() != self.vocab_size || h_rms.len() != self.config.d_r {
            return Err("exact_update_w_from_token shape mismatch".to_string());
        }
        let params = ExactWUpdateParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            step_t,
            _pad0: 0,
            lr_bits: lr.to_bits(),
            beta1_bits: beta1.to_bits(),
            beta2_bits: beta2.to_bits(),
            eps_bits: eps.to_bits(),
            grad_scale_bits: grad_scale.to_bits(),
            _pad1: [0; 7],
        };
        queue.write_buffer(
            &self.exact_w_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        queue.write_buffer(
            &self.exact_w_update_dl_buf,
            0,
            bytemuck::cast_slice(dl_scaled),
        );
        queue.write_buffer(
            &self.exact_w_update_h_rms_buf,
            0,
            bytemuck::cast_slice(h_rms),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact W Update BG"),
            layout: &self.exact_w_update_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_w_update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.moments_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.exact_w_update_dl_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.exact_w_update_h_rms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.exact_w_update_out_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact W Update Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact W Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_w_update_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((self.config.d_r as u32).div_ceil(64), 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        self.read_buffer_prefix(
            device,
            queue,
            &self.exact_w_update_out_buf,
            self.config.d_r,
            "LM exact w update out",
        )
    }

    pub fn upload_bias_gain_only(&self, queue: &wgpu::Queue, b_cpu: &[f32], g_cpu: &[f32]) {
        queue.write_buffer(&self.b_buf, 0, bytemuck::cast_slice(b_cpu));
        queue.write_buffer(&self.g_buf, 0, bytemuck::cast_slice(g_cpu));
    }

    pub fn exact_apply_w_from_grad(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grad_w: &[f32],
        step_t: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        grad_scale: f32,
    ) -> Result<(), String> {
        let num_weights = self.vocab_size * self.config.d_r;
        if grad_w.len() != num_weights {
            return Err("exact_apply_w_from_grad shape mismatch".to_string());
        }
        let groups_x = ((num_weights as u32).div_ceil(256)).min(65_535);
        let groups_y = (num_weights as u32).div_ceil(groups_x * 256);
        let params = ExactWApplyParams {
            num_weights: num_weights as u32,
            step_t,
            groups_x,
            _pad0: 0,
            lr_bits: lr.to_bits(),
            beta1_bits: beta1.to_bits(),
            beta2_bits: beta2.to_bits(),
            eps_bits: eps.to_bits(),
            grad_scale_bits: grad_scale.to_bits(),
            _pad1: [0; 7],
        };
        queue.write_buffer(
            &self.exact_w_apply_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        queue.write_buffer(&self.exact_w_grad_buf, 0, bytemuck::cast_slice(grad_w));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact W Apply BG"),
            layout: &self.exact_w_apply_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_w_apply_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.moments_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.exact_w_grad_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact W Apply Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact W Apply Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_w_apply_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn exact_apply_bg_from_grad(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grad_b: &[f32],
        grad_g: &[f32],
        step_t: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        grad_scale: f32,
    ) -> Result<(), String> {
        if grad_b.len() != self.vocab_size || grad_g.len() != self.config.d_r {
            return Err("exact_apply_bg_from_grad shape mismatch".to_string());
        }
        let max_len = self.vocab_size.max(self.config.d_r);
        let groups_x = ((max_len as u32).div_ceil(256)).min(65_535);
        let groups_y = (max_len as u32).div_ceil(groups_x * 256);
        let params = ExactBgApplyParams {
            vocab_size: self.vocab_size as u32,
            d_model: self.config.d_r as u32,
            step_t,
            groups_x,
            lr_bits: lr.to_bits(),
            beta1_bits: beta1.to_bits(),
            beta2_bits: beta2.to_bits(),
            eps_bits: eps.to_bits(),
            grad_scale_bits: grad_scale.to_bits(),
            _pad1: [0; 7],
        };
        queue.write_buffer(
            &self.exact_bg_apply_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        queue.write_buffer(&self.exact_db_grad_buf, 0, bytemuck::cast_slice(grad_b));
        queue.write_buffer(&self.exact_dg_grad_buf, 0, bytemuck::cast_slice(grad_g));
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact BG Apply BG"),
            layout: &self.exact_bg_apply_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_bg_apply_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.moments_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.moments_g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.exact_db_grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.exact_dg_grad_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact BG Apply Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact BG Apply Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_bg_apply_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn exact_zero_accumulators(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        seq_len: u32,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact Zero Accumulators"),
        });
        encoder.clear_buffer(&self.exact_w_grad_buf, 0, None);
        encoder.clear_buffer(&self.exact_db_grad_buf, 0, None);
        encoder.clear_buffer(&self.exact_dg_grad_buf, 0, None);
        encoder.clear_buffer(
            &self.dl_dh_buf,
            0,
            Some((seq_len as u64) * (self.config.d_r as u64) * 4),
        );
        queue.submit(Some(encoder.finish()));
    }

    pub fn exact_compute_token_loss_grad(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_buf: &wgpu::Buffer,
        token_index: u32,
        target_index: u32,
        seq_scale: f32,
    ) -> Result<f32, String> {
        let params = ExactTokenParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            token_index,
            target_index,
            seq_scale_bits: seq_scale.to_bits(),
            _pad1: [0; 19],
        };
        queue.write_buffer(&self.exact_token_params_buf, 0, bytemuck::bytes_of(&params));
        let logits_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact Token Logits BG"),
            layout: &self.exact_token_logits_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_token_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.exact_token_logits_buf.as_entire_binding(),
                },
            ],
        });
        let lossgrad_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact Token LossGrad BG"),
            layout: &self.exact_token_lossgrad_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_token_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.exact_token_logits_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.exact_w_update_dl_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.exact_token_loss_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact Token LossGrad Encoder"),
        });
        encoder.clear_buffer(&self.exact_token_loss_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact Token Logits Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_token_logits_pipeline);
            pass.set_bind_group(0, &logits_bg, &[]);
            pass.dispatch_workgroups((self.vocab_size as u32).div_ceil(256), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact Token LossGrad Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_token_lossgrad_pipeline);
            pass.set_bind_group(0, &lossgrad_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.exact_token_loss_buf,
            0,
            &self.exact_token_loss_staging_buf,
            0,
            4,
        );
        queue.submit(Some(encoder.finish()));

        let slice = self.exact_token_loss_staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let loss = bytemuck::cast_slice::<u8, f32>(&data)[0];
            drop(data);
            self.exact_token_loss_staging_buf.unmap();
            Ok(loss)
        } else {
            Err("LM exact token loss map failed".to_string())
        }
    }

    pub fn exact_accumulate_token(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_buf: &wgpu::Buffer,
        token_index: u32,
        dl_scaled: &[f32],
    ) -> Result<(), String> {
        if !dl_scaled.is_empty() && dl_scaled.len() != self.vocab_size {
            return Err("exact_accumulate_token shape mismatch".to_string());
        }
        let params = ExactAccumParams {
            d_model: self.config.d_r as u32,
            vocab_size: self.vocab_size as u32,
            token_index,
            _pad0: 0,
            rms_bits: 0,
            _pad1: [0; 15],
        };
        queue.write_buffer(&self.exact_accum_params_buf, 0, bytemuck::bytes_of(&params));
        if !dl_scaled.is_empty() {
            queue.write_buffer(
                &self.exact_w_update_dl_buf,
                0,
                bytemuck::cast_slice(dl_scaled),
            );
        }

        let token_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact Accum Token BG"),
            layout: &self.exact_accum_token_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_accum_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.exact_w_update_dl_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.exact_w_grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.exact_dg_grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.exact_w_update_out_buf.as_entire_binding(),
                },
            ],
        });
        let bias_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact Accum Bias BG"),
            layout: &self.exact_accum_bias_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_accum_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.exact_w_update_dl_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.exact_db_grad_buf.as_entire_binding(),
                },
            ],
        });
        let dldh_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact Accum dL/dh BG"),
            layout: &self.exact_accum_dldh_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_accum_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.exact_w_update_out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.dl_dh_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact Accum Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact Accum Token Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_accum_token_pipeline);
            pass.set_bind_group(0, &token_bg, &[]);
            pass.dispatch_workgroups((self.config.d_r as u32).div_ceil(64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact Accum Bias Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_accum_bias_pipeline);
            pass.set_bind_group(0, &bias_bg, &[]);
            pass.dispatch_workgroups((self.vocab_size as u32).div_ceil(256), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact Accum dL/dh Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_accum_dldh_pipeline);
            pass.set_bind_group(0, &dldh_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn exact_apply_w_from_internal_grad(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        step_t: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        grad_scale: f32,
    ) -> Result<(), String> {
        let num_weights = self.vocab_size * self.config.d_r;
        let groups_x = ((num_weights as u32).div_ceil(256)).min(65_535);
        let groups_y = (num_weights as u32).div_ceil(groups_x * 256);
        let params = ExactWApplyParams {
            num_weights: num_weights as u32,
            step_t,
            groups_x,
            _pad0: 0,
            lr_bits: lr.to_bits(),
            beta1_bits: beta1.to_bits(),
            beta2_bits: beta2.to_bits(),
            eps_bits: eps.to_bits(),
            grad_scale_bits: grad_scale.to_bits(),
            _pad1: [0; 7],
        };
        queue.write_buffer(
            &self.exact_w_apply_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact W Apply Internal BG"),
            layout: &self.exact_w_apply_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_w_apply_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.moments_w_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.exact_w_grad_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact W Apply Internal Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact W Apply Internal Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_w_apply_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn exact_apply_bg_from_internal_grad(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        step_t: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        grad_scale: f32,
    ) -> Result<(), String> {
        let max_len = self.vocab_size.max(self.config.d_r);
        let groups_x = ((max_len as u32).div_ceil(256)).min(65_535);
        let groups_y = (max_len as u32).div_ceil(groups_x * 256);
        let params = ExactBgApplyParams {
            vocab_size: self.vocab_size as u32,
            d_model: self.config.d_r as u32,
            step_t,
            groups_x,
            lr_bits: lr.to_bits(),
            beta1_bits: beta1.to_bits(),
            beta2_bits: beta2.to_bits(),
            eps_bits: eps.to_bits(),
            grad_scale_bits: grad_scale.to_bits(),
            _pad1: [0; 7],
        };
        queue.write_buffer(
            &self.exact_bg_apply_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LM Exact BG Apply Internal BG"),
            layout: &self.exact_bg_apply_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.exact_bg_apply_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.moments_b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.moments_g_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.exact_db_grad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.exact_dg_grad_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LM Exact BG Apply Internal Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Exact BG Apply Internal Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.exact_bg_apply_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn write_moments_w_only(&self, queue: &wgpu::Queue, m_w: &[f32], v_w: &[f32]) {
        let interleave: Vec<f32> = m_w
            .iter()
            .zip(v_w.iter())
            .flat_map(|(&a, &b)| [a, b])
            .collect();
        queue.write_buffer(&self.moments_w_buf, 0, bytemuck::cast_slice(&interleave));
    }

    pub fn last_sampled_indices(&self) -> &[u32] {
        &self.last_sampled_indices
    }

    pub fn last_num_samples(&self) -> u32 {
        self.last_num_samples
    }

    pub fn assoc_grad_buf(&self) -> &wgpu::Buffer {
        &self.s_h_rms_buf
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

        // Sampled Softmax: targets + random negatives — reuse buffer to avoid heap alloc per step.
        self.sampled_indices_reuse.clear();
        self.sampled_indices_reuse.reserve(self.sample_capacity);
        let actual_num_samples = self.build_sampled_indices(targets, step_t);
        if self.cfg_lm_debug {
            let tgt0 = targets.get(0).copied().unwrap_or(0);
            eprintln!(
                "    [LM-DEBUG] seq_len={} num_samples={} target0={}",
                seq_len, actual_num_samples, tgt0
            );
        }
        std::mem::swap(
            &mut self.sampled_indices_reuse,
            &mut self.last_sampled_indices,
        );
        self.last_num_samples = actual_num_samples;
        let sampled_indices = &self.last_sampled_indices;

        if seq_len > self.config.ctx_len {
            return Err(format!(
                "seq_len {} exceeds architectural limit of {}",
                seq_len, self.config.ctx_len
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
            active_targets: self.active_target_count(targets),
            tie_lambda_bits: 0.0f32.to_bits(),
            assoc_logit_lambda_bits: 0.0f32.to_bits(),
            full_vocab_estimate: if self.cfg_full_vocab_estimate { 1 } else { 0 },
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
        encoder.clear_buffer(&self.loss_buf, 0, None);
        let seq_len_u32 = seq_len as u32;
        let t_grid_x = seq_len_u32.min(65535).max(1);
        let t_grid_y = seq_len_u32.div_ceil(65535).max(1);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Probs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probs_pipeline);
            pass.set_bind_group(0, &self.probs_bg, &[]);
            pass.dispatch_workgroups(t_grid_x, t_grid_y, 1);
        }
        let t_parts_x = t_grid_x;
        let t_parts_y = t_grid_y;
        if self.fused_b19 {
            // Fused path: use the same update kernel as non-fused to guarantee
            // numerical equivalence (avoids drift seen with lm_dw_accum_main).
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LM Update Pass (fused)"),
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
                pass.dispatch_workgroups(t_parts_x, t_parts_y, 1);
            }
        } else {
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
                pass.dispatch_workgroups(t_parts_x, t_parts_y, 1);
            }
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
        // Immediate CPU readback path: block until mapped dl/loss are ready.
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let dl = bytemuck::cast_slice(&dl_slice.get_mapped_range()).to_vec();
            let loss_int = bytemuck::cast_slice::<u8, i32>(&loss_slice.get_mapped_range())[0];
            let loss = loss_int as f32 / 10000.0;
            if self.cfg_lm_debug {
                eprintln!("    [LM-DEBUG] loss_int={}", loss_int);
            }
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

        self.sampled_indices_reuse.clear();
        self.sampled_indices_reuse.reserve(self.sample_capacity);
        let actual_num_samples = self.build_sampled_indices(targets, step_t);
        if self.cfg_lm_debug {
            let tgt0 = targets.get(0).copied().unwrap_or(0);
            eprintln!(
                "    [LM-DEBUG] seq_len={} num_samples={} target0={}",
                seq_len, actual_num_samples, tgt0
            );
        }
        // Swap buffers: sampled_indices_reuse ↔ last_sampled_indices (no copy, just pointer swap).
        std::mem::swap(
            &mut self.sampled_indices_reuse,
            &mut self.last_sampled_indices,
        );
        self.last_num_samples = actual_num_samples;
        // Use last_sampled_indices for the rest of this call.
        let sampled_indices = &self.last_sampled_indices;
        if self.cfg_lm_debug {
            let tgt0 = targets.first().copied().unwrap_or(Self::IGNORE_TARGET);
            let target_pos = sampled_indices.iter().position(|&v| v == tgt0);
            let preview_len = sampled_indices.len().min(12);
            eprintln!(
                "    [LM-DEBUG] buffer_input sampled contains_target={} target_pos={:?} preview={:?}",
                target_pos.is_some(),
                target_pos,
                &sampled_indices[..preview_len]
            );
        }

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
            active_targets: self.active_target_count(targets),
            tie_lambda_bits: 0.0f32.to_bits(),
            assoc_logit_lambda_bits: 0.0f32.to_bits(),
            full_vocab_estimate: if self.cfg_full_vocab_estimate { 1 } else { 0 },
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
        queue.write_buffer(&self.target_indices_buf, 0, bytemuck::cast_slice(targets));
        queue.write_buffer(
            &self.sampled_indices_buf,
            0,
            bytemuck::cast_slice(sampled_indices),
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

        let probs_bg = if h_offset == 0 {
            self.make_probs_bg_for_h(device, h_src)
        } else {
            let copy_size = (self.config.d_r * seq_len * 4) as u64;
            encoder.copy_buffer_to_buffer(h_src, h_offset, &self.h_buf, 0, copy_size);
            self.make_probs_bg_for_h(device, &self.h_buf)
        };
        let seq_len_u32 = seq_len as u32;
        let t_grid_x = seq_len_u32.min(65535).max(1);
        let t_grid_y = seq_len_u32.div_ceil(65535).max(1);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Probs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probs_pipeline);
            pass.set_bind_group(0, &probs_bg, &[]);
            pass.dispatch_workgroups(t_grid_x, t_grid_y, 1);
        }
        let t_parts_x = t_grid_x;
        let t_parts_y = t_grid_y;
        if self.fused_b19 {
            // Fused path: use the same update kernel as non-fused to guarantee
            // numerical equivalence (avoids drift seen with lm_dw_accum_main).
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LM Update Pass (fused, buffer input)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.update_pipeline);
                pass.set_bind_group(0, &probs_bg, &[]);
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
                pass.set_bind_group(0, &probs_bg, &[]);
                pass.dispatch_workgroups(t_parts_x, t_parts_y, 1);
            }
        } else {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LM Update Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.update_pipeline);
                pass.set_bind_group(0, &probs_bg, &[]);
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
                pass.set_bind_group(0, &probs_bg, &[]);
                pass.dispatch_workgroups(t_parts_x, t_parts_y, 1);
            }
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
        // Immediate CPU readback path: block until mapped dl/loss are ready.
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

    /// Runs LM Head training without reading results back to the CPU.
    /// Ideal for chaining with the DEQ's backpropagation on the GPU.
    pub fn train_step_no_readback(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        h_src: &wgpu::Buffer,
        h_offset: u64,
        assoc_src: &wgpu::Buffer,
        assoc_offset: u64,
        emb_ref_src: Option<&wgpu::Buffer>,
        targets: &[u32],
        lr: f32,
        step_t: u32,
        ternary: bool,
        read_loss: bool,
    ) -> Result<f32, String> {
        let seq_len = targets.len();

        self.sampled_indices_reuse.clear();
        self.sampled_indices_reuse.reserve(self.sample_capacity);
        let actual_num_samples = self.build_sampled_indices(targets, step_t);
        if self.cfg_lm_debug {
            let tgt0 = targets.get(0).copied().unwrap_or(0);
            eprintln!(
                "    [LM-DEBUG] seq_len={} num_samples={} target0={}",
                seq_len, actual_num_samples, tgt0
            );
        }
        // Swap buffers: sampled_indices_reuse ↔ last_sampled_indices (no copy, just pointer swap).
        std::mem::swap(
            &mut self.sampled_indices_reuse,
            &mut self.last_sampled_indices,
        );
        self.last_num_samples = actual_num_samples;
        // Use last_sampled_indices for the rest of this call.
        let sampled_indices = &self.last_sampled_indices;
        if self.cfg_lm_debug {
            let tgt0 = targets.first().copied().unwrap_or(Self::IGNORE_TARGET);
            let target_pos = sampled_indices.iter().position(|&v| v == tgt0);
            let preview_len = sampled_indices.len().min(12);
            eprintln!(
                "    [LM-DEBUG] no_readback sampled contains_target={} target_pos={:?} preview={:?}",
                target_pos.is_some(),
                target_pos,
                &sampled_indices[..preview_len]
            );
        }

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
            active_targets: self.active_target_count(targets),
            tie_lambda_bits: self.cfg_lm_emb_tie_lambda.to_bits(),
            assoc_logit_lambda_bits: self.cfg_assoc_logit_lambda.to_bits(),
            full_vocab_estimate: if self.cfg_full_vocab_estimate { 1 } else { 0 },
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

        if self.cfg_lm_emb_tie_lambda > 0.0 || self.cfg_assoc_logit_lambda > 0.0 {
            if let Some(src) = emb_ref_src {
                let copy_size = (self.vocab_size * self.config.d_r * 4) as u64;
                encoder.copy_buffer_to_buffer(src, 0, &self.emb_ref_buf, 0, copy_size);
            }
        }

        let copy_size = (self.config.d_r * seq_len * 4) as u64;
        if h_offset == 0 {
            encoder.copy_buffer_to_buffer(h_src, 0, &self.h_buf, 0, copy_size);
        } else {
            encoder.copy_buffer_to_buffer(h_src, h_offset, &self.h_buf, 0, copy_size);
        }
        if assoc_offset == 0 {
            encoder.copy_buffer_to_buffer(assoc_src, 0, &self.dl_dh_temp_buf, 0, copy_size);
        } else {
            encoder.copy_buffer_to_buffer(
                assoc_src,
                assoc_offset,
                &self.dl_dh_temp_buf,
                0,
                copy_size,
            );
        }
        let probs_bg = self.make_probs_bg_for_h(device, &self.h_buf);
        encoder.clear_buffer(&self.loss_buf, 0, None);
        let seq_len_u32 = seq_len as u32;
        let t_grid_x = seq_len_u32.min(65535).max(1);
        let t_grid_y = seq_len_u32.div_ceil(65535).max(1);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LM Probs Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.probs_pipeline);
            pass.set_bind_group(0, &probs_bg, &[]);
            pass.dispatch_workgroups(t_grid_x, t_grid_y, 1);
        }
        let t_parts_x = t_grid_x;
        let t_parts_y = t_grid_y;
        if self.fused_b19 {
            // Fused path: use the same update kernel as non-fused to guarantee
            // numerical equivalence (avoids drift seen with lm_dw_accum_main).
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LM Update Pass (fused, no readback)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.update_pipeline);
                pass.set_bind_group(0, &probs_bg, &[]);
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
                pass.set_bind_group(0, &probs_bg, &[]);
                pass.dispatch_workgroups(t_parts_x, t_parts_y, 1);
            }
        } else {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("LM Update Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.update_pipeline);
                pass.set_bind_group(0, &probs_bg, &[]);
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
                pass.set_bind_group(0, &probs_bg, &[]);
                pass.dispatch_workgroups(t_parts_x, t_parts_y, 1);
            }
        }
        // Ternary project is now fused into update_pipeline

        // Always copy loss to staging so read_cached_loss() can retrieve it later.
        encoder.copy_buffer_to_buffer(&self.loss_buf, 0, &self.loss_staging_buf, 0, 4);
        queue.submit(Some(encoder.finish()));

        if !read_loss {
            return Ok(0.0); // Don't block — caller reads loss via read_cached_loss()
        }

        // Read loss (only if requested)
        let slice = self.loss_staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        // Immediate CPU readback path: caller consumes the mapped loss now.
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let loss_int = i32::from_le_bytes(data[0..4].try_into().unwrap());
            drop(data);
            self.loss_staging_buf.unmap();
            if self.cfg_lm_debug {
                eprintln!("    [LM-DEBUG] no_readback loss_int={}", loss_int);
            }
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
        // Use Wait to ensure we can unmap before the next submission.
        // Immediate CPU readback path: caller consumes the mapped loss now.
        device.poll(wgpu::Maintain::Wait);
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
        // Immediate CPU readback helper.
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
        // Immediate CPU readback helper.
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

    /// TEMPORARY DIAGNOSTIC: read arbitrary prefix of final dL/dh.
    pub fn read_dl_dh_n(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n_floats: usize,
    ) -> Result<Vec<f32>, String> {
        self.read_buffer_prefix(device, queue, &self.dl_dh_buf, n_floats, "LM dl_dh_n")
    }

    /// TEMPORARY DIAGNOSTIC: read arbitrary prefix of assoc RHS scratch.
    pub fn read_assoc_grad_n(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n_floats: usize,
    ) -> Result<Vec<f32>, String> {
        self.read_buffer_prefix(
            device,
            queue,
            &self.s_h_rms_buf,
            n_floats,
            "LM assoc_grad_n",
        )
    }

    /// TEMPORARY DIAGNOSTIC: read sampled-softmax probabilities/logits prefix.
    pub fn read_probs_n(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n_floats: usize,
    ) -> Result<Vec<f32>, String> {
        self.read_buffer_prefix(device, queue, &self.lm_scratch_buf, n_floats, "LM probs_n")
    }

    /// TEMPORARY DIAGNOSTIC: read per-token RMSNorm denominators.
    pub fn read_rms_n(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n_floats: usize,
    ) -> Result<Vec<f32>, String> {
        self.read_buffer_prefix(device, queue, &self.rms_buf, n_floats, "LM rms_n")
    }

    fn read_buffer_prefix(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &wgpu::Buffer,
        n_floats: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        let bytes = (n_floats * std::mem::size_of::<f32>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let out = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
            drop(data);
            staging.unmap();
            Ok(out)
        } else {
            Err(format!("{label} map failed"))
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
        // Immediate CPU readback helper.
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

    /// Reads the Adam moments (m, v) from the GPU for checkpointing.
    /// Returns (m_w, v_w, m_b, v_b, m_g, v_g) — all as Vec<f32>.
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
        // Immediate CPU readback helper.
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
    /// Expects the same 6 vecs that `read_moments` returns.
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
    /// Use to initialize weights for the first time before `train_step_no_readback`.
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
