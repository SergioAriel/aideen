use aideen_block::cg_bridge::RustCgBridge;
use aideen_block::deq_bridge::{DeqComputeShape, RustDeqBridge};
use aideen_core::state::ArchitectureConfig;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct UpdateUniforms {
    d_model: u32,
    h_slots: u32,
    lr: f32,
    grad_scale: f32,
    ternary_flag: u32,
    weight_decay: f32,
    seq_len: u32,
    damping: f32, // v14: Crucial for Picard Adjoint stability
    residual_alpha: f32,
}

/// Abstracción del DEQ vía GPU (WGPU).
pub struct GpuDeqBackend {
    pub config: ArchitectureConfig,
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub bridge: RustDeqBridge,
    pub cg_bridge: RustCgBridge,

    // Fused update pipeline
    fused_adjoint_picard_pipeline: wgpu::ComputePipeline,
    staged_picard_init_pipeline: wgpu::ComputePipeline,
    staged_picard_gcomb_pipeline: wgpu::ComputePipeline,
    staged_picard_gmix_pipeline: wgpu::ComputePipeline,
    staged_picard_gscore_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_v_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_k_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_q_pipeline: wgpu::ComputePipeline,
    fused_update_stage1a_pipeline: wgpu::ComputePipeline,
    fused_update_stage1b_pipeline: wgpu::ComputePipeline,
    fused_update_stage2_pipeline: wgpu::ComputePipeline,
    fused_update_stage3_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_pipeline: wgpu::ComputePipeline,
    fused_update_hist_prep_pipeline: wgpu::ComputePipeline,
    fused_update_hist_mat_pipeline: wgpu::ComputePipeline,
    fused_update_hist_scale_pipeline: wgpu::ComputePipeline,
    fused_update_hist_bias_pipeline: wgpu::ComputePipeline,
    fused_update_hist_gate_pipeline: wgpu::ComputePipeline,
    fused_update_hist_mprev_pipeline: wgpu::ComputePipeline,
    fused_update_hist_tbptt_pipeline: wgpu::ComputePipeline,
    fused_update_hist_xprep_pipeline: wgpu::ComputePipeline,
    fused_update_hist_hrhs_pipeline: wgpu::ComputePipeline,
    fused_update_hist_wout_pipeline: wgpu::ComputePipeline,
    fused_update_hist_wx_pipeline: wgpu::ComputePipeline,
    fused_update_hist_alog_pipeline: wgpu::ComputePipeline,
    fused_update_hist_wdelta_pipeline: wgpu::ComputePipeline,
    fused_update_hist_bdelta_pipeline: wgpu::ComputePipeline,
    fused_adjoint_bg: wgpu::BindGroup,
    fused_adjoint_weights_bg: wgpu::BindGroup,
    staged_picard_bg: wgpu::BindGroup,
    staged_picard_bg1: wgpu::BindGroup,
    fused_update_bg0: wgpu::BindGroup,
    fused_update_bg1: wgpu::BindGroup,
    pub fused_update_params_buf: wgpu::Buffer,
    fused_v_next_buf: wgpu::Buffer, // State for Picard Adjoint
    fused_mix_buf: wgpu::Buffer,
    fused_weighted_h_buf: wgpu::Buffer,
    fused_gmix_buf: wgpu::Buffer,
    fused_hist_ctx_buf: wgpu::Buffer,
    fused_hist_delta_buf: wgpu::Buffer,
    fused_gscore_buf: wgpu::Buffer,
    fused_qgrad_buf: wgpu::Buffer,
    staged_wq_t_buf: wgpu::Buffer,
    staged_wk_t_buf: wgpu::Buffer,
    staged_wv_t_buf: wgpu::Buffer,
    staged_wo_t_buf: wgpu::Buffer,
}

impl GpuDeqBackend {
    fn transpose_square(src: &[f32], d: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        for r in 0..d {
            for c in 0..d {
                out[c * d + r] = src[r * d + c];
            }
        }
        out
    }

    fn upload_staged_transposed_weights(
        &self,
        queue: &wgpu::Queue,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
    ) {
        let d = self.config.d_r;
        let wq_t = Self::transpose_square(wq, d);
        let wk_t = Self::transpose_square(wk, d);
        let wv_t = Self::transpose_square(wv, d);
        let wo_t = Self::transpose_square(wo, d);
        queue.write_buffer(&self.staged_wq_t_buf, 0, bytemuck::cast_slice(&wq_t));
        queue.write_buffer(&self.staged_wk_t_buf, 0, bytemuck::cast_slice(&wk_t));
        queue.write_buffer(&self.staged_wv_t_buf, 0, bytemuck::cast_slice(&wv_t));
        queue.write_buffer(&self.staged_wo_t_buf, 0, bytemuck::cast_slice(&wo_t));
    }

    fn pack_history_params(
        &self,
        w_hist_shared: &[f32],
        hist_slot_scale: &[f32],
        hist_slot_bias: &[f32],
        hist_gate_logit: &[f32],
        slot_anchor: &[f32],
        w_delta: &[f32],
        b_delta: &[f32],
    ) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let mut out = Vec::with_capacity(Self::history_params_len(self.config.d_r, self.config.h_slots));
        out.extend_from_slice(w_hist_shared);
        out.extend_from_slice(hist_slot_scale);
        out.extend_from_slice(hist_slot_bias);
        out.extend_from_slice(hist_gate_logit);
        if Self::env_flag("AIDEEN_DEQ_SLOT_ANCHOR_ZERO") {
            out.extend(std::iter::repeat(0.0).take(h * d));
        } else {
            out.extend_from_slice(slot_anchor);
        }
        out.extend_from_slice(w_delta);
        out.extend_from_slice(b_delta);
        out.push(if Self::hist_selective_from_env() { 1.0 } else { 0.0 });
        // Warmup factor for alpha_min (0..1). Default to 1.0 so inference is unaffected.
        out.push(1.0);
        let rms_floor = Self::env_f32("AIDEEN_DEQ_RMS_FLOOR").unwrap_or(0.0).max(0.0);
        let contr_floor = Self::env_f32("AIDEEN_DEQ_CONTR_RMS_FLOOR").unwrap_or(0.0).max(0.0);
        out.push(rms_floor);
        out.push(contr_floor);
        let hist_inject = if Self::env_flag("AIDEEN_DEQ_HIST_ZERO") {
            0.0
        } else {
            1.0
        };
        out.push(hist_inject);
        let hist_minner_zero = if Self::env_flag("AIDEEN_DEQ_HIST_MINNER_ZERO") {
            1.0
        } else {
            0.0
        };
        out.push(hist_minner_zero);
        let hist_force_nomamba = if Self::env_flag("AIDEEN_DEQ_HIST_FORCE_NOMAMBA") {
            1.0
        } else {
            0.0
        };
        out.push(hist_force_nomamba);
        let hist_prelude_skip = if Self::env_flag("AIDEEN_DEQ_HIST_PRELUDE_SKIP") {
            1.0
        } else {
            0.0
        };
        out.push(hist_prelude_skip);
        let hist_loop_force_nomamba = if Self::env_flag("AIDEEN_DEQ_HIST_LOOP_FORCE_NOMAMBA") {
            1.0
        } else {
            0.0
        };
        out.push(hist_loop_force_nomamba);
        out
    }

    fn history_params_len(d: usize, h: usize) -> usize {
        2 * d * d + 3 * h * d + h + d + 9
    }

    pub fn set_hist_warmup_factor(&self, factor: f32) {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let idx = Self::history_params_len(d, h) - 1;
        self.queue
            .write_buffer(&self.bridge.hist_params_buf, (idx * 4) as u64, bytemuck::bytes_of(&factor));
    }

    fn hist_selective_from_env() -> bool {
        std::env::var("AIDEEN_HIST_SELECTIVE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false)
    }

    fn env_flag(name: &str) -> bool {
        std::env::var(name)
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false)
    }

    fn env_f32(name: &str) -> Option<f32> {
        std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok())
    }

    fn residual_alpha_from_env() -> f32 {
        // Stage modes via sentinel passed to WGSL:
        //  -2.0 => DEQ-only (no attention, no mamba)
        //  -1.0 => attention ON, mamba OFF
        //  -0.75 => attention ON, mamba history only as h0 init
        //  -0.5 => attention ON, gated per-slot historical context
        //  -0.25 => attention ON, fixed mamba bias per token
        if std::env::var("AIDEEN_DEQ_ONLY").ok().as_deref() == Some("1") {
            return -2.0;
        }
        if std::env::var("AIDEEN_DEQ_NO_MAMBA").ok().as_deref() == Some("1") {
            return -1.0;
        }
        if std::env::var("AIDEEN_DEQ_INIT_MAMBA").ok().as_deref() == Some("1") {
            return -0.75;
        }
        if std::env::var("AIDEEN_DEQ_HIST_GATED").ok().as_deref() == Some("1") {
            return -0.5;
        }
        if std::env::var("AIDEEN_DEQ_FIXED_MAMBA").ok().as_deref() == Some("1") {
            return -0.25;
        }

        let alpha = std::env::var("AIDEEN_DEQ_RESIDUAL_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or_else(|| {
                // v14 Default: 0.0 is mathematically proven to converge 100% of the time.
                0.0
            });
        alpha
    }

    /// Inicializa la conexión con Apple Metal / Vulkan y compila los Shaders WGSL del DEQ
    pub async fn new_async(config: ArchitectureConfig) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
        {
            Some(adapter) => adapter,
            None => {
                eprintln!("[GpuDeqBackend] No compatible GPU adapter found.");
                return None;
            }
        };

        println!("[GpuDeqBackend] Adapter: {}", adapter.get_info().name);
        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage = 16;

        // 3. Crear Device
        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("AIDEEN DEQ GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
        {
            Ok((device, queue)) => (device, queue),
            Err(err) => {
                eprintln!("[GpuDeqBackend] request_device failed: {err:?}");
                return None;
            }
        };

        // Forward DEQ and backward CG do not use the same geometry:
        // - Forward DEQ processes one sequence as batch=1, seq_len=T
        // - CG/fused backward processes per-token pooled gradients as batch=T
        // Sizing both bridges with the same "safe_batch" corrupts the intended layout.
        let forward_batch_cap = 1u32;
        let forward_seq_cap = config.ctx_len.max(1) as u32;
        let cg_batch_cap = config.ctx_len.max(1) as u32;

        // RustDeqBridge::new(device, d_model, h_slots, max_batch_size, max_seq_len)
        let bridge = RustDeqBridge::new(
            &device,
            config.d_r as u32,
            config.h_slots as u32,
            forward_batch_cap,
            forward_seq_cap,
        );

        let cg_bridge = RustCgBridge::new(
            &device,
            config.d_r as u32,
            config.h_slots as u32,
            cg_batch_cap,
        );

        // Fused Update Pipeline setup
        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused DEQ Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fused_deq_update.wgsl").into()),
        });
        let adjoint_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused Picard Adjoint Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/fused_adjoint_picard.wgsl").into(),
            ),
        });
        let staged_picard_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Staged Picard Adjoint Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/staged_adjoint_picard.wgsl").into(),
            ),
        });

        let bg_adjoint_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused Picard Adjoint BG Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, // params
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, // b_in (dl_dh_pooled)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, // H_star
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, // v_state
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, // v_final (result)
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5, // NormScale
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6, // Scratch
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

        let bg0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused Update BG0 Layout"),
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
                // v14 + Phase 3: Mamba Autograd required buffers
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
            ],
        });

        let bg_adjoint_weights_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fused Picard Adjoint Weights BG Layout"),
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
                ],
            });


        let bg1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused Update BG1 Layout (Weights)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused Update PL"),
            bind_group_layouts: &[&bg0_layout, &bg1_layout],
            push_constant_ranges: &[],
        });

        let fused_update_stage1a_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage1a Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage1a_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage1b_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage1b Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage1b_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage2_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage2 Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage2_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage3_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage3 Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage3_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage4_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_prep_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Prep Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_prep_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_mat_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Mat Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_mat_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_scale_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Scale Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_scale_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_bias_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Bias Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_bias_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_gate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Gate Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_gate_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_mprev_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist MPrev Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_mprev_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_tbptt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist TBPTT Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_tbptt_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_xprep_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist XPrep Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_xprep_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_hrhs_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist HRhs Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_hrhs_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_wout_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Wout Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_wout_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_wx_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Wx Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_wx_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_alog_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist ALog Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_alog_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_wdelta_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist WDelta Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_wdelta_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_bdelta_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist BDelta Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_bdelta_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_adjoint_picard_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused Picard Adjoint Pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Picard Adjoint PL"),
                        bind_group_layouts: &[&bg_adjoint_layout, &bg_adjoint_weights_layout],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &adjoint_shader,
                entry_point: Some("fused_adjoint_picard_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_pl =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Staged Picard PL"),
                bind_group_layouts: &[&bg0_layout, &bg1_layout],
                push_constant_ranges: &[],
            });
        let staged_picard_gcomb_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard GComb Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_gcomb_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_init_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard Init Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_init_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_gmix_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard GMix Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_gmix_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_gscore_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard GScore Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_gscore_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_accum_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard Accum Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_accum_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_accum_v_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard Accum V Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_accum_v_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_accum_k_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard Accum K Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_accum_k_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_accum_q_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard Accum Q Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_accum_q_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Update Params"),
            size: std::mem::size_of::<UpdateUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let attn_entries = (config.ctx_len.max(1) * config.h_slots * config.d_r) as u64;
        let fused_mix_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Attn Mix Buffer"),
            size: attn_entries * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fused_weighted_h_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Attn Weighted H Buffer"),
            size: attn_entries * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fused_gmix_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Attn Gmix Buffer"),
            size: attn_entries * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fused_hist_ctx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Hist Context Buffer"),
            size: attn_entries * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fused_hist_delta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Hist Delta Buffer"),
            size: attn_entries * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let gscore_entries = (config.ctx_len.max(1) * config.h_slots * config.h_slots) as u64;
        let fused_gscore_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Attn Gscore Buffer"),
            size: gscore_entries * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fused_qgrad_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Attn Qgrad Buffer"),
            size: attn_entries * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let fused_v_next_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Picard Adjoint State Buffer"),
            size: attn_entries * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let w_mat_bytes = (config.d_r * config.d_r * std::mem::size_of::<f32>()) as u64;
        let staged_wq_t_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staged Picard Wq^T Buffer"),
            size: w_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staged_wk_t_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staged Picard Wk^T Buffer"),
            size: w_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staged_wv_t_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staged Picard Wv^T Buffer"),
            size: w_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staged_wo_t_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staged Picard Wo^T Buffer"),
            size: w_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fused_update_bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Update BG0"),
            layout: &bg0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fused_update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cg_bridge.b_v_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bridge.s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bridge.hnext_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bridge.debug_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cg_bridge.b_dl.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bridge.scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: fused_mix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: fused_weighted_h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: fused_gmix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: fused_gscore_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: fused_qgrad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: fused_hist_ctx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: fused_hist_delta_buf.as_entire_binding(),
                },
            ],
        });

        let staged_picard_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Staged Picard BG0"),
            layout: &bg0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fused_update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cg_bridge.b_v_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bridge.s_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bridge.hnext_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bridge.debug_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cg_bridge.b_dl.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bridge.scratch_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: fused_mix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: fused_weighted_h_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: fused_gmix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: fused_gscore_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: fused_qgrad_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: fused_hist_ctx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: fused_hist_delta_buf.as_entire_binding(),
                },
            ],
        });
        let staged_picard_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Staged Picard BG1"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: staged_wq_t_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: staged_wk_t_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: staged_wv_t_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: staged_wo_t_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bridge.win_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bridge.wx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bridge.wout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bridge.a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bridge.n_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: bridge.hist_params_buf.as_entire_binding(),
                },
            ],
        });

        let fused_adjoint_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Picard Adjoint BG"),
            layout: &bg_adjoint_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fused_update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cg_bridge.b_dl.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bridge.hnext_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cg_bridge.b_v_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: fused_v_next_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bridge.n_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bridge.scratch_buf.as_entire_binding(),
                },
            ],
        });

        let fused_adjoint_weights_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Picard Adjoint Weights BG"),
            layout: &bg_adjoint_weights_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bridge.wq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bridge.wk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bridge.wv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bridge.wo_buf.as_entire_binding(),
                },
            ],
        });

        let fused_update_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Update BG1"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bridge.wq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bridge.wk_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bridge.wv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bridge.wo_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bridge.win_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bridge.wx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bridge.wout_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bridge.a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bridge.n_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: bridge.hist_params_buf.as_entire_binding(),
                },
            ],
        });
        Some(Self {
            config,
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            bridge,
            cg_bridge,
            fused_adjoint_picard_pipeline,
            staged_picard_init_pipeline,
            staged_picard_gcomb_pipeline,
            staged_picard_gmix_pipeline,
            staged_picard_gscore_pipeline,
            staged_picard_accum_pipeline,
            staged_picard_accum_v_pipeline,
            staged_picard_accum_k_pipeline,
            staged_picard_accum_q_pipeline,
            fused_update_stage1a_pipeline,
            fused_update_stage1b_pipeline,
            fused_update_stage2_pipeline,
            fused_update_stage3_pipeline,
            fused_update_stage4_pipeline,
            fused_update_hist_prep_pipeline,
            fused_update_hist_mat_pipeline,
            fused_update_hist_scale_pipeline,
            fused_update_hist_bias_pipeline,
            fused_update_hist_gate_pipeline,
            fused_update_hist_mprev_pipeline,
            fused_update_hist_tbptt_pipeline,
            fused_update_hist_xprep_pipeline,
            fused_update_hist_hrhs_pipeline,
            fused_update_hist_wout_pipeline,
            fused_update_hist_wx_pipeline,
            fused_update_hist_alog_pipeline,
            fused_update_hist_wdelta_pipeline,
            fused_update_hist_bdelta_pipeline,
            fused_adjoint_bg,
            fused_adjoint_weights_bg,
            staged_picard_bg,
            staged_picard_bg1,
            fused_update_bg0,
            fused_update_bg1,
            fused_update_params_buf,
            fused_v_next_buf,
            fused_mix_buf,
            fused_weighted_h_buf,
            fused_gmix_buf,
            fused_hist_ctx_buf,
            fused_hist_delta_buf,
            fused_gscore_buf,
            fused_qgrad_buf,
            staged_wq_t_buf,
            staged_wk_t_buf,
            staged_wv_t_buf,
            staged_wo_t_buf,
        })
    }

    /// Método sincrónico por si necesitamos llamarlo desde el main hilo (pollster) V1.
    pub fn new_blocking(config: ArchitectureConfig) -> Option<Self> {
        pollster::block_on(Self::new_async(config))
    }

    /// Reinicia los estados ocultos (slots) en la GPU a cero.
    pub fn reset_state(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GPU Reset State"),
            });
        encoder.clear_buffer(&self.bridge.hcurr_buf, 0, None);
        encoder.clear_buffer(&self.bridge.hnext_buf, 0, None);
        // v14: En wgsl, utilizamos Scratch[mamba_base] como M_t persistente.
        // Es estrictamente necesario limpiar Scratch al iniciar nueva secuencia.
        encoder.clear_buffer(&self.bridge.scratch_buf, 0, None);
        self.queue.submit(Some(encoder.finish()));
    }

    // --- HELPER UNIFICADO ---
    fn build_compute_shape(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
    ) -> DeqComputeShape {
        DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            residual_alpha: Self::residual_alpha_from_env(),
        }
    }

    fn cg_damping_from_env() -> f32 {
        if std::env::var("AIDEEN_DEQ_ONLY").ok().as_deref() == Some("1") {
            0.95
        } else if std::env::var("AIDEEN_DEQ_HIST_GATED").ok().as_deref() == Some("1") {
            0.90
        } else if std::env::var("AIDEEN_DEQ_INIT_MAMBA").ok().as_deref() == Some("1") {
            0.90
        } else if std::env::var("AIDEEN_DEQ_FIXED_MAMBA").ok().as_deref() == Some("1") {
            0.90
        } else if std::env::var("AIDEEN_DEQ_NO_MAMBA").ok().as_deref() == Some("1") {
            0.90
        } else {
            0.85
        }
    }

    pub fn build_cg_shape(
        &self,
        batch_size: u32,
        cg_iters: u32,
    ) -> aideen_block::cg_bridge::CGComputeShape {
        aideen_block::cg_bridge::CGComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            cg_iters,
            epsilon: self.config.deq_epsilon,
            damping: Self::cg_damping_from_env(),
            curr_iter: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
        }
    }

    // --- WEIGHTS ---

    pub fn upload_weights(
        &self,
        queue: &wgpu::Queue,
        wq: &[f32],
        wk: &[f32],
        wv: &[f32],
        wo: &[f32],
        win: &[f32],
        wx: &[f32],
        wout: &[f32],
        alog: &[f32],
        nscale: &[f32],
        w_hist_shared: &[f32],
        hist_slot_scale: &[f32],
        hist_slot_bias: &[f32],
        hist_gate_logit: &[f32],
        slot_anchor: &[f32],
        w_delta: &[f32],
        b_delta: &[f32],
    ) {
        queue.write_buffer(&self.bridge.wq_buf, 0, bytemuck::cast_slice(wq));
        queue.write_buffer(&self.bridge.wk_buf, 0, bytemuck::cast_slice(wk));
        queue.write_buffer(&self.bridge.wv_buf, 0, bytemuck::cast_slice(wv));
        queue.write_buffer(&self.bridge.wo_buf, 0, bytemuck::cast_slice(wo));
        self.upload_staged_transposed_weights(queue, wq, wk, wv, wo);
        queue.write_buffer(&self.bridge.win_buf, 0, bytemuck::cast_slice(win));
        queue.write_buffer(&self.bridge.wx_buf, 0, bytemuck::cast_slice(wx));
        queue.write_buffer(&self.bridge.wout_buf, 0, bytemuck::cast_slice(wout));
        queue.write_buffer(&self.bridge.a_buf, 0, bytemuck::cast_slice(alog));
        queue.write_buffer(&self.bridge.n_buf, 0, bytemuck::cast_slice(nscale));
        let hist_params = self.pack_history_params(
            w_hist_shared,
            hist_slot_scale,
            hist_slot_bias,
            hist_gate_logit,
            slot_anchor,
            w_delta,
            b_delta,
        );
        queue.write_buffer(
            &self.bridge.hist_params_buf,
            0,
            bytemuck::cast_slice(hist_params.as_slice()),
        );
        queue.submit([]);
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok((wq_check, _, _, _, win_check, _, _, _, _)) = self.read_weights() {
            let stats = |v: &[f32]| -> (f32, f32, f32) {
                let min = v.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let abs_mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                (min, max, abs_mean)
            };
            let (q_min, q_max, q_abs) = stats(&wq_check);
            let (in_min, in_max, in_abs) = stats(&win_check);
            eprintln!(
                "[GPU-VERIFY] Post-upload:\n    W_q:  min={:.4}, max={:.4}, abs_mean={:.6}\n    W_in: min={:.4}, max={:.4}, abs_mean={:.6}",
                q_min, q_max, q_abs, in_min, in_max, in_abs
            );
        }
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
    }

    // --- MÉTODOS DE EJECUCIÓN ---

    pub fn run_forward_deq(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<u32>), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    // ESTE ES EL QUE ESPERA EL TRAINER (16 argumentos)
    pub fn run_forward_deq_pooled(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<u32>), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward_pooled(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_forward_single_token(
        &self,
        batch_size: u32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool, // <--- ARREGLADO: Agregado el argumento faltante
    ) -> Result<Vec<f32>, &'static str> {
        let shape =
            self.build_compute_shape(batch_size, 1, self.config.max_deq_iters as u32, 5e-4, 0.9);

        // ARREGLADO: Extraemos el primer elemento de la tupla (Vec<f32>, Vec<u32>)
        match self.bridge.run_forward(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        ) {
            Ok((h_data, _)) => Ok(h_data),
            Err(e) => Err(e),
        }
    }

    pub fn run_forward_deq_pooled_with_state(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward_pooled_with_state(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_forward_deq_no_readback(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_in: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        update_weights: bool,
    ) -> Result<(), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.bridge.run_forward_no_readback(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_forward_from_seq_buf(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        epsilon: f32,
        damping: f32,
        s_buf_gpu: &wgpu::Buffer,
        update_weights: bool,
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
    ) -> Result<(), &'static str> {
        let shape = self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping);
        self.queue
            .write_buffer(&self.bridge.uniform_buf, 0, bytemuck::bytes_of(&shape));

        if update_weights {
            self.queue
                .write_buffer(&self.bridge.wq_buf, 0, bytemuck::cast_slice(w_q));
            self.queue
                .write_buffer(&self.bridge.wk_buf, 0, bytemuck::cast_slice(w_k));
            self.queue
                .write_buffer(&self.bridge.wv_buf, 0, bytemuck::cast_slice(w_v));
            self.queue
                .write_buffer(&self.bridge.wo_buf, 0, bytemuck::cast_slice(w_o));
            self.upload_staged_transposed_weights(&self.queue, w_q, w_k, w_v, w_o);
            self.queue
                .write_buffer(&self.bridge.win_buf, 0, bytemuck::cast_slice(w_in));
            self.queue
                .write_buffer(&self.bridge.wx_buf, 0, bytemuck::cast_slice(w_x));
            self.queue
                .write_buffer(&self.bridge.wout_buf, 0, bytemuck::cast_slice(w_out));
            self.queue
                .write_buffer(&self.bridge.a_buf, 0, bytemuck::cast_slice(a_log));
            self.queue
                .write_buffer(&self.bridge.n_buf, 0, bytemuck::cast_slice(norm));
        }

        let copy_size = (batch_size as u64) * (seq_len as u64) * (self.config.d_r as u64) * 4;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("DEQ Forward (GPU s_buf) Encoder"),
            });
        encoder.copy_buffer_to_buffer(s_buf_gpu, 0, &self.bridge.s_buf, 0, copy_size);
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DEQ Forward Pass (GPU s_buf)"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bridge.pipeline);
            cpass.set_bind_group(0, &self.bridge.bind_group, &[]);
            cpass.dispatch_workgroups(batch_size.max(1), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    // --- BACKWARD ---

    pub fn run_backward_deq(
        &self,
        batch_size: u32,
        s_in: &[f32],
        h_star: &[f32],
        dl_dh_pooled: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        cg_iters: u32,
        update_weights: bool,
    ) -> Result<Vec<f32>, String> {
        let shape = self.build_cg_shape(batch_size, cg_iters);
        self.cg_bridge.run_backward(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            h_star,
            dl_dh_pooled,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn run_backward_deq_from_forward_state(
        &self,
        batch_size: u32,
        s_in: &[f32],
        h_offset: u64,
        dl_dh_pooled: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_in: &[f32],
        w_x: &[f32],
        w_out: &[f32],
        a_log: &[f32],
        norm: &[f32],
        cg_iters: u32,
        update_weights: bool,
    ) -> Result<Vec<f32>, String> {
        let shape = self.build_cg_shape(batch_size, cg_iters);
        self.cg_bridge.run_backward_from_buffer(
            &self.device,
            &self.queue,
            &shape,
            s_in,
            &self.bridge.hnext_buf,
            h_offset,
            dl_dh_pooled,
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm,
            update_weights,
        )
    }

    pub fn apply_deq_sgd_update_and_sync_cg(
        &self,
        lr: f32,
        grad_mat: &[f32],
        grad_vec: &[f32],
    ) -> Result<(), String> {
        self.bridge
            .apply_sgd_update(&self.device, &self.queue, lr, grad_mat, grad_vec)
            .map_err(|e| e.to_string())?;
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            &self.queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
        Ok(())
    }

    pub fn run_backward_no_readback(
        &self,
        batch_size: u32,
        _num_tokens: u32,
        dl_dh_src: &wgpu::Buffer,
        cg_iters: u32,
    ) -> Result<(), String> {
        let shape = self.build_cg_shape(batch_size, cg_iters);
        let h_offset = 0;
        self.cg_bridge.run_backward_no_readback(
            &self.device,
            &self.queue,
            &shape,
            &self.bridge.hcurr_buf,
            h_offset,
            dl_dh_src,
        )
    }

    pub fn run_fused_adjoint_picard_no_readback(
        &self,
        seq_len: u32,
        damping: f32,
    ) -> Result<(), String> {
        let params = UpdateUniforms {
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            lr: 0.0,
            grad_scale: 0.0,
            ternary_flag: 0,
            weight_decay: 0.0,
            seq_len,
            damping,
            residual_alpha: Self::residual_alpha_from_env(),
        };
        self.queue.write_buffer(
            &self.fused_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );

        let zero_len = (seq_len * self.config.h_slots as u32 * self.config.d_r as u32) as usize;
        let zeros = vec![0.0f32; zero_len];
        self.queue
            .write_buffer(&self.cg_bridge.b_v_out, 0, bytemuck::cast_slice(&zeros));
        self.queue
            .write_buffer(&self.fused_v_next_buf, 0, bytemuck::cast_slice(&zeros));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Picard Adjoint Only"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Picard Adjoint Only"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fused_adjoint_picard_pipeline);
            pass.set_bind_group(0, &self.fused_adjoint_bg, &[]);
            pass.set_bind_group(1, &self.fused_adjoint_weights_bg, &[]);
            pass.dispatch_workgroups(seq_len, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    pub fn run_staged_adjoint_picard_no_readback(
        &self,
        seq_len: u32,
        damping: f32,
        iters: u32,
        dl_dh_src: Option<&wgpu::Buffer>,
        clear_slot_rhs: bool,
    ) -> Result<(), String> {
        let profile_picard = std::env::var("AIDEEN_PICARD_PROFILE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        let profile_picard_stages = std::env::var("AIDEEN_PICARD_STAGE_PROFILE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        let profile_picard_accum_split = std::env::var("AIDEEN_PICARD_ACCUM_SPLIT_PROFILE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        let picard_internal_probe = std::env::var("AIDEEN_PICARD_INTERNAL_PROBE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        let total_t0 = std::time::Instant::now();
        let params = UpdateUniforms {
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            lr: 0.0,
            grad_scale: 0.0,
            ternary_flag: 0,
            weight_decay: 0.0,
            seq_len,
            damping,
            residual_alpha: Self::residual_alpha_from_env(),
        };
        self.queue.write_buffer(
            &self.fused_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );

        // Mirror the CG path contract: staged Picard consumes cg_bridge.b_dl as
        // the pooled upstream gradient source. Without this copy, Picard can
        // iterate on a stale/zero rhs even when the LM head produced dl/dh.
        if let Some(dl_dh_src) = dl_dh_src {
            let dl_bytes = (seq_len as u64) * (self.config.d_r as u64) * 4;
            let mut dl_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Staged Picard dl_dh Copy"),
                    });
            dl_encoder.copy_buffer_to_buffer(dl_dh_src, 0, &self.cg_bridge.b_dl, 0, dl_bytes);
            self.queue.submit(Some(dl_encoder.finish()));
        }

        let prep_t0 = std::time::Instant::now();
        let attn_len = (seq_len * self.config.h_slots as u32 * self.config.d_r as u32) as usize;
        let zero_t0 = std::time::Instant::now();
        let mut zero_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Staged Picard Zero Buffers"),
            });
        zero_encoder.clear_buffer(&self.fused_v_next_buf, 0, None);
        // `fused_mix_buf` carries the staged Picard `g_comb` output when we come from the
        // precomputed adjoint path. Hist-gated temporal kernels consume that signal first;
        // clearing it here would silently zero the historical backward path before it runs.
        // Stage1b overwrites the whole buffer later, so no explicit clear is needed.
        zero_encoder.clear_buffer(&self.fused_weighted_h_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_gmix_buf, 0, None);
        if clear_slot_rhs {
            zero_encoder.clear_buffer(&self.fused_hist_ctx_buf, 0, None);
        }
        zero_encoder.clear_buffer(&self.fused_hist_delta_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_qgrad_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_gscore_buf, 0, None);
        self.queue.submit(Some(zero_encoder.finish()));
        let zero_ms = zero_t0.elapsed().as_millis();
        let prep_ms = prep_t0.elapsed().as_millis();

        let n_entries = seq_len * self.config.h_slots as u32;
        let d = self.config.d_r as u32;
        let init_t0 = std::time::Instant::now();
        {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Staged Picard Init"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Staged Picard Init Pass"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.staged_picard_bg, &[]);
                pass.set_bind_group(1, &self.staged_picard_bg1, &[]);
                pass.set_pipeline(&self.staged_picard_init_pipeline);
                pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
            }
            let bytes = (attn_len * std::mem::size_of::<f32>()) as u64;
            // The staged Picard shader writes its init/output state into binding 8.
            // In the current staged bind-group that binding is `fused_weighted_h_buf`,
            // not `fused_v_next_buf`. Copying `fused_v_next_buf` here seeds `b_v_out`
            // with zeros and kills the rerun branch before the first iteration.
            encoder.copy_buffer_to_buffer(
                &self.fused_weighted_h_buf,
                0,
                &self.cg_bridge.b_v_out,
                0,
                bytes,
            );
            self.queue.submit(Some(encoder.finish()));
        }
        let init_ms = init_t0.elapsed().as_millis();
        if picard_internal_probe {
            let sample_t = (seq_len as usize / 2).max(1).min(seq_len.saturating_sub(1) as usize);
            let dl = self.read_storage_buffer(
                &self.cg_bridge.b_dl,
                seq_len as usize * self.config.d_r,
                "Picard Probe b_dl Readback",
            );
            let v_state = self.read_storage_buffer(
                &self.cg_bridge.b_v_out,
                attn_len,
                "Picard Probe v_state Readback",
            );
            let (dl_mean, dl_max, dl_nz) =
                Self::summarize_token_vec(&dl, sample_t, self.config.d_r);
            let (v_mean, v_max, v_nz) = Self::summarize_token_block(
                &v_state,
                sample_t,
                self.config.h_slots,
                self.config.d_r,
            );
            eprintln!(
                "[PICARD-INTERNAL] post-init sample_t={} bdl(mean/max/nz)={:.6e}/{:.6e}/{}/{} vstate(mean/max/nz)={:.6e}/{:.6e}/{}/{}",
                sample_t,
                dl_mean,
                dl_max,
                dl_nz,
                self.config.d_r,
                v_mean,
                v_max,
                v_nz,
                self.config.h_slots * self.config.d_r,
            );
        }
        let mut submit_ms = 0u128;
        let mut stage_gcomb_ms = 0u128;
        let mut stage_gmix_ms = 0u128;
        let mut stage_gscore_ms = 0u128;
        let mut stage_accum_ms = 0u128;
        let mut stage_copy_ms = 0u128;
        let mut stage_accum_v_ms = 0u128;
        let mut stage_accum_k_ms = 0u128;
        let mut stage_accum_q_ms = 0u128;
        for _ in 0..iters {
            if profile_picard_stages {
                let run_stage = |label: &str,
                                 pipeline: &wgpu::ComputePipeline,
                                 x: u32,
                                 y: u32,
                                 device: &wgpu::Device,
                                 queue: &wgpu::Queue,
                                 bg0: &wgpu::BindGroup,
                                 bg1: &wgpu::BindGroup| {
                    let t0 = std::time::Instant::now();
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some(label),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(label),
                            timestamp_writes: None,
                        });
                        pass.set_bind_group(0, bg0, &[]);
                        pass.set_bind_group(1, bg1, &[]);
                        pass.set_pipeline(pipeline);
                        pass.dispatch_workgroups(x, y, 1);
                    }
                    queue.submit(Some(encoder.finish()));
                    device.poll(wgpu::Maintain::Wait);
                    t0.elapsed().as_millis()
                };
                stage_gcomb_ms += run_stage(
                    "Staged Picard GComb",
                    &self.staged_picard_gcomb_pipeline,
                    n_entries.max(1),
                    1,
                    &self.device,
                    &self.queue,
                    &self.staged_picard_bg,
                    &self.staged_picard_bg1,
                );
                stage_gmix_ms += run_stage(
                    "Staged Picard GMix",
                    &self.staged_picard_gmix_pipeline,
                    n_entries.max(1),
                    1,
                    &self.device,
                    &self.queue,
                    &self.staged_picard_bg,
                    &self.staged_picard_bg1,
                );
                stage_gscore_ms += run_stage(
                    "Staged Picard GScore",
                    &self.staged_picard_gscore_pipeline,
                    self.config.h_slots.div_ceil(16) as u32,
                    n_entries.div_ceil(16).max(1),
                    &self.device,
                    &self.queue,
                    &self.staged_picard_bg,
                    &self.staged_picard_bg1,
                );
                stage_accum_ms += run_stage(
                    "Staged Picard Accum",
                    &self.staged_picard_accum_pipeline,
                    d.div_ceil(16),
                    n_entries.div_ceil(16).max(1),
                    &self.device,
                    &self.queue,
                    &self.staged_picard_bg,
                    &self.staged_picard_bg1,
                );
                if profile_picard_accum_split {
                    stage_accum_v_ms += run_stage(
                        "Staged Picard Accum V",
                        &self.staged_picard_accum_v_pipeline,
                        d.div_ceil(16),
                        n_entries.div_ceil(16).max(1),
                        &self.device,
                        &self.queue,
                        &self.staged_picard_bg,
                        &self.staged_picard_bg1,
                    );
                    stage_accum_k_ms += run_stage(
                        "Staged Picard Accum K",
                        &self.staged_picard_accum_k_pipeline,
                        d.div_ceil(16),
                        n_entries.div_ceil(16).max(1),
                        &self.device,
                        &self.queue,
                        &self.staged_picard_bg,
                        &self.staged_picard_bg1,
                    );
                    stage_accum_q_ms += run_stage(
                        "Staged Picard Accum Q",
                        &self.staged_picard_accum_q_pipeline,
                        d.div_ceil(16),
                        n_entries.div_ceil(16).max(1),
                        &self.device,
                        &self.queue,
                        &self.staged_picard_bg,
                        &self.staged_picard_bg1,
                    );
                }
                let t0 = std::time::Instant::now();
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Staged Picard Copy"),
                        });
                let bytes = (attn_len * std::mem::size_of::<f32>()) as u64;
                encoder.copy_buffer_to_buffer(
                    &self.fused_weighted_h_buf,
                    0,
                    &self.cg_bridge.b_v_out,
                    0,
                    bytes,
                );
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                stage_copy_ms += t0.elapsed().as_millis();
            } else {
                let iter_t0 = std::time::Instant::now();
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Staged Picard Adjoint"),
                    });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Staged Picard GComb"),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, &self.staged_picard_bg, &[]);
                    pass.set_bind_group(1, &self.staged_picard_bg1, &[]);
                    pass.set_pipeline(&self.staged_picard_gcomb_pipeline);
                    pass.dispatch_workgroups(n_entries.max(1), 1, 1);
                    pass.set_pipeline(&self.staged_picard_gmix_pipeline);
                    pass.dispatch_workgroups(n_entries.max(1), 1, 1);
                    pass.set_pipeline(&self.staged_picard_gscore_pipeline);
                    pass.dispatch_workgroups(
                        self.config.h_slots.div_ceil(16) as u32,
                        n_entries.div_ceil(16).max(1),
                        1,
                    );
                    pass.set_pipeline(&self.staged_picard_accum_pipeline);
                    pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                }
                let bytes = (attn_len * std::mem::size_of::<f32>()) as u64;
                encoder.copy_buffer_to_buffer(
                    &self.fused_weighted_h_buf,
                    0,
                    &self.cg_bridge.b_v_out,
                    0,
                    bytes,
                );
                self.queue.submit(Some(encoder.finish()));
                submit_ms += iter_t0.elapsed().as_millis();
            }
        }
        let poll_t0 = std::time::Instant::now();
        if !profile_picard_stages {
            self.device.poll(wgpu::Maintain::Wait);
        }
        let poll_ms = poll_t0.elapsed().as_millis();
        if profile_picard {
            eprintln!(
                "[PICARD-PROFILE] seq_len={} iters={} prep={}ms init={}ms zero={}ms submit={}ms poll={}ms total={}ms",
                seq_len,
                iters,
                prep_ms,
                init_ms,
                zero_ms,
                submit_ms,
                poll_ms,
                total_t0.elapsed().as_millis()
            );
            if profile_picard_stages {
                eprintln!(
                    "[PICARD-STAGES] gcomb={}ms gmix={}ms gscore={}ms accum={}ms copy={}ms",
                    stage_gcomb_ms,
                    stage_gmix_ms,
                    stage_gscore_ms,
                    stage_accum_ms,
                    stage_copy_ms
                );
                if profile_picard_accum_split {
                    eprintln!(
                        "[PICARD-ACCUM-SPLIT] v={}ms k={}ms q={}ms",
                        stage_accum_v_ms,
                        stage_accum_k_ms,
                        stage_accum_q_ms
                    );
                }
            }
        }
        Ok(())
    }

    pub fn apply_fused_deq_update(
        &self,
        lr: f32,
        grad_scale: f32,
        ternary: bool,
        weight_decay: f32,
        seq_len: u32,
        damping: f32,
    ) -> Result<(), String> {
        let params = UpdateUniforms {
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            lr,
            grad_scale,
            ternary_flag: if ternary { 1 } else { 0 },
            weight_decay,
            seq_len,
            damping,
            residual_alpha: Self::residual_alpha_from_env(),
        };
        self.queue.write_buffer(
            &self.fused_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        let profile_fused = std::env::var("AIDEEN_FUSED_PROFILE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        // El trainer siempre corre run_staged_adjoint_picard_no_readback antes de llamar
        // esta función — el adjoint siempre está precomputado. El inline Picard de abajo
        // queda comentado como referencia del diseño original.
        // let use_precomputed_adjoint =
        //     std::env::var("AIDEEN_DEQ_NO_MAMBA").ok().as_deref() == Some("1")
        //         || std::env::var("AIDEEN_DEQ_HIST_GATED").ok().as_deref() == Some("1")
        //         || std::env::var("AIDEEN_DEQ_FIXED_MAMBA").ok().as_deref() == Some("1")
        //         || std::env::var("AIDEEN_DEQ_INIT_MAMBA").ok().as_deref() == Some("1");
        let hist_gated = std::env::var("AIDEEN_DEQ_HIST_GATED").ok().as_deref() == Some("1");
        let hist_selective = hist_gated && Self::hist_selective_from_env();
        let hist_internal_probe = std::env::var("AIDEEN_HIST_INTERNAL_PROBE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        if profile_fused {
            // Drain any previously queued GPU work (notably CG) so per-stage fused timings
            // do not absorb unrelated latency from earlier submissions.
            self.device.poll(wgpu::Maintain::Wait);
        }
        let mut zero_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fused Update Zero Buffers"),
            });
        // fused_mix_buf NO se limpia — el staged Picard ya lo llenó con g_comb correcto.
        // (antes se limpiaba para el path inline Picard que ya no se usa)
        // if !hist_gated {
        //     zero_encoder.clear_buffer(&self.fused_mix_buf, 0, None);
        // }
        zero_encoder.clear_buffer(&self.fused_weighted_h_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_gmix_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_hist_ctx_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_hist_delta_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_gscore_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_qgrad_buf, 0, None);
        // b_v_out y fused_v_next_buf ya no se usan — eran output del inline Picard.
        // if !use_precomputed_adjoint {
        //     zero_encoder.clear_buffer(&self.cg_bridge.b_v_out, 0, None);
        //     zero_encoder.clear_buffer(&self.fused_v_next_buf, 0, None);
        // }
        self.queue.submit(Some(zero_encoder.finish()));
        if profile_fused {
            self.device.poll(wgpu::Maintain::Wait);
        }
        let d = self.config.d_r as u32;
        let n = seq_len * self.config.h_slots as u32;
        let hs = self.config.h_slots as u32;

        let run_stage = |device: &wgpu::Device,
                         queue: &wgpu::Queue,
                         label: &str,
                         pipeline: &wgpu::ComputePipeline,
                         bg0: &wgpu::BindGroup,
                         bg1: &wgpu::BindGroup,
                         x: u32,
                         y: u32,
                         profile: bool| {
            let t0 = std::time::Instant::now();
            let mut encoder = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg0, &[]);
                pass.set_bind_group(1, bg1, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
            queue.submit(Some(encoder.finish()));
            if profile {
                device.poll(wgpu::Maintain::Wait);
                eprintln!("[FUSED-PROFILE] {label}: {} ms", t0.elapsed().as_millis());
            }
        };

        // INLINE PICARD — comentado. El trainer siempre corre run_staged_adjoint_picard_no_readback
        // antes de llamar esta función. El staged Picard (staged_adjoint_picard.wgsl) escribe
        // correctamente a fused_mix_buf. El inline Picard escribía a cg_bridge.b_v_out (buffer
        // equivocado), dejando fused_mix_buf en cero → stage1a computaba gradiente cero.
        // if !use_precomputed_adjoint {
        //     run_stage(
        //         &self.device, &self.queue,
        //         "Picard Adjoint (BPTT)",
        //         &self.fused_adjoint_picard_pipeline,
        //         &self.fused_adjoint_bg,
        //         &self.fused_adjoint_weights_bg,
        //         seq_len, 1, profile_fused,
        //     );
        // }

        if hist_gated {
            let train_hist_carrier = std::env::var("AIDEEN_HIST_TRAIN_CARRIER")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                // Root-cause finding: even with consistent selective TBPTT, opening W_x/W_out
                // from step 0 lets the carrier learn against an interface that is still weak.
                // Keep the carrier frozen by default so the historical interface learns first;
                // explicit opt-in remains available via env override.
                .unwrap_or(false);
            let train_hist_wx = std::env::var("AIDEEN_HIST_TRAIN_WX")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(train_hist_carrier);
            let train_hist_wout = std::env::var("AIDEEN_HIST_TRAIN_WOUT")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                // Root-cause finding: reopening W_out together with W_x makes the memory-space
                // basis itself drift while the historical interface is still adapting. That is
                // a different failure mode from "carrier collapse", and it destabilizes long
                // runs even when the carrier magnitude is healthy. Keep W_out frozen by default
                // in the first carrier-reopen phase; explicit opt-in remains available.
                .unwrap_or(false);
            let train_hist_alog = std::env::var("AIDEEN_HIST_TRAIN_ALOG")
                .ok()
                .map(|v| {
                    let vl = v.trim().to_ascii_lowercase();
                    vl == "1" || vl == "true" || vl == "yes"
                })
                .unwrap_or(false);
            run_stage(
                &self.device,
                &self.queue,
                "hist_prep",
                &self.fused_update_hist_prep_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                n,
                1,
                profile_fused,
            );
            let train_hist_temporal = train_hist_carrier || train_hist_alog;
            if train_hist_temporal {
                run_stage(
                    &self.device,
                    &self.queue,
                    "hist_mprev",
                    &self.fused_update_hist_mprev_pipeline,
                    &self.fused_update_bg0,
                    &self.fused_update_bg1,
                    n,
                    1,
                    profile_fused,
                );
                run_stage(
                    &self.device,
                    &self.queue,
                    "hist_tbptt",
                    &self.fused_update_hist_tbptt_pipeline,
                    &self.fused_update_bg0,
                    &self.fused_update_bg1,
                    hs,
                    1,
                    profile_fused,
                );
                if hist_internal_probe {
                    let sample_t =
                        (seq_len as usize / 2).max(1).min(seq_len.saturating_sub(1) as usize);
                    let n_entries = seq_len as usize * self.config.h_slots;
                    let n_floats = n_entries * self.config.d_r;
                    let hist_rhs = self.read_storage_buffer(
                        &self.fused_hist_ctx_buf,
                        n_floats,
                        "Hist Probe RHS Readback",
                    );
                    let hist_delta = self.read_storage_buffer(
                        &self.fused_hist_delta_buf,
                        n_floats,
                        "Hist Probe Delta Readback",
                    );
                    let (rhs_mean, rhs_max, rhs_nz) = Self::summarize_token_block(
                        &hist_rhs,
                        sample_t,
                        self.config.h_slots,
                        self.config.d_r,
                    );
                    let (delta_mean, delta_max, delta_nz) = Self::summarize_token_block(
                        &hist_delta,
                        sample_t,
                        self.config.h_slots,
                        self.config.d_r,
                    );
                    eprintln!(
                        "[HIST-INTERNAL] pre-rerun sample_t={} rhs(mean/max/nz)={:.6e}/{:.6e}/{}/{} delta(mean/max/nz)={:.6e}/{:.6e}/{}/{}",
                        sample_t,
                        rhs_mean,
                        rhs_max,
                        rhs_nz,
                        self.config.h_slots * self.config.d_r,
                        delta_mean,
                        delta_max,
                        delta_nz,
                        self.config.h_slots * self.config.d_r,
                    );
                }
                let picard_t0 = std::time::Instant::now();
                self.run_staged_adjoint_picard_no_readback(seq_len, damping, 1, None, false)?;
                // The staged Picard helper reuses the same uniform buffer and writes a
                // solver-only UpdateUniforms payload with lr/grad_scale/weight_decay = 0.
                // All subsequent history/fused update kernels must see the original
                // training update params, otherwise HistParams/W_x/W_out/W_delta steps
                // collapse to exact zero despite nonzero gradients.
                self.queue.write_buffer(
                    &self.fused_update_params_buf,
                    0,
                    bytemuck::bytes_of(&params),
                );
                if profile_fused {
                    eprintln!(
                        "[FUSED-PROFILE] hist_picard_rerun: {} ms",
                        picard_t0.elapsed().as_millis()
                    );
                }
                if hist_internal_probe {
                    let sample_t =
                        (seq_len as usize / 2).max(1).min(seq_len.saturating_sub(1) as usize);
                    let rerun_gcomb = self.read_storage_buffer(
                        &self.fused_mix_buf,
                        seq_len as usize * self.config.h_slots * self.config.d_r,
                        "Hist Probe Rerun GComb Readback",
                    );
                    let (rerun_mean, rerun_max, rerun_nz) = Self::summarize_token_block(
                        &rerun_gcomb,
                        sample_t,
                        self.config.h_slots,
                        self.config.d_r,
                    );
                    eprintln!(
                        "[HIST-INTERNAL] post-rerun sample_t={} gcomb(mean/max/nz)={:.6e}/{:.6e}/{}/{}",
                        sample_t,
                        rerun_mean,
                        rerun_max,
                        rerun_nz,
                        self.config.h_slots * self.config.d_r,
                    );
                }
                run_stage(
                    &self.device,
                    &self.queue,
                    "hist_prep_final",
                    &self.fused_update_hist_prep_pipeline,
                    &self.fused_update_bg0,
                    &self.fused_update_bg1,
                    n,
                    1,
                    profile_fused,
                );
            }
            // Interface-only training must consume the projected historical context
            // directly from hist_prep. Running TBPTT first overwrites gmix_buf with
            // carrier gradients and changes the objective of the gate update.
            run_stage(
                &self.device,
                &self.queue,
                "hist_gate",
                &self.fused_update_hist_gate_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                hs.div_ceil(64),
                1,
                profile_fused,
            );
            // HistParams gradients are defined on the projected context produced by
            // hist_prep. They do not depend on the temporal TBPTT state and must be
            // computed before any carrier-stage kernels reuse the shared scratch.
            run_stage(
                &self.device,
                &self.queue,
                "hist_mat",
                &self.fused_update_hist_mat_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                d.div_ceil(16),
                profile_fused,
            );
            run_stage(
                &self.device,
                &self.queue,
                "hist_scale",
                &self.fused_update_hist_scale_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                hs.div_ceil(16),
                profile_fused,
            );
            if train_hist_temporal {
                run_stage(
                    &self.device,
                    &self.queue,
                    "hist_mprev_final",
                    &self.fused_update_hist_mprev_pipeline,
                    &self.fused_update_bg0,
                    &self.fused_update_bg1,
                    n,
                    1,
                    profile_fused,
                );
                run_stage(
                    &self.device,
                    &self.queue,
                    "hist_tbptt_final",
                    &self.fused_update_hist_tbptt_pipeline,
                    &self.fused_update_bg0,
                    &self.fused_update_bg1,
                    hs,
                    1,
                    profile_fused,
                );
                if train_hist_alog {
                    run_stage(
                        &self.device,
                        &self.queue,
                        "hist_alog",
                        &self.fused_update_hist_alog_pipeline,
                        &self.fused_update_bg0,
                        &self.fused_update_bg1,
                        d.div_ceil(64),
                        1,
                        profile_fused,
                    );
                }
                if train_hist_wout {
                    run_stage(
                        &self.device,
                        &self.queue,
                        "hist_wout",
                        &self.fused_update_hist_wout_pipeline,
                        &self.fused_update_bg0,
                        &self.fused_update_bg1,
                        d.div_ceil(16),
                        d.div_ceil(16),
                        profile_fused,
                    );
                }
                if train_hist_wx {
                    run_stage(
                        &self.device,
                        &self.queue,
                        "hist_wx",
                        &self.fused_update_hist_wx_pipeline,
                        &self.fused_update_bg0,
                        &self.fused_update_bg1,
                        d.div_ceil(16),
                        d.div_ceil(16),
                        profile_fused,
                    );
                    if hist_selective {
                        run_stage(
                            &self.device,
                            &self.queue,
                            "hist_wdelta",
                            &self.fused_update_hist_wdelta_pipeline,
                            &self.fused_update_bg0,
                            &self.fused_update_bg1,
                            d.div_ceil(16),
                            d.div_ceil(16),
                            profile_fused,
                        );
                        run_stage(
                            &self.device,
                            &self.queue,
                            "hist_bdelta",
                            &self.fused_update_hist_bdelta_pipeline,
                            &self.fused_update_bg0,
                            &self.fused_update_bg1,
                            d.div_ceil(64),
                            1,
                            profile_fused,
                        );
                    }
                }
            }
        }
        run_stage(
            &self.device,
            &self.queue,
            "stage1a",
            &self.fused_update_stage1a_pipeline,
            &self.fused_update_bg0,
            &self.fused_update_bg1,
            n,
            1,
            profile_fused,
        );
        run_stage(
            &self.device,
            &self.queue,
            "stage1b",
            &self.fused_update_stage1b_pipeline,
            &self.fused_update_bg0,
            &self.fused_update_bg1,
            n,
            1,
            profile_fused,
        );
        run_stage(
            &self.device,
            &self.queue,
            "stage2",
            &self.fused_update_stage2_pipeline,
            &self.fused_update_bg0,
            &self.fused_update_bg1,
            hs.div_ceil(16),
            n.div_ceil(16),
            profile_fused,
        );
        run_stage(
            &self.device,
            &self.queue,
            "stage3",
            &self.fused_update_stage3_pipeline,
            &self.fused_update_bg0,
            &self.fused_update_bg1,
            d.div_ceil(16),
            n.div_ceil(16),
            profile_fused,
        );
        run_stage(
            &self.device,
            &self.queue,
            "stage4",
            &self.fused_update_stage4_pipeline,
            &self.fused_update_bg0,
            &self.fused_update_bg1,
            d.div_ceil(16),
            d.div_ceil(16),
            profile_fused,
        );
        if !profile_fused {
            self.device.poll(wgpu::Maintain::Poll);
        }
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            &self.queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
        Ok(())
    }

    /// Spectral renormalization fully on GPU: power iteration on W_q..W_out,
    /// then syncs updated weights to the CG bridge.
    pub fn renormalize_spectral(&self) -> Result<(), String> {
        self.bridge.renormalize_spectral(
            &self.device,
            &self.queue,
            self.config.d_r as u32,
            0.10, // W_q/W_k/W_v/W_o/W_in: DEQ operator must stay contractive
            0.70, // W_x/W_out: temporal carrier, does not enter J_h of the DEQ solve
            12,
        );
        self.cg_bridge.sync_weights_from_deq_buffers(
            &self.device,
            &self.queue,
            &self.bridge.wq_buf,
            &self.bridge.wk_buf,
            &self.bridge.wv_buf,
            &self.bridge.wo_buf,
            &self.bridge.win_buf,
            &self.bridge.wx_buf,
            &self.bridge.wout_buf,
            &self.bridge.a_buf,
            &self.bridge.n_buf,
            self.config.d_r as u32,
        );
        Ok(())
    }

    pub fn read_weights(
        &self,
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
        self.bridge
            .read_weights(&self.device, &self.queue, self.config.d_r as u32)
    }

    pub fn read_debug_buffer(&self) -> Vec<f32> {
        self.bridge.read_debug_buffer(&self.device, &self.queue)
    }

    pub fn read_scratch_buffer(&self) -> Vec<f32> {
        self.bridge.read_scratch_buffer(&self.device, &self.queue)
    }

    pub fn read_cg_debug_buffer(&self) -> Vec<f32> {
        self.cg_bridge.read_debug_buffer(&self.device, &self.queue)
    }

    fn read_storage_buffer(&self, buffer: &wgpu::Buffer, n_floats: usize, label: &str) -> Vec<f32> {
        let byte_size = (n_floats * 4) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(label),
            });
        enc.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = rx.recv() {
            let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            out
        } else {
            vec![0.0; n_floats]
        }
    }

    fn summarize_token_block(
        vals: &[f32],
        token: usize,
        h_slots: usize,
        d: usize,
    ) -> (f32, f32, usize) {
        if vals.is_empty() || h_slots == 0 || d == 0 {
            return (0.0, 0.0, 0);
        }
        let base = token.saturating_mul(h_slots).saturating_mul(d);
        let end = (base + h_slots * d).min(vals.len());
        if base >= end {
            return (0.0, 0.0, 0);
        }
        let mut sum = 0.0f32;
        let mut max = 0.0f32;
        let mut nz = 0usize;
        for &v in &vals[base..end] {
            let a = v.abs();
            sum += a;
            max = max.max(a);
            if a > 1e-12 {
                nz += 1;
            }
        }
        let mean = sum / ((end - base).max(1) as f32);
        (mean, max, nz)
    }

    fn summarize_token_vec(vals: &[f32], token: usize, d: usize) -> (f32, f32, usize) {
        if vals.is_empty() || d == 0 {
            return (0.0, 0.0, 0);
        }
        let base = token.saturating_mul(d);
        let end = (base + d).min(vals.len());
        if base >= end {
            return (0.0, 0.0, 0);
        }
        let mut sum = 0.0f32;
        let mut max = 0.0f32;
        let mut nz = 0usize;
        for &v in &vals[base..end] {
            let a = v.abs();
            sum += a;
            max = max.max(a);
            if a > 1e-12 {
                nz += 1;
            }
        }
        let mean = sum / ((end - base).max(1) as f32);
        (mean, max, nz)
    }

    pub fn read_cg_v_out(&self, seq_len: u32) -> Vec<f32> {
        let n_floats = seq_len as usize * self.config.h_slots * self.config.d_r;
        self.read_storage_buffer(&self.cg_bridge.b_v_out, n_floats, "CG V_out Readback Staging")
    }

    pub fn read_dl_dh(&self, seq_len: u32) -> Vec<f32> {
        let n_floats = seq_len as usize * self.config.d_r;
        self.read_storage_buffer(&self.cg_bridge.b_dl, n_floats, "dl_dh Readback Staging")
    }

    /// Reads h_star (hnext_buf) from GPU for the first batch sample.
    /// Returns Vec<f32> of length h_slots * d_r. Used by CPU CG in strict path.
    pub fn read_hnext(&self) -> Vec<f32> {
        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;
        let n_floats = h_slots * d_r;
        self.read_storage_buffer(&self.bridge.hnext_buf, n_floats, "H_next Readback Staging")
    }

    /// Reads exact H*_t for the first batch sample across `seq_len` tokens.
    /// Layout matches `hnext_buf`: [token][slot][d].
    pub fn read_hnext_seq(&self, seq_len: u32) -> Vec<f32> {
        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;
        let n_floats = seq_len as usize * h_slots * d_r;
        self.read_storage_buffer(
            &self.bridge.hnext_buf,
            n_floats,
            "H_next Sequence Readback Staging",
        )
    }

    /// Reads pooled state (hpooled_buf) from GPU for the first batch sample.
    /// Returns Vec<f32> of length d_r.
    pub fn read_hpooled(&self) -> Vec<f32> {
        let d_r = self.config.d_r;
        self.read_storage_buffer(&self.bridge.hpooled_buf, d_r, "H_pooled Readback Staging")
    }

    pub fn read_staged_gcomb(&self, seq_len: u32) -> Vec<f32> {
        let n_floats = seq_len as usize * self.config.h_slots * self.config.d_r;
        self.read_storage_buffer(
            &self.fused_mix_buf,
            n_floats,
            "Staged Picard GComb Readback Staging",
        )
    }

    pub fn read_fixed_mamba_hist_grad(&self, seq_len: u32) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let gcomb = self.read_staged_gcomb(seq_len);
        let mut hist = vec![0.0f32; seq_len as usize * d];
        for t in 0..seq_len as usize {
            for s in 0..h_slots {
                let base = (t * h_slots + s) * d;
                for i in 0..d {
                    hist[t * d + i] += gcomb[base + i];
                }
            }
        }
        hist
    }

    pub fn read_hist_gated_ctx(&self, seq_len: u32) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let n_entries = seq_len as usize * h_slots;
        self.read_storage_buffer(
            &self.fused_hist_ctx_buf,
            n_entries * d,
            "Hist Gated Context Readback Staging",
        )
    }

    /// Reconstructs the forward historical context `c_{t,k}` used by the
    /// hist-gated DEQ branch from persisted scratch state and history params.
    /// This is the semantically correct debug signal for hist_gated; the
    /// fused_hist_ctx_buf storage is reused by backward and cannot be trusted
    /// as a forward-context probe after training steps.
    pub fn read_hist_gated_ctx_forward(&self, seq_len: u32) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let scratch = self.read_scratch_buffer();
        let hist_params = self.read_storage_buffer(
            &self.bridge.hist_params_buf,
            Self::history_params_len(d, h_slots),
            "Hist Gated Params Readback Staging",
        );
        let scratch_stride = d * (h_slots * 6 + 1) + h_slots * h_slots;
        let mamba_base = h_slots * 4 * d;
        let signal_base = mamba_base + h_slots * d;
        let hist_mat_len = d * d;
        let hist_scale_base = hist_mat_len;
        let hist_bias_base = hist_scale_base + h_slots * d;
        let hist_gate_base = hist_bias_base + h_slots * d;

        let mut out = vec![0.0f32; seq_len as usize * h_slots * d];
        for t in 0..seq_len as usize {
            let token_base = t * scratch_stride;
            let mut inj_sumsq = 0.0f32;
            for i in 0..d {
                let inj = scratch[token_base + signal_base + i];
                inj_sumsq += inj * inj;
            }
            let inj_rms = (inj_sumsq / d.max(1) as f32 + 1e-6).sqrt();
            for s in 0..h_slots {
                let gate_logit = hist_params[hist_gate_base + s];
                let alpha = 0.08 + 0.20 * (1.0 / (1.0 + (-gate_logit).exp()));
                let out_base = (t * h_slots + s) * d;
                if t == 0 {
                    continue;
                }
                let prev_base = (t - 1) * scratch_stride + mamba_base + s * d;
                let scale_base = hist_scale_base + s * d;
                let mut prev_sumsq = 0.0f32;
                for j in 0..d {
                    let prev = scratch[prev_base + j];
                    prev_sumsq += prev * prev;
                }
                let prev_rms = (prev_sumsq / d.max(1) as f32 + 1e-6).sqrt();
                let mut hist_sumsq = 0.0f32;
                for dim_out in 0..d {
                    let mut u =
                        hist_params[scale_base + dim_out] * (scratch[prev_base + dim_out] / prev_rms);
                    let row_base = dim_out * d;
                    for j in 0..d {
                        u += hist_params[row_base + j] * (scratch[prev_base + j] / prev_rms);
                    }
                    out[out_base + dim_out] = u;
                    hist_sumsq += u * u;
                }
                let hist_rms = (hist_sumsq / d.max(1) as f32 + 1e-6).sqrt();
                let tau = inj_rms;
                let hist_scale = (tau / hist_rms.max(1e-6)).min(1.0);
                for dim_out in 0..d {
                    out[out_base + dim_out] *= alpha * hist_scale;
                }
            }
        }
        out
    }

    pub fn read_hist_gate_alpha(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let hist_params = self.read_storage_buffer(
            &self.bridge.hist_params_buf,
            Self::history_params_len(d, h_slots),
            "Hist Gate Readback Staging",
        );
        let hist_gate_base = d * d + 2 * h_slots * d;
        (0..h_slots)
            .map(|slot| {
                let gate_logit = hist_params[hist_gate_base + slot];
                0.08 + 0.20 * (1.0 / (1.0 + (-gate_logit).exp()))
            })
            .collect()
    }

    pub fn read_hist_delta_signal(&self, seq_len: u32) -> Vec<f32> {
        let n_floats = seq_len as usize * self.config.h_slots * self.config.d_r;
        self.read_storage_buffer(
            &self.fused_hist_delta_buf,
            n_floats,
            "Hist Selective GPre Readback Staging",
        )
    }

    pub fn read_hist_qgrad_signal(&self, seq_len: u32) -> Vec<f32> {
        let n_floats = seq_len as usize * self.config.h_slots * self.config.d_r;
        self.read_storage_buffer(
            &self.fused_qgrad_buf,
            n_floats,
            "Hist Selective QGrad Readback Staging",
        )
    }

    pub fn read_hist_params_full(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        self.read_storage_buffer(
            &self.bridge.hist_params_buf,
            Self::history_params_len(d, h_slots),
            "Hist Params Full Readback Staging",
        )
    }

    pub fn read_hist_selective_param_stats(&self) -> ((f32, f32), (f32, f32), (f32, f32)) {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let hist_params = self.read_hist_params_full();
        let hist_mat_base = 0usize;
        let hist_mat_len = d * d;
        let slot_scale_base = hist_mat_base + hist_mat_len;
        let hist_bias_base = slot_scale_base + h_slots * d;
        let hist_gate_base = hist_bias_base + h_slots * d;
        let slot_anchor_base = hist_gate_base + h_slots;
        let w_delta_base = slot_anchor_base + h_slots * d;
        let b_delta_base = w_delta_base + d * d;
        let stats = |slice: &[f32]| -> (f32, f32) {
            if slice.is_empty() {
                return (0.0, 0.0);
            }
            let mean_abs = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
            let max_abs = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            (mean_abs, max_abs)
        };
        let w_hist_stats = stats(&hist_params[hist_mat_base..slot_scale_base]);
        let w_delta_stats = stats(&hist_params[w_delta_base..b_delta_base]);
        let b_delta_stats = stats(&hist_params[b_delta_base..b_delta_base + d]);
        (w_hist_stats, w_delta_stats, b_delta_stats)
    }

    pub fn read_hist_carrier_params_full(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let wx = self.read_storage_buffer(&self.bridge.wx_buf, d * d, "Hist Carrier W_x Readback");
        let wout =
            self.read_storage_buffer(&self.bridge.wout_buf, d * d, "Hist Carrier W_out Readback");
        let a_log = self.read_storage_buffer(&self.bridge.a_buf, d, "Hist Carrier ALog Readback");
        let mut out = Vec::with_capacity(wx.len() + wout.len() + a_log.len());
        out.extend_from_slice(&wx);
        out.extend_from_slice(&wout);
        out.extend_from_slice(&a_log);
        out
    }

    pub fn read_hist_carrier_param_stats(
        &self,
    ) -> ((f32, f32), (f32, f32), (f32, f32)) {
        let d = self.config.d_r;
        let full = self.read_hist_carrier_params_full();
        let wx_base = 0usize;
        let wout_base = wx_base + d * d;
        let a_base = wout_base + d * d;
        let stats = |slice: &[f32]| -> (f32, f32) {
            if slice.is_empty() {
                return (0.0, 0.0);
            }
            let mean_abs = slice.iter().map(|v| v.abs()).sum::<f32>() / slice.len() as f32;
            let max_abs = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            (mean_abs, max_abs)
        };
        (
            stats(&full[wx_base..wout_base]),
            stats(&full[wout_base..a_base]),
            stats(&full[a_base..a_base + d]),
        )
    }

    pub fn read_hist_selective_forward_stats(
        &self,
        seq_len: u32,
    ) -> ((f32, f32, usize), (f32, f32, f32)) {
        let a_floor = 0.10f32;
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        if seq_len == 0 || d == 0 || h_slots == 0 {
            return ((0.0, 0.0, 0), (0.0, 0.0, 0.0));
        }

        let h_seq = self.read_hnext_seq(seq_len);
        let hist_params = self.read_hist_params_full();
        let a_log = self.read_storage_buffer(
            &self.bridge.a_buf,
            d,
            "Hist Selective ALog Readback Staging",
        );

        let hist_mat_len = d * d;
        let slot_scale_base = hist_mat_len;
        let hist_bias_base = slot_scale_base + h_slots * d;
        let hist_gate_base = hist_bias_base + h_slots * d;
        let slot_anchor_base = hist_gate_base + h_slots;
        let w_delta_base = slot_anchor_base + h_slots * d;
        let b_delta_base = w_delta_base + d * d;

        let mut delta_sum = 0.0f32;
        let mut delta_max = 0.0f32;
        let mut delta_nz = 0usize;
        let mut a_sum = 0.0f32;
        let mut a_min = f32::INFINITY;
        let mut a_max = 0.0f32;
        let mut count = 0usize;
        for t in 0..seq_len as usize {
            for slot in 0..h_slots {
                let base = (t * h_slots + slot) * d;
                let mut h_sumsq = 0.0f32;
                for dim in 0..d {
                    let h = h_seq[base + dim];
                    h_sumsq += h * h;
                }
                let h_rms = (h_sumsq / d.max(1) as f32 + 1e-6).sqrt();
                for dim in 0..d {
                    let mut delta_pre = hist_params[b_delta_base + dim];
                    for j in 0..d {
                        delta_pre += hist_params[w_delta_base + j * d + dim]
                            * (h_seq[base + j] / h_rms);
                    }
                    let delta = (1.0 + delta_pre.exp()).ln();
                    let delta_abs = delta.abs();
                    delta_sum += delta_abs;
                    delta_max = delta_max.max(delta_abs);
                    if delta_abs > 1e-12 {
                        delta_nz += 1;
                    }
                    let a_base = 1.0 / (1.0 + a_log[dim].exp());
                    let a_core = (delta * a_base.max(1.0e-6).ln()).exp();
                    let a_t = a_floor + (1.0 - a_floor) * a_core;
                    a_sum += a_t;
                    a_min = a_min.min(a_t);
                    a_max = a_max.max(a_t);
                    count += 1;
                }
            }
        }
        if count == 0 {
            return ((0.0, 0.0, 0), (0.0, 0.0, 0.0));
        }
        (
            (delta_sum / count as f32, delta_max, delta_nz),
            (a_sum / count as f32, a_min, a_max),
        )
    }

    pub fn read_fixed_mamba_state_rms(&self, seq_len: u32) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let scratch = self.read_scratch_buffer();
        let scratch_stride = d * (h_slots * 6 + 1) + h_slots * h_slots;
        let mamba_base = h_slots * 4 * d;
        let mut rms = vec![0.0f32; seq_len as usize];
        for t in 0..seq_len as usize {
            let base = t * scratch_stride + mamba_base;
            let mut sumsq = 0.0f32;
            let mut count = 0usize;
            for i in 0..(h_slots * d) {
                let v = scratch[base + i];
                sumsq += v * v;
                count += 1;
            }
            rms[t] = (sumsq / count.max(1) as f32 + 1e-12).sqrt();
        }
        rms
    }

    pub fn run_forward(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        damping: f32,
        epsilon: f32,
    ) -> Result<(), &'static str> {
        let shape = aideen_block::deq_bridge::DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32, // ✅ consistencia total con el pipeline
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            residual_alpha: Self::residual_alpha_from_env(),
        };

        self.bridge
            .run_forward_gpu_only(&self.device, &self.queue, &shape);
        Ok(())
    }
}
