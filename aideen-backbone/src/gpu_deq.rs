use aideen_block::deq_bridge::{
    aw_alog_byte_off, aw_hist_byte_off, aw_nscale_byte_off, aw_total_bytes, aw_win_byte_off,
    aw_wk_byte_off, aw_wo_byte_off, aw_wout_byte_off, aw_wv_byte_off, aw_wx_byte_off,
    DeqComputeShape, RustDeqBridge,
};
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
    grad_accum_mode: u32, // 0=direct update, 1=accumulate into AllGradients
    n_accum: u32,         // accumulation count (for apply_grad_update_main)
    n_total_weights: u32, // total AllWeights elements (for apply_grad_update_main)
    batch_size: u32,      // number of sequences processed in parallel
    apply_accum: u32,     // 1=apply accumulated gradients (apply_grad_update_main)
    _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct AndersonParams {
    m: u32,
    k: u32,
    slots_per_segment: u32,
    _pad0: u32,
}

/// Minimal buffers needed by the Picard adjoint path (replaces the old RustCgBridge).
pub struct AdjointBuffers {
    pub b_dl: wgpu::Buffer, // ∂L/∂h input for the Picard adjoint (size: ctx_len × d_model)
    pub b_v_out: wgpu::Buffer, // adjoint state v output       (size: ctx_len × h_slots × d_model)
}

/// Abstracción del DEQ vía GPU (WGPU).
pub struct GpuDeqBackend {
    pub config: ArchitectureConfig,
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub bridge: RustDeqBridge,
    pub adj_bufs: AdjointBuffers,
    tps_timestamp_enabled: bool,
    tps_timestamp_period: f32,
    tps_timestamp_query: Option<wgpu::QuerySet>,
    tps_timestamp_resolve_buf: Option<wgpu::Buffer>,
    tps_timestamp_readback_buf: Option<wgpu::Buffer>,

    // Fused update pipeline
    staged_picard_init_pipeline: wgpu::ComputePipeline,
    staged_picard_gcomb_pipeline: wgpu::ComputePipeline,
    staged_picard_gmix_pipeline: wgpu::ComputePipeline,
    staged_picard_gmix_gscore_pipeline: wgpu::ComputePipeline,
    staged_picard_gscore_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_base_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_opt_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_v_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_k_pipeline: wgpu::ComputePipeline,
    staged_picard_accum_q_pipeline: wgpu::ComputePipeline,
    fused_update_stage1a_pipeline: wgpu::ComputePipeline,
    fused_update_stage1b_pipeline: wgpu::ComputePipeline,
    fused_update_stage2_pipeline: wgpu::ComputePipeline,
    fused_update_stage3_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_wo_win_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_wq_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_prep_wk_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_prep_wv_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_wk_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_wv_pipeline: wgpu::ComputePipeline,
    fused_update_stage4_bias_pipeline: wgpu::ComputePipeline,
    fused_update_hist_prep_pipeline: wgpu::ComputePipeline,
    fused_update_hist_mat_pipeline: wgpu::ComputePipeline,
    fused_update_hist_scale_pipeline: wgpu::ComputePipeline,
    fused_update_hist_gate_pipeline: wgpu::ComputePipeline,
    fused_update_hist_mprev_pipeline: wgpu::ComputePipeline,
    fused_update_hist_tbptt_pipeline: wgpu::ComputePipeline,
    fused_update_hist_wout_pipeline: wgpu::ComputePipeline,
    fused_update_hist_wx_pipeline: wgpu::ComputePipeline,
    fused_update_hist_alog_pipeline: wgpu::ComputePipeline,
    fused_update_hist_wdelta_pipeline: wgpu::ComputePipeline,
    fused_update_hist_bdelta_pipeline: wgpu::ComputePipeline,
    fused_update_hist_wgate_pipeline: wgpu::ComputePipeline,
    fused_update_hist_forget_pipeline: wgpu::ComputePipeline,
    fused_update_apply_grad_pipeline: wgpu::ComputePipeline,
    staged_picard_bg: wgpu::BindGroup,
    staged_picard_bg_alt: wgpu::BindGroup,
    staged_picard_bg1: wgpu::BindGroup,
    fused_update_bg0: wgpu::BindGroup,
    fused_update_bg1: wgpu::BindGroup,
    pub fused_update_params_buf: wgpu::Buffer,
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
    tbptt_carry_buf: wgpu::Buffer,
    pub all_gradients_buf: wgpu::Buffer,

    // Anderson acceleration for adjoint Picard
    anderson_m: u32, // ring buffer depth (effective window = m-1)
    anderson_hist_bufs: Vec<wgpu::Buffer>, // segmented ring buffer slots
    anderson_slots_per_segment: u32,
    anderson_params_bufs: Vec<wgpu::Buffer>, // one 16-byte uniform per iteration slot
    anderson_bgs: Vec<wgpu::BindGroup>,      // one bind group per iteration slot
    anderson_store_pipeline: wgpu::ComputePipeline,
    anderson_mix_pipeline: wgpu::ComputePipeline,

    // --- Hot-path cached env vars (parsed once at construction, not per training step) ---
    // Avoids ~52 syscalls/step from std::env::var calls in apply_fused_deq_update and adjoint.
    pub cached_residual_alpha: f32,
    cfg_hist_gated: bool,
    cfg_hist_selective: bool,
    cfg_slot_anchor_zero: bool,
    cfg_rms_floor: f32,
    cfg_contr_floor: f32,
    cfg_hist_zero: bool,
    cfg_hist_minner_zero: bool,
    cfg_hist_force_nomamba: bool,
    cfg_hist_prelude_skip: bool,
    cfg_hist_loop_force_nomamba: bool,
    cfg_signal_zero: bool,
    cfg_attn_out_mode: f32,
    cfg_attn_uniform: bool,
    cfg_attn_freeze: bool,
    cfg_v_fixed: bool,
    cfg_v_lag: bool,
    cfg_v_scale: f32,
    cfg_signal_scale: f32,
    cfg_v_gate_scale: f32,
    cfg_v_gate_bias: f32,
    cfg_v_norm: bool,
    cfg_v_norm_scale: f32,
    cfg_hist_train_carrier: bool,
    cfg_hist_train_wx: bool,
    cfg_hist_train_wout: bool,
    cfg_hist_train_alog: bool,
    cfg_hist_train_delta: bool,
    cfg_hist_internal_probe: bool,
    cfg_fused_profile: bool,
    cfg_picard_profile: bool,
    cfg_picard_stage_profile: bool,
    cfg_picard_accum_split: bool,
    cfg_picard_internal_probe: bool,
    cfg_picard_gscore_fused: bool,
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
        // wq/wk may include per-slot bias appended after d*d matrix — use only matrix portion.
        let h = self.config.h_slots;
        let wq_t = Self::transpose_square(&wq[..d * d], d);
        let wk_t = Self::transpose_square(&wk[..d * d], d);
        // W_v is now per-slot: h_slots blocks of d×d, each must be transposed separately.
        let mut wv_t = Vec::with_capacity(h * d * d);
        for s in 0..h {
            wv_t.extend_from_slice(&Self::transpose_square(&wv[s * d * d..(s + 1) * d * d], d));
        }
        // W_o is per-slot: h_slots blocks of d×d, each must be transposed separately.
        let mut wo_t = Vec::with_capacity(h * d * d);
        for s in 0..h {
            wo_t.extend_from_slice(&Self::transpose_square(&wo[s * d * d..(s + 1) * d * d], d));
        }
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
        w_gate_hist: &[f32],
        w_forget: &[f32],
        b_forget: &[f32],
    ) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let mut out = Vec::with_capacity(Self::history_params_len(
            self.config.d_r,
            self.config.h_slots,
        ));
        out.extend_from_slice(w_hist_shared);
        out.extend_from_slice(hist_slot_scale);
        out.extend_from_slice(hist_slot_bias);
        out.extend_from_slice(hist_gate_logit);
        if self.cfg_slot_anchor_zero {
            out.extend(std::iter::repeat(0.0).take(h * d));
        } else {
            out.extend_from_slice(slot_anchor);
        }
        out.extend_from_slice(w_delta);
        out.extend_from_slice(b_delta);
        out.push(if self.cfg_hist_selective { 1.0 } else { 0.0 });
        // Warmup factor for alpha_min (0..1). Default to 1.0 so inference is unaffected.
        out.push(1.0);
        out.push(self.cfg_rms_floor);
        out.push(self.cfg_contr_floor);
        out.push(if self.cfg_hist_zero { 0.0 } else { 1.0 });
        out.push(if self.cfg_hist_minner_zero { 1.0 } else { 0.0 });
        out.push(if self.cfg_hist_force_nomamba {
            1.0
        } else {
            0.0
        });
        out.push(if self.cfg_hist_prelude_skip { 1.0 } else { 0.0 });
        out.push(if self.cfg_hist_loop_force_nomamba {
            1.0
        } else {
            0.0
        });
        out.push(if self.cfg_signal_zero { 1.0 } else { 0.0 });
        out.push(self.cfg_attn_out_mode);
        out.push(if self.cfg_attn_uniform { 1.0 } else { 0.0 });
        out.push(if self.cfg_attn_freeze { 1.0 } else { 0.0 });
        out.push(if self.cfg_v_fixed { 1.0 } else { 0.0 });
        out.push(if self.cfg_v_lag { 1.0 } else { 0.0 });
        out.push(self.cfg_v_scale);
        out.push(self.cfg_signal_scale);
        out.push(self.cfg_v_gate_scale);
        out.push(self.cfg_v_gate_bias);
        out.push(if self.cfg_v_norm { 1.0 } else { 0.0 });
        out.push(self.cfg_v_norm_scale);
        // W_gate_hist: h_slots × d_model dynamic gate query matrix (after 21 scalars)
        out.extend_from_slice(w_gate_hist);
        // Forget gate parameters: W_forget (h×d) and b_forget (h)
        out.extend_from_slice(w_forget);
        out.extend_from_slice(b_forget);
        out
    }

    fn history_params_len(d: usize, h: usize) -> usize {
        (h + 1) * d * d + 5 * h * d + 2 * h + d + 21
    }

    pub fn set_hist_warmup_factor(&self, factor: f32) {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let base = (h + 1) * d * d + 3 * h * d + h + d;
        let idx = base + 1; // warmup is the second scalar after hist_selective flag
        let d64 = d as u64;
        let h64 = h as u64;
        let hist_off = aw_hist_byte_off(d64, h64) + (idx * 4) as u64;
        self.queue.write_buffer(
            &self.bridge.all_weights_buf,
            hist_off,
            bytemuck::bytes_of(&factor),
        );
    }

    fn hist_selective_from_env() -> bool {
        std::env::var("AIDEEN_HIST_SELECTIVE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(true)
    }

    fn env_f32(name: &str) -> Option<f32> {
        std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok())
    }

    fn env_u32(name: &str) -> Option<u32> {
        std::env::var(name)
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
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
        if std::env::var("AIDEEN_DEQ_FIXED_MAMBA").ok().as_deref() == Some("1") {
            return -0.25;
        }

        // hist_gated (-0.5) is the default mode. Override only if another mode is set.
        let alpha = std::env::var("AIDEEN_DEQ_RESIDUAL_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or_else(|| {
                // Default: hist_gated mode (residual_alpha = -0.5 sentinel).
                -0.5
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
        let wants_timestamps = std::env::var("AIDEEN_TPS_GPU_TIMESTAMPS")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        let adapter_features = adapter.features();
        let timestamps_supported = adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let tps_timestamp_enabled = wants_timestamps && timestamps_supported;
        let mut required_features = wgpu::Features::SUBGROUP;
        if tps_timestamp_enabled {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        } else if wants_timestamps {
            eprintln!(
                "[GpuDeqBackend] TIMESTAMP_QUERY not supported or missing TIMESTAMP_QUERY_INSIDE_ENCODERS."
            );
        }
        let mut limits = adapter.limits();
        limits.max_storage_buffers_per_shader_stage = 16;

        // 3. Crear Device
        let (device, queue) = match adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("AIDEEN DEQ GPU"),
                    required_features,
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

        let (
            tps_timestamp_enabled,
            tps_timestamp_period,
            tps_timestamp_query,
            tps_timestamp_resolve_buf,
            tps_timestamp_readback_buf,
        ) = if tps_timestamp_enabled {
            let query = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("AIDEEN TPS Timestamp QuerySet"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("AIDEEN TPS Timestamp Resolve"),
                size: 2 * std::mem::size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("AIDEEN TPS Timestamp Readback"),
                size: 2 * std::mem::size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (
                true,
                queue.get_timestamp_period(),
                Some(query),
                Some(resolve),
                Some(readback),
            )
        } else {
            (false, 0.0, None, None, None)
        };

        // Read batch_size from env — scales all forward+backward buffers.
        let forward_batch_cap = std::env::var("AIDEEN_BATCH_SIZE")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or(1)
            .max(1);
        let forward_seq_cap = config.ctx_len.max(1) as u32;
        // RustDeqBridge::new(device, d_model, h_slots, max_batch_size, max_seq_len)
        let bridge = RustDeqBridge::new(
            &device,
            config.d_r as u32,
            config.h_slots as u32,
            forward_batch_cap,
            forward_seq_cap,
        );

        // AdjointBuffers: only the two buffers actually used by the Picard adjoint.
        let adj_batch = forward_batch_cap as u64;
        let adj_seq_cap = config.ctx_len.max(1) as u64;
        let adj_d = config.d_r as u64;
        let adj_h = config.h_slots as u64;
        let adj_dl_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjoint b_dl"),
            size: adj_batch * adj_seq_cap * adj_d * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let adj_v_out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Adjoint b_v_out"),
            size: adj_batch * adj_seq_cap * adj_h * adj_d * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Fused Update Pipeline setup
        let update_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused DEQ Update Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fused_deq_update.wgsl").into()),
        });
        let staged_picard_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Staged Picard Adjoint Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/staged_adjoint_picard.wgsl").into(),
            ),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            ],
        });


        // bg1_layout: 10 read_write bindings — used by staged_picard_pl (staged_adjoint_picard.wgsl)
        // fused_update_bg1_layout: 1 binding — AllWeights for fused_deq_update.wgsl @group(1)
        let fused_update_bg1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fused Update BG1 Layout (AllWeights)"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bg1_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Staged Picard BG1 Layout (Weights)"),
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
            bind_group_layouts: &[&bg0_layout, &fused_update_bg1_layout],
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
        let fused_update_stage4_wo_win_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 WO/WIN Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_wo_win_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage4_wq_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 WQ Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_wq_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage4_prep_wk_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 Prep WK Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_prep_wk_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage4_prep_wv_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 Prep WV Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_prep_wv_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage4_wk_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 WK Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_wk_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage4_wv_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 WV Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_wv_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_stage4_bias_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Stage4 Bias Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_attn_stage4_bias_main"),
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
        let fused_update_hist_wgate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist WGate Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_wgate_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_hist_forget_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Update Hist Forget Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("fused_hist_stage_forget_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fused_update_apply_grad_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Fused DEQ Apply Gradient Pipeline"),
                layout: Some(&pl),
                module: &update_shader,
                entry_point: Some("apply_grad_update_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        let staged_picard_gmix_gscore_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard GMix+GScore Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_gmix_gscore_main"),
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
        let staged_picard_accum_base_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard Accum Base Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_accum_base_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let staged_picard_accum_opt_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Staged Picard Accum Opt Pipeline"),
                layout: Some(&staged_picard_pl),
                module: &staged_picard_shader,
                entry_point: Some("picard_accum_opt_main"),
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
        // Anderson acceleration setup
        let anderson_m_val = Self::env_u32("AIDEEN_ANDERSON_M").unwrap_or(4);
        let anderson_bg1_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Anderson BG1 Layout"),
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
                ],
            });
        let anderson_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Anderson PL"),
            bind_group_layouts: &[&bg0_layout, &anderson_bg1_layout],
            push_constant_ranges: &[],
        });
        // Ring buffer: m × max_attn_len floats (segment into <=4 buffers to stay under buffer limits)
        let anderson_attn_len_max = (forward_batch_cap as usize).max(1)
            * config.ctx_len.max(1)
            * config.h_slots
            * config.d_r;
        let anderson_m_alloc = anderson_m_val.max(1);
        let max_storage_bytes = device.limits().max_storage_buffer_binding_size as usize;
        let slot_bytes = anderson_attn_len_max * 4;
        if slot_bytes > max_storage_bytes {
            return None;
        }
        let slots_per_segment = (max_storage_bytes / slot_bytes).max(1);
        let segment_count = (anderson_m_alloc as usize + slots_per_segment - 1) / slots_per_segment;
        if segment_count > 4 {
            return None;
        }
        let mut anderson_hist_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(4);
        for seg in 0..4 {
            let slots_in_seg = if seg < segment_count {
                slots_per_segment
            } else {
                1
            };
            let size = (slots_in_seg * slot_bytes) as u64;
            anderson_hist_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Anderson Hist Buffer seg {}", seg)),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        const ANDERSON_MAX_ITERS: usize = 16;
        let mut anderson_params_bufs: Vec<wgpu::Buffer> = Vec::with_capacity(ANDERSON_MAX_ITERS);
        let mut anderson_bgs: Vec<wgpu::BindGroup> = Vec::with_capacity(ANDERSON_MAX_ITERS);
        for k in 0..ANDERSON_MAX_ITERS {
            let pbuf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Anderson Params k={}", k)),
                size: std::mem::size_of::<AndersonParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Anderson BG k={}", k)),
                layout: &anderson_bg1_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: pbuf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: anderson_hist_bufs[0].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: anderson_hist_bufs[1].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: anderson_hist_bufs[2].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: anderson_hist_bufs[3].as_entire_binding(),
                    },
                ],
            });
            anderson_params_bufs.push(pbuf);
            anderson_bgs.push(bg);
        }
        let anderson_store_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Anderson Store Pipeline"),
                layout: Some(&anderson_pl),
                module: &staged_picard_shader,
                entry_point: Some("anderson_store_main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let anderson_mix_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Anderson Mix Pipeline"),
                layout: Some(&anderson_pl),
                module: &staged_picard_shader,
                entry_point: Some("anderson_mix_main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let fused_update_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fused Update Params"),
            size: std::mem::size_of::<UpdateUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let attn_entries =
            (forward_batch_cap as usize * config.ctx_len.max(1) * config.h_slots * config.d_r)
                as u64;
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
        let gscore_entries =
            (forward_batch_cap as usize * config.ctx_len.max(1) * config.h_slots * config.h_slots)
                as u64;
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
        let w_mat_bytes = (config.d_r * config.d_r * std::mem::size_of::<f32>()) as u64;
        let wv_mat_bytes =
            (config.h_slots * config.d_r * config.d_r * std::mem::size_of::<f32>()) as u64;
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
            size: wv_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staged_wo_t_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staged Picard Wo^T Buffer"),
            size: wv_mat_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tbptt_carry_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TBPTT Carry Buffer"),
            size: (forward_batch_cap as u64) * (config.h_slots as u64) * (config.d_r as u64) * 4u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // AllGradients: same layout as AllWeights; accumulates raw gradients across
        // n_accum forward passes in mode=1. Zero-initialized (GPU zero-inits storage buffers).
        let ag_d64 = config.d_r as u64;
        let ag_h64 = config.h_slots as u64;
        let ag_hist_len = (config.h_slots + 1) * config.d_r * config.d_r
            + 5 * config.h_slots * config.d_r
            + 2 * config.h_slots
            + config.d_r
            + 21;
        let all_gradients_size = aw_total_bytes(ag_d64, ag_h64, ag_hist_len as u64);
        let all_gradients_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AllGradients"),
            size: all_gradients_size,
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
                    resource: adj_v_out_buf.as_entire_binding(),
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
                    resource: adj_dl_buf.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: tbptt_carry_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: all_gradients_buf.as_entire_binding(),
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
                    resource: adj_v_out_buf.as_entire_binding(),
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
                    resource: adj_dl_buf.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: tbptt_carry_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: all_gradients_buf.as_entire_binding(),
                },
            ],
        });
        let staged_picard_bg_alt = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Staged Picard BG0 Alt"),
            layout: &bg0_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fused_update_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: fused_weighted_h_buf.as_entire_binding(),
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
                    resource: adj_dl_buf.as_entire_binding(),
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
                    resource: adj_v_out_buf.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: tbptt_carry_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: all_gradients_buf.as_entire_binding(),
                },
            ],
        });
        let d64 = config.d_r as u64;
        let h64 = config.h_slots as u64;
        let aw_mat_sz = std::num::NonZeroU64::new(d64 * d64 * 4);
        let aw_win_sz = std::num::NonZeroU64::new(h64 * d64 * d64 * 4);
        let aw_alog_sz = std::num::NonZeroU64::new(h64 * d64 * 4);
        let aw_nscale_sz = std::num::NonZeroU64::new(d64 * 4);
        let aw_hist_len = (config.h_slots + 1) * config.d_r * config.d_r
            + 5 * config.h_slots * config.d_r
            + 2 * config.h_slots
            + config.d_r
            + 21;
        let aw_hist_sz = std::num::NonZeroU64::new(aw_hist_len as u64 * 4);
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
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bridge.all_weights_buf,
                        offset: aw_win_byte_off(d64, h64),
                        size: aw_win_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bridge.all_weights_buf,
                        offset: aw_wx_byte_off(d64, h64),
                        size: aw_mat_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bridge.all_weights_buf,
                        offset: aw_wout_byte_off(d64, h64),
                        size: aw_mat_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bridge.all_weights_buf,
                        offset: aw_alog_byte_off(d64, h64),
                        size: aw_alog_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bridge.all_weights_buf,
                        offset: aw_nscale_byte_off(d64, h64),
                        size: aw_nscale_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &bridge.all_weights_buf,
                        offset: aw_hist_byte_off(d64, h64),
                        size: aw_hist_sz,
                    }),
                },
            ],
        });

        let fused_update_bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fused Update BG1 (AllWeights)"),
            layout: &fused_update_bg1_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: bridge.all_weights_buf.as_entire_binding(),
            }],
        });
        // Parse all hot-path env vars once at construction.
        let cached_residual_alpha = Self::residual_alpha_from_env();
        let cfg_hist_gated = std::env::var("AIDEEN_DEQ_HIST_GATED").ok().as_deref() != Some("0");
        let cfg_hist_selective = Self::hist_selective_from_env();
        let bool_flag = |s: &str| -> Option<bool> {
            std::env::var(s).ok().map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
        };
        let f32_flag =
            |s: &str| -> Option<f32> { std::env::var(s).ok().and_then(|v| v.parse::<f32>().ok()) };
        let cfg_hist_train_carrier = bool_flag("AIDEEN_HIST_TRAIN_CARRIER").unwrap_or(false);
        let cfg_hist_train_wx = bool_flag("AIDEEN_HIST_TRAIN_WX").unwrap_or(cfg_hist_train_carrier);
        let cfg_hist_train_wout = bool_flag("AIDEEN_HIST_TRAIN_WOUT").unwrap_or(false);
        let cfg_hist_train_alog = bool_flag("AIDEEN_HIST_TRAIN_ALOG").unwrap_or(cfg_hist_selective);
        let cfg_hist_train_delta =
            bool_flag("AIDEEN_HIST_TRAIN_DELTA").unwrap_or(cfg_hist_selective);
        let cfg_hist_internal_probe = bool_flag("AIDEEN_HIST_INTERNAL_PROBE").unwrap_or(false);
        let cfg_fused_profile = bool_flag("AIDEEN_FUSED_PROFILE").unwrap_or(false);
        let cfg_slot_anchor_zero = bool_flag("AIDEEN_DEQ_SLOT_ANCHOR_ZERO").unwrap_or(false);
        let cfg_rms_floor = f32_flag("AIDEEN_DEQ_RMS_FLOOR").unwrap_or(0.0).max(0.0);
        let cfg_contr_floor = f32_flag("AIDEEN_DEQ_CONTR_RMS_FLOOR")
            .unwrap_or(0.0)
            .max(0.0);
        let cfg_hist_zero = bool_flag("AIDEEN_DEQ_HIST_ZERO").unwrap_or(false);
        let cfg_hist_minner_zero = bool_flag("AIDEEN_DEQ_HIST_MINNER_ZERO").unwrap_or(false);
        let cfg_hist_force_nomamba = bool_flag("AIDEEN_DEQ_HIST_FORCE_NOMAMBA").unwrap_or(false);
        let cfg_hist_prelude_skip = bool_flag("AIDEEN_DEQ_HIST_PRELUDE_SKIP").unwrap_or(false);
        let cfg_hist_loop_force_nomamba =
            bool_flag("AIDEEN_DEQ_HIST_LOOP_FORCE_NOMAMBA").unwrap_or(false);
        let cfg_signal_zero = bool_flag("AIDEEN_DEQ_SIGNAL_ZERO").unwrap_or(false);
        let cfg_attn_out_mode = f32_flag("AIDEEN_DEQ_ATTN_OUT_MODE")
            .unwrap_or(0.0)
            .clamp(0.0, 2.0);
        let cfg_attn_uniform = bool_flag("AIDEEN_DEQ_ATTN_UNIFORM").unwrap_or(false);
        let cfg_attn_freeze = bool_flag("AIDEEN_DEQ_ATTN_FREEZE").unwrap_or(false);
        let cfg_v_fixed = bool_flag("AIDEEN_DEQ_V_FIXED").unwrap_or(false);
        let cfg_v_lag = bool_flag("AIDEEN_DEQ_V_LAG").unwrap_or(false);
        let cfg_v_scale = f32_flag("AIDEEN_DEQ_V_SCALE")
            .unwrap_or(1.0)
            .clamp(0.01, 10.0);
        let cfg_signal_scale = f32_flag("AIDEEN_DEQ_SIGNAL_SCALE")
            .unwrap_or(1.0)
            .clamp(0.01, 10.0);
        let cfg_v_gate_scale = f32_flag("AIDEEN_DEQ_V_GATE_SCALE").unwrap_or(0.0);
        let cfg_v_gate_bias = f32_flag("AIDEEN_DEQ_V_GATE_BIAS").unwrap_or(0.0);
        let cfg_v_norm = bool_flag("AIDEEN_DEQ_V_NORM").unwrap_or(false);
        let cfg_v_norm_scale = f32_flag("AIDEEN_DEQ_V_NORM_SCALE").unwrap_or(1.0);
        let cfg_picard_profile = bool_flag("AIDEEN_PICARD_PROFILE").unwrap_or(false);
        let cfg_picard_stage_profile = bool_flag("AIDEEN_PICARD_STAGE_PROFILE").unwrap_or(false);
        let cfg_picard_accum_split = bool_flag("AIDEEN_PICARD_ACCUM_SPLIT")
            .or_else(|| bool_flag("AIDEEN_PICARD_ACCUM_SPLIT_PROFILE"))
            .unwrap_or(false);
        let cfg_picard_internal_probe = bool_flag("AIDEEN_PICARD_INTERNAL_PROBE").unwrap_or(false);
        let cfg_picard_gscore_fused = bool_flag("AIDEEN_PICARD_GSCORE_FUSED").unwrap_or(false);

        Some(Self {
            config,
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            bridge,
            adj_bufs: AdjointBuffers {
                b_dl: adj_dl_buf,
                b_v_out: adj_v_out_buf,
            },
            tps_timestamp_enabled,
            tps_timestamp_period,
            tps_timestamp_query,
            tps_timestamp_resolve_buf,
            tps_timestamp_readback_buf,
            staged_picard_init_pipeline,
            staged_picard_gcomb_pipeline,
            staged_picard_gmix_pipeline,
            staged_picard_gmix_gscore_pipeline,
            staged_picard_gscore_pipeline,
            staged_picard_accum_base_pipeline,
            staged_picard_accum_opt_pipeline,
            staged_picard_accum_pipeline,
            staged_picard_accum_v_pipeline,
            staged_picard_accum_k_pipeline,
            staged_picard_accum_q_pipeline,
            fused_update_stage1a_pipeline,
            fused_update_stage1b_pipeline,
            fused_update_stage2_pipeline,
            fused_update_stage3_pipeline,
            fused_update_stage4_wo_win_pipeline,
            fused_update_stage4_wq_pipeline,
            fused_update_stage4_prep_wk_pipeline,
            fused_update_stage4_prep_wv_pipeline,
            fused_update_stage4_wk_pipeline,
            fused_update_stage4_wv_pipeline,
            fused_update_stage4_bias_pipeline,
            fused_update_hist_prep_pipeline,
            fused_update_hist_mat_pipeline,
            fused_update_hist_scale_pipeline,
            fused_update_hist_gate_pipeline,
            fused_update_hist_mprev_pipeline,
            fused_update_hist_tbptt_pipeline,
            fused_update_hist_wout_pipeline,
            fused_update_hist_wx_pipeline,
            fused_update_hist_alog_pipeline,
            fused_update_hist_wdelta_pipeline,
            fused_update_hist_bdelta_pipeline,
            fused_update_hist_wgate_pipeline,
            fused_update_hist_forget_pipeline,
            fused_update_apply_grad_pipeline,
            staged_picard_bg,
            staged_picard_bg_alt,
            staged_picard_bg1,
            fused_update_bg0,
            fused_update_bg1,
            fused_update_params_buf,
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
            tbptt_carry_buf,
            all_gradients_buf,
            anderson_m: anderson_m_val,
            anderson_hist_bufs,
            anderson_slots_per_segment: slots_per_segment as u32,
            anderson_params_bufs,
            anderson_bgs,
            anderson_store_pipeline,
            anderson_mix_pipeline,
            cached_residual_alpha,
            cfg_hist_gated,
            cfg_hist_selective,
            cfg_slot_anchor_zero,
            cfg_rms_floor,
            cfg_contr_floor,
            cfg_hist_zero,
            cfg_hist_minner_zero,
            cfg_hist_force_nomamba,
            cfg_hist_prelude_skip,
            cfg_hist_loop_force_nomamba,
            cfg_signal_zero,
            cfg_attn_out_mode,
            cfg_attn_uniform,
            cfg_attn_freeze,
            cfg_v_fixed,
            cfg_v_lag,
            cfg_v_scale,
            cfg_signal_scale,
            cfg_v_gate_scale,
            cfg_v_gate_bias,
            cfg_v_norm,
            cfg_v_norm_scale,
            cfg_hist_train_carrier,
            cfg_hist_train_wx,
            cfg_hist_train_wout,
            cfg_hist_train_alog,
            cfg_hist_train_delta,
            cfg_hist_internal_probe,
            cfg_fused_profile,
            cfg_picard_profile,
            cfg_picard_stage_profile,
            cfg_picard_accum_split,
            cfg_picard_internal_probe,
            cfg_picard_gscore_fused,
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
        // Clear TBPTT carry on document reset.
        // M_carry lives in second half of hcurr_buf — already cleared above.
        encoder.clear_buffer(&self.tbptt_carry_buf, 0, None);
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
        debug_enable: bool,
    ) -> DeqComputeShape {
        DeqComputeShape {
            batch_size,
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            max_iters,
            epsilon,
            damping,
            seq_len,
            residual_alpha: self.cached_residual_alpha,
            debug_enable: if debug_enable { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
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
        w_gate_hist: &[f32],
        w_forget: &[f32],
        b_forget: &[f32],
    ) {
        let d = self.config.d_r as u64;
        let h = self.config.h_slots as u64;
        queue.write_buffer(&self.bridge.all_weights_buf, 0, bytemuck::cast_slice(wq));
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_wk_byte_off(d, h),
            bytemuck::cast_slice(wk),
        );
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_wv_byte_off(d, h),
            bytemuck::cast_slice(wv),
        );
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_wo_byte_off(d, h),
            bytemuck::cast_slice(wo),
        );
        self.upload_staged_transposed_weights(queue, wq, wk, wv, wo);
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_win_byte_off(d, h),
            bytemuck::cast_slice(win),
        );
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_wx_byte_off(d, h),
            bytemuck::cast_slice(wx),
        );
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_wout_byte_off(d, h),
            bytemuck::cast_slice(wout),
        );
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_alog_byte_off(d, h),
            bytemuck::cast_slice(alog),
        );
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_nscale_byte_off(d, h),
            bytemuck::cast_slice(nscale),
        );
        let hist_params = self.pack_history_params(
            w_hist_shared,
            hist_slot_scale,
            hist_slot_bias,
            hist_gate_logit,
            slot_anchor,
            w_delta,
            b_delta,
            w_gate_hist,
            w_forget,
            b_forget,
        );
        queue.write_buffer(
            &self.bridge.all_weights_buf,
            aw_hist_byte_off(d, h),
            bytemuck::cast_slice(hist_params.as_slice()),
        );
        queue.submit([]);
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok((wq_check, wk_check, wv_check, wo_check, win_check, _, _, _, _)) =
            self.read_weights()
        {
            let stats = |v: &[f32]| -> (f32, f32, f32) {
                let min = v.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let abs_mean = v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                (min, max, abs_mean)
            };
            let (q_min, q_max, q_abs) = stats(&wq_check);
            let (k_min, k_max, k_abs) = stats(&wk_check);
            let (v_min, v_max, v_abs) = stats(&wv_check);
            let (o_min, o_max, o_abs) = stats(&wo_check);
            let (in_min, in_max, in_abs) = stats(&win_check);
            eprintln!(
                "[GPU-VERIFY] Post-upload:\n    W_q:  min={:.4}, max={:.4}, abs_mean={:.6}\n    W_k:  min={:.4}, max={:.4}, abs_mean={:.6}\n    W_v:  min={:.4}, max={:.4}, abs_mean={:.6}\n    W_o:  min={:.4}, max={:.4}, abs_mean={:.6}\n    W_in: min={:.4}, max={:.4}, abs_mean={:.6}",
                q_min, q_max, q_abs,
                k_min, k_max, k_abs,
                v_min, v_max, v_abs,
                o_min, o_max, o_abs,
                in_min, in_max, in_abs
            );
        }
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
        let shape =
            self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping, false);
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
        let shape =
            self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping, false);
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
        let shape = self.build_compute_shape(
            batch_size,
            1,
            self.config.max_deq_iters as u32,
            5e-4,
            0.9,
            true,
        );

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
        let shape =
            self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping, false);
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
        let shape =
            self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping, false);
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
        let shape =
            self.build_compute_shape(batch_size, seq_len, max_iters, epsilon, damping, false);
        self.queue
            .write_buffer(&self.bridge.uniform_buf, 0, bytemuck::bytes_of(&shape));

        if update_weights {
            let d = self.config.d_r as u64;
            let h = self.config.h_slots as u64;
            self.queue
                .write_buffer(&self.bridge.all_weights_buf, 0, bytemuck::cast_slice(w_q));
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_wk_byte_off(d, h),
                bytemuck::cast_slice(w_k),
            );
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_wv_byte_off(d, h),
                bytemuck::cast_slice(w_v),
            );
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_wo_byte_off(d, h),
                bytemuck::cast_slice(w_o),
            );
            self.upload_staged_transposed_weights(&self.queue, w_q, w_k, w_v, w_o);
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_win_byte_off(d, h),
                bytemuck::cast_slice(w_in),
            );
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_wx_byte_off(d, h),
                bytemuck::cast_slice(w_x),
            );
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_wout_byte_off(d, h),
                bytemuck::cast_slice(w_out),
            );
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_alog_byte_off(d, h),
                bytemuck::cast_slice(a_log),
            );
            self.queue.write_buffer(
                &self.bridge.all_weights_buf,
                aw_nscale_byte_off(d, h),
                bytemuck::cast_slice(norm),
            );
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

    pub fn run_staged_adjoint_picard_no_readback(
        &self,
        seq_len: u32,
        damping: f32,
        iters: u32,
        dl_dh_src: Option<&wgpu::Buffer>,
        clear_slot_rhs: bool,
        batch_size: u32,
    ) -> Result<(), String> {
        let profile_picard = self.cfg_picard_profile;
        let profile_picard_stages = self.cfg_picard_stage_profile;
        let profile_picard_accum_split = self.cfg_picard_accum_split;
        let picard_internal_probe = self.cfg_picard_internal_probe;
        let total_t0 = std::time::Instant::now();
        let fused_gscore = self.cfg_picard_gscore_fused;
        let params = UpdateUniforms {
            d_model: self.config.d_r as u32,
            h_slots: self.config.h_slots as u32,
            lr: 0.0,
            grad_scale: 0.0,
            ternary_flag: 0,
            weight_decay: 0.0,
            seq_len,
            damping,
            residual_alpha: self.cached_residual_alpha,
            grad_accum_mode: 0,
            n_accum: 1,
            n_total_weights: 0,
            batch_size,
            apply_accum: 0,
            _pad0: 0,
        };
        self.queue.write_buffer(
            &self.fused_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );

        let attn_len =
            (batch_size * seq_len * self.config.h_slots as u32 * self.config.d_r as u32) as usize;
        let n_entries = batch_size * seq_len * self.config.h_slots as u32;
        let d = self.config.d_r as u32;
        let dl_bytes = (batch_size as u64) * (seq_len as u64) * (self.config.d_r as u64) * 4;
        let bytes = (attn_len * std::mem::size_of::<f32>()) as u64;

        // Active training adjoint path:
        // - v_state / v_next ping-pong between adj_bufs.b_v_out and fused_weighted_h_buf
        // - fused_hist_delta_buf is not read by staged_adjoint_picard.wgsl
        //
        // T1-B: Batch dl_copy + zero + init + all Picard iterations into a single encoder
        // in the normal path, eliminating iters+3 queue.submit() calls per adjoint call.
        if !profile_picard_stages && !picard_internal_probe {
            let anderson_m = self.anderson_m;
            let n_tokens = batch_size * seq_len;
            let needs_final_copy = (iters & 1) == 1;

            // Pre-write Anderson params for each iteration before building the encoder.
            if anderson_m > 0 {
                for k in 0..(iters as usize) {
                    let ap = AndersonParams {
                        m: anderson_m,
                        k: k as u32,
                        slots_per_segment: self.anderson_slots_per_segment,
                        _pad0: 0,
                    };
                    self.queue.write_buffer(
                        &self.anderson_params_bufs[k],
                        0,
                        bytemuck::bytes_of(&ap),
                    );
                }
            }

            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Staged Picard Adjoint — All Iters"),
                });
            // Optional dl_dh copy (replaces separate dl_encoder submit).
            if let Some(dl_dh_src) = dl_dh_src {
                enc.copy_buffer_to_buffer(dl_dh_src, 0, &self.adj_bufs.b_dl, 0, dl_bytes);
            }
            // Zero buffers (replaces separate zero_encoder submit).
            enc.clear_buffer(&self.adj_bufs.b_v_out, 0, None);
            enc.clear_buffer(&self.fused_weighted_h_buf, 0, None);
            enc.clear_buffer(&self.fused_gmix_buf, 0, None);
            if clear_slot_rhs {
                enc.clear_buffer(&self.fused_hist_ctx_buf, 0, None);
            }
            enc.clear_buffer(&self.fused_qgrad_buf, 0, None);
            enc.clear_buffer(&self.fused_gscore_buf, 0, None);
            // Clear Anderson history ring buffer
            if anderson_m > 0 {
                for buf in &self.anderson_hist_bufs {
                    enc.clear_buffer(buf, 0, None);
                }
            }
            // Init pass (replaces separate init_encoder submit).
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Staged Picard Init"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.staged_picard_bg, &[]);
                pass.set_bind_group(1, &self.staged_picard_bg1, &[]);
                pass.set_pipeline(&self.staged_picard_init_pipeline);
                pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
            }
            enc.copy_buffer_to_buffer(
                &self.fused_weighted_h_buf,
                0,
                &self.adj_bufs.b_v_out,
                0,
                bytes,
            );
            // All Picard iterations + optional Anderson mixing per iteration.
            for k in 0..(iters as usize) {
                let bg0 = if (k & 1) == 0 {
                    &self.staged_picard_bg
                } else {
                    &self.staged_picard_bg_alt
                };
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Staged Picard Iter"),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, bg0, &[]);
                    pass.set_bind_group(1, &self.staged_picard_bg1, &[]);
                    pass.set_pipeline(&self.staged_picard_gcomb_pipeline);
                    pass.dispatch_workgroups(n_entries.max(1), 1, 1);
                    if fused_gscore {
                        pass.set_pipeline(&self.staged_picard_gmix_gscore_pipeline);
                        pass.dispatch_workgroups(n_entries.max(1), 1, 1);
                    } else {
                        pass.set_pipeline(&self.staged_picard_gmix_pipeline);
                        pass.dispatch_workgroups(n_entries.max(1), 1, 1);
                        pass.set_pipeline(&self.staged_picard_gscore_pipeline);
                        pass.dispatch_workgroups(
                            self.config.h_slots.div_ceil(16) as u32,
                            n_entries.div_ceil(16).max(1),
                            1,
                        );
                    }
                    if self.cfg_picard_accum_split {
                        pass.set_pipeline(&self.staged_picard_accum_base_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        pass.set_pipeline(&self.staged_picard_accum_v_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        pass.set_pipeline(&self.staged_picard_accum_k_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        pass.set_pipeline(&self.staged_picard_accum_q_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                    } else {
                        if self.config.d_r <= 512 {
                            pass.set_pipeline(&self.staged_picard_accum_opt_pipeline);
                            pass.dispatch_workgroups(n_entries.max(1), 1, 1);
                        } else {
                            pass.set_pipeline(&self.staged_picard_accum_pipeline);
                            pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        }
                    }
                    // Anderson: store current v_next into ring buffer, then mix.
                    if anderson_m > 0 {
                        pass.set_bind_group(1, &self.anderson_bgs[k], &[]);
                        // Store v_next into hist[k % m]
                        pass.set_pipeline(&self.anderson_store_pipeline);
                        pass.dispatch_workgroups((attn_len as u32).div_ceil(256).max(1), 1, 1);
                        // Mix when ≥2 valid pseudo-residuals (k ≥ 2) — produces better gradient
                        if k >= 2 {
                            pass.set_pipeline(&self.anderson_mix_pipeline);
                            pass.dispatch_workgroups(n_tokens.max(1), 1, 1);
                        }
                    }
                }
            }
            if needs_final_copy {
                enc.copy_buffer_to_buffer(
                    &self.fused_weighted_h_buf,
                    0,
                    &self.adj_bufs.b_v_out,
                    0,
                    bytes,
                );
            }
            self.queue.submit(Some(enc.finish()));
            if profile_picard {
                // Profiling path only: stall CPU so stage timings measure GPU completion,
                // not just command submission latency.
                self.device.poll(wgpu::Maintain::Wait);
                eprintln!(
                    "[PICARD-PROFILE] seq_len={} iters={} total={}ms (batched encoder)",
                    seq_len,
                    iters,
                    total_t0.elapsed().as_millis()
                );
            }
            return Ok(());
        }

        // Profile / probe path: per-stage/per-iter encoders for accurate timings and readbacks.
        // Mirror the adjoint contract: staged Picard consumes adj_bufs.b_dl as
        // the pooled upstream gradient source. Without this copy, Picard can
        // iterate on a stale/zero rhs even when the LM head produced dl/dh.
        if let Some(dl_dh_src) = dl_dh_src {
            let mut dl_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Staged Picard dl_dh Copy"),
                    });
            dl_encoder.copy_buffer_to_buffer(dl_dh_src, 0, &self.adj_bufs.b_dl, 0, dl_bytes);
            self.queue.submit(Some(dl_encoder.finish()));
        }

        let prep_t0 = std::time::Instant::now();
        let attn_len =
            (batch_size * seq_len * self.config.h_slots as u32 * self.config.d_r as u32) as usize;
        let zero_t0 = std::time::Instant::now();
        let mut zero_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Staged Picard Zero Buffers"),
                });
        // Picard adjoint must start from v_state = 0 each call to solve (I - J^T)v = b.
        // Leaving b_v_out stale carries state across steps and breaks the linear solve.
        zero_encoder.clear_buffer(&self.adj_bufs.b_v_out, 0, None);
        zero_encoder.clear_buffer(&self.fused_weighted_h_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_gmix_buf, 0, None);
        if clear_slot_rhs {
            zero_encoder.clear_buffer(&self.fused_hist_ctx_buf, 0, None);
        }
        zero_encoder.clear_buffer(&self.fused_qgrad_buf, 0, None);
        zero_encoder.clear_buffer(&self.fused_gscore_buf, 0, None);
        self.queue.submit(Some(zero_encoder.finish()));
        let zero_ms = zero_t0.elapsed().as_millis();
        let prep_ms = prep_t0.elapsed().as_millis();

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
            encoder.copy_buffer_to_buffer(
                &self.fused_weighted_h_buf,
                0,
                &self.adj_bufs.b_v_out,
                0,
                bytes,
            );
            self.queue.submit(Some(encoder.finish()));
        }
        let init_ms = init_t0.elapsed().as_millis();
        let needs_final_copy = (iters & 1) == 1;
        if picard_internal_probe {
            let sample_t = (seq_len as usize / 2)
                .max(1)
                .min(seq_len.saturating_sub(1) as usize);
            let dl = self.read_storage_buffer(
                &self.adj_bufs.b_dl,
                seq_len as usize * self.config.d_r,
                "Picard Probe b_dl Readback",
            );
            let v_state = self.read_storage_buffer(
                &self.adj_bufs.b_v_out,
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
        for iter in 0..iters {
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
                    if (iter & 1) == 0 {
                        &self.staged_picard_bg
                    } else {
                        &self.staged_picard_bg_alt
                    },
                    &self.staged_picard_bg1,
                );
                stage_gmix_ms += run_stage(
                    "Staged Picard GMix",
                    &self.staged_picard_gmix_pipeline,
                    n_entries.max(1),
                    1,
                    &self.device,
                    &self.queue,
                    if (iter & 1) == 0 {
                        &self.staged_picard_bg
                    } else {
                        &self.staged_picard_bg_alt
                    },
                    &self.staged_picard_bg1,
                );
                stage_gscore_ms += run_stage(
                    "Staged Picard GScore",
                    &self.staged_picard_gscore_pipeline,
                    self.config.h_slots.div_ceil(16) as u32,
                    n_entries.div_ceil(16).max(1),
                    &self.device,
                    &self.queue,
                    if (iter & 1) == 0 {
                        &self.staged_picard_bg
                    } else {
                        &self.staged_picard_bg_alt
                    },
                    &self.staged_picard_bg1,
                );
                if profile_picard_accum_split {
                    stage_accum_ms += run_stage(
                        "Staged Picard Accum Base",
                        &self.staged_picard_accum_base_pipeline,
                        d.div_ceil(16),
                        n_entries.div_ceil(16).max(1),
                        &self.device,
                        &self.queue,
                        if (iter & 1) == 0 {
                            &self.staged_picard_bg
                        } else {
                            &self.staged_picard_bg_alt
                        },
                        &self.staged_picard_bg1,
                    );
                } else {
                    stage_accum_ms += run_stage(
                        if self.config.d_r <= 512 { "Staged Picard Accum Opt" } else { "Staged Picard Accum" },
                        if self.config.d_r <= 512 {
                            &self.staged_picard_accum_opt_pipeline
                        } else {
                            &self.staged_picard_accum_pipeline
                        },
                        if self.config.d_r <= 512 { n_entries.max(1) } else { d.div_ceil(16) },
                        if self.config.d_r <= 512 { 1 } else { n_entries.div_ceil(16).max(1) },
                        &self.device,
                        &self.queue,
                        if (iter & 1) == 0 {
                            &self.staged_picard_bg
                        } else {
                            &self.staged_picard_bg_alt
                        },
                        &self.staged_picard_bg1,
                    );
                }
                if profile_picard_accum_split {
                    stage_accum_v_ms += run_stage(
                        "Staged Picard Accum V",
                        &self.staged_picard_accum_v_pipeline,
                        d.div_ceil(16),
                        n_entries.div_ceil(16).max(1),
                        &self.device,
                        &self.queue,
                        if (iter & 1) == 0 {
                            &self.staged_picard_bg
                        } else {
                            &self.staged_picard_bg_alt
                        },
                        &self.staged_picard_bg1,
                    );
                    stage_accum_k_ms += run_stage(
                        "Staged Picard Accum K",
                        &self.staged_picard_accum_k_pipeline,
                        d.div_ceil(16),
                        n_entries.div_ceil(16).max(1),
                        &self.device,
                        &self.queue,
                        if (iter & 1) == 0 {
                            &self.staged_picard_bg
                        } else {
                            &self.staged_picard_bg_alt
                        },
                        &self.staged_picard_bg1,
                    );
                    stage_accum_q_ms += run_stage(
                        "Staged Picard Accum Q",
                        &self.staged_picard_accum_q_pipeline,
                        d.div_ceil(16),
                        n_entries.div_ceil(16).max(1),
                        &self.device,
                        &self.queue,
                        if (iter & 1) == 0 {
                            &self.staged_picard_bg
                        } else {
                            &self.staged_picard_bg_alt
                        },
                        &self.staged_picard_bg1,
                    );
                }
            } else {
                let iter_t0 = std::time::Instant::now();
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Staged Picard Adjoint"),
                        });
                let bg0 = if (iter & 1) == 0 {
                    &self.staged_picard_bg
                } else {
                    &self.staged_picard_bg_alt
                };
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Staged Picard GComb"),
                        timestamp_writes: None,
                    });
                    pass.set_bind_group(0, bg0, &[]);
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
                    if self.cfg_picard_accum_split {
                        pass.set_pipeline(&self.staged_picard_accum_base_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        pass.set_pipeline(&self.staged_picard_accum_v_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        pass.set_pipeline(&self.staged_picard_accum_k_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        pass.set_pipeline(&self.staged_picard_accum_q_pipeline);
                        pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                    } else {
                        if self.config.d_r <= 512 {
                            pass.set_pipeline(&self.staged_picard_accum_opt_pipeline);
                            pass.dispatch_workgroups(n_entries.max(1), 1, 1);
                        } else {
                            pass.set_pipeline(&self.staged_picard_accum_pipeline);
                            pass.dispatch_workgroups(d.div_ceil(16), n_entries.div_ceil(16).max(1), 1);
                        }
                    }
                }
                self.queue.submit(Some(encoder.finish()));
                submit_ms += iter_t0.elapsed().as_millis();
            }
        }
        if needs_final_copy {
            let bytes = (attn_len * std::mem::size_of::<f32>()) as u64;
            let t0 = std::time::Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Staged Picard Final Copy"),
                });
            encoder.copy_buffer_to_buffer(
                &self.fused_weighted_h_buf,
                0,
                &self.adj_bufs.b_v_out,
                0,
                bytes,
            );
            self.queue.submit(Some(encoder.finish()));
            if profile_picard_stages {
                self.device.poll(wgpu::Maintain::Wait);
            }
            stage_copy_ms += t0.elapsed().as_millis();
        }
        let poll_t0 = std::time::Instant::now();
        if profile_picard_stages {
            // Only poll(Wait) in profiling mode for per-stage timing accuracy.
            // In normal operation, the caller (apply_gradient_update) provides the
            // final sync barrier — no need to stall CPU here between adjoint and update.
            // Profiling path only: caller in normal training keeps adjoint->update asynchronous.
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
                    stage_gcomb_ms, stage_gmix_ms, stage_gscore_ms, stage_accum_ms, stage_copy_ms
                );
                if profile_picard_accum_split {
                    eprintln!(
                        "[PICARD-ACCUM-SPLIT] v={}ms k={}ms q={}ms",
                        stage_accum_v_ms, stage_accum_k_ms, stage_accum_q_ms
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
        grad_accum_mode: u32,
        n_accum: u32,
        batch_size: u32,
        apply_accum: bool,
    ) -> Result<(), String> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let hist_len = Self::history_params_len(d, h);
        let n_total_weights = (aw_hist_byte_off(d as u64, h as u64) / 4) as u32 + hist_len as u32;
        let params = UpdateUniforms {
            d_model: d as u32,
            h_slots: h as u32,
            lr,
            grad_scale,
            ternary_flag: if ternary { 1 } else { 0 },
            weight_decay,
            seq_len,
            damping,
            residual_alpha: self.cached_residual_alpha,
            grad_accum_mode,
            n_accum,
            n_total_weights,
            batch_size,
            apply_accum: if apply_accum { 1 } else { 0 },
            _pad0: 0,
        };
        self.queue.write_buffer(
            &self.fused_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        // Use cached config flags — no env::var syscalls in hot path.
        let profile_fused = self.cfg_fused_profile;
        // hist_gated is the default mode. Disable explicitly with AIDEEN_DEQ_HIST_GATED=0.
        let hist_gated = self.cfg_hist_gated;
        let hist_selective = self.cfg_hist_selective;
        let hist_internal_probe = self.cfg_hist_internal_probe;
        if profile_fused {
            // Drain any previously queued GPU work (notably CG) so per-stage fused timings
            // do not absorb unrelated latency from earlier submissions.
            self.device.poll(wgpu::Maintain::Wait);
        }
        if profile_fused || hist_internal_probe {
            let mut zero_encoder =
                self.device
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
            // Profile/probe path is intentionally conservative: we zero every temp that may be
            // inspected mid-flight so debug reads never depend on prior work left in buffers.
            zero_encoder.clear_buffer(&self.fused_hist_ctx_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_hist_delta_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_gscore_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_qgrad_buf, 0, None);
            self.queue.submit(Some(zero_encoder.finish()));
            if profile_fused {
                self.device.poll(wgpu::Maintain::Wait);
            }
        }
        let d = self.config.d_r as u32;
        let n = batch_size * seq_len * self.config.h_slots as u32;
        let hs = self.config.h_slots as u32;
        // T1-A: Batch all compute stages into a single command encoder for the normal
        // (non-profile, non-probe) training path, eliminating ~20 queue.submit() calls/step.
        if !profile_fused && !hist_internal_probe {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Fused Update — All Stages"),
                });
            // weighted_h/gmix/gscore/qgrad are fully overwritten by stage1b/stage1a/stage2/stage3
            // before any consumer reads them in the normal fused-update path.
            // Keep hist_ctx/hist_delta clears: their producers do not cover every regime as cleanly.
            // Normal hot path still zeros hist_ctx/hist_delta because the history producers do
            // not provably cover every selective/gated regime. qgrad/gscore stay untouched here
            // because stage2/stage3 fully overwrite them before any read.
            enc.clear_buffer(&self.fused_hist_ctx_buf, 0, None);
            enc.clear_buffer(&self.fused_hist_delta_buf, 0, None);
            // add_pass!: append one compute pass (one pipeline) to `enc`.
            // Macro expansion is inline so `enc` / `self` are captured correctly
            // without closure-capture lifetime issues.
            macro_rules! add_pass {
                ($pipeline:expr, $x:expr, $y:expr) => {{
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline($pipeline);
                    pass.set_bind_group(0, &self.fused_update_bg0, &[]);
                    pass.set_bind_group(1, &self.fused_update_bg1, &[]);
                    pass.dispatch_workgroups($x, $y, 1);
                }};
            }
            macro_rules! add_pass_3d {
                ($pipeline:expr, $x:expr, $y:expr, $z:expr) => {{
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline($pipeline);
                    pass.set_bind_group(0, &self.fused_update_bg0, &[]);
                    pass.set_bind_group(1, &self.fused_update_bg1, &[]);
                    pass.dispatch_workgroups($x, $y, $z);
                }};
            }
            if hist_gated {
                let train_hist_carrier = self.cfg_hist_train_carrier;
                let train_hist_wx = self.cfg_hist_train_wx;
                let train_hist_wout = self.cfg_hist_train_wout;
                let train_hist_alog = self.cfg_hist_train_alog;
                let train_hist_delta = self.cfg_hist_train_delta;
                let train_hist_temporal = train_hist_carrier || train_hist_alog || train_hist_delta;
                add_pass!(&self.fused_update_hist_prep_pipeline, n, 1);
                add_pass!(&self.fused_update_hist_gate_pipeline, hs.div_ceil(64), 1);
                add_pass!(
                    &self.fused_update_hist_mat_pipeline,
                    d.div_ceil(16),
                    d.div_ceil(16)
                );
                add_pass!(
                    &self.fused_update_hist_scale_pipeline,
                    d.div_ceil(16),
                    hs.div_ceil(16)
                );
                add_pass!(
                    &self.fused_update_hist_wgate_pipeline,
                    d.div_ceil(16),
                    hs.div_ceil(16)
                );
                if train_hist_temporal {
                    add_pass!(&self.fused_update_hist_mprev_pipeline, n, 1);
                    add_pass!(&self.fused_update_hist_tbptt_pipeline, batch_size * hs, 1);
                    add_pass!(
                        &self.fused_update_hist_forget_pipeline,
                        d.div_ceil(16),
                        hs.div_ceil(16)
                    );
                    if train_hist_alog {
                        add_pass!(&self.fused_update_hist_alog_pipeline, d.div_ceil(64), 1);
                    }
                    if train_hist_wout {
                        add_pass!(
                            &self.fused_update_hist_wout_pipeline,
                            d.div_ceil(16),
                            d.div_ceil(16)
                        );
                    }
                    if train_hist_wx {
                        add_pass!(
                            &self.fused_update_hist_wx_pipeline,
                            d.div_ceil(16),
                            d.div_ceil(16)
                        );
                    }
                    if train_hist_delta && hist_selective {
                        add_pass!(
                            &self.fused_update_hist_wdelta_pipeline,
                            d.div_ceil(16),
                            d.div_ceil(16)
                        );
                        add_pass!(&self.fused_update_hist_bdelta_pipeline, d.div_ceil(64), 1);
                    }
                }
            }
            add_pass!(&self.fused_update_stage1a_pipeline, n, 1);
            add_pass!(&self.fused_update_stage1b_pipeline, n, 1);
            add_pass!(
                &self.fused_update_stage2_pipeline,
                hs.div_ceil(16),
                n.div_ceil(16)
            );
            add_pass!(
                &self.fused_update_stage3_pipeline,
                d.div_ceil(16),
                n.div_ceil(16)
            );
            add_pass!(
                &self.fused_update_stage4_wo_win_pipeline,
                d.div_ceil(16),
                d.div_ceil(16)
            );
            add_pass_3d!(
                &self.fused_update_stage4_wq_pipeline,
                d.div_ceil(16),
                d.div_ceil(16),
                hs
            );
            add_pass!(
                &self.fused_update_stage4_prep_wk_pipeline,
                d.div_ceil(16),
                (batch_size * seq_len * hs).div_ceil(16)
            );
            add_pass!(
                &self.fused_update_stage4_prep_wv_pipeline,
                d.div_ceil(16),
                (batch_size * seq_len * hs).div_ceil(16)
            );
            add_pass_3d!(
                &self.fused_update_stage4_wk_pipeline,
                d.div_ceil(16),
                d.div_ceil(16),
                hs
            );
            add_pass_3d!(
                &self.fused_update_stage4_wv_pipeline,
                d.div_ceil(16),
                d.div_ceil(16),
                hs
            );
            add_pass!(&self.fused_update_stage4_bias_pipeline, d.div_ceil(64), 1);
            if grad_accum_mode == 1 && apply_accum {
                add_pass!(
                    &self.fused_update_apply_grad_pipeline,
                    n_total_weights.div_ceil(256),
                    1
                );
            }
            self.queue.submit(Some(enc.finish()));
            // Non-blocking queue drain for long batched encoders. This preserves overlap while
            // letting Metal advance work and release memory pressure.
            self.device.poll(wgpu::Maintain::Poll);
        } else {
            // Profile / probe path: per-stage encoders for accurate per-stage timings
            // and mid-sequence GPU readbacks (hist_internal_probe).
            let mut zero_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Fused Update Zero Buffers"),
                    });
            zero_encoder.clear_buffer(&self.fused_weighted_h_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_gmix_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_hist_ctx_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_hist_delta_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_gscore_buf, 0, None);
            zero_encoder.clear_buffer(&self.fused_qgrad_buf, 0, None);
            self.queue.submit(Some(zero_encoder.finish()));
            if profile_fused {
                // Profiling path only: per-stage timings require completion, not queued work.
                self.device.poll(wgpu::Maintain::Wait);
            }
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
            let run_stage_3d = |device: &wgpu::Device,
                                queue: &wgpu::Queue,
                                label: &str,
                                pipeline: &wgpu::ComputePipeline,
                                bg0: &wgpu::BindGroup,
                                bg1: &wgpu::BindGroup,
                                x: u32,
                                y: u32,
                                z: u32,
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
                    pass.dispatch_workgroups(x, y, z);
                }
                queue.submit(Some(encoder.finish()));
                if profile {
                    device.poll(wgpu::Maintain::Wait);
                    eprintln!("[FUSED-PROFILE] {label}: {} ms", t0.elapsed().as_millis());
                }
            };
            if hist_gated {
                let train_hist_carrier = self.cfg_hist_train_carrier;
                let train_hist_wx = self.cfg_hist_train_wx;
                let train_hist_wout = self.cfg_hist_train_wout;
                let train_hist_alog = self.cfg_hist_train_alog;
                let train_hist_delta = self.cfg_hist_train_delta;
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
                let train_hist_temporal = train_hist_carrier || train_hist_alog || train_hist_delta;
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
                run_stage(
                    &self.device,
                    &self.queue,
                    "hist_wgate",
                    &self.fused_update_hist_wgate_pipeline,
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
                        batch_size * hs,
                        1,
                        profile_fused,
                    );
                    run_stage(
                        &self.device,
                        &self.queue,
                        "hist_forget",
                        &self.fused_update_hist_forget_pipeline,
                        &self.fused_update_bg0,
                        &self.fused_update_bg1,
                        d.div_ceil(16),
                        hs.div_ceil(16),
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
                    }
                    if train_hist_delta && hist_selective {
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
                "stage4_wo_win",
                &self.fused_update_stage4_wo_win_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                d.div_ceil(16),
                profile_fused,
            );
            run_stage_3d(
                &self.device,
                &self.queue,
                "stage4_wq",
                &self.fused_update_stage4_wq_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                d.div_ceil(16),
                hs,
                profile_fused,
            );
            run_stage(
                &self.device,
                &self.queue,
                "stage4_prep_wk",
                &self.fused_update_stage4_prep_wk_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                (batch_size * seq_len * hs).div_ceil(16),
                profile_fused,
            );
            run_stage(
                &self.device,
                &self.queue,
                "stage4_prep_wv",
                &self.fused_update_stage4_prep_wv_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                (batch_size * seq_len * hs).div_ceil(16),
                profile_fused,
            );
            run_stage_3d(
                &self.device,
                &self.queue,
                "stage4_wk",
                &self.fused_update_stage4_wk_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                d.div_ceil(16),
                hs,
                profile_fused,
            );
            run_stage_3d(
                &self.device,
                &self.queue,
                "stage4_wv",
                &self.fused_update_stage4_wv_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(16),
                d.div_ceil(16),
                hs,
                profile_fused,
            );
            run_stage(
                &self.device,
                &self.queue,
                "stage4_bias",
                &self.fused_update_stage4_bias_pipeline,
                &self.fused_update_bg0,
                &self.fused_update_bg1,
                d.div_ceil(64),
                1,
                profile_fused,
            );
            if grad_accum_mode == 1 && apply_accum {
                run_stage(
                    &self.device,
                    &self.queue,
                    "apply_grad",
                    &self.fused_update_apply_grad_pipeline,
                    &self.fused_update_bg0,
                    &self.fused_update_bg1,
                    n_total_weights.div_ceil(256),
                    1,
                    profile_fused,
                );
            }
            if !profile_fused {
                // Same rationale as the normal hot path: keep queue moving without inserting a
                // hard CPU barrier when profiling is disabled.
                self.device.poll(wgpu::Maintain::Poll);
            }
        }
        Ok(())
    }

    /// Applies accumulated gradients (from AllGradients) to AllWeights and zeroes AllGradients.
    /// Called once after n_accum passes of apply_fused_deq_update with grad_accum_mode=1.
    pub fn apply_gradient_update(
        &self,
        lr: f32,
        weight_decay: f32,
        n_accum: u32,
    ) -> Result<(), String> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let hist_len = Self::history_params_len(d, h);
        let n_total_weights = (aw_hist_byte_off(d as u64, h as u64) / 4) as u32 + hist_len as u32;
        let params = UpdateUniforms {
            d_model: d as u32,
            h_slots: h as u32,
            lr,
            grad_scale: 1.0,
            ternary_flag: 0,
            weight_decay,
            seq_len: 1,
            damping: 0.0,
            residual_alpha: self.cached_residual_alpha,
            grad_accum_mode: 0, // unused by apply_grad_update_main but set for clarity
            n_accum,
            n_total_weights,
            batch_size: 1,
            apply_accum: 0,
            _pad0: 0,
        };
        self.queue.write_buffer(
            &self.fused_update_params_buf,
            0,
            bytemuck::bytes_of(&params),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Apply Gradient Update"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Apply Gradient Update"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fused_update_apply_grad_pipeline);
            pass.set_bind_group(0, &self.fused_update_bg0, &[]);
            pass.set_bind_group(1, &self.fused_update_bg1, &[]);
            pass.dispatch_workgroups(n_total_weights.div_ceil(256), 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        // Explicit API boundary: callers expect accumulated gradients to be fully applied when
        // this method returns, so this is a real synchronization point, not a hot-path stall.
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    /// Spectral renormalization fully on GPU.
    pub fn renormalize_spectral(&self) -> Result<(), String> {
        let attn_threshold = Self::env_f32("AIDEEN_DEQ_ATTN_THRESHOLD")
            .unwrap_or(0.10)
            .max(0.01);
        let win_threshold = Self::env_f32("AIDEEN_DEQ_WIN_THRESHOLD")
            .unwrap_or(0.30)
            .max(0.05);
        let wv_threshold = Self::env_f32("AIDEEN_DEQ_WV_THRESHOLD")
            .unwrap_or(0.50)
            .max(0.01);
        let wo_threshold = Self::env_f32("AIDEEN_DEQ_WO_THRESHOLD")
            .unwrap_or(0.50)
            .max(0.01);
        self.bridge.renormalize_spectral(
            &self.device,
            &self.queue,
            self.config.d_r as u32,
            attn_threshold, // W_q/W_k: keep attention logits stable
            win_threshold,  // W_in only — does not enter J_h, can be larger to carry signal
            0.70,           // W_x/W_out: temporal carrier, does not enter J_h of the DEQ solve
            wv_threshold,
            wo_threshold,
            12,
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

    pub fn tps_epoch_begin(&self) {
        if !self.tps_timestamp_enabled {
            return;
        }
        let Some(qs) = self.tps_timestamp_query.as_ref() else {
            return;
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("AIDEEN TPS Timestamp Begin"),
            });
        encoder.write_timestamp(qs, 0);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn tps_epoch_end(&self) {
        if !self.tps_timestamp_enabled {
            return;
        }
        let (Some(qs), Some(resolve), Some(readback)) = (
            self.tps_timestamp_query.as_ref(),
            self.tps_timestamp_resolve_buf.as_ref(),
            self.tps_timestamp_readback_buf.as_ref(),
        ) else {
            return;
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("AIDEEN TPS Timestamp End"),
            });
        encoder.write_timestamp(qs, 1);
        encoder.resolve_query_set(qs, 0..2, resolve, 0);
        encoder.copy_buffer_to_buffer(
            resolve,
            0,
            readback,
            0,
            2 * std::mem::size_of::<u64>() as u64,
        );
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn read_tps_epoch_ns(&self) -> Option<f64> {
        if !self.tps_timestamp_enabled {
            return None;
        }
        let buf = self.tps_timestamp_readback_buf.as_ref()?;
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        // Readback helper: Wait is required because the CPU consumes mapped data immediately.
        self.device.poll(wgpu::Maintain::Wait);
        if rx.recv().ok().and_then(|r| r.ok()).is_none() {
            buf.unmap();
            return None;
        }
        let data = slice.get_mapped_range();
        let ts: &[u64] = bytemuck::cast_slice(&data);
        if ts.len() < 2 {
            drop(data);
            buf.unmap();
            return None;
        }
        let start = ts[0];
        let end = ts[1];
        drop(data);
        buf.unmap();
        if end <= start {
            return None;
        }
        let period = self.tps_timestamp_period as f64;
        Some((end - start) as f64 * period)
    }

    pub fn read_scratch_buffer(&self) -> Vec<f32> {
        self.bridge.read_scratch_buffer(&self.device, &self.queue)
    }

    fn read_storage_buffer_at(
        &self,
        buffer: &wgpu::Buffer,
        src_offset: u64,
        n_floats: usize,
        label: &str,
    ) -> Vec<f32> {
        let byte_size = (n_floats * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        enc.copy_buffer_to_buffer(buffer, src_offset, &staging, 0, byte_size);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Readback helper: Wait is required because the CPU consumes mapped data immediately.
        self.device.poll(wgpu::Maintain::Wait);
        if let Ok(Ok(())) = rx.recv() {
            let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
            staging.unmap();
            out
        } else {
            vec![0.0; n_floats]
        }
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
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
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

    pub fn read_adj_v_out(&self, seq_len: u32) -> Vec<f32> {
        let n_floats = seq_len as usize * self.config.h_slots * self.config.d_r;
        self.read_storage_buffer(&self.adj_bufs.b_v_out, n_floats, "Adjoint V_out Readback")
    }

    pub fn read_dl_dh(&self, seq_len: u32) -> Vec<f32> {
        let n_floats = seq_len as usize * self.config.d_r;
        self.read_storage_buffer(&self.adj_bufs.b_dl, n_floats, "dl_dh Readback")
    }

    /// Reads h_star (hnext_buf) from GPU for the first batch sample.
    /// Returns Vec<f32> of length h_slots * d_r.
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

    /// Reads the exact forward historical context `c_{t,k}` retained in Scratch.
    /// This matches the constant term injected into the DEQ loop, including the
    /// dynamic `hist_mod` scaling applied in the forward prelude.
    pub fn read_hist_gated_ctx_forward(&self, seq_len: u32) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let scratch = self.read_scratch_buffer();
        let scratch_stride = d * h_slots * 8 + h_slots * h_slots + h_slots;
        let hist_ctx_base = h_slots * d * 7;

        let mut out = vec![0.0f32; seq_len as usize * h_slots * d];
        for t in 0..seq_len as usize {
            let token_base = t * scratch_stride;
            let src = token_base + hist_ctx_base;
            let dst = t * h_slots * d;
            out[dst..dst + h_slots * d].copy_from_slice(&scratch[src..src + h_slots * d]);
        }
        out
    }

    pub fn read_hist_gate_alpha(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h_slots = self.config.h_slots;
        let hist_params = self.read_storage_buffer_at(
            &self.bridge.all_weights_buf,
            aw_hist_byte_off(d as u64, h_slots as u64),
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
        self.read_storage_buffer_at(
            &self.bridge.all_weights_buf,
            aw_hist_byte_off(d as u64, h_slots as u64),
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
        let b_delta_base = w_delta_base + h_slots * d * d;
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
        let h_slots = self.config.h_slots;
        let d64 = d as u64;
        let h64 = h_slots as u64;
        let wx = self.read_storage_buffer_at(
            &self.bridge.all_weights_buf,
            aw_wx_byte_off(d64, h64),
            d * d,
            "Hist Carrier W_x Readback",
        );
        let wout = self.read_storage_buffer_at(
            &self.bridge.all_weights_buf,
            aw_wout_byte_off(d64, h64),
            d * d,
            "Hist Carrier W_out Readback",
        );
        let a_log = self.read_storage_buffer_at(
            &self.bridge.all_weights_buf,
            aw_alog_byte_off(d64, h64),
            h_slots * d,
            "Hist Carrier ALog Readback",
        );
        let mut out = Vec::with_capacity(wx.len() + wout.len() + a_log.len());
        out.extend_from_slice(&wx);
        out.extend_from_slice(&wout);
        out.extend_from_slice(&a_log);
        out
    }

    pub fn read_hist_carrier_param_stats(&self) -> ((f32, f32), (f32, f32), (f32, f32)) {
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
        let a_log = self.read_storage_buffer_at(
            &self.bridge.all_weights_buf,
            aw_alog_byte_off(d as u64, h_slots as u64),
            h_slots * d,
            "Hist Selective ALog Readback Staging",
        );

        let hist_mat_len = d * d;
        let slot_scale_base = hist_mat_len;
        let hist_bias_base = slot_scale_base + h_slots * d;
        let hist_gate_base = hist_bias_base + h_slots * d;
        let slot_anchor_base = hist_gate_base + h_slots;
        let w_delta_base = slot_anchor_base + h_slots * d;
        let b_delta_base = w_delta_base + h_slots * d * d;

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
                        delta_pre += hist_params[w_delta_base + slot * d * d + j * d + dim]
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
        let scratch_stride = d * h_slots * 8 + h_slots * h_slots + h_slots;
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
        debug_enable: bool,
    ) -> Result<(), &'static str> {
        let shape = self.build_compute_shape(
            batch_size,
            seq_len,
            max_iters,
            epsilon,
            damping,
            debug_enable,
        );

        self.bridge
            .run_forward_gpu_only(&self.device, &self.queue, &shape);
        Ok(())
    }
}
