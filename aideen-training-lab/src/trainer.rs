//! Training loop principal de AIDEEN.
//!
//! Pipeline por step:
//!   ① tokenizer.embed_context(tokens) → query D_R
//!   ② query → DEQ forward → H*
//!   ③ H* → LmHead → logits
//!   ④ loss = cross_entropy(logits, target)
//!   ⑤ backward LmHead + backward embedding (analítico)
//!   ⑥ backward DEQ (implicit diff via CG)
//!   ⑦ Adam update (embeddings + LmHead + DEQ)
//!   ⑧ renormalize_weights() (spectral norm)

use aideen_core::{
    deq_mode::DeqSolveMode,
    reasoning::Reasoning,
    state::{ArchitectureConfig, HSlots},
};

use crate::{gradients, loss, optimizer::Adam};
use aideen_backbone::{
    lm_head::LmHead, mamba_slot_reasoning::MambaSlotReasoning, tokenizer::Tokenizer,
};

#[cfg(feature = "wgpu")]
use aideen_backbone::gpu_deq::GpuDeqBackend;
#[cfg(feature = "wgpu")]
use aideen_backbone::gpu_embedding::GpuEmbeddingTrainer;
#[cfg(feature = "wgpu")]
use aideen_backbone::gpu_lm_head::GpuLmHeadTrainer;

/// Configuración del training (hiperparámetros).
pub struct TrainingConfig {
    pub lr: f32,
    /// LR mínimo al final del cosine schedule (default: lr/10).
    pub lr_min: f32,
    pub epochs: usize,
    pub log_every: usize,
    /// Warmup epochs: LR sube linealmente de lr_min a lr.
    pub warmup_epochs: usize,
    /// Experimento "Bit-Dieta": Proyecta pesos a valores ternarios (-1, 0, 1).
    pub ternary: bool,
    pub emb_lr_mult: f32,
    pub lm_lr_mult: f32,
    pub deq_lr_mult: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lr: 0.0001,
            lr_min: 0.00001,
            epochs: 1,
            log_every: 10,
            warmup_epochs: 1,
            ternary: false,
            emb_lr_mult: 1.0,
            lm_lr_mult: 1.0,
            deq_lr_mult: 0.01,
        }
    }
}

pub struct Trainer {
    pub config: ArchitectureConfig,
    pub training_config: TrainingConfig,
    pub reasoning: MambaSlotReasoning,
    #[cfg(feature = "wgpu")]
    pub gpu_deq: Option<GpuDeqBackend>,
    #[cfg(feature = "wgpu")]
    gpu_weights_uploaded: bool,
    #[cfg(feature = "wgpu")]
    gpu_cg_weights_uploaded: bool,
    #[cfg(feature = "wgpu")]
    pub gpu_lm: Option<GpuLmHeadTrainer>,
    #[cfg(feature = "wgpu")]
    pub gpu_emb: Option<GpuEmbeddingTrainer>,
    #[cfg(feature = "wgpu")]
    gpu_lm_weights_uploaded: bool,
    #[cfg(feature = "wgpu")]
    gpu_emb_weights_uploaded: bool,
    #[cfg(feature = "wgpu")]
    lm_head_cpu_stale: bool,
    pub lm_head: LmHead,
    pub tokenizer: Tokenizer,
    pub optimizer: Adam,
    // Flags de ablatación y validación
    pub frozen_deq: bool,
    pub frozen_emb: bool,
    pub frozen_lm: bool,
    pub eval_mode: bool,
    // --- v13.2 Stability Oracle (v13.1 plus upgrade) ---
    pub adaptive_max_iters: u32,
    pub adaptive_damping: f32,
    pub adaptive_adj_iters: u32,
    pub hit_hi_streak: u32,
    pub hit_lo_streak: u32,
    pub cg_res_hi_streak: u32,
    pub damping_boost_left: u32,
    pub emergency_left: u32,
    pub last_max_h: f32,
    pub max_h_growth_streak: u32,
    pub max_delta_hi_streak: u32,
    pub invalid_hi_streak: u32,
    pub contractivity_hi_streak: u32,
    pub force_renorm_done: bool,

    // --- v14 Temporal Memory State ---
    pub m_prev: Option<HSlots>,

    // --- Gradient Accumulation ---
    grad_accum_counter: u32, // steps accumulated so far in the current window

    // --- TPS tracking for GPU-DEBUG log ---
    debug_last_time: Option<std::time::Instant>,
    debug_tokens_accum: u32,   // tokens processed since last GPU-DEBUG print

    // --- Debug buffer cache (avoid blocking GPU readback every step) ---
    // read_debug_buffer() calls device.poll(Maintain::Wait) — blocks CPU until GPU finishes.
    // Now deferred to end-of-step (after apply_gradient_update) so GPU is already idle.
    cached_debug_buf: Vec<f32>,
    // --- Cached GPU loss (avoid sync readback every step) ---
    last_gpu_loss: f32,

    // --- Cached hot-path env vars (parsed once at construction) ---
    // Avoids ~26 env::var syscalls per training step.
    cfg_fwd_batch_size: u32,       // AIDEEN_BATCH_SIZE (for forward dispatch)
    cfg_debug_sample_every: usize, // AIDEEN_DEBUG_SAMPLE
    cfg_loss_readback_every: usize, // AIDEEN_LOSS_READBACK_EVERY
    cfg_tps_sync_every: usize, // AIDEEN_TPS_SYNC_EVERY
    cfg_grad_accum: u32,           // AIDEEN_GRAD_ACCUM
    cfg_hist_min_iters: u32,       // AIDEEN_HIST_MIN_ITERS
    cfg_val_every: usize,          // AIDEEN_VAL_EVERY
    cfg_progress_every: usize,     // AIDEEN_PROGRESS_EVERY
    cfg_progress_strict_sync: bool, // AIDEEN_PROGRESS_STRICT_SYNC
    cfg_wv_debug: bool,            // AIDEEN_DEQ_WV_DEBUG
    cfg_ssm_debug: bool,           // AIDEEN_SSM_DEBUG
    cfg_max_chunks: usize,         // AIDEEN_MAX_CHUNKS
    cfg_adj_iters_override: Option<u32>, // AIDEEN_ADJ_ITERS_OVERRIDE
    cfg_system_cost_audit: bool,   // AIDEEN_SYSTEM_COST_AUDIT
    cfg_system_cost_wait: bool,    // AIDEEN_SYSTEM_COST_WAIT

    // --- CSV Metrics Logging (High Resolution) ---
    pub metrics_log_path: Option<String>,
    metrics_pending: std::collections::VecDeque<PendingChunkMetric>,

    cfg_force_renorm: bool,        // AIDEEN_DEQ_FORCE_RENORM
    cfg_slot_attn_unified: bool,   // AIDEEN_DEQ_SLOT_ATTN_REAL_UNIFIED
    cfg_dynamic_qkv: bool,         // AIDEEN_DEQ_SLOT_ATTN_DYNAMIC_QKV
    cfg_lm_force_cpu_dldh: bool,   // AIDEEN_LM_FORCE_CPU_DLDH
    cfg_lm_dldh_parity: bool,      // AIDEEN_LM_DLDH_PARITY
    cfg_clean_deq_mode: bool,      // derived from DEQ env at construction
    cfg_log_emb_stats: bool,       // AIDEEN_LOG_EMB_STATS
}

#[derive(Debug)]
struct PendingChunkMetric {
    epoch: usize,
    chunk_id: usize,
    loss: f32,
    tokens: usize,
    slot_start: u32,
    interval_start_time: std::time::Instant,
}

impl Trainer {
    fn visible_loss_text(&self, fallback: Option<f32>) -> String {
        if self.last_gpu_loss.is_finite() && self.last_gpu_loss > 0.0 {
            format!("{:.4}", self.last_gpu_loss)
        } else if let Some(v) = fallback.filter(|v| v.is_finite() && *v > 0.0) {
            format!("{:.4}", v)
        } else {
            "n/a".to_string()
        }
    }

    /// Recolecta métricas de la GPU de forma asíncrona y las escribe al CSV si están listas.
    fn reap_metrics(&mut self, force_wait: bool) {
        #[cfg(feature = "wgpu")]
        {
            let gpu = if let Some(g) = self.gpu_deq.as_ref() { g } else { return };
            let path = if let Some(p) = self.metrics_log_path.as_ref() { p } else { return };

            while let Some(pending) = self.metrics_pending.front() {
                let ns_opt = if force_wait {
                    // Forzar sincronización al final para no perder los últimos chunks
                    gpu.device.poll(wgpu::Maintain::Wait);
                    gpu.read_tps_range_ns_async(pending.slot_start)
                } else {
                    gpu.read_tps_range_ns_async(pending.slot_start)
                };

                if let Some(ns) = ns_opt {
                    let duration_sec = ns / 1e9;
                    let tps_g = (pending.tokens as f64) / duration_sec;
                    let tps_w = (pending.tokens as f32) / pending.interval_start_time.elapsed().as_secs_f32().max(1e-9);
                    
                    // Abrir en modo append
                    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(path) {
                        use std::io::Write;
                        let _ = writeln!(f, "{},{},{:.6},{:.2},{:.2}", 
                            pending.epoch, pending.chunk_id, pending.loss, tps_w, tps_g);
                    }
                    
                    self.metrics_pending.pop_front();
                } else {
                    break;
                }
            }
        }
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

    fn fixed_history_reference_mode() -> bool {
        Self::env_flag("AIDEEN_DEQ_FIXED_HISTORY_REFERENCE")
    }

    fn solve_query_with_fixed_history(
        &self,
        query: &nalgebra::DVector<f32>,
        m_prev: Option<&HSlots>,
        max_iters: usize,
    ) -> HSlots {
        let hist_ctx = self.reasoning.fixed_hist_ctx(m_prev, query);
        let mut h = self.reasoning.init(query);
        for _ in 0..max_iters.max(1) {
            h = self
                .reasoning
                .step_with_fixed_hist_ctx(&h, query, &hist_ctx, None);
        }
        h
    }

    fn temporal_update_from_h(&self, m_prev: Option<&HSlots>, h_star: &HSlots) -> HSlots {
        let zero_state;
        let prev = if let Some(m_prev) = m_prev {
            m_prev
        } else {
            zero_state = HSlots::zeros(&self.config);
            &zero_state
        };
        self.reasoning.temporal_step(prev, h_star)
    }

    #[cfg(feature = "wgpu")]
    fn cached_loss_after_sync(&self, gpu: &GpuDeqBackend) -> Option<f32> {
        let every = self.cfg_loss_readback_every;
        let should_read =
            every != 0 && (every == 1 || (self.optimizer.step_count() % every == 0));
        if should_read {
            self
                .gpu_lm
                .as_ref()
                .map(|gpu_lm| gpu_lm.read_cached_loss(&gpu.device))
        } else {
            None
        }
    }

    fn prime_fixed_history_state(&self, tokens: &[u32]) -> Option<HSlots> {
        if tokens.is_empty() {
            return None;
        }

        let mut m_prev: Option<HSlots> = None;
        for &token in tokens {
            let query = self.tokenizer.embed(token);
            let h_star = self.solve_query_with_fixed_history(
                &query,
                m_prev.as_ref(),
                self.config.max_deq_iters,
            );
            m_prev = Some(self.temporal_update_from_h(m_prev.as_ref(), &h_star));
        }
        m_prev
    }

    #[cfg(feature = "wgpu")]
    fn upload_reasoning_weights_with_slot_anchor(
        &self,
        gpu: &GpuDeqBackend,
        slot_anchor_rm: &[f32],
    ) {
        let (
            w_hist_shared_rm,
            hist_slot_scale_rm,
            hist_slot_bias_rm,
            hist_gate_logit,
            _slot_anchor_base_rm,
            w_delta_rm,
            b_delta,
            w_gate_hist_rm,
            w_forget_rm,
            b_forget_rm,
        ) = self.reasoning.history_params_gpu_layout();
        gpu.upload_weights(
            &gpu.queue,
            &self.reasoning.w_q_gpu_flat(),
            &self.reasoning.w_k_gpu_flat(),
            &self.reasoning.w_v_gpu_flat(),
            &self.reasoning.w_o_gpu_flat(),
            &self.reasoning.w_in_gpu_flat(),
            self.reasoning.w_x.as_slice(),
            self.reasoning.w_out.as_slice(),
            &self.reasoning.a_log_gpu_flat(),
            self.reasoning.norm_scale.as_slice(),
            w_hist_shared_rm.as_slice(),
            hist_slot_scale_rm.as_slice(),
            hist_slot_bias_rm.as_slice(),
            hist_gate_logit.as_slice(),
            slot_anchor_rm,
            w_delta_rm.as_slice(),
            b_delta.as_slice(),
            w_gate_hist_rm.as_slice(),
            w_forget_rm.as_slice(),
            b_forget_rm.as_slice(),
        );
    }

    #[cfg(feature = "wgpu")]
    fn gpu_solve_token_with_fixed_history(
        &self,
        gpu: &GpuDeqBackend,
        token: u32,
        m_prev: Option<&HSlots>,
    ) -> Result<(nalgebra::DVector<f32>, HSlots, HSlots), &'static str> {
        let query = self.tokenizer.embed(token);
        let hist_ctx = self.reasoning.fixed_hist_ctx(m_prev, &query);
        let (
            _w_hist_shared_rm,
            _hist_slot_scale_rm,
            _hist_slot_bias_rm,
            _hist_gate_logit,
            slot_anchor_rm,
            _w_delta_rm,
            _b_delta,
            _w_gate_hist_rm,
            _w_forget_rm,
            _b_forget_rm,
        ) = self.reasoning.history_params_gpu_layout();
        let hist_flat = hist_ctx.to_flat();
        let mut slot_anchor_eff = slot_anchor_rm;
        for (dst, add) in slot_anchor_eff.iter_mut().zip(hist_flat.iter()) {
            *dst += *add;
        }

        self.upload_reasoning_weights_with_slot_anchor(gpu, slot_anchor_eff.as_slice());
        let s_sequence = self.tokenizer.embed_sequence(&[token]);
        let (h_pooled, h_star_flat, _) = gpu.run_forward_deq_pooled_with_state(
            1,
            1,
            self.config.max_deq_iters as u32,
            self.config.deq_epsilon,
            self.reasoning.damping,
            &s_sequence,
            &self.reasoning.w_q_gpu_flat(),
            &self.reasoning.w_k_gpu_flat(),
            &self.reasoning.w_v_gpu_flat(),
            &self.reasoning.w_o_gpu_flat(),
            &self.reasoning.w_in_gpu_flat(),
            self.reasoning.w_x.as_slice(),
            self.reasoning.w_out.as_slice(),
            &self.reasoning.a_log_gpu_flat(),
            self.reasoning.norm_scale.as_slice(),
            false,
        )?;

        let h_star = HSlots::from_flat(&h_star_flat, &self.config);
        let m_next = self.temporal_update_from_h(m_prev, &h_star);
        Ok((nalgebra::DVector::from_vec(h_pooled), h_star, m_next))
    }

    fn gib(bytes: f64) -> f64 {
        bytes / (1024.0 * 1024.0 * 1024.0)
    }

    fn clean_deq_mode_active() -> bool {
        if Self::env_flag("AIDEEN_DEQ_ONLY") {
            return true;
        }
        if Self::env_flag("AIDEEN_DEQ_SLOT_ATTN_REAL_STAGED")
            || Self::env_flag("AIDEEN_DEQ_SLOT_ATTN_REAL_UNIFIED")
        {
            return false;
        }
        DeqSolveMode::from_env().is_clean_core()
    }

    fn default_hist_min_iters() -> u32 {
        if Self::clean_deq_mode_active() { 1 } else { 20 }
    }

    fn estimate_forward_bandwidth_bytes(
        &self,
        batch_size: u32,
        seq_len: u32,
        max_iters: u32,
        d_r: usize,
        h_slots: usize,
    ) -> (f64, f64, f64, f64) {
        let d = d_r as f64;
        let h = h_slots as f64;
        let b = batch_size as f64;
        let t = seq_len as f64;
        let iters = max_iters as f64;
        let f32_bytes = std::mem::size_of::<f32>() as f64;
        let slot_attn_unified = self.cfg_slot_attn_unified;
        let dynamic_qkv = self.cfg_dynamic_qkv;

        if self.cfg_clean_deq_mode {
            let win_bytes = b * t * (h * d * d * f32_bytes);
            return (0.0, 0.0, win_bytes, 0.0);
        }

        // Lower bounds derived from the hot dense matrices touched in deq_forward.wgsl.
        let attn_iters = if slot_attn_unified && !dynamic_qkv {
            1.0
        } else {
            iters
        };
        let qkv_bytes = b * t * attn_iters * (3.0 * h * d * d * f32_bytes);
        let wo_bytes = b * t * attn_iters * (h * d * d * f32_bytes);
        let win_bytes = b * t * (h * d * d * f32_bytes);
        let hist_lower_bound = b * t * ((d * d + h * d * d + d * d) * f32_bytes);

        (qkv_bytes, wo_bytes, win_bytes, hist_lower_bound)
    }

    fn decode_forward_debug_metrics(
        fw: &[f32],
        h_slots: usize,
    ) -> (f32, f32, f32, f32, f32, f32, f32) {
        if fw.len() > 11 && fw[8] == 901.0 {
            let seq = fw[10].max(1.0);
            let slot_count = fw[11]
                .max(1.0)
                .min(h_slots as f32)
                .round() as usize;
            let mut max_delta = 0.0f32;
            let mut hit_count = 0.0f32;
            let mut avg_iters_sum = 0.0f32;
            let mut contractivity = 0.0f32;
            let mut max_h = 0.0f32;
            for slot in 0..slot_count {
                let base = 32 + slot * 5;
                if fw.len() <= base + 4 {
                    break;
                }
                max_delta = max_delta.max(fw[base]);
                hit_count += fw[base + 1];
                avg_iters_sum += fw[base + 2];
                contractivity = contractivity.max(fw[base + 3]);
                max_h = max_h.max(fw[base + 4]);
            }
            let avg_iters = if slot_count > 0 {
                avg_iters_sum / slot_count as f32
            } else {
                0.0
            };
            let hit_den = seq * slot_count.max(1) as f32;
            return (seq, max_h, avg_iters, hit_count, max_delta, contractivity, hit_den);
        }

        let heartbeat = if fw.len() > 10 { fw[10] } else { 0.0 };
        let max_h = if fw.len() > 11 { fw[11] } else { 0.0 };
        let avg_iters = if fw.len() > 13 { fw[13] } else { 0.0 };
        let hit_count = if fw.len() > 15 { fw[15] } else { 0.0 };
        let max_delta = if fw.len() > 16 { fw[16] } else { 0.0 };
        let contractivity = if fw.len() > 21 { fw[21] } else { 0.0 };
        (
            heartbeat.max(1.0),
            max_h,
            avg_iters,
            hit_count,
            max_delta,
            contractivity,
            heartbeat.max(1.0),
        )
    }

    fn lmhead_backward_sampled(
        sampled_indices: &[u32],
        target: u32,
        h_pooled: &nalgebra::DVector<f32>,
        w_lm: &nalgebra::DMatrix<f32>,
        b_lm: &nalgebra::DVector<f32>,
        g: &nalgebra::DVector<f32>,
    ) -> nalgebra::DVector<f32> {
        let eps = 1e-5_f32;
        let d = h_pooled.len() as f32;
        let mean_sq = h_pooled.map(|v| v * v).mean();
        let rms = (mean_sq + eps).sqrt();
        let h_norm = h_pooled.map(|v| v / rms);
        let h_rms = h_norm.component_mul(g);

        let mut logits = Vec::with_capacity(sampled_indices.len());
        let mut max_logit = f32::NEG_INFINITY;
        for &v in sampled_indices {
            let mut logit = b_lm[v as usize];
            for i in 0..h_rms.len() {
                logit += w_lm[(v as usize, i)] * h_rms[i];
            }
            max_logit = max_logit.max(logit);
            logits.push(logit);
        }

        let mut sum = 0.0;
        for val in logits.iter_mut() {
            *val = (*val - max_logit).exp();
            sum += *val;
        }
        let inv_sum = 1.0 / sum.max(1e-8);

        let mut dl_dh_rms = nalgebra::DVector::zeros(h_pooled.len());
        for (idx, &v) in sampled_indices.iter().enumerate() {
            let mut p = logits[idx] * inv_sum;
            if v == target {
                p -= 1.0;
            }
            for i in 0..dl_dh_rms.len() {
                dl_dh_rms[i] += w_lm[(v as usize, i)] * p;
            }
        }

        let dx = dl_dh_rms.component_mul(g) / rms;
        let sum_dx_h = dx.dot(h_pooled);
        dx - h_pooled.map(|v| v * sum_dx_h / (d * rms * rms))
    }

    fn apply_experimental_profile_from_env(&mut self) {
        let exp = Self::env_flag("AIDEEN_DEQ_EXPERIMENTAL");
        let alpha_env = Self::env_f32("AIDEEN_DEQ_RESIDUAL_ALPHA").map(|v| v.clamp(0.0, 1.0));
        let alpha = alpha_env.unwrap_or(0.0); // v14: Matemáticamente probado que requiere 0.0
        self.reasoning.residual_alpha = alpha;

        if exp {
            // Experimental DEQ stabilization profile.
            self.config.renorm_every_steps = 1;
            self.config.max_deq_iters = self.config.max_deq_iters.min(8).max(4);
            self.config.adj_iters = self.config.adj_iters.min(8).max(4);
            self.adaptive_max_iters = self.adaptive_max_iters.min(8).max(4);
            self.adaptive_adj_iters = self.adaptive_adj_iters.min(8).max(4);
            self.adaptive_damping = 0.80;
            self.reasoning.damping = self.adaptive_damping;
        }
    }

    /// Crea un Trainer con un tokenizer pre-construido.
    pub fn from_tokenizer(tokenizer: Tokenizer, lr: f32) -> Self {
        let config = tokenizer.config.clone();

        #[cfg(feature = "wgpu")]
        let gpu_deq = GpuDeqBackend::new_blocking(config.clone());

        let mut trainer = Self {
            reasoning: MambaSlotReasoning::new(config.clone()),
            #[cfg(feature = "wgpu")]
            gpu_deq,
            #[cfg(feature = "wgpu")]
            gpu_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            gpu_cg_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            gpu_lm: None,
            #[cfg(feature = "wgpu")]
            gpu_emb: None,
            #[cfg(feature = "wgpu")]
            gpu_lm_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            gpu_emb_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            lm_head_cpu_stale: false,
            lm_head: LmHead::new(config.clone()),
            tokenizer,
            optimizer: Adam::new(lr),
            training_config: TrainingConfig {
                lr,
                ..Default::default()
            },
            config,
            frozen_deq: false,
            frozen_emb: false,
            frozen_lm: false,
            eval_mode: false,
            // --- v13.2 Stability Oracle ---
            adaptive_max_iters: 12,
            adaptive_damping: 0.85,
            adaptive_adj_iters: 8,
            hit_hi_streak: 0,
            hit_lo_streak: 0,
            cg_res_hi_streak: 0,
            damping_boost_left: 0,
            emergency_left: 0,
            last_max_h: 0.0,
            max_h_growth_streak: 0,
            max_delta_hi_streak: 0,
            invalid_hi_streak: 0,
            contractivity_hi_streak: 0,
            force_renorm_done: false,
            m_prev: None,
            grad_accum_counter: 0,
            debug_last_time: None,
            debug_tokens_accum: 0,
            cached_debug_buf: Vec::new(),
            last_gpu_loss: 0.0,
            cfg_fwd_batch_size: std::env::var("AIDEEN_BATCH_SIZE")
                .ok().and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(1).max(1),
            cfg_debug_sample_every: std::env::var("AIDEEN_DEBUG_SAMPLE")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0),
            cfg_loss_readback_every: std::env::var("AIDEEN_LOSS_READBACK_EVERY")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0),
            cfg_tps_sync_every: std::env::var("AIDEEN_TPS_SYNC_EVERY")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0),
            cfg_grad_accum: std::env::var("AIDEEN_GRAD_ACCUM")
                .ok().and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(1).max(1),
            cfg_hist_min_iters: std::env::var("AIDEEN_HIST_MIN_ITERS")
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or_else(Self::default_hist_min_iters)
                .max(1),
            cfg_val_every: std::env::var("AIDEEN_VAL_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            cfg_progress_every: std::env::var("AIDEEN_PROGRESS_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            cfg_progress_strict_sync: Self::env_flag("AIDEEN_PROGRESS_STRICT_SYNC"),
            cfg_wv_debug: Self::env_flag("AIDEEN_DEQ_WV_DEBUG"),
            cfg_ssm_debug: Self::env_flag("AIDEEN_SSM_DEBUG"),
            cfg_max_chunks: std::env::var("AIDEEN_MAX_CHUNKS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(usize::MAX),
            cfg_adj_iters_override: std::env::var("AIDEEN_ADJ_ITERS_OVERRIDE")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok()),
            cfg_system_cost_audit: Self::env_flag("AIDEEN_SYSTEM_COST_AUDIT"),
            cfg_system_cost_wait: Self::env_flag("AIDEEN_SYSTEM_COST_WAIT"),
            cfg_force_renorm: Self::env_flag("AIDEEN_DEQ_FORCE_RENORM"),
            cfg_slot_attn_unified: Self::env_flag("AIDEEN_DEQ_SLOT_ATTN_REAL_UNIFIED"),
            cfg_dynamic_qkv: Self::env_flag("AIDEEN_DEQ_SLOT_ATTN_DYNAMIC_QKV"),
            cfg_lm_force_cpu_dldh: Self::env_flag("AIDEEN_LM_FORCE_CPU_DLDH"),
            cfg_lm_dldh_parity: Self::env_flag("AIDEEN_LM_DLDH_PARITY"),
            cfg_clean_deq_mode: Self::clean_deq_mode_active(),
            cfg_log_emb_stats: Self::env_flag("AIDEEN_LOG_EMB_STATS"),
            metrics_log_path: None,
            metrics_pending: std::collections::VecDeque::new(),
        };
        trainer.apply_experimental_profile_from_env();
        trainer
    }

    /// Igual que `from_tokenizer`, pero forzando inicialización determinística
    /// de los pesos de reasoning (DEQ core) para reproducibilidad por seed.
    pub fn from_tokenizer_seeded(tokenizer: Tokenizer, lr: f32, seed: u64) -> Self {
        let config = tokenizer.config.clone();

        #[cfg(feature = "wgpu")]
        let gpu_deq = GpuDeqBackend::new_blocking(config.clone());

        let mut trainer = Self {
            reasoning: MambaSlotReasoning::new_with_seed(config.clone(), seed),
            #[cfg(feature = "wgpu")]
            gpu_deq,
            #[cfg(feature = "wgpu")]
            gpu_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            gpu_cg_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            gpu_lm: None,
            #[cfg(feature = "wgpu")]
            gpu_emb: None,
            #[cfg(feature = "wgpu")]
            gpu_lm_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            gpu_emb_weights_uploaded: false,
            #[cfg(feature = "wgpu")]
            lm_head_cpu_stale: false,
            lm_head: LmHead::new(config.clone()),
            tokenizer,
            optimizer: Adam::new(lr),
            training_config: TrainingConfig {
                lr,
                ..Default::default()
            },
            config,
            frozen_deq: false,
            frozen_emb: false,
            frozen_lm: false,
            eval_mode: false,
            adaptive_max_iters: 12,
            adaptive_damping: 0.85,
            adaptive_adj_iters: 8,
            hit_hi_streak: 0,
            hit_lo_streak: 0,
            cg_res_hi_streak: 0,
            damping_boost_left: 0,
            emergency_left: 0,
            last_max_h: 0.0,
            max_h_growth_streak: 0,
            max_delta_hi_streak: 0,
            invalid_hi_streak: 0,
            contractivity_hi_streak: 0,
            force_renorm_done: false,
            m_prev: None,
            grad_accum_counter: 0,
            debug_last_time: None,
            debug_tokens_accum: 0,
            cached_debug_buf: Vec::new(),
            last_gpu_loss: 0.0,
            cfg_fwd_batch_size: std::env::var("AIDEEN_BATCH_SIZE")
                .ok().and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(1).max(1),
            cfg_debug_sample_every: std::env::var("AIDEEN_DEBUG_SAMPLE")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0),
            cfg_loss_readback_every: std::env::var("AIDEEN_LOSS_READBACK_EVERY")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0),
            cfg_tps_sync_every: std::env::var("AIDEEN_TPS_SYNC_EVERY")
                .ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0),
            cfg_grad_accum: std::env::var("AIDEEN_GRAD_ACCUM")
                .ok().and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(1).max(1),
            cfg_hist_min_iters: std::env::var("AIDEEN_HIST_MIN_ITERS")
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or_else(Self::default_hist_min_iters)
                .max(1),
            cfg_val_every: std::env::var("AIDEEN_VAL_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            cfg_progress_every: std::env::var("AIDEEN_PROGRESS_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            cfg_progress_strict_sync: Self::env_flag("AIDEEN_PROGRESS_STRICT_SYNC"),
            cfg_wv_debug: Self::env_flag("AIDEEN_DEQ_WV_DEBUG"),
            cfg_ssm_debug: Self::env_flag("AIDEEN_SSM_DEBUG"),
            cfg_max_chunks: std::env::var("AIDEEN_MAX_CHUNKS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(usize::MAX),
            cfg_adj_iters_override: std::env::var("AIDEEN_ADJ_ITERS_OVERRIDE")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok()),
            cfg_system_cost_audit: Self::env_flag("AIDEEN_SYSTEM_COST_AUDIT"),
            cfg_system_cost_wait: Self::env_flag("AIDEEN_SYSTEM_COST_WAIT"),
            cfg_force_renorm: Self::env_flag("AIDEEN_DEQ_FORCE_RENORM"),
            cfg_slot_attn_unified: Self::env_flag("AIDEEN_DEQ_SLOT_ATTN_REAL_UNIFIED"),
            cfg_dynamic_qkv: Self::env_flag("AIDEEN_DEQ_SLOT_ATTN_DYNAMIC_QKV"),
            cfg_lm_force_cpu_dldh: Self::env_flag("AIDEEN_LM_FORCE_CPU_DLDH"),
            cfg_lm_dldh_parity: Self::env_flag("AIDEEN_LM_DLDH_PARITY"),
            cfg_clean_deq_mode: Self::clean_deq_mode_active(),
            cfg_log_emb_stats: Self::env_flag("AIDEEN_LOG_EMB_STATS"),
            metrics_log_path: None,
            metrics_pending: std::collections::VecDeque::new(),
        };
        trainer.apply_experimental_profile_from_env();
        trainer
    }

    #[cfg(feature = "wgpu")]
    fn take_gpu(&mut self) -> Option<GpuDeqBackend> {
        self.gpu_deq.take()
    }

    #[cfg(feature = "wgpu")]
    fn put_gpu(&mut self, gpu: GpuDeqBackend) {
        self.gpu_deq = Some(gpu);
    }

    fn damping_eff(&self) -> f32 {
        if self.emergency_left > 0 {
            0.60
        } else if self.damping_boost_left > 0 {
            0.75
        } else {
            self.adaptive_damping.clamp(0.55, 0.95)
        }
    }

    #[cfg(feature = "wgpu")]
    fn ensure_gpu_trainers(&mut self, gpu: &GpuDeqBackend) {
        if self.gpu_lm.is_none() {
            self.gpu_lm = Some(GpuLmHeadTrainer::new(
                &gpu.device,
                self.lm_head.b.len(),
                self.config.clone(),
            ));
        }

        if self.gpu_emb.is_none() {
            let batch_size = self.cfg_fwd_batch_size.max(1) as usize;
            let safe_ctx = self.config.ctx_len.max(1024) * batch_size;
            self.gpu_emb = Some(GpuEmbeddingTrainer::new(
                &gpu.device,
                self.tokenizer.vocab_size(),
                safe_ctx,
                self.config.clone(),
            ));
        }
    }

    /// Reinicia los estados cognitivos (slots) tanto en CPU como en GPU.
    pub fn reset_state(&mut self) {
        // MambaSlotReasoning es stateless entre llamadas: el DEQ recomputa h* desde cero
        // en cada forward pass, por lo que no hay estado oculto persistente que limpiar.
        // reset_state sirve para forzar que la próxima secuencia no comparta contexto.
        self.m_prev = None;
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = self.gpu_deq.as_ref() {
            gpu.reset_state();
        }
    }

    /// Ejecuta un paso de entrenamiento dado un slice de tokens.
    /// `context`: tokens de contexto (input)
    /// `target`: token a predecir
    /// `reset_state`: si es true, limpia el estado oculto antes de procesar.
    pub fn train_step(&mut self, context: &[u32], target: u32, reset_state: bool) -> f32 {
        if reset_state {
            self.reset_state();
        }
        // IMPORTANT:
        // AIDEEN_DEQ_FIXED_HISTORY_REFERENCE is an eval/inference reference path only.
        // Training must stay on the fused GPU path so history experiments do not silently
        // change the execution regime we use for performance and quality comparisons.

        #[cfg(feature = "wgpu")]
        {
            if self.gpu_deq.is_some() {
                let gpu = self.take_gpu().expect("gpu_deq checked as Some");

                let out = (|| {
                    self.ensure_gpu_trainers(&gpu);

                    let Some(gpu_emb) = self.gpu_emb.as_ref() else {
                        return 0.0;
                    };

                    let emb_needs_upload = !self.gpu_emb_weights_uploaded;
                    let (_s_sequence, query_vec) = match gpu_emb.prepare_sequence_and_query(
                        &gpu.device,
                        &gpu.queue,
                        context,
                        self.config.ctx_len,
                        self.tokenizer.embeddings.as_slice(),
                        emb_needs_upload,
                    ) {
                        Ok(v) => {
                            self.gpu_emb_weights_uploaded = true;
                            v
                        }
                        Err(_) => {
                            self.gpu_emb_weights_uploaded = false;
                            return 0.0;
                        }
                    };

                    let query = nalgebra::DVector::from_vec(query_vec);
                    self.apply_training_update_from_gpu_buffers(
                        context,
                        &[target],
                        &query,
                        Some(&gpu),
                        self.config.deq_epsilon,
                    )
                })();

                self.put_gpu(gpu);
                return out;
            }
        }
        0.0
    }

    /// Paso de entrenamiento para una secuencia completa (Sequence Fusing).
    /// Procesa 1..N tokens en una única ráfaga GPU.
    /// `reset_state`: si es true, limpia el estado oculto antes de empezar la secuencia.
    pub fn train_sequence(
        &mut self,
        tokens: &[u32],
        targets: &[u32],
        reset_state: bool,
        epsilon: f32,
    ) -> f32 {
        if reset_state {
            self.reset_state();
        }
        // Keep fixed-history out of the hot training path. The reference mode is useful to
        // validate semantics in eval/generation, but training comparisons must remain fused
        // so we isolate the value of history from the cost of a different execution engine.

        // hist_gated is default — always enforce min_iters for stable history injection.
        {
            let min_iters = self.cfg_hist_min_iters;
            if self.adaptive_max_iters < min_iters {
                self.adaptive_max_iters = min_iters;
            }
        }

        #[cfg(feature = "wgpu")]
        if self.gpu_deq.is_some() {
            let gpu = self.take_gpu().expect("gpu_deq checked as Some");

            let out = (|| {
                self.ensure_gpu_trainers(&gpu);

                // Arreglo defensivo para evitar underflow si seq_len < ctx_len.
                // Para batch > 1 el training loop pasa B*ctx_len tokens — no truncar.
                let fwd_batch_size_ts: usize = self.cfg_fwd_batch_size.max(1) as usize;
                let seq_len = tokens.len().min(targets.len());
                let actual_ctx_len = if fwd_batch_size_ts > 1 {
                    seq_len
                } else {
                    seq_len.min(self.config.ctx_len)
                };
                let ctx = &tokens[seq_len - actual_ctx_len..];
                let ctx_targets = &targets[seq_len - actual_ctx_len..];

                let Some(_gpu_emb) = self.gpu_emb.as_ref() else {
                    return 0.0;
                };

                // Reasoning weights synchronization (identico)
                if !self.gpu_weights_uploaded {
                    let (
                        w_hist_shared_rm,
                        hist_slot_scale_rm,
                        hist_slot_bias_rm,
                        hist_gate_logit,
                        slot_anchor_rm,
                        w_delta_rm,
                        b_delta,
                        w_gate_hist_rm,
                        w_forget_rm,
                        b_forget_rm,
                    ) = self.reasoning.history_params_gpu_layout();
                    gpu.upload_weights(
                        &gpu.queue,
                        &self.reasoning.w_q_gpu_flat(),
                        &self.reasoning.w_k_gpu_flat(),
                        &self.reasoning.w_v_gpu_flat(),
                        &self.reasoning.w_o_gpu_flat(),
                        &self.reasoning.w_in_gpu_flat(),
                        self.reasoning.w_x.as_slice(),
                        self.reasoning.w_out.as_slice(),
                        &self.reasoning.a_log_gpu_flat(),
                        self.reasoning.norm_scale.as_slice(),
                        w_hist_shared_rm.as_slice(),
                        hist_slot_scale_rm.as_slice(),
                        hist_slot_bias_rm.as_slice(),
                        hist_gate_logit.as_slice(),
                        slot_anchor_rm.as_slice(),
                        w_delta_rm.as_slice(),
                        b_delta.as_slice(),
                        w_gate_hist_rm.as_slice(),
                        w_forget_rm.as_slice(),
                        b_forget_rm.as_slice(),
                    );
                    self.gpu_weights_uploaded = true;
                    self.gpu_cg_weights_uploaded = true;
                }
                // Dynamic slot-attention re-enters the DEQ Jacobian; ensure the GPU buffers
                // start from the strict spectral regime before the first training step.
                if !self.force_renorm_done
                    && (self.cfg_force_renorm || self.cfg_slot_attn_unified)
                {
                    let _ = gpu.renormalize_spectral();
                    self.force_renorm_done = true;
                }

                // Pipeline GPU fused
                self.apply_training_update_from_gpu_buffers(
                    ctx,
                    ctx_targets,
                    &nalgebra::DVector::zeros(0),
                    Some(&gpu),
                    epsilon,
                )
            })();

            self.put_gpu(gpu);
            return out;
        }
        // CPU fallback: recorre la secuencia por pasos autoregresivos.
        let seq_len = tokens.len().min(targets.len());
        if seq_len == 0 {
            return 0.0;
        }

        let mut loss_sum = 0.0f32;
        let mut count = 0usize;
        for i in 0..seq_len {
            let ctx_start = i.saturating_sub(self.config.ctx_len.saturating_sub(1));
            let context = &tokens[ctx_start..=i];
            let loss = self.train_step(context, targets[i], false);
            if loss.is_finite() {
                loss_sum += loss;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            loss_sum / count as f32
        }
    }

    #[cfg(feature = "wgpu")]
    fn apply_training_update_from_gpu_buffers(
        &mut self,
        context: &[u32],
        targets: &[u32],
        query: &nalgebra::DVector<f32>,
        gpu_ctx: Option<&GpuDeqBackend>,
        epsilon: f32,
    ) -> f32 {
        let damping_eff = self.damping_eff();
        self.reasoning.damping = damping_eff;
        let base_lr = if self.emergency_left > 0 {
            self.optimizer.lr * 0.5
        } else {
            self.optimizer.lr
        };
        let mut lm_lr = if self.frozen_lm || self.eval_mode {
            0.0
        } else {
            base_lr * self.training_config.lm_lr_mult
        };
        let audit_cost = self.cfg_system_cost_audit;
        let mut audit_gather_ms = 0.0f64;
        let mut audit_forward_ms = 0.0f64;
        let mut audit_lm_ms = 0.0f64;
        let mut audit_picard_ms = 0.0f64;
        let mut audit_update_ms = 0.0f64;
        let mut audit_embed_ms = 0.0f64;
        let mut audit_sync_ms = 0.0f64;
        let mut audit_renorm_ms = 0.0f64;
        let mut audit_renorm_calls = 0u32;

        // Sync CPU lm_head only when frozen/debug paths need it — not on every training step.
        #[cfg(feature = "wgpu")]
        if self.frozen_lm
            || self.cfg_lm_force_cpu_dldh
            || self.cfg_lm_dldh_parity
        {
            self.sync_lm_head_from_gpu_if_needed();
        }

        let num_tokens = targets.len();
        let fwd_batch_size = self.cfg_fwd_batch_size;
        let per_seq_len = if fwd_batch_size == 0 {
            0
        } else {
            (num_tokens as u32) / fwd_batch_size
        };
        let audit_forward_bytes = self.estimate_forward_bandwidth_bytes(
            fwd_batch_size,
            per_seq_len,
            self.adaptive_max_iters,
            self.config.d_r,
            self.config.h_slots,
        );

        if let (Some(gpu), Some(gpu_lm), Some(gpu_emb)) =
            (gpu_ctx, self.gpu_lm.as_mut(), self.gpu_emb.as_ref())
        {
            self.optimizer.tick();
            if num_tokens == 0 {
                return 0.0;
            }
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();

            // 1. Prepare DEQ input on GPU
            // per_seq_len = tokens per sequence (num_tokens / B for batch mode, or 1 for query mode)
            let single_query_mode = query.len() == self.config.d_r && num_tokens == 1;
            if single_query_mode {
                // train_step path: match CPU semantics by feeding the pooled query vector.
                let q_bytes = bytemuck::cast_slice(query.as_slice());
                gpu.queue.write_buffer(&gpu.bridge.s_buf, 0, q_bytes);
                // Align DEQ init with CPU `reasoning.init(query)`: broadcast query to all slots.
                // Without this, GPU starts from zeroed H_curr after reset and diverges in h* parity.
                let d_r = self.config.d_r;
                let h_slots = self.config.h_slots;
                let mut h_init = vec![0.0f32; h_slots * d_r];
                for k in 0..h_slots {
                    let off = k * d_r;
                    h_init[off..off + d_r].copy_from_slice(query.as_slice());
                }
                gpu.queue
                    .write_buffer(&gpu.bridge.hcurr_buf, 0, bytemuck::cast_slice(&h_init));
                gpu.queue
                    .write_buffer(&gpu.bridge.hnext_buf, 0, bytemuck::cast_slice(&h_init));
            } else {
                // train_sequence path: use token sequence embeddings.
                let gather_t0 = std::time::Instant::now();
                let emb_needs_upload = !self.gpu_emb_weights_uploaded;
                let _ = gpu_emb.gather_only_to_sbuf(
                    &gpu.queue,
                    &gpu.device,
                    context,
                    self.tokenizer.embeddings.as_slice(),
                    emb_needs_upload,
                    &gpu.bridge.s_buf,
                );
                if emb_needs_upload {
                    self.gpu_emb_weights_uploaded = true;
                }
                if audit_cost {
                    audit_gather_ms += gather_t0.elapsed().as_secs_f64() * 1e3;
                }
            }

            // 2. DEQ Forward (GPU-Only) - v13.1 Adaptive
            let debug_every = self.cfg_debug_sample_every;
            let debug_enable = debug_every != 0
                && (self.optimizer.step_count() % debug_every == 0);
            let forward_t0 = std::time::Instant::now();
            let _ = gpu.run_forward(
                fwd_batch_size,
                per_seq_len,
                self.adaptive_max_iters,
                damping_eff,
                epsilon,
                debug_enable,
            );
            if audit_cost && self.cfg_system_cost_wait {
                // Audit-only: force GPU completion so forward_ms reflects execution, not enqueue.
                gpu.device.poll(wgpu::Maintain::Wait);
            }
            if audit_cost {
                audit_forward_ms += forward_t0.elapsed().as_secs_f64() * 1e3;
            }
            // Use cached debug buffer — refresh deferred to end of step (after GPU is idle).
            // The DEQ-INVALID streak check needs 3 consecutive failures, so 1-step lag is safe.
            let fw = self.cached_debug_buf.clone();
            let (seq, _max_h_dbg, _avg_iters_dbg, hit_count, max_delta, contractivity, hit_den) =
                Self::decode_forward_debug_metrics(&fw, self.config.h_slots);
            let hit_ratio = hit_count.max(0.0) / hit_den.max(1.0);
            // DEQ-INVALID: only when the system FAILED to converge (maxΔ >> epsilon) while
            // also being non-contractive. Non-monotone convergence (contr transiently > 1
            // but maxΔ ≈ epsilon) is a normal property of non-linear Picard iterations and
            // does NOT indicate an invalid fixed point — the system DID find h*.
            let invalid_fixed_point =
                contractivity > 1.0 && max_delta > self.config.deq_epsilon * 10.0;
            if invalid_fixed_point {
                self.invalid_hi_streak += 1;
            } else {
                self.invalid_hi_streak = 0;
            }
            if self.invalid_hi_streak >= 3 {
                eprintln!(
                    "    [DEQ-INVALID] step={} contr={:.3} hit_ratio={:.3} maxΔ={:.3e} seq={:.0}",
                    self.optimizer.step_count(),
                    contractivity,
                    hit_ratio,
                    max_delta,
                    seq
                );
                self.invalid_hi_streak = 0;
                self.emergency_left = 3;
                self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(48);
                #[cfg(feature = "wgpu")]
                {
                    let renorm_t0 = std::time::Instant::now();
                    let _ = gpu.renormalize_spectral();
                    if audit_cost {
                        audit_renorm_ms += renorm_t0.elapsed().as_secs_f64() * 1e3;
                        audit_renorm_calls += 1;
                    }
                }
                lm_lr = 0.0;
            }
            if !self.gpu_lm_weights_uploaded {
                // Export weights in Row-Major format as required by LM Head shader
                let w_head = self.lm_head.export_weights();
                let w_raw = w_head.get("head.w").unwrap();
                let b_raw = w_head.get("head.b").unwrap();
                let g_raw = w_head.get("head.g").unwrap();

                let w_sum: f32 = w_raw.iter().map(|&x| x.abs()).sum();
                println!(
                    "[GPU-LM] Sincronizando pesos del LM Head... (abs_sum={:.4})",
                    w_sum
                );

                gpu_lm.upload_weights_only(&gpu.queue, w_raw, b_raw, g_raw);
                self.gpu_lm_weights_uploaded = true;
            }

            // Submit LM forward+backward without blocking for loss readback.
            // For eval_mode (validation) we still read synchronously since there's no adjoint.
            // For train mode, loss is read after apply_gradient_update when GPU is already idle.
            let read_loss_now = self.eval_mode;
            let lm_t0 = std::time::Instant::now();
            let current_loss_sync = gpu_lm
                .train_step_no_readback(
                    &gpu.device,
                    &gpu.queue,
                    &gpu.bridge.hpooled_buf,
                    0,
                    &targets_u32,
                    lm_lr,
                    self.optimizer.step_count() as u32,
                    self.training_config.ternary,
                    read_loss_now,
                )
                .unwrap_or(0.0);
            if audit_cost {
                audit_lm_ms += lm_t0.elapsed().as_secs_f64() * 1e3;
            }
            if self.eval_mode {
                self.last_gpu_loss = current_loss_sync;
            }
            let current_loss = self.last_gpu_loss;
            if lm_lr > 0.0 {
                self.lm_head_cpu_stale = true;
            }

            // 4. Embedding Update from GPU dl_dh buffer (Moved to step 6 to avoid duplication)

            // 5. DEQ Reasoning Core Update (Picard Adjoint + Fused GPU Weight Update)
            if self.eval_mode {
                return current_loss;
            }

            if !self.frozen_deq && !invalid_fixed_point {
                // ⑥ Backward DEQ — Picard Adjoint (GPU, siempre).
                // Skip when DEQ diverged (invalid_fixed_point): gradients from a non-converged
                // forward pass are unreliable (∂L/∂θ via implicit diff requires h* to exist).
                // staged Picard llena fused_mix_buf con g_comb, luego apply_fused_deq_update
                // aplica el weight update completo en GPU. Un solo path, siempre correcto.
                let batch_size = fwd_batch_size;
                let picard_t0 = std::time::Instant::now();
                let _ = gpu.run_staged_adjoint_picard_no_readback(
                    per_seq_len,
                    self.reasoning.damping,
                    self.config.adj_iters as u32,
                    Some(&gpu_lm.dl_dh_buf),
                    true, // clear fused_hist_ctx_buf (rhs_slot) before adjoint — eliminates hist rerun
                    batch_size,
                );
                if audit_cost {
                    audit_picard_ms += picard_t0.elapsed().as_secs_f64() * 1e3;
                }
                let grad_accum = self.cfg_grad_accum;
                // Cross-step gradient accumulation:
                // Each train_step() call accumulates gradients from a different sequence.
                // Weight update is applied only every grad_accum steps.
                let mode = if grad_accum == 1 { 0u32 } else { 1u32 };
                let apply_accum = grad_accum == 1 || self.grad_accum_counter + 1 >= grad_accum;
                let update_t0 = std::time::Instant::now();
                let _ = gpu.apply_fused_deq_update(
                    base_lr,
                    self.config.deq_grad_scale,
                    self.training_config.ternary,
                    self.config.weight_decay,
                    per_seq_len,
                    self.reasoning.damping,
                    mode,
                    grad_accum,
                    batch_size,
                    apply_accum,
                );
                if audit_cost {
                    audit_update_ms += update_t0.elapsed().as_secs_f64() * 1e3;
                }
                self.grad_accum_counter += 1;
                if self.grad_accum_counter >= grad_accum {
                    self.grad_accum_counter = 0;
                    // Refresh debug buffer cache if this is a sample step.
                    let debug_every = self.cfg_debug_sample_every;
                    if debug_every != 0
                        && (self.optimizer.step_count() % debug_every == 0) {
                        let sync_t0 = std::time::Instant::now();
                        self.cached_debug_buf = gpu.read_debug_buffer();
                        if audit_cost {
                            audit_sync_ms += sync_t0.elapsed().as_secs_f64() * 1e3;
                        }
                    }
                }
                self.gpu_weights_uploaded = true;
                self.gpu_cg_weights_uploaded = true;

                // =============================================================================
                // [LEGACY — NO UTILIZADO] CPU fallback path (CG en CPU con h* leído del GPU).
                //
                // Reemplazado por: run_staged_adjoint_picard_no_readback() + apply_fused_deq_update()
                // (path GPU completo, líneas arriba de este bloque).
                //
                // Este bloque nunca ejecuta (if false). Lo que hace:
                //   1. Lee h* del GPU, recalcula en CPU via reasoning.step()
                //   2. Corre deq_implicit_grad (CG numérico con finite differences)
                //   3. Aplica weight update en CPU y re-sube todo a GPU
                //
                // Problemas: latencia de readback, CG escalar (O(N²) por iter),
                // no soporta W_hist ni hist_gate (solo W_q/W_k/W_v/W_o/W_in/NormScale).
                // =============================================================================
                if false {
                    // Strict path: CPU CG with GPU's h_star.
                    // Reads h_star from hnext_buf after GPU forward, then uses numerical
                    // JVP through reasoning.step() — captures all weights correctly and
                    // matches the CPU update direction (fixes update_parity_cpu_vs_gpu).
                    let query_vec = self.tokenizer.embed_context(context, self.config.ctx_len);

                    // Strict path stability: recompute h* in CPU with the exact same
                    // reasoning.step() used by deq_implicit_grad. This avoids inheriting
                    // any CPU↔GPU forward mismatch into DEQ gradient direction.
                    let mut h_star = self.reasoning.init(&query_vec);
                    for _ in 0..self.adaptive_max_iters.max(1) {
                        h_star = self.reasoning.step(&h_star, &query_vec, None);
                    }
                    let h_star_flat = h_star.to_flat();

                    let force_cpu_lm_dldh = self.cfg_lm_force_cpu_dldh;
                    let parity_check = self.cfg_lm_dldh_parity;
                    let mut dl_dh_cpu_opt: Option<nalgebra::DVector<f32>> = None;
                    let mut dl_dh_cpu_parity_opt: Option<nalgebra::DVector<f32>> = None;

                    if self.frozen_lm || force_cpu_lm_dldh || parity_check {
                        // Compute CPU reference dl_dh (matches CPU lm_head.forward + RMSNorm).
                        let d_r = self.config.d_r;
                        let h_slots = self.config.h_slots;
                        let mut h_pooled_buf = vec![0.0f32; d_r];
                        for k in 0..h_slots {
                            for d in 0..d_r {
                                h_pooled_buf[d] += h_star_flat[k * d_r + d];
                            }
                        }
                        for v in h_pooled_buf.iter_mut() {
                            *v /= h_slots as f32;
                        }
                        let h_pooled = nalgebra::DVector::from_vec(h_pooled_buf.clone());
                        let logits = self.lm_head.forward_on_flat(&h_pooled_buf);
                        let target = *targets_u32.last().unwrap_or(&0);
                        let dl_dlogits = loss::cross_entropy_grad(&logits, target);
                        let (_, dl_dh_cpu) = gradients::lmhead_backward(
                            &dl_dlogits,
                            &h_pooled,
                            &self.lm_head.w,
                            &self.lm_head.g,
                        );
                        dl_dh_cpu_opt = Some(dl_dh_cpu);
                    }
                    if parity_check {
                        let h_pooled_gpu = gpu.read_hpooled();
                        if !h_pooled_gpu.is_empty() {
                            let h_pooled = nalgebra::DVector::from_vec(h_pooled_gpu);
                            let sampled = gpu_lm.last_sampled_indices();
                            if !sampled.is_empty() {
                                let target = *targets_u32.first().unwrap_or(&0);
                                let dl_dh_cpu_parity = Self::lmhead_backward_sampled(
                                    sampled,
                                    target,
                                    &h_pooled,
                                    &self.lm_head.w,
                                    &self.lm_head.b,
                                    &self.lm_head.g,
                                );
                                dl_dh_cpu_parity_opt = Some(dl_dh_cpu_parity);
                            }
                        }
                    }

                    let dl_dh_vec = if self.frozen_lm || force_cpu_lm_dldh {
                        dl_dh_cpu_opt
                            .as_ref()
                            .cloned()
                            .unwrap_or_else(|| nalgebra::DVector::zeros(self.config.d_r))
                    } else {
                        match gpu_lm.read_dl_dh(&gpu.device, &gpu.queue, self.config.d_r) {
                            Ok(v) => nalgebra::DVector::from_vec(v),
                            Err(_) => nalgebra::DVector::zeros(self.config.d_r),
                        }
                    };

                    if parity_check {
                        // Compare GPU dl_dh with CPU reference to detect LM backward mismatch.
                        let dl_dh_cpu = dl_dh_cpu_parity_opt.as_ref().or(dl_dh_cpu_opt.as_ref());
                        if let Some(dl_dh_cpu) = dl_dh_cpu {
                            let dl_dh_gpu = if self.frozen_lm || force_cpu_lm_dldh {
                                match gpu_lm.read_dl_dh(&gpu.device, &gpu.queue, self.config.d_r) {
                                    Ok(v) => nalgebra::DVector::from_vec(v),
                                    Err(_) => nalgebra::DVector::zeros(self.config.d_r),
                                }
                            } else {
                                dl_dh_vec.clone()
                            };
                            let dot = dl_dh_cpu.dot(&dl_dh_gpu);
                            let n_cpu = dl_dh_cpu.norm();
                            let n_gpu = dl_dh_gpu.norm();
                            let cos = if n_cpu > 0.0 && n_gpu > 0.0 {
                                dot / (n_cpu * n_gpu)
                            } else {
                                0.0
                            };
                            let d_r = self.config.d_r;
                            let rms_cpu = n_cpu / (d_r as f32).sqrt();
                            let rms_gpu = n_gpu / (d_r as f32).sqrt();
                            let ratio = rms_gpu / (rms_cpu + 1e-12);
                            eprintln!(
                                "[LM-DLDH-PARITY] cos={:.6} rms_cpu={:.6e} rms_gpu={:.6e} ratio={:.6} step={}",
                                cos,
                                rms_cpu,
                                rms_gpu,
                                ratio,
                                self.optimizer.step_count()
                            );
                        }
                    }

                    // CPU CG: full Jacobian via finite differences through reasoning.step().
                    let mut v = gradients::deq_implicit_grad(
                        &self.reasoning,
                        &h_star,
                        &query_vec,
                        &dl_dh_vec,
                        self.config.adj_iters,
                    );
                    Self::clip_grad_norm(&mut v, 1.0);

                    let grad_mat = v.clone() * query_vec.transpose() * self.config.deq_grad_scale;
                    let grad_vec = v * self.config.deq_grad_scale;
                    self.optimizer
                        .step_matrix("deq_wq", &mut self.reasoning.w_q, &grad_mat);
                    self.optimizer
                        .step_matrix("deq_wk", &mut self.reasoning.w_k, &grad_mat);
                    self.optimizer
                        .step_matrix("deq_wv", &mut self.reasoning.w_v, &grad_mat);
                    self.optimizer
                        .step_matrix("deq_wo", &mut self.reasoning.w_o, &grad_mat);
                    self.optimizer
                        .step_matrix("deq_win", &mut self.reasoning.w_in, &grad_mat);
                    self.optimizer.step_vector(
                        "deq_norm",
                        &mut self.reasoning.norm_scale,
                        &grad_vec,
                    );

                    // Sync updated weights back to GPU.
                    let (
                        w_hist_shared_rm,
                        hist_slot_scale_rm,
                        hist_slot_bias_rm,
                        hist_gate_logit,
                        slot_anchor_rm,
                        w_delta_rm,
                        b_delta,
                        w_gate_hist_rm,
                        w_forget_rm,
                        b_forget_rm,
                    ) = self.reasoning.history_params_gpu_layout();
                    gpu.upload_weights(
                        &gpu.queue,
                        &self.reasoning.w_q_gpu_flat(),
                        &self.reasoning.w_k_gpu_flat(),
                        &self.reasoning.w_v_gpu_flat(),
                        &self.reasoning.w_o_gpu_flat(),
                        &self.reasoning.w_in_gpu_flat(),
                        self.reasoning.w_x.as_slice(),
                        self.reasoning.w_out.as_slice(),
                        &self.reasoning.a_log_gpu_flat(),
                        self.reasoning.norm_scale.as_slice(),
                        w_hist_shared_rm.as_slice(),
                        hist_slot_scale_rm.as_slice(),
                        hist_slot_bias_rm.as_slice(),
                        hist_gate_logit.as_slice(),
                        slot_anchor_rm.as_slice(),
                        w_delta_rm.as_slice(),
                        b_delta.as_slice(),
                        w_gate_hist_rm.as_slice(),
                        w_forget_rm.as_slice(),
                        b_forget_rm.as_slice(),
                    );
                    self.gpu_weights_uploaded = true;
                    self.gpu_cg_weights_uploaded = true;
                }

                // Spectral Renormalization (Periodic)
                let renorm_every = self.config.renorm_every_steps.max(1);
                if self.optimizer.step_count() % renorm_every == 0 {
                    let renorm_t0 = std::time::Instant::now();
                    let _ = gpu.renormalize_spectral();
                    if audit_cost {
                        audit_renorm_ms += renorm_t0.elapsed().as_secs_f64() * 1e3;
                        audit_renorm_calls += 1;
                    }
                }
            }

            if !self.frozen_emb {
                let emb_lr = base_lr * self.training_config.emb_lr_mult;
                let embed_t0 = std::time::Instant::now();
                let _ = gpu_emb.apply_embedding_update_from_buffer(
                    &gpu.device,
                    &gpu.queue,
                    context,
                    self.config.ctx_len,
                    &gpu_lm.dl_dh_buf,
                    emb_lr,
                    self.optimizer.beta1,
                    self.optimizer.beta2,
                    self.optimizer.eps,
                    self.optimizer.step_count() as u32,
                    self.training_config.ternary,
                );
                if audit_cost {
                    audit_embed_ms += embed_t0.elapsed().as_secs_f64() * 1e3;
                }
                self.gpu_emb_weights_uploaded = true;
            }

            // --- TPS tracking ---
            self.debug_tokens_accum += num_tokens as u32;

            // --- DIAGNÓSTICOS GPU (v13.1 Auto-Healing) ---
            // Reuse cached debug buffer to avoid blocking GPU every diagnostic step.
            let debug_every = self.cfg_debug_sample_every;
            if debug_every != 0
                && (self.optimizer.step_count() % debug_every == 0) {
                let sync_t0 = std::time::Instant::now();
                self.cached_debug_buf = gpu.read_debug_buffer();
                if audit_cost {
                    audit_sync_ms += sync_t0.elapsed().as_secs_f64() * 1e3;
                }
            }
            if !self.cached_debug_buf.is_empty() {
                let fw = &self.cached_debug_buf;

                let rs_cg = 0.0f32;

                let (heartbeat, max_h, avg_iters, hit_count, max_delta, contractivity, hit_den) =
                    Self::decode_forward_debug_metrics(fw, self.config.h_slots);
                let _last_delta = if fw.len() > 17 { fw[17] } else { 0.0 };
                let trunc_flag = if fw.len() > 18 { fw[18] } else { 0.0 };
                let total_elems = if fw.len() > 19 { fw[19] } else { 0.0 };
                let inj_rms = if fw.len() > 22 { fw[22] } else { 0.0 };
                let hist_rms = if fw.len() > 23 { fw[23] } else { 0.0 };
                let hist_ratio = if fw.len() > 24 { fw[24] } else { 0.0 };
                let mamba_rms = if fw.len() > 25 { fw[25] } else { 0.0 };
                let q_rms = if fw.len() > 26 { fw[26] } else { 0.0 };
                let k_rms = if fw.len() > 27 { fw[27] } else { 0.0 };
                let v_rms = if fw.len() > 28 { fw[28] } else { 0.0 };
                let mix_rms = if fw.len() > 29 { fw[29] } else { 0.0 };
                let attn_out_rms = if fw.len() > 30 { fw[30] } else { 0.0 };
                let attn_max = if fw.len() > 31 { fw[31] } else { 0.0 };
                let attn_entropy = if fw.len() > 32 { fw[32] } else { 0.0 };
                let combined_rms = if fw.len() > 33 { fw[33] } else { 0.0 };
                let hist0 = if fw.len() > 100 { fw[100] } else { 0.0 };
                let hist1 = if fw.len() > 101 { fw[101] } else { 0.0 };
                let hist2 = if fw.len() > 102 { fw[102] } else { 0.0 };
                let hist_anchor0 = if fw.len() > 103 { fw[103] } else { 0.0 };
                let hist_anchor1 = if fw.len() > 104 { fw[104] } else { 0.0 };
                let hist_rms_floor = if fw.len() > 105 { fw[105] } else { 0.0 };
                let hist_contr_floor = if fw.len() > 106 { fw[106] } else { 0.0 };
                let hist_inject = if fw.len() > 107 { fw[107] } else { 0.0 };
                let hist_minner_zero = if fw.len() > 108 { fw[108] } else { 0.0 };
                let hist_force_nomamba = if fw.len() > 109 { fw[109] } else { 0.0 };
                let hist_prelude_skip = if fw.len() > 110 { fw[110] } else { 0.0 };
                let hist_loop_force_nomamba = if fw.len() > 111 { fw[111] } else { 0.0 };

                let trunc_str = if trunc_flag >= 0.5 { "TRUNC" } else { "OK" };

                let now = std::time::Instant::now();
                let tps_debug = if let Some(t) = self.debug_last_time {
                    let elapsed = t.elapsed().as_secs_f32();
                    if elapsed > 0.0 {
                        self.debug_tokens_accum as f32 / elapsed
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                self.debug_last_time = Some(now);
                self.debug_tokens_accum = 0;

                if self.cfg_wv_debug {
                    if let Ok((_, _, wv, _, _, _, _, _, _)) = gpu.read_weights() {
                        let mut max_abs = 0.0f32;
                        let mut sum_abs = 0.0f32;
                        for v in &wv {
                            let av = v.abs();
                            if av > max_abs {
                                max_abs = av;
                            }
                            sum_abs += av;
                        }
                        let mean_abs = sum_abs / (wv.len().max(1) as f32);
                        println!(
                            "    \x1b[90m[GPU-WV] abs_mean={:.6e} max_abs={:.6e}\x1b[0m",
                            mean_abs, max_abs
                        );
                    }
                }

                let seq = heartbeat.max(1.0);
                let hit = hit_count.max(0.0);
                let hit_ratio = hit / hit_den.max(1.0);

                // ---------- v13.2 Stability Oracle Logic ----------

                // 1. CG Adaptation
                if rs_cg > 1e-4 {
                    self.cg_res_hi_streak += 1;
                } else {
                    self.cg_res_hi_streak = 0;
                }
                if self.cg_res_hi_streak >= 3 {
                    self.adaptive_adj_iters = (self.adaptive_adj_iters + 4).min(20);
                    self.cg_res_hi_streak = 0;
                }

                // 2. Contractivity Monitor & Emergency Renorm
                if contractivity > 1.02 {
                    self.contractivity_hi_streak += 1;
                } else {
                    self.contractivity_hi_streak = 0;
                }
                if self.contractivity_hi_streak >= 3 || contractivity > 1.20 {
                    let renorm_t0 = std::time::Instant::now();
                    let _ = gpu.renormalize_spectral();
                    if audit_cost {
                        audit_renorm_ms += renorm_t0.elapsed().as_secs_f64() * 1e3;
                        audit_renorm_calls += 1;
                    }
                    self.contractivity_hi_streak = 0;
                    if contractivity > 1.20 {
                        self.emergency_left = 2; // 2 windows de debug (~20 steps)
                    }
                }

                // 3. Forward Iters Hysteresis (v13.3)
                // For very short sequences (e.g. seq=1), hit_ratio is not informative
                // and tends to force unnecessary iteration growth.
                if seq >= 8.0 {
                    if hit_ratio > 0.08 {
                        self.hit_hi_streak += 1;
                        self.hit_lo_streak = 0;
                    } else if hit_ratio < 0.03 {
                        self.hit_lo_streak += 1;
                        self.hit_hi_streak = 0;
                    } else {
                        self.hit_hi_streak = 0;
                        self.hit_lo_streak = 0;
                    }
                } else {
                    let hi_delta = (self.config.deq_epsilon * 12.0).max(1.2e-3);
                    let lo_delta = (self.config.deq_epsilon * 3.0).max(3e-4);
                    if max_delta > hi_delta {
                        self.hit_hi_streak += 1;
                        self.hit_lo_streak = 0;
                    } else if max_delta < lo_delta {
                        self.hit_lo_streak += 1;
                        self.hit_hi_streak = 0;
                    } else {
                        self.hit_hi_streak = 0;
                        self.hit_lo_streak = 0;
                    }
                }

                // Subir iters si se está pegando al techo sostenidamente
                if self.hit_hi_streak >= 2 {
                    self.adaptive_max_iters = (self.adaptive_max_iters + 1).min(12);
                    self.hit_hi_streak = 0;
                }

                // Bajar iters si está sobrado por un buen rato
                if self.hit_lo_streak >= 10 {
                    self.adaptive_max_iters = self.adaptive_max_iters.saturating_sub(1).max(4);
                    self.hit_lo_streak = 0;
                }

                // BOOST de Damping por inestabilidad puntual o hit ratio alto
                if hit_ratio > 0.20 || max_delta > 1e-3 {
                    self.damping_boost_left = 2; // 2 windows de debug (~20 steps)
                }

                // Detector de crecimiento explosivo o inestabilidad pura (Stability Guardrail)
                let growth = if self.last_max_h > 0.0 {
                    max_h / self.last_max_h
                } else {
                    1.0
                };
                self.last_max_h = max_h;

                // Evita falsos positivos cuando las activaciones aún son diminutas.
                if self.last_max_h > 1e-2 && max_h > 1e-2 && growth > 1.20 {
                    self.max_h_growth_streak += 1;
                } else {
                    self.max_h_growth_streak = 0;
                }

                if max_delta > 5e-1 {
                    self.max_delta_hi_streak += 1;
                } else {
                    self.max_delta_hi_streak = 0;
                }

                // EMERGENCY Triggers: crecimiento rápido, NaNs, residuo inaceptable sostenido
                // o Divergencia (>1.20).
                if self.max_h_growth_streak >= 3
                    || self.max_delta_hi_streak >= 3
                    || max_h.is_nan()
                    || max_delta.is_nan()
                    || contractivity > 1.20
                {
                    self.emergency_left = 3; // 3 windows de debug (~30 steps)
                    self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(48);
                    self.max_h_growth_streak = 0;
                    self.max_delta_hi_streak = 0;
                    // Trigger spectral renorm immediately
                    #[cfg(feature = "wgpu")]
                    {
                        let renorm_t0 = std::time::Instant::now();
                        let _ = gpu.renormalize_spectral();
                        if audit_cost {
                            audit_renorm_ms += renorm_t0.elapsed().as_secs_f64() * 1e3;
                            audit_renorm_calls += 1;
                        }
                    }
                }

                // Countdown de modos temporales
                if self.damping_boost_left > 0 {
                    self.damping_boost_left -= 1;
                }
                if self.emergency_left > 0 {
                    self.emergency_left -= 1;
                }

                // Determinar modo y damping efectivo
                let damping_eff = self.damping_eff();
                let mode_str = if self.emergency_left > 0 {
                    "EMERG"
                } else if self.damping_boost_left > 0 {
                    "BOOST"
                } else {
                    "NORMAL"
                };
                let conv_ok =
                    hit_ratio <= 0.05 || max_delta <= (self.config.deq_epsilon * 4.0).max(3e-4);
                let conv_str = if conv_ok { "OK" } else { "FAIL" };
                let hit_i = hit.round() as i32;
                println!(
                    "    \x1b[90m[GPU-DEBUG] Step {:>2}: hit={:>3}/{:.0} ({:>5.1}%) contr={:.3} maxΔ={:.3e} rs_cg={:.1e} iters={:.1} cap={} damp={:.2} mode={} conv={} tps={:.1} max_h={:.6} inj_rms={:.3e} hist_rms={:.3e} hist/inj={:.3e} mamba_rms={:.3e} q/k/v={:.3e}/{:.3e}/{:.3e} mix/attn={:.3e}/{:.3e} attn_max={:.3} attn_ent={:.3} comb_rms={:.3e} hist=[{:.3e},{:.3e},{:.3e}] anchor=[{:.3e},{:.3e}] floors=[{:.3e},{:.3e}] flags=[{:.0},{:.0},{:.0},{:.0},{:.0}] shared={} total={:.0}\x1b[0m",
                    self.optimizer.step_count() % 100,
                    hit_i,
                    hit_den,
                    100.0 * hit_ratio,
                    contractivity,
                    max_delta,
                    rs_cg,
                    avg_iters,
                    self.adaptive_max_iters,
                    damping_eff,
                    mode_str,
                    conv_str,
                    tps_debug,
                    max_h,
                    inj_rms,
                    hist_rms,
                    hist_ratio,
                    mamba_rms,
                    q_rms,
                    k_rms,
                    v_rms,
                    mix_rms,
                    attn_out_rms,
                    attn_max,
                    attn_entropy,
                    combined_rms,
                    hist0,
                    hist1,
                    hist2,
                    hist_anchor0,
                    hist_anchor1,
                    hist_rms_floor,
                    hist_contr_floor,
                    hist_inject,
                    hist_minner_zero,
                    hist_force_nomamba,
                    hist_prelude_skip,
                    hist_loop_force_nomamba,
                    trunc_str,
                    total_elems
                );
                let hist_w_grad = fw.get(64).copied().unwrap_or(0.0);
                let hist_w_step = fw.get(65).copied().unwrap_or(0.0);
                let hist_w_before = fw.get(66).copied().unwrap_or(0.0);
                let hist_w_after = fw.get(67).copied().unwrap_or(0.0);
                let hist_gate_grad = fw.get(68).copied().unwrap_or(0.0);
                let hist_gate_step = fw.get(69).copied().unwrap_or(0.0);
                let hist_gate_before = fw.get(70).copied().unwrap_or(0.0);
                let hist_gate_after = fw.get(71).copied().unwrap_or(0.0);
                let hist_lr = fw.get(80).copied().unwrap_or(0.0);
                let hist_grad_scale = fw.get(81).copied().unwrap_or(0.0);
                let hist_wd = fw.get(82).copied().unwrap_or(0.0);
                println!(
                    "    \x1b[90m[GPU-HIST] W_hist grad/step/before/after={:.3e}/{:.3e}/{:.3e}/{:.3e} gate grad/step/before/after={:.3e}/{:.3e}/{:.3e}/{:.3e} lr={:.3e} gscale={:.3e} wd={:.3e}\x1b[0m",
                    hist_w_grad,
                    hist_w_step,
                    hist_w_before,
                    hist_w_after,
                    hist_gate_grad,
                    hist_gate_step,
                    hist_gate_before,
                    hist_gate_after,
                    hist_lr,
                    hist_grad_scale,
                    hist_wd
                );

                // GPU-SSM per-slot decay diagnostics (activar con AIDEEN_SSM_DEBUG=1).
                if self.cfg_ssm_debug {
                    let carrier = gpu.read_hist_carrier_params_full();
                    let d_r = self.config.d_r;
                    let h_slots = self.config.h_slots;
                    let a_offset = 2 * d_r * d_r; // after wx + wout
                    if carrier.len() >= a_offset + h_slots * d_r {
                        let mut a_means = Vec::with_capacity(h_slots);
                        let mut a_spreads = Vec::with_capacity(h_slots);
                        for s in 0..h_slots {
                            let slice = &carrier[a_offset + s * d_r..a_offset + (s + 1) * d_r];
                            let a_vals: Vec<f32> =
                                slice.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();
                            let mean = a_vals.iter().sum::<f32>() / d_r as f32;
                            let spread =
                                a_vals.iter().map(|&v| (v - mean).abs()).sum::<f32>() / d_r as f32;
                            a_means.push(mean);
                            a_spreads.push(spread);
                        }
                        let fmt_vec = |v: &[f32]| {
                            v.iter()
                                .map(|x| format!("{:.3}", x))
                                .collect::<Vec<_>>()
                                .join(",")
                        };
                        println!(
                            "    \x1b[90m[GPU-SSM] Step {}: a_mean=[{}] a_spread=[{}] mamba_rms={:.3e}\x1b[0m",
                            self.optimizer.step_count() % 100,
                            fmt_vec(&a_means),
                            fmt_vec(&a_spreads),
                            mamba_rms,
                        );
                    }
                }

                // Per-token debug (slot 0) for small sequences.
                let seq_len = heartbeat.max(1.0).round() as usize;
                if seq_len > 0 && seq_len <= 16 {
                    let base = 200usize;
                    let mut per_token = String::new();
                    for t in 0..seq_len {
                        let idx = base + t * 3;
                        let h_rms = if fw.len() > idx { fw[idx] } else { 0.0 };
                        let v_rms = if fw.len() > idx + 1 { fw[idx + 1] } else { 0.0 };
                        let a_rms = if fw.len() > idx + 2 { fw[idx + 2] } else { 0.0 };
                        let _ = std::fmt::Write::write_fmt(
                            &mut per_token,
                            format_args!(
                                " t{:02} h={:.3e} v={:.3e} a={:.3e}",
                                t, h_rms, v_rms, a_rms
                            ),
                        );
                    }
                    println!("    \x1b[90m[GPU-TOKENS]{}\x1b[0m", per_token);
                }
            }

            if audit_cost {
                eprintln!(
                    "[SYSTEM-COST] step={} gather={:.2}ms forward={:.2}ms lm={:.2}ms picard={:.2}ms update={:.2}ms embed={:.2}ms sync={:.2}ms renorm={:.2}ms calls={} total={:.2}ms",
                    self.optimizer.step_count(),
                    audit_gather_ms,
                    audit_forward_ms,
                    audit_lm_ms,
                    audit_picard_ms,
                    audit_update_ms,
                    audit_embed_ms,
                    audit_sync_ms,
                    audit_renorm_ms,
                    audit_renorm_calls,
                    audit_gather_ms
                        + audit_forward_ms
                        + audit_lm_ms
                        + audit_picard_ms
                        + audit_update_ms
                        + audit_embed_ms
                        + audit_sync_ms
                        + audit_renorm_ms,
                );
                let (qkv_bytes, wo_bytes, win_bytes, hist_lower_bound) = audit_forward_bytes;
                let dense_lower_bound = qkv_bytes + wo_bytes + win_bytes + hist_lower_bound;
                let forward_secs = (audit_forward_ms / 1e3).max(1e-9);
                eprintln!(
                    "[FORWARD-BW] step={} batch={} seq={} iters_cap={} qkv_lb={:.3}GiB wo_lb={:.3}GiB win_once={:.3}GiB hist_lb={:.3}GiB dense_lb={:.3}GiB dense_lb_bw={:.1}GiB/s",
                    self.optimizer.step_count(),
                    fwd_batch_size,
                    per_seq_len,
                    self.adaptive_max_iters,
                    Self::gib(qkv_bytes),
                    Self::gib(wo_bytes),
                    Self::gib(win_bytes),
                    Self::gib(hist_lower_bound),
                    Self::gib(dense_lower_bound),
                    Self::gib(dense_lower_bound) / forward_secs,
                );
            }

            return current_loss;
        }
        0.0
    }

    #[cfg(not(feature = "wgpu"))]
    fn apply_training_update_from_gpu_buffers(
        &mut self,
        _context: &[u32],
        _targets: &[u32],
        _query: &nalgebra::DVector<f32>,
        _gpu_ctx: Option<()>,
    ) -> f32 {
        0.0
    }

    /// Cosine LR schedule con warmup.
    fn cosine_lr(&self, epoch: usize, total_epochs: usize) -> f32 {
        let lr_max = self.training_config.lr;
        let lr_min = self.training_config.lr_min;
        let warmup = self.training_config.warmup_epochs;

        if epoch < warmup {
            lr_min + (lr_max - lr_min) * (epoch as f32 / warmup.max(1) as f32)
        } else {
            let progress = (epoch - warmup) as f32 / (total_epochs - warmup).max(1) as f32;
            lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }

    /// Schedule de Epsilon progresivo para el solver DEQ.
    fn progressive_epsilon(&self, epoch: usize, total_epochs: usize) -> f32 {
        let progress = epoch as f32 / total_epochs.max(1) as f32;
        if progress < 0.20 {
            1e-3
        } else if progress < 0.50 {
            5e-4
        } else if progress < 0.80 {
            1e-4
        } else {
            1e-5
        }
    }

    /// Ejecuta el training loop sobre un corpus tokenizado.
    pub fn train_on_tokens(&mut self, tokens: &[u32], epochs: usize, log_every: usize) {
        if tokens.len() < 2 {
            return;
        }
        let train_tokens = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        for epoch in 0..epochs {
            let t_start = std::time::Instant::now();
            let current_lr = self.cosine_lr(epoch, epochs);
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                gpu.tps_epoch_begin();
            }

            self.optimizer.lr = current_lr;

            // Progressive DEQ schedule: fewer iters when weights are still random
            let deq_progress = epoch as f32 / epochs as f32;
            let (floor, cap) = if deq_progress < 0.25 {
                (8, 12)
            } else if deq_progress < 0.60 {
                (10, 14)
            } else {
                (12, 16)
            };
            self.adaptive_max_iters = self.adaptive_max_iters.clamp(floor, cap);
            self.config.adj_iters = if deq_progress < 0.25 {
                6
            } else if deq_progress < 0.60 {
                8
            } else {
                10
            };

            // Training chunk by chunk over the whole corpus
            let mut epoch_loss = 0.0;
            let mut num_chunks = 0;
            let mut interval_start = std::time::Instant::now();
            let mut interval_tokens = 0;
            let mut total_tokens = 0usize;
            let mut last_progress_chunk_logged = 0usize;
            let ctx_len = self.config.ctx_len.max(1);
            let batch_size = self.cfg_fwd_batch_size.max(1) as usize;
            let step = ctx_len * batch_size;

            for i in (0..train_tokens.len()).step_by(step) {
                let end = (i + step).min(train_tokens.len());
                let batch_ctx = &train_tokens[i..end];
                let batch_tgt = &targets[i..end];
                if self.cfg_log_emb_stats && epoch % log_every == 0 && i == 0 {
                    use std::collections::HashSet;
                    let uniq: HashSet<u32> = batch_ctx.iter().copied().collect();
                    eprintln!(
                        "[EMB] uniq_tokens={}/{} ({:.1}%)",
                        uniq.len(),
                        batch_ctx.len(),
                        100.0 * (uniq.len() as f32) / (batch_ctx.len() as f32)
                    );
                }

                // Detectar si el primer token es un separador/EOS (ej: 0)
                // En muchos datasets tokenizados, el 0 se usa para marcar inicio/fin de doc.
                let mut reset_requested = false;
                if i == 0 {
                    reset_requested = true; // Siempre reset al inicio del epoch
                } else if !batch_ctx.is_empty() && (batch_ctx[0] == 0 || batch_ctx[0] == 2) {
                    reset_requested = true;
                }

                // Implementar Warmup lineal de 100 pasos
                let current_step = self.optimizer.step_count();
                let eps = self.progressive_epsilon(epoch, epochs);
                if current_step < 100 {
                    let warmup_factor = (current_step as f32 + 1.0) / 100.0;
                    let original_lr = self.optimizer.lr;
                    self.optimizer.lr *= warmup_factor;
                    epoch_loss += self.train_sequence(batch_ctx, batch_tgt, reset_requested, eps);
                    self.optimizer.lr = original_lr;
                } else {
                    epoch_loss += self.train_sequence(batch_ctx, batch_tgt, reset_requested, eps);
                }
                num_chunks += 1;
                interval_tokens += batch_ctx.len();
                total_tokens += batch_ctx.len();
                #[cfg(feature = "wgpu")]
                if self.cfg_tps_sync_every != 0
                    && num_chunks % self.cfg_tps_sync_every == 0 {
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        // Progress cadence is observability, not model math. Use a non-blocking
                        // drain here so progress reporting does not stall the fused hot path.
                        gpu.device.poll(wgpu::Maintain::Poll);
                    }
                }

                if self.cfg_progress_every != 0
                    && num_chunks % self.cfg_progress_every == 0
                    && num_chunks != last_progress_chunk_logged
                {
                    #[cfg(feature = "wgpu")]
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        if self.cfg_progress_strict_sync {
                            gpu.device.poll(wgpu::Maintain::Wait);
                        } else {
                            // Progress is observability, not model math. Avoid serializing the
                            // GPU pipeline on every cadence; trustworthy GPU throughput is
                            // reported at epoch boundaries via timestamp queries.
                            gpu.device.poll(wgpu::Maintain::Poll);
                        }
                    }
                    let interval_elapsed = interval_start.elapsed().as_secs_f32();
                    let window_tps = interval_tokens as f32 / interval_elapsed.max(1e-9);
                    let epoch_elapsed = t_start.elapsed().as_secs_f32();
                    let epoch_tps = total_tokens as f32 / epoch_elapsed.max(1e-9);
                    let current_loss_disp = self.visible_loss_text(Some(epoch_loss / num_chunks as f32));

                    println!(
                        "    \x1b[95m[progress]\x1b[0m chunk {:>5}  \x1b[92mloss={}\x1b[0m  \x1b[96mtps_win={:>8.1}\x1b[0m  \x1b[94mtps_run={:>8.1}\x1b[0m  \x1b[90mtime={:.1}s\x1b[0m",
                        num_chunks, current_loss_disp, window_tps, epoch_tps, epoch_elapsed
                    );

                    // Reset interval timers
                    interval_start = std::time::Instant::now();
                    interval_tokens = 0;
                    last_progress_chunk_logged = num_chunks;
                }
            }

            let total_loss = if num_chunks > 0 {
                epoch_loss / num_chunks as f32
            } else {
                0.0
            };

            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                gpu.tps_epoch_end();
            }
            // Drain GPU queue every epoch — prevents Metal command queue overflow and
            // makes elapsed/TPS measurements reflect actual GPU execution time.
            #[cfg(feature = "wgpu")]
            {
                if let Some(gpu) = self.gpu_deq.as_ref() {
                    // Epoch boundary: make end-of-epoch metrics/checkpoints observe completed GPU work.
                    gpu.device.poll(wgpu::Maintain::Wait);
                }
            }
            #[cfg(feature = "wgpu")]
            if let Some(loss) = self
                .gpu_deq
                .as_ref()
                .and_then(|gpu| self.cached_loss_after_sync(gpu))
            {
                self.last_gpu_loss = loss;
            }
            let elapsed = t_start.elapsed().as_secs_f32();
            // Usar tokens reales procesados, no num_chunks * ctx_len (que sobre-cuenta el último chunk).
            let tokens_processed = train_tokens.len();
            let tps = tokens_processed as f32 / elapsed.max(1e-9);

            if epoch % log_every == 0 {
                // GPU already idle (poll above) — read_cached_loss is near-instant here
                let display_loss = self.visible_loss_text(Some(total_loss));
                let mut gpu_suffix = String::new();
                #[cfg(feature = "wgpu")]
                if let Some(gpu) = self.gpu_deq.as_ref() {
                    if let Some(ns) = gpu.read_tps_epoch_ns() {
                        let tps_gpu = (tokens_processed as f64) / (ns / 1e9);
                        gpu_suffix = format!("  tps_gpu={:>8.1}", tps_gpu);
                    }
                }
                println!(
                    "  epoch {epoch:>4}/{epochs}  loss={}  lr={:.6}  tps={:>8.1}  time={:.2}s{}",
                    display_loss, current_lr, tps, elapsed, gpu_suffix
                );
            }
        }
    }

    /// Genera texto a partir de un prompt.
    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
    ) -> String {
        #[cfg(feature = "wgpu")]
        {
            // Política de inferencia: priorizar GPU; fallback a CPU si no hay dispositivo.
            if !self.reasoning.has_backend() {
                let _ = self.configure_inference_backend(true);
            }
            self.sync_lm_head_from_gpu_if_needed();
        }

        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() {
            return String::new();
        }
        let prompt_len = tokens.len();
        let use_fixed_history_ref = Self::fixed_history_reference_mode();
        let mut ref_m_prev = if use_fixed_history_ref {
            self.prime_fixed_history_state(&tokens)
        } else {
            None
        };

        for _ in 0..max_tokens {
            let ctx_start = tokens.len().saturating_sub(self.config.ctx_len);
            let context = &tokens[ctx_start..];

            #[cfg(feature = "wgpu")]
            if self.gpu_deq.is_some() && use_fixed_history_ref {
                let gpu = self.gpu_deq.take().expect("gpu_deq checked as Some");
                let current_token = *context.last().unwrap_or(&0);
                if let Ok((h_pooled, _h_star, m_next)) = self.gpu_solve_token_with_fixed_history(
                    &gpu,
                    current_token,
                    ref_m_prev.as_ref(),
                ) {
                    ref_m_prev = Some(m_next);
                    self.sync_lm_head_from_gpu_if_needed();
                    let d_r = self.config.d_r;
                    let last_h = h_pooled.as_slice().chunks(d_r).last().unwrap();
                    let h_pooled = nalgebra::DVector::from_column_slice(last_h);
                    let logits = &self.lm_head.w * h_pooled + &self.lm_head.b;
                    let next_token = LmHead::sample(
                        &logits,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penalty,
                        context,
                    );
                    tokens.push(next_token);
                    self.gpu_deq = Some(gpu);
                    continue;
                }
                self.gpu_deq = Some(gpu);
            } else if self.gpu_deq.is_some() && !use_fixed_history_ref {
                let gpu = self.gpu_deq.take().expect("gpu_deq checked as Some");
                if self.gpu_emb.is_none() {
                    let safe_ctx = self.config.ctx_len.max(1024);
                    self.gpu_emb = Some(GpuEmbeddingTrainer::new(
                        &gpu.device,
                        self.tokenizer.vocab_size(),
                        safe_ctx,
                        self.config.clone(),
                    ));
                }
                let Some(gpu_emb) = self.gpu_emb.as_ref() else {
                    self.gpu_deq = Some(gpu);
                    continue;
                };
                let emb_needs_upload = !self.gpu_emb_weights_uploaded;
                let (s_sequence, _) = match gpu_emb.prepare_sequence_and_query(
                    &gpu.device,
                    &gpu.queue,
                    context,
                    self.config.ctx_len,
                    self.tokenizer.embeddings.as_slice(),
                    emb_needs_upload,
                ) {
                    Ok(v) => {
                        self.gpu_emb_weights_uploaded = true;
                        v
                    }
                    Err(_) => {
                        self.gpu_emb_weights_uploaded = false;
                        self.gpu_deq = Some(gpu);
                        continue;
                    }
                };
                let needs_upload = !self.gpu_weights_uploaded;

                if let Ok((h_pooled, _)) = gpu.run_forward_deq_pooled(
                    1,
                    context.len() as u32,
                    self.config.max_deq_iters as u32,
                    self.config.deq_epsilon,
                    self.reasoning.damping,
                    &s_sequence,
                    &self.reasoning.w_q_gpu_flat(),
                    &self.reasoning.w_k_gpu_flat(),
                    &self.reasoning.w_v_gpu_flat(),
                    &self.reasoning.w_o_gpu_flat(),
                    &self.reasoning.w_in_gpu_flat(),
                    self.reasoning.w_x.as_slice(),
                    self.reasoning.w_out.as_slice(),
                    &self.reasoning.a_log_gpu_flat(),
                    self.reasoning.norm_scale.as_slice(),
                    needs_upload,
                ) {
                    self.gpu_weights_uploaded = true;
                    self.sync_lm_head_from_gpu_if_needed();
                    let d_r = self.config.d_r;
                    let last_h = h_pooled.chunks(d_r).last().unwrap();
                    let h_pooled = nalgebra::DVector::from_column_slice(last_h);
                    let logits = &self.lm_head.w * h_pooled + &self.lm_head.b;
                    let next_token = LmHead::sample(
                        &logits,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penalty,
                        context,
                    );
                    tokens.push(next_token);
                    self.gpu_deq = Some(gpu);
                    continue;
                }
                self.gpu_deq = Some(gpu);
            }

            // Fallback CPU
            let query = if use_fixed_history_ref {
                self.tokenizer.embed(*context.last().unwrap_or(&0))
            } else {
                self.tokenizer.embed_context(context, self.config.ctx_len)
            };
            let h = if use_fixed_history_ref {
                let h_star = self.solve_query_with_fixed_history(
                    &query,
                    ref_m_prev.as_ref(),
                    self.config.max_deq_iters,
                );
                ref_m_prev = Some(self.temporal_update_from_h(ref_m_prev.as_ref(), &h_star));
                h_star
            } else {
                let mut h = self.reasoning.init(&query);
                for _ in 0..10 {
                    h = self.reasoning.step(&h, &query, None);
                }
                h
            };
            let logits = self.lm_head.forward(&h);
            tokens.push(LmHead::sample(
                &logits,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                context,
            ));
        }
        if use_fixed_history_ref {
            self.m_prev = ref_m_prev;
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                let (_, _, _, _, slot_anchor_rm, _, _, _, _, _) =
                    self.reasoning.history_params_gpu_layout();
                self.upload_reasoning_weights_with_slot_anchor(gpu, slot_anchor_rm.as_slice());
                self.gpu_weights_uploaded = true;
            }
        }
        self.tokenizer.decode(&tokens[prompt_len..])
    }

    /// Versión streaming de generate: llama a `on_token` con cada fragmento de texto
    /// generado en tiempo real, sin esperar a que termine la generación completa.
    pub fn generate_stream<F: FnMut(&str)>(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        mut on_token: F,
    ) -> String {
        #[cfg(feature = "wgpu")]
        {
            if !self.reasoning.has_backend() {
                let _ = self.configure_inference_backend(true);
            }
            self.sync_lm_head_from_gpu_if_needed();
        }

        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() {
            return String::new();
        }
        let prompt_len = tokens.len();
        let mut decoded_len = 0usize;
        let use_fixed_history_ref = Self::fixed_history_reference_mode();
        let mut ref_m_prev = if use_fixed_history_ref {
            self.prime_fixed_history_state(&tokens)
        } else {
            None
        };

        for _ in 0..max_tokens {
            let ctx_start = tokens.len().saturating_sub(self.config.ctx_len);
            let context = &tokens[ctx_start..];

            #[cfg(feature = "wgpu")]
            if self.gpu_deq.is_some() && use_fixed_history_ref {
                let gpu = self.gpu_deq.take().expect("gpu_deq checked as Some");
                let current_token = *context.last().unwrap_or(&0);
                if let Ok((h_pooled, _h_star, m_next)) = self.gpu_solve_token_with_fixed_history(
                    &gpu,
                    current_token,
                    ref_m_prev.as_ref(),
                ) {
                    ref_m_prev = Some(m_next);
                    self.sync_lm_head_from_gpu_if_needed();
                    let d_r = self.config.d_r;
                    let last_h = h_pooled.as_slice().chunks(d_r).last().unwrap();
                    let h_pooled = nalgebra::DVector::from_column_slice(last_h);
                    let logits = &self.lm_head.w * h_pooled + &self.lm_head.b;
                    let next_token = LmHead::sample(
                        &logits,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penalty,
                        context,
                    );
                    tokens.push(next_token);
                    self.gpu_deq = Some(gpu);

                    let current = self.tokenizer.decode(&tokens[prompt_len..]);
                    if current.len() > decoded_len {
                        on_token(&current[decoded_len..]);
                        decoded_len = current.len();
                    }
                    continue;
                }
                self.gpu_deq = Some(gpu);
            } else if self.gpu_deq.is_some() && !use_fixed_history_ref {
                let gpu = self.gpu_deq.take().expect("gpu_deq checked as Some");
                if self.gpu_emb.is_none() {
                    let safe_ctx = self.config.ctx_len.max(1024);
                    self.gpu_emb = Some(GpuEmbeddingTrainer::new(
                        &gpu.device,
                        self.tokenizer.vocab_size(),
                        safe_ctx,
                        self.config.clone(),
                    ));
                }
                let Some(gpu_emb) = self.gpu_emb.as_ref() else {
                    self.gpu_deq = Some(gpu);
                    continue;
                };
                let emb_needs_upload = !self.gpu_emb_weights_uploaded;
                let (s_sequence, _) = match gpu_emb.prepare_sequence_and_query(
                    &gpu.device,
                    &gpu.queue,
                    context,
                    self.config.ctx_len,
                    self.tokenizer.embeddings.as_slice(),
                    emb_needs_upload,
                ) {
                    Ok(v) => {
                        self.gpu_emb_weights_uploaded = true;
                        v
                    }
                    Err(_) => {
                        self.gpu_emb_weights_uploaded = false;
                        self.gpu_deq = Some(gpu);
                        continue;
                    }
                };
                let needs_upload = !self.gpu_weights_uploaded;

                if let Ok((h_pooled, _)) = gpu.run_forward_deq_pooled(
                    1,
                    context.len() as u32,
                    self.config.max_deq_iters as u32,
                    self.config.deq_epsilon,
                    self.reasoning.damping,
                    &s_sequence,
                    &self.reasoning.w_q_gpu_flat(),
                    &self.reasoning.w_k_gpu_flat(),
                    &self.reasoning.w_v_gpu_flat(),
                    &self.reasoning.w_o_gpu_flat(),
                    &self.reasoning.w_in_gpu_flat(),
                    self.reasoning.w_x.as_slice(),
                    self.reasoning.w_out.as_slice(),
                    &self.reasoning.a_log_gpu_flat(),
                    self.reasoning.norm_scale.as_slice(),
                    needs_upload,
                ) {
                    self.gpu_weights_uploaded = true;
                    self.sync_lm_head_from_gpu_if_needed();
                    let d_r = self.config.d_r;
                    let last_h = h_pooled.chunks(d_r).last().unwrap();
                    let h_pooled = nalgebra::DVector::from_column_slice(last_h);
                    let logits = &self.lm_head.w * h_pooled + &self.lm_head.b;
                    let next_token = LmHead::sample(
                        &logits,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penalty,
                        context,
                    );
                    tokens.push(next_token);
                    self.gpu_deq = Some(gpu);

                    // Emitir texto nuevo
                    let current = self.tokenizer.decode(&tokens[prompt_len..]);
                    if current.len() > decoded_len {
                        on_token(&current[decoded_len..]);
                        decoded_len = current.len();
                    }
                    continue;
                }
                self.gpu_deq = Some(gpu);
            }

            // Fallback CPU
            let query = if use_fixed_history_ref {
                self.tokenizer.embed(*context.last().unwrap_or(&0))
            } else {
                self.tokenizer.embed_context(context, self.config.ctx_len)
            };
            let h = if use_fixed_history_ref {
                let h_star = self.solve_query_with_fixed_history(
                    &query,
                    ref_m_prev.as_ref(),
                    self.config.max_deq_iters,
                );
                ref_m_prev = Some(self.temporal_update_from_h(ref_m_prev.as_ref(), &h_star));
                h_star
            } else {
                let mut h = self.reasoning.init(&query);
                for _ in 0..10 {
                    h = self.reasoning.step(&h, &query, None);
                }
                h
            };
            let logits = self.lm_head.forward(&h);
            let next_token = LmHead::sample(
                &logits,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                context,
            );
            tokens.push(next_token);

            let current = self.tokenizer.decode(&tokens[prompt_len..]);
            if current.len() > decoded_len {
                on_token(&current[decoded_len..]);
                decoded_len = current.len();
            }
        }
        if use_fixed_history_ref {
            self.m_prev = ref_m_prev;
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                let (_, _, _, _, slot_anchor_rm, _, _, _, _, _) =
                    self.reasoning.history_params_gpu_layout();
                self.upload_reasoning_weights_with_slot_anchor(gpu, slot_anchor_rm.as_slice());
                self.gpu_weights_uploaded = true;
            }
        }

        self.tokenizer.decode(&tokens[prompt_len..])
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }

    /// Calcula la cross-entropy loss sobre una secuencia sin actualizar pesos.
    /// Útil para validación limpia (forward-only, sin backprop).
    pub fn eval_loss(&self, tokens: &[u32]) -> f32 {
        if tokens.len() < 2 {
            return f32::NAN;
        }
        let inputs = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        let mut total_loss = 0.0f32;
        let mut count = 0usize;
        let use_fixed_history_ref = Self::fixed_history_reference_mode();
        let mut m_prev: Option<HSlots> = None;

        for (i, &target) in targets.iter().enumerate() {
            let ctx_start = i.saturating_sub(self.config.ctx_len.saturating_sub(1));
            let context = &inputs[ctx_start..=i];

            let query = if use_fixed_history_ref {
                self.tokenizer.embed(*context.last().unwrap_or(&0))
            } else {
                self.tokenizer.embed_context(context, self.config.ctx_len)
            };
            let h = if use_fixed_history_ref {
                let h_star = self.solve_query_with_fixed_history(
                    &query,
                    m_prev.as_ref(),
                    self.config.max_deq_iters,
                );
                m_prev = Some(self.temporal_update_from_h(m_prev.as_ref(), &h_star));
                h_star
            } else {
                let mut h = self.reasoning.init(&query);
                for _ in 0..self.config.max_deq_iters.max(1) {
                    h = self.reasoning.step(&h, &query, None);
                }
                h
            };
            let logits = self.lm_head.forward(&h);

            // Cross-entropy: -log(softmax[target])
            let max_l = logits.max();
            let exp: nalgebra::DVector<f32> = logits.map(|v| (v - max_l).exp());
            let sum = exp.sum();
            let prob = exp[target as usize] / sum;
            total_loss += -prob.max(1e-10).ln();
            count += 1;
        }

        if count == 0 {
            f32::NAN
        } else {
            total_loss / count as f32
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gradient clipping global (L2 norm)
    // ─────────────────────────────────────────────────────────────────────────

    /// Escala `grad` in-place si su norma L2 supera `max_norm`.
    /// Equivalente a `clip_grad_norm_` de PyTorch para un único tensor.
    fn clip_grad_norm(grad: &mut nalgebra::DVector<f32>, max_norm: f32) {
        let norm = grad.norm();
        if norm > max_norm {
            *grad *= max_norm / (norm + 1e-6);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Streaming dataloader — entrenamiento desde archivo binario de tokens
    // ─────────────────────────────────────────────────────────────────────────

    /// Entrena sobre un archivo binario de tokens u32 (little-endian).
    ///
    /// El archivo puede tener cualquier tamaño: se lee en chunks de `ctx_len + 1`
    /// tokens con una ventana solapada de `overlap` tokens para preservar contexto
    /// entre chunks. Cuando aparece el token `eos_token` se reinicia el contexto.
    ///
    /// `save_every`: guarda checkpoint cada N epochs (0 = nunca).
    pub fn train_on_file(
        &mut self,
        path: &str,
        epochs: usize,
        log_every: usize,
        eos_token: u32,
        save_every: usize,
        checkpoint_path: &str,
    ) -> std::io::Result<()> {
        use std::io::Read;

        let file_size = std::fs::metadata(path)?.len() as usize;
        let total_file_tokens = file_size / 4;
        if total_file_tokens < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "El archivo tiene menos de 2 tokens",
            ));
        }

        let ctx_len = self.config.ctx_len.max(1);
        // Ventana solapada: leemos ctx_len + 1 tokens, avanzamos ctx_len/2 para contexto continuo.
        let stride = (ctx_len / 2).max(1);
        let chunk_tokens = ctx_len + 1;
        let chunk_bytes = chunk_tokens * 4;
        // Batch accumulation: collect batch_size chunks before training.
        // Without this, AIDEEN_BATCH_SIZE=N with 256-token chunks would give per_seq_len=256/N=32,
        // training on 32-token windows instead of 256. Fix: accumulate N chunks → N×256 tokens →
        // per_seq_len = (N×256)/N = 256 (correct).
        let batch_size_file = self.cfg_fwd_batch_size.max(1) as usize;
        let mut batch_train_buf: Vec<u32> = Vec::with_capacity(batch_size_file * ctx_len);
        let mut batch_tgt_buf: Vec<u32> = Vec::with_capacity(batch_size_file * ctx_len);

        for epoch in 0..epochs {
            let t_start = std::time::Instant::now();
            let current_lr = self.cosine_lr(epoch, epochs);
            self.optimizer.lr = current_lr;
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                gpu.tps_epoch_begin();
            }

            // v13.1 Adaptive Epoch Schedule (Piso de iteraciones)
            let deq_progress = epoch as f32 / epochs.max(1) as f32;
            let sched_floor = if deq_progress < 0.25 {
                8
            } else if deq_progress < 0.60 {
                10
            } else {
                12
            };
            self.adaptive_max_iters = self.adaptive_max_iters.max(sched_floor);
            // Early large-file training was over-solving the adjoint: short and chunk-10
            // validations preserve loss with 2 Picard adjoint iterations while materially
            // improving throughput. Keep later phases conservative until we validate them too.
            self.config.adj_iters = if deq_progress < 0.25 {
                2
            } else if deq_progress < 0.60 {
                6
            } else {
                8
            };
            if let Some(adj) = self.cfg_adj_iters_override {
                let adj_usize = (adj.max(1)) as usize;
                self.config.adj_iters = adj_usize;
                self.adaptive_adj_iters = adj_usize as u32;
            }

            // Prefetch next chunk on a background thread to overlap disk I/O with GPU work.
            let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<u8>, usize)>(2);
            let path_owned = path.to_string();
            std::thread::spawn(move || {
                let mut f = match std::fs::File::open(&path_owned) {
                    Ok(f) => f,
                    Err(_) => {
                        let _ = tx.send((Vec::new(), 0));
                        return;
                    }
                };
                loop {
                    let mut buf = vec![0u8; chunk_bytes];
                    let n = match f.read(&mut buf) {
                        Ok(n) => n,
                        Err(_) => 0,
                    };
                    if n == 0 {
                        let _ = tx.send((Vec::new(), 0));
                        break;
                    }
                    buf.truncate(n);
                    if tx.send((buf, n)).is_err() {
                        break;
                    }
                }
            });
            let mut epoch_loss = 0.0f32;
            let mut num_chunks = 0usize;
            let mut total_tokens = 0usize;
            let mut interval_start = std::time::Instant::now();
            let mut interval_tokens = 0usize;
            let mut last_progress_chunk_logged = 0usize;
            #[cfg(feature = "wgpu")]
            let mut interval_tokens_prev = 0usize;
            
            // Inicializar archivo de métricas CSV si está configurado
            if let Some(path) = self.metrics_log_path.as_ref() {
                // Si no estamos reanudando (resume_path), creamos un archivo nuevo.
                // Si estamos reanudando, lo dejamos como está o añadimos (por ahora creamos nuevo para simplicidad del gráfico).
                if let Ok(mut f) = std::fs::File::create(path) {
                    use std::io::Write;
                    let _ = writeln!(f, "epoch,chunk,loss,tps_win,tps_gpu");
                }
            }

            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                // Iniciar el primer slot de la cola circular
                gpu.tps_chunk_record(0);
            }
            // Buffer de tokens no consumidos del chunk anterior (para ventana solapada).
            let mut carry: Vec<u32> = Vec::with_capacity(stride);
            // Pre-allocate token window to avoid per-chunk heap allocations.
            let mut tokens: Vec<u32> = Vec::with_capacity(chunk_tokens + stride);
            let mut last_save_time = std::time::Instant::now();

            loop {
                // Prepend carry del chunk anterior + leer nuevos bytes (prefetch thread).
                let (read_buf, n) = match rx.recv() {
                    Ok(v) => v,
                    Err(_) => (Vec::new(), 0),
                };
                if n == 0 && carry.is_empty() {
                    break;
                }

                // Reset de estado estricto en límites de documento (opcional según eos_token)
                if eos_token != 0 && !carry.is_empty() && carry[0] == eos_token {
                    self.reset_state();
                }

                // Reuse pre-allocated tokens buffer to avoid per-chunk allocation.
                tokens.clear();
                tokens.extend_from_slice(&carry);
                carry.clear();
                if n > 0 {
                    let aligned = n & !3;
                    tokens.extend_from_slice(bytemuck::cast_slice(&read_buf[..aligned]));
                }

                if tokens.len() < 2 {
                    break;
                }

                // Detectar EOS: dividir el chunk en sub-secuencias de documento.
                let mut seg_start = 0;
                while seg_start < tokens.len().saturating_sub(1) {
                    // Encontrar el siguiente EOS dentro de este chunk.
                    let seg_end = tokens[seg_start..]
                        .iter()
                        .position(|&t| t == eos_token)
                        .map(|p| seg_start + p)
                        .unwrap_or(tokens.len());

                    if seg_end > seg_start {
                        let train_end = seg_end.min(tokens.len().saturating_sub(1));
                        let tgt_end = seg_end.min(tokens.len());
                        let train_seg = tokens.get(seg_start..train_end).unwrap_or(&[]);
                        let tgt_seg = tokens.get(seg_start + 1..tgt_end).unwrap_or(&[]);

                        if !train_seg.is_empty() {
                            // Accumulate chunks for batch training.
                            // Each call contributes one ctx_len-sized segment; once batch_size_file
                            // segments are collected (or EOS forces a flush), train on the batch.
                            let max_batch_tokens = batch_size_file * ctx_len;
                            let seg_len = train_seg.len().min(tgt_seg.len());
                            let mut offset = 0usize;
                            while offset < seg_len {
                                let remaining = max_batch_tokens.saturating_sub(batch_train_buf.len());
                                if remaining == 0 {
                                    // Flush full batch before consuming more.
                                    let is_val = self.cfg_val_every != 0
                                        && num_chunks > 0
                                        && num_chunks % self.cfg_val_every == 0;
                                    if is_val {
                                        self.eval_mode = true;
                                    }
                                    let eps = self.progressive_epsilon(epoch, epochs);
                                    let t_chunk_start = std::time::Instant::now();
                                    #[cfg(feature = "wgpu")]
                                    let slot_start = (num_chunks % 64) as u32 * 2;
                                    
                                    let loss = self.train_sequence(
                                        &batch_train_buf,
                                        &batch_tgt_buf,
                                        seg_start > 0,
                                        eps,
                                    );
                                    
                                    #[cfg(feature = "wgpu")]
                                    if let Some(gpu) = self.gpu_deq.as_ref() {
                                        gpu.tps_chunk_record(slot_start + 1); // Fin de este chunk
                                        gpu.tps_resolve_range(slot_start, 2);
                                        gpu.tps_chunk_record((slot_start + 2) % 128); // Inicio del siguiente
                                    }
                                    
                                    // Registrar en el buffer de métricas pendientes
                                    if self.metrics_log_path.is_some() {
                                        self.metrics_pending.push_back(PendingChunkMetric {
                                            epoch,
                                            chunk_id: num_chunks,
                                            loss,
                                            tokens: batch_train_buf.len(),
                                            slot_start: (num_chunks % 64) as u32 * 2,
                                            interval_start_time: t_chunk_start,
                                        });
                                        self.reap_metrics(false);
                                    }
                                    if is_val {
                                        self.eval_mode = false;
                                        println!(
                                            "    \x1b[93m[VAL] chunk {:>5}  val_loss={:.4}\x1b[0m",
                                            num_chunks, loss
                                        );
                                    } else {
                                        epoch_loss += loss;
                                    }
                                    num_chunks += 1;
                                    total_tokens += batch_train_buf.len();
                                    interval_tokens += batch_train_buf.len();
                                    batch_train_buf.clear();
                                    batch_tgt_buf.clear();
                                    #[cfg(feature = "wgpu")]
                                    if self.cfg_tps_sync_every != 0
                                        && num_chunks % self.cfg_tps_sync_every == 0 {
                                        if let Some(gpu) = self.gpu_deq.as_ref() {
                                            gpu.device.poll(wgpu::Maintain::Poll);
                                        }
                                    }
                                    #[cfg(feature = "wgpu")]
                                    if self.cfg_progress_every != 0 && num_chunks % self.cfg_progress_every == 0 {
                                        if let Some(gpu) = self.gpu_deq.as_ref() {
                                            let win_idx = ((num_chunks / self.cfg_progress_every) - 1) % 64;
                                            let slot_start = (win_idx as u32) * 2;
                                            gpu.tps_chunk_record(slot_start + 1); // Mark end
                                            gpu.tps_resolve_range(slot_start, 2);
                                            gpu.tps_chunk_record((slot_start + 2) % 128); // Mark start of next
                                        }
                                    }
                                    continue;
                                }

                                let take = remaining.min(seg_len - offset);
                                batch_train_buf.extend_from_slice(&train_seg[offset..offset + take]);
                                batch_tgt_buf.extend_from_slice(&tgt_seg[offset..offset + take]);
                                offset += take;
                            }

                            let flush = batch_train_buf.len() >= max_batch_tokens
                                || (eos_token != 0 && seg_start > 0); // flush on document boundary

                            if flush && !batch_train_buf.is_empty() {
                                let is_val = self.cfg_val_every != 0
                                    && num_chunks > 0
                                    && num_chunks % self.cfg_val_every == 0;
                                if is_val {
                                    self.eval_mode = true;
                                }
                                let t_chunk_start = std::time::Instant::now();
                                #[cfg(feature = "wgpu")]
                                let slot_start = (num_chunks % 64) as u32 * 2;
                                
                                let eps = self.progressive_epsilon(epoch, epochs);
                                let loss =
                                    self.train_sequence(&batch_train_buf, &batch_tgt_buf, seg_start > 0, eps);
                                
                                #[cfg(feature = "wgpu")]
                                if let Some(gpu) = self.gpu_deq.as_ref() {
                                    gpu.tps_chunk_record(slot_start + 1);
                                    gpu.tps_resolve_range(slot_start, 2);
                                }

                                if self.metrics_log_path.is_some() {
                                    self.metrics_pending.push_back(PendingChunkMetric {
                                        epoch,
                                        chunk_id: num_chunks,
                                        loss,
                                        tokens: batch_train_buf.len(),
                                        slot_start: (num_chunks % 64) as u32 * 2,
                                        interval_start_time: t_chunk_start,
                                    });
                                    self.reap_metrics(false);
                                }
                                if is_val {
                                    self.eval_mode = false;
                                    println!(
                                        "    \x1b[93m[VAL] chunk {:>5}  val_loss={:.4}\x1b[0m",
                                        num_chunks, loss
                                    );
                                } else {
                                    epoch_loss += loss;
                                }
                                num_chunks += 1;
                                total_tokens += batch_train_buf.len();
                                interval_tokens += batch_train_buf.len();
                                batch_train_buf.clear();
                                batch_tgt_buf.clear();
                                #[cfg(feature = "wgpu")]
                                if self.cfg_tps_sync_every != 0
                                    && num_chunks % self.cfg_tps_sync_every == 0 {
                                    if let Some(gpu) = self.gpu_deq.as_ref() {
                                        gpu.device.poll(wgpu::Maintain::Poll);
                                    }
                                }
                                // Marcado por-chunk manejado arriba
                                // Marcado por-chunk manejado arriba
                            }
                        }
                    }

                    // Saltar el token EOS y continuar en la siguiente sub-secuencia.
                    seg_start = if seg_end < tokens.len() {
                        seg_end + 1
                    } else {
                        tokens.len()
                    };
                }

                // Carry: solapar los últimos min(stride, tokens.len()) tokens para contexto.
                let overlap_start = tokens.len().saturating_sub(stride.min(tokens.len()));
                carry.clear();
                carry.extend_from_slice(&tokens[overlap_start..]);

                // Early stop for quick benchmarks / debugging.
                if num_chunks >= self.cfg_max_chunks {
                    break;
                }

                if n == 0 {
                    // EOF: flush any remaining accumulated chunks (< batch_size_file)
                    if !batch_train_buf.is_empty() {
                        let eps = self.progressive_epsilon(epoch, epochs);
                        let loss = self.train_sequence(&batch_train_buf, &batch_tgt_buf, false, eps);
                        epoch_loss += loss;
                        num_chunks += 1;
                        total_tokens += batch_train_buf.len();
                        batch_train_buf.clear();
                        batch_tgt_buf.clear();
                        #[cfg(feature = "wgpu")]
                        if self.cfg_tps_sync_every != 0
                            && num_chunks % self.cfg_tps_sync_every == 0 {
                            if let Some(gpu) = self.gpu_deq.as_ref() {
                                gpu.device.poll(wgpu::Maintain::Poll);
                            }
                        }
                        #[cfg(feature = "wgpu")]
                        if self.cfg_progress_every != 0 && num_chunks % self.cfg_progress_every == 0 {
                            // Handled by per-chunk marks
                        }
                    }
                    break; // EOF
                }

                // Mini-log cada 10 chunks para ver progreso en tiempo real
                if self.cfg_progress_every != 0
                    && num_chunks % self.cfg_progress_every == 0
                    && num_chunks > 0
                    && num_chunks != last_progress_chunk_logged
                {
                    #[cfg(feature = "wgpu")]
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        if self.cfg_progress_strict_sync {
                            gpu.device.poll(wgpu::Maintain::Wait);
                        } else {
                            gpu.device.poll(wgpu::Maintain::Poll);
                        }
                    }
                    let elapsed = t_start.elapsed().as_secs_f32();
                    let tps_run = total_tokens as f32 / elapsed.max(1e-9);
                    let tps_win = interval_tokens as f32 / interval_start.elapsed().as_secs_f32().max(1e-9);

                    let mut tps_gpu_text = String::new();
                    #[cfg(feature = "wgpu")]
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        // Intentamos leer la ventana que acaba de cerrarse hace 1 intervalo
                        // (o 2 para asegurar cero colisión de mapping)
                        let win_to_read = if num_chunks >= self.cfg_progress_every * 2 {
                            ((num_chunks / self.cfg_progress_every) - 2) % 64
                        } else { 999 };
                        if win_to_read < 64 {
                            if let Some(ns) = gpu.read_tps_range_ns_async((win_to_read as u32) * 2) {
                                let tps_g = (interval_tokens_prev as f64) / (ns / 1e9);
                                tps_gpu_text = format!("  \x1b[93mtps_gpu={:>8.1}\x1b[0m", tps_g);
                            }
                        }
                        interval_tokens_prev = interval_tokens;
                    }

                    let current_loss = self.visible_loss_text(Some(epoch_loss / num_chunks as f32));

                    println!(
                        "    \x1b[95m[progress]\x1b[0m chunk {:>5}  \x1b[92mloss={}\x1b[0m  \x1b[96mtps_win={:>8.1}\x1b[0m{}  \x1b[94mtps_run={:>8.1}\x1b[0m  \x1b[90mtime={:.1}s\x1b[0m",
                        num_chunks, current_loss, tps_win, tps_gpu_text, tps_run, elapsed
                    );
                    last_progress_chunk_logged = num_chunks;
                    interval_start = std::time::Instant::now();
                    interval_tokens = 0;

                    // Auto-save intra-epoch: solo por tiempo (cada 30 min).
                    // El guardado por `save_every` ya se aplica por epoch al final del loop.
                    let time_save = last_save_time.elapsed().as_secs() > 1800;
                    if time_save && !checkpoint_path.is_empty() {
                        println!(
                            "    \x1b[93m[auto-save]\x1b[0m Guardando progreso (chunk {})...",
                            num_chunks
                        );
                        if let Err(e) = self.save_checkpoint(checkpoint_path) {
                            eprintln!("    \x1b[31m[error]\x1b[0m No se pudo guardar: {}", e);
                        } else {
                            println!(
                                "    \x1b[32m[success]\x1b[0m Checkpoint '{}' actualizado.",
                                checkpoint_path
                            );
                            last_save_time = std::time::Instant::now();
                        }
                    }
                }
            }

            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                gpu.tps_epoch_end();
            }
            // Vaciar cola GPU al final de cada epoch.
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                // Validation/end-of-run boundary. We intentionally synchronize before final metrics.
                gpu.device.poll(wgpu::Maintain::Wait);
            }
            #[cfg(feature = "wgpu")]
            if let Some(loss) = self
                .gpu_deq
                .as_ref()
                .and_then(|gpu| self.cached_loss_after_sync(gpu))
            {
                self.last_gpu_loss = loss;
            }

            let elapsed = t_start.elapsed().as_secs_f32();
            let tps = if elapsed > 0.001 {
                total_tokens as f32 / elapsed
            } else {
                0.0
            };

            if epoch % log_every == 0 {
                let mut gpu_stats = String::new();
                #[cfg(feature = "wgpu")]
                if self.cfg_debug_sample_every != 0 {
                    let nz_h = self.cached_debug_buf.get(9).copied().unwrap_or(0.0);
                    gpu_stats = format!(" nz_h={:.4}", nz_h);
                }
                #[cfg(feature = "wgpu")]
                if let Some(gpu) = self.gpu_deq.as_ref() {
                    if let Some(ns) = gpu.read_tps_epoch_ns() {
                        let tps_gpu = (total_tokens as f64) / (ns / 1e9);
                        gpu_stats.push_str(&format!(" tps_gpu={:.1}", tps_gpu));
                    }
                }

                let display_loss = self.visible_loss_text(if num_chunks > 0 {
                    Some(epoch_loss / num_chunks as f32)
                } else {
                    None
                });

                println!(
                    "  epoch {epoch:>4}/{epochs}  loss={}  lr={:.6}  tps_epoch={:>8.1}  time={:.2}s  tokens={} {}",
                    display_loss, current_lr, tps, elapsed, total_tokens, gpu_stats
                );
            }

            if save_every > 0 && (epoch + 1) % save_every == 0 && !checkpoint_path.is_empty() {
                if let Err(e) = self.save_checkpoint(checkpoint_path) {
                    eprintln!(
                        "[checkpoint] Error guardando en '{}': {}",
                        checkpoint_path, e
                    );
                } else {
                    eprintln!(
                        "[checkpoint] Guardado en '{}' (epoch {})",
                        checkpoint_path,
                        epoch + 1
                    );
                }
            }
        }

        // Final drain de métricas para asegurar que el CSV esté completo antes de retornar
        self.reap_metrics(true);

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Checkpointing: pesos + estado del optimizador
    // ─────────────────────────────────────────────────────────────────────────

    /// Guarda el modelo completo más el estado del optimizador.
    /// Formato: <path>.aidn (pesos) + <path>.opt (momentos Adam).
    pub fn save_checkpoint(&mut self, base_path: &str) -> std::io::Result<()> {
        // Pesos del modelo
        self.save_full(&format!("{base_path}.aidn"))?;

        // Sincronizar momentos GPU → CPU Adam antes de guardar
        #[cfg(feature = "wgpu")]
        self.sync_gpu_moments_to_cpu();

        // Estado del optimizador (con momentos ya poblados)
        self.optimizer
            .save_state(&format!("{base_path}.opt"))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Carga un checkpoint completo (pesos + estado del optimizador).
    pub fn load_checkpoint(base_path: &str) -> std::io::Result<Self> {
        let mut trainer = Self::load_full(&format!("{base_path}.aidn"))?;
        let opt_path = format!("{base_path}.opt");
        if std::path::Path::new(&opt_path).exists() {
            trainer
                .optimizer
                .load_state(&opt_path)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            // Restaurar momentos Adam en GPU si está disponible
            #[cfg(feature = "wgpu")]
            trainer.sync_cpu_moments_to_gpu();
        }
        Ok(trainer)
    }

    /// Descarga los momentos Adam desde GPU → CPU Adam struct para poder guardarlos.
    #[cfg(feature = "wgpu")]
    fn sync_gpu_moments_to_cpu(&mut self) {
        use nalgebra::{DMatrix, DVector};
        let gpu = match self.gpu_deq.as_ref() {
            Some(g) => g,
            None => return,
        };

        println!("[GPU-CHECK] Iniciando Checksum de VRAM antes de guardar...");
        // Checkpoint checksum path: CPU is about to read GPU moments synchronously.
        gpu.device.poll(wgpu::Maintain::Wait);

        // LM Head moments
        if let Some(gpu_lm) = self.gpu_lm.as_ref() {
            if let Ok((m_w, v_w, m_b, v_b, m_g, v_g)) = gpu_lm.read_moments(&gpu.device, &gpu.queue)
            {
                let sum_mw: f32 = m_w.iter().map(|x| x.abs()).sum();
                let sum_vw: f32 = v_w.iter().map(|x| x.abs()).sum();

                // VRAM Checksum: Si los momentos son exactamente cero, algo falló en el mapeo
                if sum_mw == 0.0 && sum_vw == 0.0 && self.optimizer.step_count() > 10 {
                    panic!("\x1b[31m[CRITICAL ERROR]\x1b[0m Checksum de VRAM falló (Momentos LM=0). Abortando para proteger checkpoint.");
                }

                println!(
                    "    LM Head Moments Checksum: m={:.4}, v={:.6e}",
                    sum_mw, sum_vw
                );

                let d_r = self.config.d_r;
                let vocab = self.config.vocab_size;
                self.optimizer
                    .set_mat("lm_w_m", DMatrix::from_vec(d_r, vocab, m_w));
                self.optimizer
                    .set_mat("lm_w_v", DMatrix::from_vec(d_r, vocab, v_w));
                self.optimizer.set_vec("lm_b_m", DVector::from_vec(m_b));
                self.optimizer.set_vec("lm_b_v", DVector::from_vec(v_b));
                self.optimizer.set_vec("lm_g_m", DVector::from_vec(m_g));
                self.optimizer.set_vec("lm_g_v", DVector::from_vec(v_g));
            }
        }

        // Embedding moments
        if let Some(gpu_emb) = self.gpu_emb.as_ref() {
            if let Ok((m_emb, v_emb)) = gpu_emb.read_moments(&gpu.device, &gpu.queue) {
                let sum_me: f32 = m_emb.iter().map(|x| x.abs()).sum();
                if sum_me == 0.0 && self.optimizer.step_count() > 10 {
                    panic!("\x1b[31m[CRITICAL ERROR]\x1b[0m Checksum de VRAM falló (Momentos EMB=0). Abortando.");
                }
                println!("    Embedding Moments Checksum: m={:.4}", sum_me);

                let d_r = self.config.d_r;
                let vocab = self.config.vocab_size;
                self.optimizer
                    .set_mat("emb_m", DMatrix::from_vec(d_r, vocab, m_emb));
                self.optimizer
                    .set_mat("emb_v", DMatrix::from_vec(d_r, vocab, v_emb));
            }
        }
    }

    /// Sube los momentos Adam CPU → GPU después de cargar un checkpoint.
    #[cfg(feature = "wgpu")]
    pub fn sync_cpu_moments_to_gpu(&mut self) {
        let gpu = match self.gpu_deq.as_ref() {
            Some(g) => g,
            None => return,
        };

        if let Some(gpu_lm) = self.gpu_lm.as_ref() {
            if let (Some(m_w), Some(v_w), Some(m_b), Some(v_b), Some(m_g), Some(v_g)) = (
                self.optimizer.get_mat("lm_w_m"),
                self.optimizer.get_mat("lm_w_v"),
                self.optimizer.get_vec("lm_b_m"),
                self.optimizer.get_vec("lm_b_v"),
                self.optimizer.get_vec("lm_g_m"),
                self.optimizer.get_vec("lm_g_v"),
            ) {
                gpu_lm.write_moments(
                    &gpu.queue,
                    m_w.as_slice(),
                    v_w.as_slice(),
                    m_b.as_slice(),
                    v_b.as_slice(),
                    m_g.as_slice(),
                    v_g.as_slice(),
                );
            }
        }

        if let Some(gpu_emb) = self.gpu_emb.as_ref() {
            if let (Some(m_emb), Some(v_emb)) = (
                self.optimizer.get_mat("emb_m"),
                self.optimizer.get_mat("emb_v"),
            ) {
                gpu_emb.write_moments(&gpu.queue, m_emb.as_slice(), v_emb.as_slice());
            }
        }
    }

    /// Guarda el modelo completo (Config + Reasoning + LmHead + Tokenizer) en un solo .aidn
    pub fn save_full(&mut self, path: &str) -> std::io::Result<()> {
        use aideen_core::model::AidenModel;

        #[cfg(feature = "wgpu")]
        self.sync_inference_weights();

        let mut model = AidenModel::new(self.config.clone());

        // Empacar pesos de Reasoning
        for (k, v) in self.reasoning.export_weights() {
            model.set_weight(&k, v);
        }

        // Empacar pesos de LmHead
        for (k, v) in self.lm_head.export_weights() {
            model.set_weight(&k, v);
        }

        // Empacar embeddings del Tokenizer
        model.set_weight(
            "tokenizer.embeddings",
            self.tokenizer.embeddings.as_slice().to_vec(),
        );

        model
            .metadata
            .insert("aideen_version".to_string(), "0.1.0".to_string());
        model
            .metadata
            .insert("type".to_string(), "FullModel".to_string());
        model.metadata.insert(
            "vocab_size".to_string(),
            self.tokenizer.vocab_size().to_string(),
        );

        // Persistir estado del Stability Oracle (v13.2)
        model.metadata.insert(
            "adaptive_max_iters".to_string(),
            self.adaptive_max_iters.to_string(),
        );
        model.metadata.insert(
            "adaptive_damping".to_string(),
            self.adaptive_damping.to_string(),
        );
        model.metadata.insert(
            "adaptive_adj_iters".to_string(),
            self.adaptive_adj_iters.to_string(),
        );
        model.metadata.insert(
            "damping_boost_left".to_string(),
            self.damping_boost_left.to_string(),
        );
        model.metadata.insert(
            "emergency_left".to_string(),
            self.emergency_left.to_string(),
        );
        model
            .metadata
            .insert("last_max_h".to_string(), self.last_max_h.to_string());

        model
            .save(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Carga el modelo completo (Config + Reasoning + LmHead + Tokenizer) desde un .aidn
    pub fn load_full(path: &str) -> std::io::Result<Self> {
        use aideen_core::model::AidenModel;

        let model = AidenModel::load(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Reconstruir Tokenizer
        let emb_data = model.get_weight("tokenizer.embeddings").ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "tokenizer.embeddings not found")
        })?;
        let vocab_size = model
            .metadata
            .get("vocab_size")
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "vocab_size not found in metadata",
                )
            })?;

        let mut tokenizer = Tokenizer::new_empty(vocab_size, model.config.clone());

        // Restaurar el HF tokenizer (BPE) si está disponible en rutas estándar.
        // Sin esto, encode() usa el fallback char-level vacío y devuelve [].
        let tok_paths = [
            "aideen-backbone/tokenizer.json",
            "tokenizer.json",
            "../aideen-backbone/tokenizer.json",
        ];
        for tok_path in &tok_paths {
            if std::path::Path::new(tok_path).exists() {
                if let Ok(hf) = tokenizers::Tokenizer::from_file(tok_path) {
                    tokenizer.hf_tokenizer = Some(hf);
                    break;
                }
            }
        }

        if emb_data.len() != tokenizer.embeddings.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Embeddings size mismatch",
            ));
        }
        tokenizer
            .embeddings
            .as_mut_slice()
            .copy_from_slice(emb_data);

        // Reconstruir Trainer
        let mut trainer = Trainer::from_tokenizer(tokenizer, 0.001); // LR por defecto
        trainer.config = model.config.clone();

        // Importar pesos en Reasoning
        trainer
            .reasoning
            .import_weights(&model.weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Importar pesos en LmHead
        trainer
            .lm_head
            .import_weights(&model.weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Restaurar estado del Stability Oracle (v13.2)
        if let Some(v) = model
            .metadata
            .get("adaptive_max_iters")
            .and_then(|s| s.parse().ok())
        {
            trainer.adaptive_max_iters = v;
        }
        if let Some(v) = model
            .metadata
            .get("adaptive_damping")
            .and_then(|s| s.parse().ok())
        {
            trainer.adaptive_damping = v;
        }
        if let Some(v) = model
            .metadata
            .get("adaptive_adj_iters")
            .and_then(|s| s.parse().ok())
        {
            trainer.adaptive_adj_iters = v;
        }
        if let Some(v) = model
            .metadata
            .get("damping_boost_left")
            .and_then(|s| s.parse().ok())
        {
            trainer.damping_boost_left = v;
        }
        if let Some(v) = model
            .metadata
            .get("emergency_left")
            .and_then(|s| s.parse().ok())
        {
            trainer.emergency_left = v;
        }
        if let Some(v) = model
            .metadata
            .get("last_max_h")
            .and_then(|s| s.parse().ok())
        {
            trainer.last_max_h = v;
        }

        Ok(trainer)
    }

    pub fn save_deq(&mut self, path: &str) -> std::io::Result<()> {
        #[cfg(feature = "wgpu")]
        self.sync_inference_weights();
        self.reasoning
            .save_checkpoint(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    #[cfg(feature = "wgpu")]
    pub fn sync_inference_weights(&mut self) {
        if let Some(gpu) = self.gpu_deq.as_ref() {
            // Sincronizar DEQ Core solo si los pesos fueron subidos/actualizados en GPU.
            if self.gpu_weights_uploaded {
                if let Ok((wq, wk, wv, wo, win, wx, wout, alog, nscale)) = gpu.read_weights() {
                let to_mat = |vec: Vec<f32>| {
                    let d_r = self.config.d_r;
                    nalgebra::DMatrix::from_column_slice(d_r, d_r, &vec)
                };
                // wq/wk contain [h_slots * d*d matrices | h_slots*d bias] — split and average.
                {
                    let d_r = self.config.d_r;
                    let h_slots = self.config.h_slots;
                    let mat_total = h_slots * d_r * d_r;
                    // Average per-slot matrices into CPU prototype (used for checkpoint only)
                    let avg_mat = |flat: &[f32]| -> Vec<f32> {
                        (0..d_r * d_r)
                            .map(|i| {
                                (0..h_slots).map(|s| flat[s * d_r * d_r + i]).sum::<f32>()
                                    / h_slots as f32
                            })
                            .collect()
                    };
                    self.reasoning.w_q =
                        nalgebra::DMatrix::from_column_slice(d_r, d_r, &avg_mat(&wq[..mat_total]));
                    self.reasoning.q_bias =
                        nalgebra::DMatrix::from_row_slice(h_slots, d_r, &wq[mat_total..]);
                    self.reasoning.w_k =
                        nalgebra::DMatrix::from_column_slice(d_r, d_r, &avg_mat(&wk[..mat_total]));
                    self.reasoning.k_bias =
                        nalgebra::DMatrix::from_row_slice(h_slots, d_r, &wk[mat_total..]);
                }
                {
                    let d_r = self.config.d_r;
                    let h_slots = self.config.h_slots;
                    // wv is now h_slots*d*d — average slots for CPU prototype (checkpoint only)
                    let avg: Vec<f32> = (0..d_r * d_r)
                        .map(|i| {
                            (0..h_slots).map(|s| wv[s * d_r * d_r + i]).sum::<f32>()
                                / h_slots as f32
                        })
                        .collect();
                    self.reasoning.w_v = nalgebra::DMatrix::from_column_slice(d_r, d_r, &avg);
                }
                {
                    let d_r = self.config.d_r;
                    let h_slots = self.config.h_slots;
                    // w_o is per-slot on GPU — average slots for CPU prototype (checkpoint only)
                    let avg: Vec<f32> = (0..d_r * d_r)
                        .map(|i| {
                            (0..h_slots).map(|s| wo[s * d_r * d_r + i]).sum::<f32>()
                                / h_slots as f32
                        })
                        .collect();
                    self.reasoning.w_o = nalgebra::DMatrix::from_column_slice(d_r, d_r, &avg);
                }
                // win is h_slots*d*d — average slots for CPU representation.
                let d_r = self.config.d_r;
                let h_slots = self.config.h_slots;
                let mut win_avg = vec![0.0f32; d_r * d_r];
                for s in 0..h_slots {
                    let base = s * d_r * d_r;
                    for i in 0..d_r * d_r {
                        win_avg[i] += win[base + i];
                    }
                }
                let inv_slots = 1.0 / h_slots as f32;
                for v in &mut win_avg {
                    *v *= inv_slots;
                }
                self.reasoning.w_in = to_mat(win_avg);
                self.reasoning.w_x = to_mat(wx);
                self.reasoning.w_out = to_mat(wout);
                {
                    let h_slots = self.reasoning.config.h_slots;
                    let d_r = self.reasoning.config.d_r;
                    self.reasoning.a_log = nalgebra::DMatrix::from_row_slice(h_slots, d_r, &alog);
                }
                self.reasoning.norm_scale = nalgebra::DVector::from_column_slice(&nscale);
                self.gpu_weights_uploaded = true; // Weights are still on GPU, just synced to CPU
                self.gpu_cg_weights_uploaded = true;
                }
            }
        }

        // Sincronizar Embeddings
        if let Some(gpu_emb) = self.gpu_emb.as_ref() {
            if let Some(gpu) = self.gpu_deq.as_ref() {
                if self.gpu_emb_weights_uploaded {
                    if let Ok(emb_data) = gpu_emb.read_weights(&gpu.device, &gpu.queue) {
                        if emb_data.len() == self.tokenizer.embeddings.len() {
                            self.tokenizer.embeddings.copy_from_slice(&emb_data);
                            self.gpu_emb_weights_uploaded = true; // Still on GPU
                        }
                    }
                }
            }
        }

        self.sync_lm_head_from_gpu_if_needed();
    }

    #[cfg(feature = "wgpu")]
    fn sync_lm_head_from_gpu_if_needed(&mut self) {
        if !self.lm_head_cpu_stale {
            return;
        }
        let Some(gpu) = self.gpu_deq.as_ref() else {
            return;
        };
        let Some(gpu_lm) = self.gpu_lm.as_ref() else {
            return;
        };
        if let Ok((w, b, g)) = gpu_lm.read_weights(&gpu.device, &gpu.queue) {
            if w.len() == self.lm_head.w.len()
                && b.len() == self.lm_head.b.len()
                && g.len() == self.lm_head.g.len()
            {
                self.lm_head.w = nalgebra::DMatrix::from_column_slice(
                    self.lm_head.w.nrows(),
                    self.config.d_r,
                    &w,
                );
                self.lm_head.b = nalgebra::DVector::from_column_slice(&b);
                self.lm_head.g = nalgebra::DVector::from_column_slice(&g);
                self.lm_head_cpu_stale = false;
            }
        }
    }

    /// Configura explícitamente el backend de inferencia.
    /// - `prefer_gpu=true`: intenta activar GPU; si no hay, queda en CPU.
    /// - `prefer_gpu=false`: fuerza CPU.
    #[cfg(feature = "wgpu")]
    pub fn configure_inference_backend(&mut self, prefer_gpu: bool) -> bool {
        if !prefer_gpu {
            self.reasoning.clear_backend();
            return false;
        }

        if let Some(gpu) = aideen_backbone::gpu_backend::WgpuBlockBackend::new_blocking() {
            self.reasoning.set_backend(gpu);
            true
        } else {
            self.reasoning.clear_backend();
            false
        }
    }

    /// En builds sin `wgpu`, siempre usa CPU.
    #[cfg(not(feature = "wgpu"))]
    pub fn configure_inference_backend(&mut self, _prefer_gpu: bool) -> bool {
        self.reasoning.clear_backend();
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trainer() -> (Trainer, Vec<u32>) {
        let text = "la inteligencia artificial distribuida razona en equilibrio";
        let config = ArchitectureConfig::default();
        let tok = Tokenizer::from_text(text, config);
        let tokens = tok.encode(text);
        let mut trainer = Trainer::from_tokenizer_seeded(tok, 0.01, 42);
        trainer.config.max_deq_iters = 20;
        trainer.config.adj_iters = 5;
        (trainer, tokens)
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn loss_decreases_with_embeddings() {
        let (mut trainer, tokens) = make_trainer();
        let ctx = &tokens[0..5];
        let target = tokens[5];
        let loss_0 = trainer.train_step(ctx, target, false);
        for _ in 0..15 {
            trainer.train_step(ctx, target, false);
        }
        let loss_15 = trainer.train_step(ctx, target, false);
        assert!(loss_15 < loss_0);
    }

    #[test]
    fn eval_loss_reports_fixed_history_reference_delta() {
        let (trainer, mut tokens) = make_trainer();
        let extra = trainer
            .tokenizer
            .encode(" la memoria temporal estabiliza el contexto token a token");
        tokens.extend_from_slice(&extra);

        std::env::remove_var("AIDEEN_DEQ_FIXED_HISTORY_REFERENCE");
        let plain = trainer.eval_loss(&tokens);
        std::env::set_var("AIDEEN_DEQ_FIXED_HISTORY_REFERENCE", "1");
        let fixed = trainer.eval_loss(&tokens);
        std::env::remove_var("AIDEEN_DEQ_FIXED_HISTORY_REFERENCE");

        eprintln!(
            "[fixed-history-eval] plain={:.6} fixed={:.6} delta={:.6}",
            plain,
            fixed,
            fixed - plain
        );
        assert!(plain.is_finite());
        assert!(fixed.is_finite());
    }

}
