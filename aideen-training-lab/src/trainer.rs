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
    reasoning::Reasoning,
    state::{ArchitectureConfig, HSlots},
};
use std::collections::VecDeque;
use std::io::Write;

use crate::optimizer::Adam;
use aideen_backbone::{
    deq_mode::DeqRuntimeConfig, lm_head::LmHead, fixed_point_memory_reasoning::FixedPointMemoryReasoning,
    tokenizer::Tokenizer,
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

#[derive(Clone, Copy, Debug, Default)]
struct FpmHealthMetrics {
    max_err_h: f32,
    max_mem_update: f32,
    max_z: f32,
    avg_z: f32,
    rescue_recovered: f32,
    dead_slots: f32,
    max_update_ratio: f32,
    write_saturation: f32,
    max_memctx_rms: f32,
    max_memctx_to_signal: f32,
    exit_err_h: f32,
    exit_iter: f32,
    rescue_entered: f32,
    pre_rescue_converged: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SolveStatus {
    HealthyConverged,
    TrivialConverged,
    Unconverged,
    NumericInvalid,
}

impl SolveStatus {
    fn label(self) -> &'static str {
        match self {
            Self::HealthyConverged => "OK",
            Self::TrivialConverged => "NULL",
            Self::Unconverged => "FAIL",
            Self::NumericInvalid => "INV",
        }
    }
}

pub struct Trainer {
    pub config: ArchitectureConfig,
    pub training_config: TrainingConfig,
    pub reasoning: FixedPointMemoryReasoning,
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
    pub completed_file_epochs: usize,
    pub plateau_best_loss: Option<f32>,
    pub plateau_bad_epochs: usize,
    pub plateau_cooldown_left: usize,
    pub plateau_lr_cap: f32,
    pub solve_stage_floor: u32,
    pub solve_stage_cap: u32,

    // --- v14 Temporal Memory State ---
    pub m_prev: Option<HSlots>,

    // --- Gradient Accumulation ---
    grad_accum_counter: u32, // steps accumulated so far in the current window

    // --- TPS tracking for GPU-DEBUG log ---
    debug_last_time: Option<std::time::Instant>,
    debug_tokens_accum: u32, // tokens processed since last GPU-DEBUG print

    // --- Debug buffer cache (avoid blocking GPU readback every step) ---
    // read_debug_buffer() calls device.poll(Maintain::Wait) — blocks CPU until GPU finishes.
    // Now deferred to end-of-step (after apply_gradient_update) so GPU is already idle.
    cached_debug_buf: Vec<f32>,
    cached_debug_gen: u64,
    invalid_eval_debug_gen: u64,
    // --- Cached GPU loss (avoid sync readback every step) ---
    last_gpu_loss: f32,
    fpm_alpha_m_current: f32,
    fpm_tau_current: f32,
    fpm_err_h_window: VecDeque<f32>,
    fpm_last_err_h_avg: f32,

    // --- Cached hot-path env vars (parsed once at construction) ---
    // Avoids ~26 env::var syscalls per training step.
    cfg_fwd_batch_size: u32,       // AIDEEN_BATCH_SIZE (for forward dispatch)
    cfg_debug_sample_every: usize, // AIDEEN_DEBUG_SAMPLE
    cfg_solve_control_every: usize, // AIDEEN_SOLVE_CONTROL_EVERY
    pub cfg_loss_readback_every: usize, // AIDEEN_LOSS_READBACK_EVERY
    cfg_tps_sync_every: usize,     // AIDEEN_TPS_SYNC_EVERY
    cfg_grad_accum: u32,           // AIDEEN_GRAD_ACCUM
    cfg_hist_min_iters: u32,       // AIDEEN_HIST_MIN_ITERS
    cfg_val_every: usize,          // AIDEEN_VAL_EVERY
    cfg_progress_every: usize,     // AIDEEN_PROGRESS_EVERY
    cfg_wv_debug: bool,            // AIDEEN_DEQ_WV_DEBUG
    cfg_ssm_debug: bool,           // AIDEEN_SSM_DEBUG
    cfg_max_chunks: usize,         // AIDEEN_MAX_CHUNKS
    cfg_adj_iters_override: Option<u32>, // AIDEEN_ADJ_ITERS_OVERRIDE
    cfg_system_cost_audit: bool,   // AIDEEN_SYSTEM_COST_AUDIT
    cfg_system_cost_wait: bool,    // AIDEEN_SYSTEM_COST_WAIT
    cfg_slot_path_audit: bool,     // AIDEEN_SLOT_PATH_AUDIT
    cfg_force_renorm: bool,        // AIDEEN_DEQ_FORCE_RENORM
    cfg_lm_force_cpu_dldh: bool,   // AIDEEN_LM_FORCE_CPU_DLDH
    cfg_lm_dldh_parity: bool,      // AIDEEN_LM_DLDH_PARITY
    cfg_log_emb_stats: bool,       // AIDEEN_LOG_EMB_STATS
    cfg_lr_plateau_enable: bool,   // AIDEEN_LR_PLATEAU_ENABLE
    cfg_lr_plateau_patience: usize, // AIDEEN_LR_PLATEAU_PATIENCE
    cfg_lr_plateau_cooldown: usize, // AIDEEN_LR_PLATEAU_COOLDOWN
    cfg_lr_plateau_factor: f32,    // AIDEEN_LR_PLATEAU_FACTOR
    cfg_lr_plateau_min_rel_improvement: f32, // AIDEEN_LR_PLATEAU_MIN_REL_IMPROVEMENT
    cfg_lr_plateau_min_lr_override: Option<f32>, // AIDEEN_LR_PLATEAU_MIN_LR
    cfg_assoc_lr_mult: f32,        // AIDEEN_ASSOC_LR_MULT
    cfg_assoc_event_lr_mult: f32,  // AIDEEN_ASSOC_EVENT_LR_MULT
    cfg_assoc_alpha_lr_mult: f32,  // AIDEEN_ASSOC_ALPHA_LR_MULT
}

impl Trainer {
    // TEMPORARY ASSOCIATIVE DIAGNOSTIC: remove after assoc recall gradient/forward path is localized.
    pub fn enable_temporary_assoc_debug_sampling(&mut self, every: usize) {
        self.cfg_debug_sample_every = every;
    }

    pub fn gpu_sequence_capacity(&self) -> usize {
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = self.gpu_deq.as_ref() {
            return gpu.forward_seq_cap.max(1) as usize;
        }
        self.config.ctx_len.max(1)
    }

    #[cfg(feature = "wgpu")]
    fn w_delta_rank_summary(
        w_delta: &nalgebra::DMatrix<f32>,
        h_slots: usize,
        d_r: usize,
    ) -> Option<(String, String, String)> {
        // Now analyses W_k_write (h*d_r × RETAIN_RANK) per slot.
        let retain_rank = w_delta.ncols();
        if h_slots == 0 || d_r == 0 || retain_rank == 0 || w_delta.nrows() != h_slots * d_r {
            return None;
        }
        let mut stable_ranks = Vec::with_capacity(h_slots);
        let mut top8 = Vec::with_capacity(h_slots);
        let mut top32 = Vec::with_capacity(h_slots);
        for slot in 0..h_slots {
            let block = w_delta.rows(slot * d_r, d_r).into_owned();
            let svd = nalgebra::linalg::SVD::new(block, false, false);
            let mut sigmas = svd.singular_values.as_slice().to_vec();
            if sigmas.is_empty() {
                stable_ranks.push("0.0".to_string());
                top8.push("0.000".to_string());
                top32.push("0.000".to_string());
                continue;
            }
            sigmas.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let energy: f32 = sigmas.iter().map(|s| s * s).sum();
            let sigma_max_sq = sigmas[0] * sigmas[0];
            let stable_rank = if sigma_max_sq > 0.0 {
                energy / sigma_max_sq
            } else {
                0.0
            };
            let top8_energy: f32 = sigmas.iter().take(8).map(|s| s * s).sum();
            let top32_energy: f32 = sigmas.iter().take(32).map(|s| s * s).sum();
            stable_ranks.push(format!("{stable_rank:.1}"));
            top8.push(format!("{:.3}", top8_energy / energy.max(1e-12)));
            top32.push(format!("{:.3}", top32_energy / energy.max(1e-12)));
        }
        Some((stable_ranks.join(","), top8.join(","), top32.join(",")))
    }

    fn visible_loss_value(&self, fallback: Option<f32>) -> Option<f32> {
        if self.last_gpu_loss.is_finite() && self.last_gpu_loss > 0.0 {
            Some(self.last_gpu_loss)
        } else {
            fallback.filter(|v| v.is_finite() && *v > 0.0)
        }
    }

    fn visible_loss_text(&self, fallback: Option<f32>) -> String {
        if let Some(v) = self.visible_loss_value(fallback) {
            format!("{:.4}", v)
        } else {
            "n/a".to_string()
        }
    }

    fn checkpoint_metrics_path(base_path: &str) -> String {
        format!("{base_path}_metrics.csv")
    }

    fn file_epoch_schedule(global_epoch: usize, total_epochs: usize) -> (f32, u32, u32, usize) {
        let total = total_epochs.max(1);
        let progress = global_epoch as f32 / total as f32;
        let (floor, cap, adj_iters) = if progress < 0.25 {
            (8, 12, 2usize)
        } else if progress < 0.60 {
            (10, 14, 6usize)
        } else {
            (12, 16, 8usize)
        };
        (progress, floor, cap, adj_iters)
    }

    fn lr_plateau_min_lr(&self) -> f32 {
        self.cfg_lr_plateau_min_lr_override
            .unwrap_or(self.training_config.lr_min)
            .max(0.0)
    }

    fn controlled_epoch_lr(&self, global_epoch: usize, total_epochs: usize) -> f32 {
        let base_lr = self.cosine_lr(global_epoch, total_epochs);
        if !self.cfg_lr_plateau_enable {
            return base_lr;
        }
        let min_lr = self.lr_plateau_min_lr();
        base_lr.min(self.plateau_lr_cap).max(min_lr)
    }

    fn update_lr_plateau_controller(&mut self, epoch_loss: f32, current_lr: f32) -> Option<f32> {
        if !self.cfg_lr_plateau_enable || !epoch_loss.is_finite() || epoch_loss <= 0.0 {
            return None;
        }

        let min_rel = self.cfg_lr_plateau_min_rel_improvement.max(0.0);
        let improved = self
            .plateau_best_loss
            .map_or(true, |best| epoch_loss < best * (1.0 - min_rel));

        if improved {
            self.plateau_best_loss = Some(epoch_loss);
            self.plateau_bad_epochs = 0;
            if self.plateau_cooldown_left > 0 {
                self.plateau_cooldown_left -= 1;
            }
            return None;
        }

        if self.plateau_cooldown_left > 0 {
            self.plateau_cooldown_left -= 1;
            return None;
        }

        self.plateau_bad_epochs += 1;
        if self.plateau_bad_epochs < self.cfg_lr_plateau_patience.max(1) {
            return None;
        }

        let min_lr = self.lr_plateau_min_lr();
        let new_cap = (current_lr * self.cfg_lr_plateau_factor.clamp(0.05, 0.95)).max(min_lr);
        if new_cap >= self.plateau_lr_cap - 1e-12 {
            self.plateau_bad_epochs = 0;
            self.plateau_cooldown_left = self.cfg_lr_plateau_cooldown;
            return None;
        }

        self.plateau_lr_cap = new_cap;
        self.plateau_bad_epochs = 0;
        self.plateau_cooldown_left = self.cfg_lr_plateau_cooldown;
        Some(new_cap)
    }

    fn apply_stage_solve_schedule(&mut self, floor: u32, cap: u32, adj_iters: usize) {
        let floor = floor.max(1);
        let cap = cap.max(floor);
        self.solve_stage_floor = floor;
        self.solve_stage_cap = cap;
        self.adaptive_max_iters = self.adaptive_max_iters.clamp(floor, cap);
        self.config.adj_iters = adj_iters.max(1);
        if self.cfg_adj_iters_override.is_none() {
            self.adaptive_adj_iters = self.config.adj_iters as u32;
        }
    }

    fn emergency_solve_cap(&self) -> u32 {
        (self.solve_stage_cap + 4).min(48).max(self.solve_stage_cap)
    }

    fn checkpoint_best_base_path(base_path: &str) -> String {
        format!("{base_path}_best_loss")
    }

    fn checkpoint_best_meta_path(base_path: &str) -> String {
        format!("{}.meta", Self::checkpoint_best_base_path(base_path))
    }

    fn load_best_loss(base_path: &str) -> Option<f32> {
        if base_path.is_empty() {
            return None;
        }
        let meta_path = Self::checkpoint_best_meta_path(base_path);
        let contents = std::fs::read_to_string(meta_path).ok()?;
        for line in contents.lines() {
            if let Some(value) = line.strip_prefix("loss=") {
                if let Ok(loss) = value.trim().parse::<f32>() {
                    return Some(loss);
                }
            }
        }
        None
    }

    fn append_epoch_metrics(
        &self,
        checkpoint_path: &str,
        epoch: usize,
        epoch_loss: f32,
        lr: f32,
        tps: f32,
        total_tokens: usize,
    ) -> std::io::Result<()> {
        if checkpoint_path.is_empty() {
            return Ok(());
        }
        let metrics_path = Self::checkpoint_metrics_path(checkpoint_path);
        if let Some(parent) = std::path::Path::new(&metrics_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let new_file = !std::path::Path::new(&metrics_path).exists();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&metrics_path)?;
        if new_file {
            writeln!(file, "epoch,loss,lr,tps_epoch,tokens")?;
        }
        writeln!(
            file,
            "{epoch},{epoch_loss:.6},{lr:.8},{tps:.3},{total_tokens}"
        )?;
        Ok(())
    }

    fn save_best_loss_checkpoint(
        &mut self,
        checkpoint_path: &str,
        epoch: usize,
        epoch_loss: f32,
        lr: f32,
        total_tokens: usize,
    ) -> std::io::Result<()> {
        if checkpoint_path.is_empty() {
            return Ok(());
        }
        let best_base = Self::checkpoint_best_base_path(checkpoint_path);
        self.save_checkpoint(&best_base)?;
        let meta_path = Self::checkpoint_best_meta_path(checkpoint_path);
        let mut meta = String::new();
        meta.push_str(&format!("loss={epoch_loss:.6}\n"));
        meta.push_str(&format!("epoch={epoch}\n"));
        meta.push_str(&format!("lr={lr:.8}\n"));
        meta.push_str(&format!("tokens={total_tokens}\n"));
        std::fs::write(meta_path, meta)
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
    fn cached_loss_after_sync(&self, gpu: &GpuDeqBackend, force: bool) -> Option<f32> {
        let every = self.cfg_loss_readback_every;
        let should_read =
            force || (every != 0 && (every == 1 || (self.optimizer.step_count() % every == 0)));
        if should_read {
            self.gpu_lm
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
            w_kv_write_rm,
            b_delta,
            w_gate_hist_rm,
            w_write_gate_rm,
            b_write_mem,
            w_retain_up_rm,
            w_retain_down_rm,
            b_retain_rm,
            w_q_mem_rm,
            w_k_mem_rm,
            b_read_mem,
            w_k_assoc_rm,
            w_v_assoc_rm,
            w_q_assoc_rm,
            alpha_assoc,
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
            w_kv_write_rm.as_slice(),
            b_delta.as_slice(),
            w_gate_hist_rm.as_slice(),
            w_write_gate_rm.as_slice(),
            b_write_mem.as_slice(),
            w_retain_up_rm.as_slice(),
            w_retain_down_rm.as_slice(),
            b_retain_rm.as_slice(),
            w_q_mem_rm.as_slice(),
            w_k_mem_rm.as_slice(),
            b_read_mem.as_slice(),
            w_k_assoc_rm.as_slice(),
            w_v_assoc_rm.as_slice(),
            w_q_assoc_rm.as_slice(),
            alpha_assoc.as_slice(),
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
            _w_kv_write_rm,
            _b_delta,
            _w_gate_hist_rm,
            _w_write_gate_rm,
            _b_write_mem,
            _w_retain_up_rm,
            _w_retain_down_rm,
            _b_retain_rm,
            _w_q_mem_rm,
            _w_k_mem_rm,
            _b_read_mem,
            _w_k_assoc_rm,
            _w_v_assoc_rm,
            _w_q_assoc_rm,
            _alpha_assoc,
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

    fn deq_runtime_config() -> DeqRuntimeConfig {
        DeqRuntimeConfig::from_env()
    }

    fn clean_deq_mode_active() -> bool {
        Self::deq_runtime_config().is_deq_only()
    }

    fn baseline_fpm_enabled() -> bool {
        std::env::var("AIDEEN_FPM")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(true)
    }

    fn slot_coord_mode_active() -> bool {
        Self::deq_runtime_config().has_explicit_slot_comparison()
    }

    fn effective_fpm_enabled() -> bool {
        Self::baseline_fpm_enabled() && !Self::clean_deq_mode_active()
    }

    fn fpm_stage_from_env() -> u32 {
        std::env::var("AIDEEN_FPM_STAGE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(4)
    }

    fn default_hist_min_iters() -> u32 {
        if Self::clean_deq_mode_active() || Self::slot_coord_mode_active() {
            1
        } else {
            20
        }
    }

    fn default_slot_coord_min_iters() -> u32 {
        1
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
        let f32_bytes = std::mem::size_of::<f32>() as f64;
        let _ = max_iters;
        let runtime_cfg = Self::deq_runtime_config();
        let deq_only = runtime_cfg.is_deq_only();
        let slot_coord_mode = runtime_cfg.has_explicit_slot_comparison();
        let fpm_enabled = Self::effective_fpm_enabled();
        if deq_only {
            let win_bytes = b * t * (h * d * d * f32_bytes);
            return (0.0, 0.0, win_bytes, 0.0);
        }

        // Lower bounds derived from the hot matrices touched in the active unified shader.
        // Slot coordination reuses the same Q/K/V/O families in both slot-coord and FPM modes.
        let qkv_bytes = b * t * (3.0 * h * d * d * f32_bytes);
        let wo_bytes = b * t * (h * d * d * f32_bytes);
        let win_bytes = b * t * (h * d * d * f32_bytes);
        let aux_lower_bound = if fpm_enabled {
            let rank = 32.0;
            // FPM forward touches low-rank read/write projections plus the dense write map.
            b * t * ((4.0 * h * d * rank + h * d * d + d + 2.0 * h * d) * f32_bytes)
        } else if slot_coord_mode {
            0.0
        } else {
            0.0
        };

        (qkv_bytes, wo_bytes, win_bytes, aux_lower_bound)
    }

    fn decode_forward_debug_metrics(
        fw: &[f32],
        h_slots: usize,
    ) -> (f32, f32, f32, f32, f32, f32, f32) {
        if fw.len() > 11 && fw[8] == 901.0 {
            let seq = fw[10].max(1.0);
            let slot_count = fw[11].max(1.0).min(h_slots as f32).round() as usize;
            let mut max_delta = 0.0f32;
            let mut max_exit_delta = 0.0f32;
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
                let diag_base = 400 + slot * 12;
                if fw.len() > diag_base + 8 {
                    let exit_delta = fw[diag_base + 8];
                    if exit_delta.is_finite() {
                        max_exit_delta = max_exit_delta.max(exit_delta);
                    }
                }
            }
            let avg_iters = if slot_count > 0 {
                avg_iters_sum / slot_count as f32
            } else {
                0.0
            };
            let hit_den = seq * slot_count.max(1) as f32;
            let solve_delta = if max_exit_delta > 0.0 {
                max_exit_delta
            } else {
                max_delta
            };
            return (
                seq,
                max_h,
                avg_iters,
                hit_count,
                solve_delta,
                contractivity,
                hit_den,
            );
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

    fn decode_slot_observability_metrics(fw: &[f32], h_slots: usize) -> Vec<(f32, f32, f32)> {
        let mut out = Vec::new();
        for slot in 0..h_slots {
            let base = 320 + slot * 3;
            if fw.len() <= base + 2 {
                break;
            }
            out.push((fw[base], fw[base + 1], fw[base + 2]));
        }
        out
    }

    fn decode_slot_max_a_delta_metrics(fw: &[f32], h_slots: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for slot in 0..h_slots {
            let idx = 384 + slot;
            if fw.len() <= idx {
                break;
            }
            out.push(fw[idx]);
        }
        out
    }

    fn decode_debug_snapshot_header(fw: &[f32]) -> Option<(f32, f32, f32, f32, f32, f32, f32)> {
        if fw.len() <= 18 {
            return None;
        }
        Some((fw[8], fw[9], fw[10], fw[11], fw[12], fw[13], fw[14]))
    }

    fn valid_debug_snapshot(fw: &[f32]) -> bool {
        fw.len() > 11 && fw[8] == 901.0 && fw[10].is_finite() && fw[10] > 0.0
    }

    #[cfg(feature = "wgpu")]
    fn slot_path_audit_stats(&self, gpu: &GpuDeqBackend, seq_len: u32) -> Option<String> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        if d == 0 || h == 0 || seq_len == 0 {
            return None;
        }
        let hd = std::env::var("AIDEEN_DEQ_SLOT_ATTN_HEAD_DIM")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|v| v.clamp(8, 32))
            .unwrap_or(32)
            .min(d);
        let scratch = gpu.read_scratch_buffer();
        let q_buf = gpu.read_slot_coord_q_forward(seq_len);
        let k_work_buf = gpu.read_slot_coord_k_work(seq_len);
        let qgrad = gpu.read_hist_qgrad_signal(seq_len);
        let gscore = gpu.read_slot_coord_gscore(seq_len);
        let (_, wk, _, _, _, _, _, _, _) = gpu.read_weights().ok()?;
        let hist_params = gpu.read_hist_params_full();

        let signal_span = d * h;
        let scratch_stride = signal_span * 2 + h * h;
        let hist_mat_len = d * d;
        let slot_scale_base = hist_mat_len;
        let hist_bias_base = slot_scale_base + h * d;
        let hist_gate_base = hist_bias_base + h * d;
        let slot_anchor_base = hist_gate_base + h;
        let slot_anchor = &hist_params[slot_anchor_base..slot_anchor_base + h * d];

        let pair_l2_owned = |data: &[f32], width: usize| -> f32 {
            let mut acc = 0.0f32;
            let mut pairs = 0u32;
            for token in 0..seq_len as usize {
                for a in 0..h {
                    let base_a = (token * h + a) * width;
                    let va = &data[base_a..base_a + width];
                    for b in (a + 1)..h {
                        let base_b = (token * h + b) * width;
                        let vb = &data[base_b..base_b + width];
                        let mut dist2 = 0.0f32;
                        let mut used = 0u32;
                        for i in 0..va.len() {
                            if !va[i].is_finite() || !vb[i].is_finite() {
                                continue;
                            }
                            let dv = va[i] - vb[i];
                            dist2 += dv * dv;
                            used += 1;
                        }
                        if used == 0 {
                            continue;
                        }
                        acc += (dist2 / used as f32).sqrt();
                        pairs += 1;
                    }
                }
            }
            if pairs == 0 { 0.0 } else { acc / pairs as f32 }
        };
        let mean_rms_owned = |data: &[f32], width: usize| -> f32 {
            let mut acc = 0.0f32;
            let mut count = 0u32;
            for token in 0..seq_len as usize {
                for slot in 0..h {
                    let base = (token * h + slot) * width;
                    let v = &data[base..base + width];
                    let mut sumsq = 0.0f32;
                    let mut used = 0u32;
                    for x in v {
                        if !x.is_finite() {
                            continue;
                        }
                        sumsq += x * x;
                        used += 1;
                    }
                    if used == 0 {
                        continue;
                    }
                    acc += (sumsq / used as f32).sqrt();
                    count += 1;
                }
            }
            if count == 0 { 0.0 } else { acc / count as f32 }
        };

        let mut src_flat = vec![0.0f32; seq_len as usize * h * d];
        for token in 0..seq_len as usize {
            let token_base = token * scratch_stride;
            for slot in 0..h {
                let signal_base = token_base + slot * d;
                let anchor_base = slot * d;
                let out_base = (token * h + slot) * d;
                for i in 0..d {
                    src_flat[out_base + i] = scratch[signal_base + i] + slot_anchor[anchor_base + i];
                }
            }
        }
        let mut q_flat = vec![0.0f32; seq_len as usize * h * hd];
        let mut k_forward_flat = vec![0.0f32; seq_len as usize * h * hd];
        let mut k_work_flat = vec![0.0f32; seq_len as usize * h * hd];
        let mut qgrad_flat = vec![0.0f32; seq_len as usize * h * hd];
        let mat_len = d * d;
        let k_bias_base = h * mat_len;
        for token in 0..seq_len as usize {
            for slot in 0..h {
                let src_base = (token * h + slot) * d;
                let dst_base = (token * h + slot) * hd;
                q_flat[dst_base..dst_base + hd].copy_from_slice(&q_buf[src_base..src_base + hd]);
                k_work_flat[dst_base..dst_base + hd]
                    .copy_from_slice(&k_work_buf[src_base..src_base + hd]);
                qgrad_flat[dst_base..dst_base + hd]
                    .copy_from_slice(&qgrad[src_base..src_base + hd]);
                let src_slice = &src_flat[src_base..src_base + d];
                let src_rms =
                    (src_slice.iter().map(|x| x * x).sum::<f32>() / d.max(1) as f32).sqrt().max(1.0e-6);
                let wk_slot = &wk[slot * mat_len..(slot + 1) * mat_len];
                let k_bias_slot = &wk[k_bias_base + slot * d..k_bias_base + (slot + 1) * d];
                for head in 0..hd {
                    let mut kval = k_bias_slot[head];
                    for j in 0..d {
                        kval += wk_slot[j * d + head] * (src_slice[j] / src_rms);
                    }
                    k_forward_flat[dst_base + head] = kval;
                }
            }
        }

        let src_pair = pair_l2_owned(&src_flat, d);
        let src_rms = mean_rms_owned(&src_flat, d);
        let q_pair = pair_l2_owned(&q_flat, hd);
        let q_rms = mean_rms_owned(&q_flat, hd);
        let k_forward_pair = pair_l2_owned(&k_forward_flat, hd);
        let k_forward_rms = mean_rms_owned(&k_forward_flat, hd);
        let k_work_pair = pair_l2_owned(&k_work_flat, hd);
        let k_work_rms = mean_rms_owned(&k_work_flat, hd);
        let qgrad_pair = pair_l2_owned(&qgrad_flat, hd);
        let qgrad_rms = mean_rms_owned(&qgrad_flat, hd);
        let gscore_pair = pair_l2_owned(&gscore, h);
        let gscore_rms = mean_rms_owned(&gscore, h);
        let mut score_flat = vec![0.0f32; seq_len as usize * h * h];
        let inv_sqrt_hd = 1.0f32 / (hd.max(1) as f32).sqrt();
        let mut score_margin_acc = 0.0f32;
        let mut self_prob_acc = 0.0f32;
        let mut score_entropy_acc = 0.0f32;
        let mut score_count = 0u32;
        let mut incoming_acc = vec![0.0f32; h];
        for token in 0..seq_len as usize {
            for qs in 0..h {
                let q_base = (token * h + qs) * hd;
                let qv = &q_flat[q_base..q_base + hd];
                let score_base = (token * h + qs) * h;
                let mut max_s = f32::NEG_INFINITY;
                for ks in 0..h {
                    let k_base = (token * h + ks) * hd;
                    let kv = &k_forward_flat[k_base..k_base + hd];
                    let score = if h > 1 && ks == qs {
                        f32::NEG_INFINITY
                    } else {
                        let mut dot = 0.0f32;
                        for i in 0..hd {
                            dot += qv[i] * kv[i];
                        }
                        dot * inv_sqrt_hd
                    };
                    score_flat[score_base + ks] = score;
                    if score.is_finite() {
                        max_s = max_s.max(score);
                    }
                }
                let mut sum_exp = 0.0f32;
                let mut top1 = f32::NEG_INFINITY;
                let mut top2 = f32::NEG_INFINITY;
                for ks in 0..h {
                    let s = score_flat[score_base + ks];
                    if !s.is_finite() {
                        continue;
                    }
                    if s > top1 {
                        top2 = top1;
                        top1 = s;
                    } else if s > top2 {
                        top2 = s;
                    }
                    sum_exp += (s - max_s).exp();
                }
                let mut entropy = 0.0f32;
                let mut self_prob = 0.0f32;
                sum_exp = sum_exp.max(1.0e-12);
                for ks in 0..h {
                    let s = score_flat[score_base + ks];
                    let p = if s.is_finite() {
                        (s - max_s).exp() / sum_exp
                    } else {
                        0.0
                    };
                    incoming_acc[ks] += p;
                    if ks == qs {
                        self_prob = p;
                    }
                    if p > 0.0 {
                        entropy -= p * p.ln();
                    }
                }
                score_margin_acc += top1 - top2.max(f32::NEG_INFINITY);
                self_prob_acc += self_prob;
                score_entropy_acc += entropy;
                score_count += 1;
            }
        }
        let score_pair = pair_l2_owned(&score_flat, h);
        let score_rms = mean_rms_owned(&score_flat, h);
        let score_margin = if score_count == 0 { 0.0 } else { score_margin_acc / score_count as f32 };
        let score_self = if score_count == 0 { 0.0 } else { self_prob_acc / score_count as f32 };
        let score_entropy =
            if score_count == 0 { 0.0 } else { score_entropy_acc / score_count as f32 };
        let incoming_den = (seq_len as usize * h).max(1) as f32;
        let incoming_mean = incoming_acc
            .iter()
            .map(|v| format!("{:.3}", *v / incoming_den))
            .collect::<Vec<_>>()
            .join(",");

        Some(format!(
            "src(pair={src_pair:.3e},rms={src_rms:.3e}) q(pair={q_pair:.3e},rms={q_rms:.3e}) kfwd(pair={k_forward_pair:.3e},rms={k_forward_rms:.3e}) kwork(pair={k_work_pair:.3e},rms={k_work_rms:.3e}) score(pair={score_pair:.3e},rms={score_rms:.3e},margin={score_margin:.3e},self={score_self:.3},ent={score_entropy:.3},incoming=[{incoming_mean}]) qgrad(pair={qgrad_pair:.3e},rms={qgrad_rms:.3e}) gscore(pair={gscore_pair:.3e},rms={gscore_rms:.3e})"
        ))
    }

    fn extract_slot_incoming(audit: &str) -> Option<&str> {
        let start_tag = "incoming=[";
        let start = audit.find(start_tag)? + start_tag.len();
        let end = audit[start..].find(']')?;
        Some(&audit[start..start + end])
    }

    #[cfg(feature = "wgpu")]
    fn slot_weight_specialization_stats(
        &self,
        gpu: &GpuDeqBackend,
    ) -> Option<(f32, f32, f32, f32, f32, f32)> {
        let (wq, wk, wv, wo, win, _, _, _, _) = gpu.read_weights().ok()?;
        let d = self.config.d_r;
        let h = self.config.h_slots;
        if h == 0 || d == 0 {
            return None;
        }
        let mat_len = d * d;
        let head_dim = std::env::var("AIDEEN_DEQ_SLOT_ATTN_HEAD_DIM")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map(|v| v.clamp(8, 32))
            .unwrap_or(32)
            .min(d);
        let mean_pair_l2_slice = |flat: &[f32], pick: &dyn Fn(&[f32]) -> Vec<f32>| -> f32 {
            let mut acc = 0.0f32;
            let mut pairs = 0u32;
            for a in 0..h {
                let sa = pick(&flat[a * mat_len..(a + 1) * mat_len]);
                for b in (a + 1)..h {
                    let sb = pick(&flat[b * mat_len..(b + 1) * mat_len]);
                    let mut dist2 = 0.0f32;
                    for i in 0..sa.len() {
                        let dv = sa[i] - sb[i];
                        dist2 += dv * dv;
                    }
                    acc += (dist2 / sa.len() as f32).sqrt();
                    pairs += 1;
                }
            }
            if pairs == 0 { 0.0 } else { acc / pairs as f32 }
        };
        let pick_qkv = |m: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(d * head_dim);
            for row in 0..d {
                let base = row * d;
                out.extend_from_slice(&m[base..base + head_dim]);
            }
            out
        };
        let pick_wo = |m: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(head_dim * d);
            for row in 0..head_dim {
                let base = row * d;
                out.extend_from_slice(&m[base..base + d]);
            }
            out
        };
        let mean_pair_l2 = |flat: &[f32]| -> f32 { mean_pair_l2_slice(flat, &pick_qkv) };
        let mean_pair_l2_bias = |flat: &[f32], base: usize| -> f32 {
            let mut acc = 0.0f32;
            let mut pairs = 0u32;
            for a in 0..h {
                let sa = &flat[base + a * d..base + (a + 1) * d];
                for b in (a + 1)..h {
                    let sb = &flat[base + b * d..base + (b + 1) * d];
                    let mut dist2 = 0.0f32;
                    for i in 0..d {
                        let dv = sa[i] - sb[i];
                        dist2 += dv * dv;
                    }
                    acc += (dist2 / d as f32).sqrt();
                    pairs += 1;
                }
            }
            if pairs == 0 { 0.0 } else { acc / pairs as f32 }
        };
        let q_bias_base = h * mat_len;
        let k_bias_base = h * mat_len;
        Some((
            mean_pair_l2(&wq[..h * mat_len]),
            mean_pair_l2(&wk[..h * mat_len]),
            mean_pair_l2(&wv[..h * mat_len]),
            mean_pair_l2_slice(&wo[..h * mat_len], &pick_wo),
            mean_pair_l2(&win[..h * mat_len]),
            0.5 * (mean_pair_l2_bias(&wq, q_bias_base) + mean_pair_l2_bias(&wk, k_bias_base)),
        ))
    }

    fn decode_fpm_diag_metrics(
        fw: &[f32],
        h_slots: usize,
    ) -> Vec<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> {
        let mut out = Vec::new();
        for slot in 0..h_slots {
            let base = 400 + slot * 12;
            if fw.len() <= base + 11 {
                break;
            }
            out.push((
                fw[base],
                fw[base + 1],
                fw[base + 2],
                fw[base + 3],
                fw[base + 4],
                fw[base + 5],
                fw[base + 6],
                fw[base + 7],
                fw[base + 8],
                fw[base + 9],
                fw[base + 10],
                fw[base + 11],
            ));
        }
        out
    }

    fn decode_fpm_read_metrics(
        fw: &[f32],
        h_slots: usize,
    ) -> Vec<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> {
        let mut out = Vec::new();
        for slot in 0..h_slots {
            let base = 520 + slot * 14;
            if fw.len() <= base + 13 {
                break;
            }
            out.push((
                fw[base],
                fw[base + 1],
                fw[base + 2],
                fw[base + 3],
                fw[base + 4],
                fw[base + 5],
                fw[base + 6],
                fw[base + 7],
                fw[base + 8],
                fw[base + 9],
                fw[base + 10],
                fw[base + 11],
                fw[base + 12],
                fw[base + 13],
            ));
        }
        out
    }

    fn decode_fpm_write_branch_metrics(
        fw: &[f32],
        h_slots: usize,
    ) -> Vec<(f32, f32, f32, f32, f32, f32)> {
        let mut out = Vec::new();
        for slot in 0..h_slots {
            let base = 640 + slot * 6;
            if fw.len() <= base + 5 {
                break;
            }
            out.push((
                fw[base],
                fw[base + 1],
                fw[base + 2],
                fw[base + 3],
                fw[base + 4],
                fw[base + 5],
            ));
        }
        out
    }

    fn decode_fpm_health_metrics(fw: &[f32], h_slots: usize) -> Option<FpmHealthMetrics> {
        if !Self::valid_debug_snapshot(fw) {
            return None;
        }
        let slot_diag = Self::decode_fpm_diag_metrics(fw, h_slots);
        if slot_diag.is_empty() {
            return None;
        }
        let slot_read = Self::decode_fpm_read_metrics(fw, h_slots);
        let finite_avg = |vals: Vec<f32>| {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for v in vals {
                if v.is_finite() {
                    sum += v;
                    count += 1;
                }
            }
            if count == 0 {
                0.0
            } else {
                sum / count as f32
            }
        };
        Some(FpmHealthMetrics {
            max_err_h: slot_diag.iter().map(|m| m.0).fold(0.0, f32::max),
            max_mem_update: slot_diag.iter().map(|m| m.1).fold(0.0, f32::max),
            max_z: slot_diag.iter().map(|m| m.2).fold(0.0, f32::max),
            avg_z: slot_diag.iter().map(|m| m.2).sum::<f32>() / slot_diag.len() as f32,
            rescue_recovered: slot_diag.iter().map(|m| m.4).sum::<f32>(),
            dead_slots: slot_diag.iter().map(|m| m.5).sum::<f32>(),
            max_update_ratio: slot_diag.iter().map(|m| m.6).fold(0.0, f32::max),
            write_saturation: slot_diag.iter().map(|m| m.7).sum::<f32>(),
            max_memctx_rms: slot_read.iter().map(|m| m.0).fold(0.0, f32::max),
            max_memctx_to_signal: slot_read.iter().map(|m| m.1).fold(0.0, f32::max),
            exit_err_h: finite_avg(slot_diag.iter().map(|m| m.8).collect()),
            exit_iter: finite_avg(slot_diag.iter().map(|m| m.9).collect()),
            rescue_entered: slot_diag.iter().map(|m| m.10).sum::<f32>(),
            pre_rescue_converged: slot_diag.iter().map(|m| m.11).sum::<f32>(),
        })
    }

    fn fpm_health_threshold(epsilon: f32) -> f32 {
        const FPM_ALPHA_H: f32 = 0.2;
        const FPM_HOMEO_ALPHA_ERR_SCALE: f32 = 0.15;
        epsilon.max(FPM_ALPHA_H * FPM_HOMEO_ALPHA_ERR_SCALE)
    }

    fn classify_fpm_solve(
        metrics: FpmHealthMetrics,
        max_h: f32,
        contractivity: f32,
        epsilon: f32,
        model_a_mode: bool,
    ) -> SolveStatus {
        let null_h_floor = 1e-4;
        let null_z_floor = 1e-4;

        if !metrics.max_err_h.is_finite()
            || !metrics.max_mem_update.is_finite()
            || !metrics.avg_z.is_finite()
            || !metrics.max_z.is_finite()
            || !metrics.max_update_ratio.is_finite()
            || !max_h.is_finite()
            || !contractivity.is_finite()
        {
            return SolveStatus::NumericInvalid;
        }

        if model_a_mode {
            if metrics.exit_err_h <= 0.15 {
                if max_h <= null_h_floor && metrics.avg_z <= null_z_floor {
                    return SolveStatus::TrivialConverged;
                }
                return SolveStatus::HealthyConverged;
            }
            if contractivity > 1.20 {
                return SolveStatus::NumericInvalid;
            }
            return SolveStatus::Unconverged;
        }

        if contractivity > 1.20 {
            return SolveStatus::NumericInvalid;
        }

        if metrics.exit_err_h <= Self::fpm_health_threshold(epsilon) {
            if max_h <= null_h_floor && metrics.avg_z <= null_z_floor {
                SolveStatus::TrivialConverged
            } else {
                SolveStatus::HealthyConverged
            }
        } else {
            SolveStatus::Unconverged
        }
    }

    #[cfg(feature = "wgpu")]
    fn configure_fpm_runtime_controls(&mut self, gpu: &mut GpuDeqBackend) {
        if std::env::var("AR_ASSOC_ONLY").ok().as_deref() == Some("1") {
            gpu.cached_fpm_alpha_m = 0.0;
            gpu.cached_fpm_stage = gpu.cached_fpm_stage.max(4);
            return;
        }
        // Allow AIDEEN_FPM_ALPHA_M to set any value; default curriculum still ramps to 0.05
        // but won't be capped below the env var value.
        let alpha_env_max = std::env::var("AIDEEN_FPM_ALPHA_M")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.1);
        gpu.cached_fpm_alpha_m = self.fpm_alpha_m_current.clamp(0.001, alpha_env_max);
        gpu.cached_fpm_tau = self.fpm_tau_current.max(0.5);
    }

    fn update_fpm_runtime_schedule(&mut self) {
        let runtime_schedule_enabled = std::env::var("AIDEEN_FPM_RUNTIME_SCHEDULE")
            .ok()
            .map(|v| {
                let vl = v.trim().to_ascii_lowercase();
                vl == "1" || vl == "true" || vl == "yes"
            })
            .unwrap_or(false);
        if !runtime_schedule_enabled {
            return;
        }
        let fpm_enabled = Self::effective_fpm_enabled();
        if !fpm_enabled || self.cached_debug_buf.is_empty() {
            return;
        }

        let Some(metrics) =
            Self::decode_fpm_health_metrics(&self.cached_debug_buf, self.config.h_slots)
        else {
            return;
        };

        let err_h_avg = metrics.max_err_h;
        self.fpm_err_h_window.push_back(err_h_avg);
        while self.fpm_err_h_window.len() > 100 {
            self.fpm_err_h_window.pop_front();
        }

        let dead_slots = metrics.dead_slots.round() as usize;
        if self.fpm_tau_current < 0.8 && dead_slots * 2 >= self.config.h_slots.max(1) {
            self.fpm_tau_current = 0.8;
        }

        if self.fpm_err_h_window.len() == 100 {
            let mean = self.fpm_err_h_window.iter().copied().sum::<f32>() / 100.0;
            let stable = mean < 0.08 && (mean - self.fpm_last_err_h_avg).abs() < 0.01;
            if stable {
                let alpha_env_max = std::env::var("AIDEEN_FPM_ALPHA_M")
                    .ok()
                    .and_then(|v| v.trim().parse::<f32>().ok())
                    .unwrap_or(0.05);
                self.fpm_alpha_m_current = (self.fpm_alpha_m_current + 0.002).min(alpha_env_max);
            }
            self.fpm_last_err_h_avg = mean;
        }
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
    fn from_tokenizer_seeded_internal(
        tokenizer: Tokenizer,
        lr: f32,
        seed: Option<u64>,
        _init_gpu: bool,
    ) -> Self {
        let config = tokenizer.config.clone();

        #[cfg(feature = "wgpu")]
        let gpu_deq = if _init_gpu {
            GpuDeqBackend::new_blocking(config.clone())
        } else {
            None
        };

        let mut trainer = Self {
            reasoning: match seed {
                Some(seed) => FixedPointMemoryReasoning::new_with_seed(config.clone(), seed),
                None => FixedPointMemoryReasoning::new(config.clone()),
            },
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
            completed_file_epochs: 0,
            plateau_best_loss: None,
            plateau_bad_epochs: 0,
            plateau_cooldown_left: 0,
            plateau_lr_cap: lr,
            solve_stage_floor: 8,
            solve_stage_cap: 12,
            m_prev: None,
            grad_accum_counter: 0,
            debug_last_time: None,
            debug_tokens_accum: 0,
            cached_debug_buf: Vec::new(),
            cached_debug_gen: 0,
            invalid_eval_debug_gen: 0,
            last_gpu_loss: 0.0,
            fpm_alpha_m_current: std::env::var("AIDEEN_FPM_ALPHA_M")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(0.01),
            fpm_tau_current: 0.5,
            fpm_err_h_window: VecDeque::with_capacity(100),
            fpm_last_err_h_avg: f32::INFINITY,
            cfg_fwd_batch_size: std::env::var("AIDEEN_BATCH_SIZE")
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(1)
                .max(1),
            cfg_debug_sample_every: std::env::var("AIDEEN_DEBUG_SAMPLE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            cfg_solve_control_every: std::env::var("AIDEEN_SOLVE_CONTROL_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(10)
                .max(1),
            cfg_loss_readback_every: std::env::var("AIDEEN_LOSS_READBACK_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            cfg_tps_sync_every: std::env::var("AIDEEN_TPS_SYNC_EVERY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0),
            cfg_grad_accum: std::env::var("AIDEEN_GRAD_ACCUM")
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(1)
                .max(1),
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
            cfg_wv_debug: Self::env_flag("AIDEEN_DEQ_WV_DEBUG"),
            cfg_ssm_debug: Self::env_flag("AIDEEN_SSM_DEBUG"),
            cfg_max_chunks: std::env::var("AIDEEN_MAX_CHUNKS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(usize::MAX),
            cfg_adj_iters_override: std::env::var("AIDEEN_ADJ_ITERS_OVERRIDE")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok())
                .or(Some(2)),
            cfg_system_cost_audit: Self::env_flag("AIDEEN_SYSTEM_COST_AUDIT"),
            cfg_slot_path_audit: Self::env_flag("AIDEEN_SLOT_PATH_AUDIT"),
            cfg_system_cost_wait: Self::env_flag("AIDEEN_SYSTEM_COST_WAIT"),
            cfg_force_renorm: Self::env_flag("AIDEEN_DEQ_FORCE_RENORM"),
            cfg_lm_force_cpu_dldh: Self::env_flag("AIDEEN_LM_FORCE_CPU_DLDH"),
            cfg_lm_dldh_parity: Self::env_flag("AIDEEN_LM_DLDH_PARITY"),
            cfg_log_emb_stats: Self::env_flag("AIDEEN_LOG_EMB_STATS"),
            cfg_lr_plateau_enable: !Self::env_flag("AIDEEN_LR_PLATEAU_DISABLE"),
            cfg_lr_plateau_patience: std::env::var("AIDEEN_LR_PLATEAU_PATIENCE")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(2),
            cfg_lr_plateau_cooldown: std::env::var("AIDEEN_LR_PLATEAU_COOLDOWN")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(1),
            cfg_lr_plateau_factor: std::env::var("AIDEEN_LR_PLATEAU_FACTOR")
                .ok()
                .and_then(|v| v.trim().parse::<f32>().ok())
                .unwrap_or(0.5),
            cfg_lr_plateau_min_rel_improvement: std::env::var(
                "AIDEEN_LR_PLATEAU_MIN_REL_IMPROVEMENT",
            )
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .unwrap_or(0.005),
            cfg_lr_plateau_min_lr_override: Self::env_f32("AIDEEN_LR_PLATEAU_MIN_LR"),
            cfg_assoc_lr_mult: Self::env_f32("AIDEEN_ASSOC_LR_MULT").unwrap_or(1.0),
            cfg_assoc_event_lr_mult: Self::env_f32("AIDEEN_ASSOC_EVENT_LR_MULT").unwrap_or(1.0),
            cfg_assoc_alpha_lr_mult: Self::env_f32("AIDEEN_ASSOC_ALPHA_LR_MULT").unwrap_or(1.0),
        };
        trainer.apply_experimental_profile_from_env();

        #[cfg(feature = "wgpu")]
        if trainer.cfg_fwd_batch_size > 4 {
            eprintln!(
                "\x1b[33m[AIDEEN] WARNING: AIDEEN_BATCH_SIZE={} may cause thermal overload on \
                 integrated GPUs (Apple Silicon, iGPU). Use AIDEEN_BATCH_SIZE=1 \
                 AIDEEN_GRAD_ACCUM={} instead.\x1b[0m",
                trainer.cfg_fwd_batch_size,
                trainer.cfg_fwd_batch_size,
            );
        }

        trainer
    }

    pub fn from_tokenizer(tokenizer: Tokenizer, lr: f32) -> Self {
        Self::from_tokenizer_seeded_internal(tokenizer, lr, None, true)
    }

    /// Igual que `from_tokenizer`, pero forzando inicialización determinística
    /// de los pesos de reasoning (DEQ core) para reproducibilidad por seed.
    pub fn from_tokenizer_seeded(tokenizer: Tokenizer, lr: f32, seed: u64) -> Self {
        Self::from_tokenizer_seeded_internal(tokenizer, lr, Some(seed), true)
    }

    /// Variante explícitamente CPU para benches/diagnósticos que no deben disparar
    /// inicialización de GPU ni mezclar paths de ejecución al medir relevancia.
    pub fn from_tokenizer_seeded_cpu(tokenizer: Tokenizer, lr: f32, seed: u64) -> Self {
        Self::from_tokenizer_seeded_internal(tokenizer, lr, Some(seed), false)
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
            let lm = GpuLmHeadTrainer::new(&gpu.device, self.lm_head.b.len(), self.config.clone());
            // Upload CPU LM head weights immediately on GPU trainer creation.
            // Without this, the GPU LM runs with zero weights until the lazy sync
            // at line ~1372, which may be hundreds of steps later — causing the DEQ
            // adjoint to receive zero gradient during those steps, then a large
            // gradient shock when the real weights are finally uploaded.
            let w_head = self.lm_head.export_weights();
            lm.upload_weights_only(
                &gpu.queue,
                w_head.get("head.w").unwrap(),
                w_head.get("head.b").unwrap(),
                w_head.get("head.g").unwrap(),
            );
            self.gpu_lm_weights_uploaded = true;
            self.gpu_lm = Some(lm);
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
        // FixedPointMemoryReasoning es stateless entre llamadas: el DEQ recomputa h* desde cero
        // en cada forward pass, por lo que no hay estado oculto persistente que limpiar.
        // reset_state sirve para forzar que la próxima secuencia no comparta contexto.
        self.m_prev = None;
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = self.gpu_deq.as_ref() {
            gpu.reset_state();
        }
    }

    /// Reinicia solo la memoria local/intra-segmento manteniendo el carry largo de FPM.
    pub fn reset_local_segment_state(&mut self) {
        self.m_prev = None;
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = self.gpu_deq.as_ref() {
            gpu.reset_local_segment_state();
        }
    }

    fn internal_segment_len(&self) -> Option<u32> {
        std::env::var("AIDEEN_INTERNAL_SEGMENT_LEN")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&v| v > 0)
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
                let mut gpu = self.take_gpu().expect("gpu_deq checked as Some");

                let out = (|| {
                    self.configure_fpm_runtime_controls(&mut gpu);
                    self.ensure_gpu_trainers(&gpu);

                    let Some(gpu_emb) = self.gpu_emb.as_ref() else {
                        return 0.0;
                    };

                    let emb_needs_upload = !self.gpu_emb_weights_uploaded;
                    let emb_upload = if emb_needs_upload {
                        Some(self.tokenizer.embeddings_row_major())
                    } else {
                        None
                    };
                    let (_s_sequence, query_vec) = match gpu_emb.prepare_sequence_and_query(
                        &gpu.device,
                        &gpu.queue,
                        context,
                        self.config.ctx_len,
                        emb_upload.as_deref().unwrap_or(&[]),
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
        // slot_coord is a one-shot initializer term in the current design. It shares the clean
        // DEQ Picard regime: frozen slot context, no history recurrence, and no extra Jacobian
        // term beyond the base solve. Keep its minimum iteration floor aligned with clean DEQ.
        if Self::slot_coord_mode_active() {
            let min_iters = Self::default_slot_coord_min_iters();
            if self.solve_stage_cap < min_iters {
                self.solve_stage_cap = min_iters;
            }
            if self.adaptive_max_iters < min_iters {
                self.adaptive_max_iters = min_iters;
            }
        }

        #[cfg(feature = "wgpu")]
        if self.gpu_deq.is_some() {
            let mut gpu = self.take_gpu().expect("gpu_deq checked as Some");

            let out = (|| {
                self.configure_fpm_runtime_controls(&mut gpu);
                self.ensure_gpu_trainers(&gpu);

                // Arreglo defensivo para evitar underflow si seq_len < ctx_len.
                // Para batch > 1 el training loop pasa B*ctx_len tokens — no truncar.
                let fwd_batch_size_ts: usize = self.cfg_fwd_batch_size.max(1) as usize;
                let seq_len = tokens.len().min(targets.len());
                let seq_cap = self.gpu_sequence_capacity();
                let actual_ctx_len = if fwd_batch_size_ts > 1 {
                    seq_len
                } else {
                    seq_len.min(seq_cap)
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
                        w_kv_write_rm,
                        b_delta,
                        w_gate_hist_rm,
                        w_write_gate_rm,
                        b_write_mem,
                        w_retain_up_rm,
                        w_retain_down_rm,
                        b_retain_rm,
                        w_q_mem_rm,
                        w_k_mem_rm,
                        b_read_mem,
                        w_k_assoc_rm,
                        w_v_assoc_rm,
                        w_q_assoc_rm,
                        alpha_assoc,
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
                        w_kv_write_rm.as_slice(),
                        b_delta.as_slice(),
                        w_gate_hist_rm.as_slice(),
                        w_write_gate_rm.as_slice(),
                        b_write_mem.as_slice(),
                        w_retain_up_rm.as_slice(),
                        w_retain_down_rm.as_slice(),
                        b_retain_rm.as_slice(),
                        w_q_mem_rm.as_slice(),
                        w_k_mem_rm.as_slice(),
                        b_read_mem.as_slice(),
                        w_k_assoc_rm.as_slice(),
                        w_v_assoc_rm.as_slice(),
                        w_q_assoc_rm.as_slice(),
                        alpha_assoc.as_slice(),
                    );
                    self.gpu_weights_uploaded = true;
                    self.gpu_cg_weights_uploaded = true;
                }
                let slot_coord_mode = Self::slot_coord_mode_active();
                if !self.force_renorm_done && (self.cfg_force_renorm || slot_coord_mode) {
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
    pub fn eval_cached_hpooled_token_loss(&mut self, token_index: usize, target: u32) -> f32 {
        let step_t = self.optimizer.step_count() as u32;
        let ternary = self.training_config.ternary;
        let d = self.config.d_r;
        let Some(gpu) = self.gpu_deq.as_ref() else {
            return f32::NAN;
        };
        let Some(gpu_lm) = self.gpu_lm.as_mut() else {
            return f32::NAN;
        };
        let h_offset = (token_index * d * std::mem::size_of::<f32>()) as u64;
        match gpu_lm.train_step_no_readback(
            &gpu.device,
            &gpu.queue,
            &gpu.bridge.hpooled_buf,
            h_offset,
            &gpu.bridge.assoc_pooled_buf,
            h_offset,
            None,
            &[target],
            0.0,
            step_t,
            ternary,
            true,
        ) {
            Ok(loss) => {
                self.last_gpu_loss = loss;
                loss
            }
            Err(_) => f32::NAN,
        }
    }

    #[cfg(not(feature = "wgpu"))]
    pub fn eval_cached_hpooled_token_loss(&mut self, _token_index: usize, _target: u32) -> f32 {
        f32::NAN
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
        if self.frozen_lm || self.cfg_lm_force_cpu_dldh || self.cfg_lm_dldh_parity {
            self.sync_lm_head_from_gpu_if_needed();
        }

        let num_tokens = targets.len();
        let fwd_batch_size = self.cfg_fwd_batch_size;
        let per_seq_len = if fwd_batch_size == 0 {
            0
        } else {
            (num_tokens as u32) / fwd_batch_size
        };
        let internal_segment_len = self.internal_segment_len();
        let audit_forward_bytes = self.estimate_forward_bandwidth_bytes(
            fwd_batch_size,
            per_seq_len,
            self.adaptive_max_iters,
            self.config.d_r,
            self.config.h_slots,
        );
        let emergency_iter_cap = self.emergency_solve_cap();

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
                let emb_upload = if emb_needs_upload {
                    Some(self.tokenizer.embeddings_row_major())
                } else {
                    None
                };
                let _ = gpu_emb.gather_only_to_sbuf(
                    &gpu.queue,
                    &gpu.device,
                    context,
                    emb_upload.as_deref().unwrap_or(&[]),
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
            let debug_enable = debug_every != 0 && (self.optimizer.step_count() % debug_every == 0);
            let debug_fpm = Self::env_flag("AIDEEN_DEBUG_FPM");
            let mut strict_debug_snapshot: Option<Vec<f32>> = None;
            if debug_enable && debug_fpm {
                println!(
                    "    \x1b[90m[STEP-SHAPE] step={} ctx_tokens={} target_tokens={} batch={} per_seq={} query_dim={} single_query={}\x1b[0m",
                    self.optimizer.step_count(),
                    context.len(),
                    num_tokens,
                    fwd_batch_size,
                    per_seq_len,
                    query.len(),
                    if query.len() == self.config.d_r && num_tokens == 1 { 1 } else { 0 }
                );
            }
            let forward_t0 = std::time::Instant::now();
            if let Some(segment_len) = internal_segment_len {
                let _ = gpu.run_forward_segmented_from_seq_buf(
                    fwd_batch_size,
                    per_seq_len,
                    segment_len,
                    self.adaptive_max_iters,
                    damping_eff,
                    epsilon,
                    debug_enable,
                    &gpu.bridge.s_buf,
                );
            } else {
                let _ = gpu.run_forward(
                    fwd_batch_size,
                    per_seq_len,
                    self.adaptive_max_iters,
                    damping_eff,
                    epsilon,
                    debug_enable,
                );
            }
            if audit_cost && self.cfg_system_cost_wait {
                // Audit-only: force GPU completion so forward_ms reflects execution, not enqueue.
                gpu.device.poll(wgpu::Maintain::Wait);
            }
            if audit_cost {
                audit_forward_ms += forward_t0.elapsed().as_secs_f64() * 1e3;
            }
            if debug_enable && debug_fpm {
                let fw_mid = gpu.read_debug_buffer();
                strict_debug_snapshot = Some(fw_mid.clone());
                let h_mid = gpu.read_hpooled();
                let h_mid_abs_mean = if h_mid.is_empty() {
                    0.0
                } else {
                    h_mid.iter().map(|v| v.abs()).sum::<f32>() / h_mid.len() as f32
                };
                let h_mid_abs_max = h_mid.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                if Self::valid_debug_snapshot(&fw_mid) {
                    println!(
                        "    \x1b[90m[DEBUG-BUF-MID] valid=1 sig={:.0} eps={:.3e} tokens={:.0} slots={:.0} hpool(mean={:.3e},max={:.3e})\x1b[0m",
                        fw_mid.get(8).copied().unwrap_or(0.0),
                        fw_mid.get(9).copied().unwrap_or(0.0),
                        fw_mid.get(10).copied().unwrap_or(0.0),
                        fw_mid.get(11).copied().unwrap_or(0.0),
                        h_mid_abs_mean,
                        h_mid_abs_max
                    );
                } else if let Some((sig, eps, tokens, slots, aux0, aux1, aux2)) =
                    Self::decode_debug_snapshot_header(&fw_mid)
                {
                    println!(
                        "    \x1b[90m[DEBUG-BUF-MID] valid=0 sig={:.3e} eps={:.3e} tokens={:.3e} slots={:.3e} aux=[{:.3e},{:.3e},{:.3e}] len={} hpool(mean={:.3e},max={:.3e})\x1b[0m",
                        sig,
                        eps,
                        tokens,
                        slots,
                        aux0,
                        aux1,
                        aux2,
                        fw_mid.len(),
                        h_mid_abs_mean,
                        h_mid_abs_max
                    );
                } else {
                    println!(
                        "    \x1b[90m[DEBUG-BUF-MID] valid=0 len={} header=short hpool(mean={:.3e},max={:.3e})\x1b[0m",
                        fw_mid.len(),
                        h_mid_abs_mean,
                        h_mid_abs_max
                    );
                }
            }
            // Use cached debug buffer — refresh deferred to end of step (after GPU is idle).
            // The DEQ-INVALID streak check needs 3 consecutive failures, so 1-step lag is safe.
            let fw = self.cached_debug_buf.clone();
            let (seq, max_h_dbg, _avg_iters_dbg, hit_count, max_delta, contractivity, hit_den) =
                Self::decode_forward_debug_metrics(&fw, self.config.h_slots);
            let hit_ratio = hit_count.max(0.0) / hit_den.max(1.0);
            let non_fpm_conv_ok = max_delta <= 0.15;
            let non_fpm_expand = contractivity > 1.02;
            let non_fpm_boost_residual = max_delta > 0.30;
            let fpm_enabled_cached = Self::effective_fpm_enabled();
            let cached_fpm_metrics = if fpm_enabled_cached {
                Self::decode_fpm_health_metrics(&fw, self.config.h_slots)
            } else {
                None
            };
            let model_a_fpm = Self::fpm_stage_from_env() < 6;
            let cached_fpm_status = cached_fpm_metrics
                .map(|m| Self::classify_fpm_solve(m, max_h_dbg, contractivity, epsilon, model_a_fpm));
            // Evaluate invalidity at most once per freshly captured debug snapshot.
            // Reusing the same cached sample for several training steps would turn one
            // bad window into an artificial streak of "consecutive" failures.
            let fresh_invalid_sample = self.cached_debug_gen != 0
                && self.cached_debug_gen != self.invalid_eval_debug_gen;
            // DEQ-INVALID: only when the system FAILED to converge (maxΔ >> epsilon) while
            // also being non-contractive. Non-monotone convergence (contr transiently > 1
            // but maxΔ ≈ epsilon) is a normal property of non-linear Picard iterations and
            // does NOT indicate an invalid fixed point — the system DID find h*.
            let invalid_fixed_point = if fpm_enabled_cached {
                matches!(cached_fpm_status, Some(SolveStatus::NumericInvalid))
                    || (matches!(cached_fpm_status, Some(SolveStatus::Unconverged))
                        && contractivity > 1.0)
            } else {
                max_delta > 0.30 && non_fpm_expand
            };
            if fresh_invalid_sample {
                self.invalid_eval_debug_gen = self.cached_debug_gen;
                if invalid_fixed_point {
                    self.invalid_hi_streak += 1;
                } else {
                    self.invalid_hi_streak = 0;
                }
            }
            // After the stale-snapshot fix, a fresh invalid sample already means the DEQ
            // forward for that window is unreliable. Waiting for several additional windows
            // only delays renorm/emergency while bad gradients keep landing.
            if fresh_invalid_sample && self.invalid_hi_streak >= 1 {
                if let Some(metrics) = cached_fpm_metrics {
                    eprintln!(
                        "    [DEQ-INVALID] step={} contr={:.3} status={} err_h={:.3e} z_avg={:.3e} max_h={:.3e} seq={:.0}",
                        self.optimizer.step_count(),
                        contractivity,
                        cached_fpm_status.unwrap_or(SolveStatus::NumericInvalid).label(),
                        metrics.max_err_h,
                        metrics.avg_z,
                        max_h_dbg,
                        seq
                    );
                } else {
                    eprintln!(
                        "    [DEQ-INVALID] step={} contr={:.3} hit_ratio={:.3} maxΔ={:.3e} seq={:.0}",
                        self.optimizer.step_count(),
                        contractivity,
                        hit_ratio,
                        max_delta,
                        seq
                    );
                }
                self.invalid_hi_streak = 0;
                self.emergency_left = 3;
                self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(emergency_iter_cap);
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
                    &gpu.bridge.assoc_pooled_buf,
                    0,
                    Some(gpu_emb.weights_buffer()),
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
            if lm_lr > 0.0 {
                self.lm_head_cpu_stale = true;
            }

            // 4. Embedding Update from GPU dl_dh buffer (Moved to step 6 to avoid duplication)

            // 5. DEQ Reasoning Core Update (Picard Adjoint + Fused GPU Weight Update)
            if self.eval_mode {
                return self.last_gpu_loss;
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
                    Some(gpu_lm.assoc_grad_buf()),
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
                let deq_grad_scale = self.config.deq_grad_scale;
                let _ = gpu.apply_fused_deq_update(
                    base_lr,
                    deq_grad_scale,
                    self.training_config.ternary,
                    self.config.weight_decay,
                    per_seq_len,
                    self.reasoning.damping,
                    mode,
                    grad_accum,
                    batch_size,
                    apply_accum,
                    self.cfg_assoc_lr_mult,
                    self.cfg_assoc_event_lr_mult,
                    self.cfg_assoc_alpha_lr_mult,
                );
                if audit_cost {
                    audit_update_ms += update_t0.elapsed().as_secs_f64() * 1e3;
                }
                self.grad_accum_counter += 1;
                if self.grad_accum_counter >= grad_accum {
                    self.grad_accum_counter = 0;
                }
                self.gpu_weights_uploaded = true;
                self.gpu_cg_weights_uploaded = true;

                if self.cfg_debug_sample_every > 0 && self.optimizer.step_count() % self.cfg_debug_sample_every == 0 {
                    let adbg = gpu.read_assoc_bwd_debug();
                    if !adbg.is_empty() {
                        let h = self.config.h_slots;
                        for s in 0..h {
                            let b = s * 16;
                            if b + 15 < adbg.len() {
                                eprintln!(
                                    "[ASSOC-BWD-STEP] s{}: val_rms={:.3e} score_grad_max={:.3e} wq_step_max={:.3e} wk_step_max={:.3e} alpha_grad_max={:.3e} key_grad_sum={:.3e} gprev_sum={:.3e} write_mass={:.3e}",
                                    s, adbg[b], adbg[b+5], adbg[b+7], adbg[b+9], adbg[b+11], adbg[b+12], adbg[b+13], adbg[b+14]
                                );
                            }
                        }
                    }
                }

                // =============================================================================

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
            // Reuse cached debug buffer, but keep the control cadence independent from
            // human-facing debug sampling so the fixed-point controller stays active in production runs.
            let debug_every = self.cfg_debug_sample_every;
            let control_every = self.cfg_solve_control_every;
            let step = self.optimizer.step_count();
            let debug_due = debug_every != 0 && step % debug_every == 0;
            let control_due = step % control_every == 0;
            let mut metrics_refreshed = false;
            if let Some(fw_mid) = strict_debug_snapshot.take() {
                if Self::valid_debug_snapshot(&fw_mid) {
                    self.cached_debug_buf = fw_mid;
                    self.cached_debug_gen = self.cached_debug_gen.wrapping_add(1);
                    metrics_refreshed = true;
                }
            }
            if !metrics_refreshed && (control_due || debug_due) {
                let sync_t0 = std::time::Instant::now();
                let fw_now = gpu.read_debug_buffer();
                if Self::valid_debug_snapshot(&fw_now) {
                    self.cached_debug_buf = fw_now;
                    self.cached_debug_gen = self.cached_debug_gen.wrapping_add(1);
                    metrics_refreshed = true;
                } else {
                    // The debug shader writes intermediate sentinels (e.g. 186) before the
                    // final 901 commit marker. Only promote completed snapshots into the
                    // control cache; otherwise the controller reacts to in-flight data.
                    let fw_retry = gpu.read_debug_buffer();
                    if Self::valid_debug_snapshot(&fw_retry) {
                        self.cached_debug_buf = fw_retry;
                        self.cached_debug_gen = self.cached_debug_gen.wrapping_add(1);
                        metrics_refreshed = true;
                    }
                }
                if audit_cost {
                    audit_sync_ms += sync_t0.elapsed().as_secs_f64() * 1e3;
                }
            }
            if metrics_refreshed && !self.cached_debug_buf.is_empty() {
                self.update_fpm_runtime_schedule();
                let fw = &self.cached_debug_buf;

                let rs_cg = 0.0f32;

                let (heartbeat, max_h, avg_iters, hit_count, max_delta, contractivity, hit_den) =
                    Self::decode_forward_debug_metrics(fw, self.config.h_slots);
                let debug_v901 = fw.len() > 18 && fw[8] == 901.0;
                let debug_fpm = Self::env_flag("AIDEEN_DEBUG_FPM");
                let _last_delta = if fw.len() > 17 { fw[17] } else { 0.0 };
                let trunc_flag = if fw.len() > 18 { fw[18] } else { 0.0 };
                let total_elems = if fw.len() > 19 { fw[19] } else { 0.0 };
                let eps_runtime_dbg = if fw.len() > 9 { fw[9] } else { 0.0 };
                let signal_rms_dbg = if debug_v901 && fw.len() > 12 { fw[12] } else { 0.0 };
                let pre_rms_dbg = if debug_v901 && fw.len() > 13 { fw[13] } else { 0.0 };
                let fh_rms_dbg = if debug_v901 && fw.len() > 14 { fw[14] } else { 0.0 };
                let hprev_rms_dbg = if debug_v901 && fw.len() > 15 { fw[15] } else { 0.0 };
                let nscale_abs_dbg = if debug_v901 && fw.len() > 16 { fw[16] } else { 0.0 };
                let pre2h_dbg = if debug_v901 && fw.len() > 17 { fw[17] } else { 0.0 };
                let fh2h_dbg = if debug_v901 && fw.len() > 18 { fw[18] } else { 0.0 };
                let attn_rms_dbg = if debug_v901 && fw.len() > 19 { fw[19] } else { 0.0 };
                let attn2sig_dbg = if debug_v901 && fw.len() > 20 { fw[20] } else { 0.0 };
                let attn_scale_dbg = if debug_v901 && fw.len() > 21 { fw[21] } else { 0.0 };
                let iter0_err_h_dbg = if debug_v901 && fw.len() > 22 { fw[22] } else { 0.0 };
                let iter1_err_h_dbg = if debug_v901 && fw.len() > 23 { fw[23] } else { 0.0 };
                let iter0_attn2sig_dbg = if debug_v901 && fw.len() > 24 { fw[24] } else { 0.0 };
                let iter1_attn2sig_dbg = if debug_v901 && fw.len() > 25 { fw[25] } else { 0.0 };
                let iter0_attn_scale_dbg = if debug_v901 && fw.len() > 26 { fw[26] } else { 0.0 };
                let iter1_attn_scale_dbg = if debug_v901 && fw.len() > 27 { fw[27] } else { 0.0 };
                let iter_max_err_dbg = if debug_v901 && fw.len() > 28 { fw[28] } else { 0.0 };
                let iter_max_attn_dbg = if debug_v901 && fw.len() > 29 { fw[29] } else { 0.0 };
                let token_max_err_dbg = if debug_v901 && fw.len() > 30 { fw[30] } else { 0.0 };
                let token_max_attn_dbg = if debug_v901 && fw.len() > 31 { fw[31] } else { 0.0 };
                let fpm_rms = if !debug_v901 && fw.len() > 25 { fw[25] } else { 0.0 };
                let hist0 = if fw.len() > 100 { fw[100] } else { 0.0 };
                let hist1 = if fw.len() > 101 { fw[101] } else { 0.0 };
                let hist2 = if fw.len() > 102 { fw[102] } else { 0.0 };
                let hist_anchor0 = if fw.len() > 103 { fw[103] } else { 0.0 };
                let hist_anchor1 = if fw.len() > 104 { fw[104] } else { 0.0 };
                let hist_rms_floor = if fw.len() > 105 { fw[105] } else { 0.0 };
                let hist_contr_floor = if fw.len() > 106 { fw[106] } else { 0.0 };
                let hist_inject = if fw.len() > 107 { fw[107] } else { 0.0 };
                let hist_minner_zero = if fw.len() > 108 { fw[108] } else { 0.0 };
                let hist_force_nofpm = if fw.len() > 109 { fw[109] } else { 0.0 };
                let hist_prelude_skip = if fw.len() > 110 { fw[110] } else { 0.0 };
                let hist_loop_force_nofpm = if fw.len() > 111 { fw[111] } else { 0.0 };
                let fpm_enabled = Self::effective_fpm_enabled();
                let mut fpm_status = SolveStatus::Unconverged;
                let mut fpm_step_diag = String::new();
                if let Some(metrics) = if fpm_enabled {
                    Self::decode_fpm_health_metrics(fw, self.config.h_slots)
                } else {
                    None
                } {
                    let model_a_fpm = Self::fpm_stage_from_env() < 6;
                    fpm_status = Self::classify_fpm_solve(
                        metrics,
                        max_h,
                        contractivity,
                        epsilon,
                        model_a_fpm,
                    );
                    fpm_step_diag = format!(
                        " status={} err_h={:.3e} exit_err_h={:.3e} exit_iter={:.2} err_M={:.3e} z_max={:.3} z_avg={:.3} u2v_max={:.3e} memctx_rms={:.3e} memctx/sig={:.3e} dead={:.0} sat={:.0} rescue={:.0}/{:.0} pre_rescue={:.0} eps={:.3e} a_m={:.3} tau={:.2}",
                        fpm_status.label(),
                        metrics.max_err_h,
                        metrics.exit_err_h,
                        metrics.exit_iter,
                        metrics.max_mem_update,
                        metrics.max_z,
                        metrics.avg_z,
                        metrics.max_update_ratio,
                        metrics.max_memctx_rms,
                        metrics.max_memctx_to_signal,
                        metrics.dead_slots,
                        metrics.write_saturation,
                        metrics.rescue_entered,
                        metrics.rescue_recovered,
                        metrics.pre_rescue_converged,
                        eps_runtime_dbg,
                        self.fpm_alpha_m_current,
                        self.fpm_tau_current
                    );
                }

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
                if fpm_enabled {
                    match fpm_status {
                        SolveStatus::HealthyConverged => {
                            self.hit_lo_streak += 1;
                            self.hit_hi_streak = 0;
                        }
                        SolveStatus::TrivialConverged => {
                            self.hit_hi_streak = 0;
                            self.hit_lo_streak = 0;
                        }
                        SolveStatus::Unconverged | SolveStatus::NumericInvalid => {
                            self.hit_hi_streak += 1;
                            self.hit_lo_streak = 0;
                        }
                    }
                } else if seq >= 8.0 {
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
                    let hi_delta = (epsilon * 12.0).max(1.2e-3);
                    let lo_delta = (epsilon * 3.0).max(3e-4);
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
                    self.adaptive_max_iters =
                        (self.adaptive_max_iters + 1).min(self.solve_stage_cap);
                    self.hit_hi_streak = 0;
                }

                // Bajar iters si está sobrado por un buen rato
                if self.hit_lo_streak >= 10 {
                    self.adaptive_max_iters = self
                        .adaptive_max_iters
                        .saturating_sub(1)
                        .max(self.solve_stage_floor);
                    self.hit_lo_streak = 0;
                }

                // BOOST de Damping por inestabilidad puntual o hit ratio alto
                if (fpm_enabled
                    && matches!(
                        fpm_status,
                        SolveStatus::Unconverged | SolveStatus::NumericInvalid
                    ))
                    || (!fpm_enabled && (hit_ratio > 0.20 || non_fpm_boost_residual))
                {
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

                if (!fpm_enabled && max_delta > 5e-1)
                    || (fpm_enabled && matches!(fpm_status, SolveStatus::NumericInvalid))
                {
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
                    || (fpm_enabled && matches!(fpm_status, SolveStatus::NumericInvalid))
                    || contractivity > 1.20
                {
                    self.emergency_left = 3; // 3 windows de debug (~30 steps)
                    self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(emergency_iter_cap);
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
                } else {
                    self.adaptive_max_iters = self
                        .adaptive_max_iters
                        .clamp(self.solve_stage_floor, self.solve_stage_cap);
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
                let conv_ok = if fpm_enabled {
                    matches!(fpm_status, SolveStatus::HealthyConverged)
                } else {
                    non_fpm_conv_ok
                };
                let conv_str = if fpm_enabled {
                    fpm_status.label()
                } else if conv_ok {
                    "OK"
                } else {
                    "FAIL"
                };
                let solve_metric = if fpm_enabled {
                    Self::decode_fpm_health_metrics(fw, self.config.h_slots)
                        .map(|m| m.max_err_h)
                        .unwrap_or(max_delta)
                } else {
                    max_delta
                };
                let hit_i = hit.round() as i32;
                if debug_due {
                    if debug_fpm && !debug_v901 {
                        if let Some((sig, eps, tokens, slots, aux0, aux1, aux2)) =
                            Self::decode_debug_snapshot_header(fw)
                        {
                            println!(
                                "    \x1b[90m[DEBUG-BUF] valid=0 sig={:.3e} eps={:.3e} tokens={:.3e} slots={:.3e} aux=[{:.3e},{:.3e},{:.3e}] len={}\x1b[0m",
                                sig,
                                eps,
                                tokens,
                                slots,
                                aux0,
                                aux1,
                                aux2,
                                fw.len()
                            );
                        } else {
                            println!(
                                "    \x1b[90m[DEBUG-BUF] valid=0 len={} header=short\x1b[0m",
                                fw.len()
                            );
                        }
                    }
                    if debug_v901 {
                        println!(
                            "    \x1b[90m[GPU-DEBUG] Step {:>2}: hit={:>3}/{:.0} ({:>5.1}%) contr={:.3} solve={:.3e} rs_cg={:.1e} iters={:.1} cap={} damp={:.2} mode={} conv={} tps={:.1} max_h={:.6} sig/pre/fh/h={:.3e}/{:.3e}/{:.3e}/{:.3e} attn={:.3e} attn/sig={:.3e} attn_scale={:.3e} i0_err={:.3e} i1_err={:.3e} i0_a/s={:.3e} i1_a/s={:.3e} i0_sc={:.3e} i1_sc={:.3e} imax_err={:.0}@tok={:.0} imax_a={:.0}@tok={:.0} nscale_max={:.3e} pre/h={:.3e} fh/h={:.3e}\x1b[0m",
                            self.optimizer.step_count() % 100,
                            hit_i,
                            hit_den,
                            100.0 * hit_ratio,
                            contractivity,
                            solve_metric,
                            rs_cg,
                            avg_iters,
                            self.adaptive_max_iters,
                            damping_eff,
                            mode_str,
                            conv_str,
                            tps_debug,
                            max_h,
                            signal_rms_dbg,
                            pre_rms_dbg,
                            fh_rms_dbg,
                            hprev_rms_dbg,
                            attn_rms_dbg,
                            attn2sig_dbg,
                            attn_scale_dbg,
                            iter0_err_h_dbg,
                            iter1_err_h_dbg,
                            iter0_attn2sig_dbg,
                            iter1_attn2sig_dbg,
                            iter0_attn_scale_dbg,
                            iter1_attn_scale_dbg,
                            iter_max_err_dbg,
                            token_max_err_dbg,
                            iter_max_attn_dbg,
                            token_max_attn_dbg,
                            nscale_abs_dbg,
                            pre2h_dbg,
                            fh2h_dbg
                        );
                    } else {
                        println!(
                            "    \x1b[90m[GPU-DEBUG] Step {:>2}: hit={:>3}/{:.0} ({:>5.1}%) contr={:.3} solve={:.3e} rs_cg={:.1e} iters={:.1} cap={} damp={:.2} mode={} conv={} tps={:.1} max_h={:.6} hist=[{:.3e},{:.3e},{:.3e}] anchor=[{:.3e},{:.3e}] floors=[{:.3e},{:.3e}] flags=[{:.0},{:.0},{:.0},{:.0},{:.0}] shared={} total={:.0}\x1b[0m",
                            self.optimizer.step_count() % 100,
                            hit_i,
                            hit_den,
                            100.0 * hit_ratio,
                            contractivity,
                            solve_metric,
                            rs_cg,
                            avg_iters,
                            self.adaptive_max_iters,
                            damping_eff,
                            mode_str,
                            conv_str,
                            tps_debug,
                            max_h,
                            hist0,
                            hist1,
                            hist2,
                            hist_anchor0,
                            hist_anchor1,
                            hist_rms_floor,
                            hist_contr_floor,
                            hist_inject,
                            hist_minner_zero,
                            hist_force_nofpm,
                            hist_prelude_skip,
                            hist_loop_force_nofpm,
                            trunc_str,
                            total_elems
                        );
                    }
                    if !fpm_step_diag.is_empty() {
                        println!("    \x1b[90m[FPM-STEP]{}\x1b[0m", fpm_step_diag);
                    }
                    if debug_v901 {
                        let slot_solve = (0..self.config.h_slots)
                            .filter_map(|slot| {
                                let base = 32 + slot * 5;
                                if fw.len() <= base + 4 {
                                    return None;
                                }
                                let exit_base = 688 + slot * 4;
                                let strict = fw.get(exit_base).copied().unwrap_or(0.0);
                                let homeo = fw.get(exit_base + 1).copied().unwrap_or(0.0);
                                let fail = fw.get(exit_base + 2).copied().unwrap_or(0.0);
                                let homeo_band = fw.get(exit_base + 3).copied().unwrap_or(0.0);
                                let diag_base = 400 + slot * 12;
                                let exit_err = fw.get(diag_base + 8).copied().unwrap_or(0.0);
                                Some(format!(
                                    "s{}:Δ={:.2e},exit_err={:.2e},hit={:.0},it={:.1},c={:.2},h={:.2e},exit={:.0}/{:.0}/{:.0},band={:.1e}",
                                    slot,
                                    fw[base],
                                    exit_err,
                                    fw[base + 1],
                                    fw[base + 2],
                                    fw[base + 3],
                                    fw[base + 4],
                                    strict,
                                    homeo,
                                    fail,
                                    homeo_band
                                ))
                            })
                            .collect::<Vec<_>>()
                            .join(" | ");
                        println!("    \x1b[90m[GPU-SLOTS] {}\x1b[0m", slot_solve);
                    }
                }
                // GPU-SSM per-slot decay diagnostics (activar con AIDEEN_SSM_DEBUG=1).
                if debug_due && self.cfg_ssm_debug {
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
                            "    \x1b[90m[GPU-SSM] Step {}: a_mean=[{}] a_spread=[{}] fpm_rms={:.3e}\x1b[0m",
                            self.optimizer.step_count() % 100,
                            fmt_vec(&a_means),
                            fmt_vec(&a_spreads),
                            fpm_rms,
                        );
                    }
                }

                // Per-token debug (slot 0) for small sequences.
                let seq_len = heartbeat.max(1.0).round() as usize;
                if debug_due && seq_len > 0 && seq_len <= 16 {
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
                let (qkv_bytes, wo_bytes, win_bytes, aux_lower_bound) = audit_forward_bytes;
                let dense_lower_bound = qkv_bytes + wo_bytes + win_bytes + aux_lower_bound;
                let forward_secs = (audit_forward_ms / 1e3).max(1e-9);
                eprintln!(
                    "[FORWARD-BW] step={} batch={} seq={} iters_cap={} qkv_lb={:.3}GiB wo_lb={:.3}GiB win_once={:.3}GiB aux_lb={:.3}GiB dense_lb={:.3}GiB dense_lb_bw={:.1}GiB/s",
                    self.optimizer.step_count(),
                    fwd_batch_size,
                    per_seq_len,
                    self.adaptive_max_iters,
                    Self::gib(qkv_bytes),
                    Self::gib(wo_bytes),
                    Self::gib(win_bytes),
                    Self::gib(aux_lower_bound),
                    Self::gib(dense_lower_bound),
                    Self::gib(dense_lower_bound) / forward_secs,
                );
            }

            if let Some(loss) = self.cached_loss_after_sync(gpu, false) {
                self.last_gpu_loss = loss;
                return loss;
            }
            // No fresh GPU loss was read this step. Returning the last cached value would
            // silently contaminate epoch_loss_avg with stale or zero data, so report "no sample"
            // and let the caller skip this step in the running average.
            return f32::NAN;
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
            self.apply_stage_solve_schedule(
                floor,
                cap,
                if deq_progress < 0.25 {
                    6
                } else if deq_progress < 0.60 {
                    8
                } else {
                    10
                },
            );

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
                    let loss = self.train_sequence(batch_ctx, batch_tgt, reset_requested, eps);
                    if loss.is_finite() && loss > 0.0 {
                        epoch_loss += loss;
                        num_chunks += 1;
                    }
                    self.optimizer.lr = original_lr;
                } else {
                    let loss = self.train_sequence(batch_ctx, batch_tgt, reset_requested, eps);
                    if loss.is_finite() && loss > 0.0 {
                        epoch_loss += loss;
                        num_chunks += 1;
                    }
                }
                interval_tokens += batch_ctx.len();
                total_tokens += batch_ctx.len();
                #[cfg(feature = "wgpu")]
                if self.cfg_tps_sync_every != 0 && num_chunks % self.cfg_tps_sync_every == 0 {
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
                        // Reporting mode must measure completed GPU work, not merely queued commands.
                        // Waiting here keeps throughput benchmarks unaffected when progress is disabled,
                        // while making visible TPS/loss trustworthy whenever the caller asks for progress.
                        gpu.device.poll(wgpu::Maintain::Wait);
                    }
                    let interval_elapsed = interval_start.elapsed().as_secs_f32();
                    let window_tps = interval_tokens as f32 / interval_elapsed.max(1e-9);
                    let epoch_elapsed = t_start.elapsed().as_secs_f32();
                    let epoch_tps = total_tokens as f32 / epoch_elapsed.max(1e-9);
                    let current_loss_disp =
                        self.visible_loss_text(Some(epoch_loss / num_chunks as f32));

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
                .and_then(|gpu| self.cached_loss_after_sync(gpu, true))
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
                let emb_upload = if emb_needs_upload {
                    Some(self.tokenizer.embeddings_row_major())
                } else {
                    None
                };
                let (s_sequence, _) = match gpu_emb.prepare_sequence_and_query(
                    &gpu.device,
                    &gpu.queue,
                    context,
                    self.config.ctx_len,
                    emb_upload.as_deref().unwrap_or(&[]),
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
                let (_, _, _, _, slot_anchor_rm, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) =
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
                let emb_upload = if emb_needs_upload {
                    Some(self.tokenizer.embeddings_row_major())
                } else {
                    None
                };
                let (s_sequence, _) = match gpu_emb.prepare_sequence_and_query(
                    &gpu.device,
                    &gpu.queue,
                    context,
                    self.config.ctx_len,
                    emb_upload.as_deref().unwrap_or(&[]),
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
                let (_, _, _, _, slot_anchor_rm, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _) =
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
        let mut best_epoch_loss = Self::load_best_loss(checkpoint_path);

        for epoch in 0..epochs {
            // Each epoch restarts the token stream from the beginning of the file.
            // Reset temporal state here so chunk/history memory does not leak across
            // epoch boundaries and contaminate the new pass over the dataset.
            self.reset_state();
            let t_start = std::time::Instant::now();
            let global_epoch = self.completed_file_epochs + epoch;
            let total_schedule_epochs = self.completed_file_epochs + epochs;
            let current_lr = self.controlled_epoch_lr(global_epoch, total_schedule_epochs);
            self.optimizer.lr = current_lr;
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                gpu.tps_epoch_begin();
            }

            // File training should use a global epoch schedule across resumes, not restart the
            // solver/LR regime on every new process launch.
            let (_deq_progress, sched_floor, sched_cap, sched_adj_iters) =
                Self::file_epoch_schedule(global_epoch, total_schedule_epochs);
            self.apply_stage_solve_schedule(sched_floor, sched_cap, sched_adj_iters);
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
            let mut train_chunks = 0usize;
            let mut total_tokens = 0usize;
            let mut interval_start = std::time::Instant::now();
            let mut interval_tokens = 0usize;
            let mut last_progress_chunk_logged = 0usize;
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
                                let remaining =
                                    max_batch_tokens.saturating_sub(batch_train_buf.len());
                                if remaining == 0 {
                                    // Flush full batch before consuming more.
                                    let is_val = self.cfg_val_every != 0
                                        && num_chunks > 0
                                        && num_chunks % self.cfg_val_every == 0;
                                    if is_val {
                                        self.eval_mode = true;
                                    }
                                    let eps = self.progressive_epsilon(epoch, epochs);
                                    let loss = self.train_sequence(
                                        &batch_train_buf,
                                        &batch_tgt_buf,
                                        seg_start > 0,
                                        eps,
                                    );
                                    if is_val {
                                        self.eval_mode = false;
                                        println!(
                                            "    \x1b[93m[VAL] chunk {:>5}  val_loss={:.4}\x1b[0m",
                                            num_chunks, loss
                                        );
                                    } else if loss.is_finite() && loss > 0.0 {
                                        epoch_loss += loss;
                                        train_chunks += 1;
                                    }
                                    num_chunks += 1;
                                    total_tokens += batch_train_buf.len();
                                    interval_tokens += batch_train_buf.len();
                                    batch_train_buf.clear();
                                    batch_tgt_buf.clear();
                                    #[cfg(feature = "wgpu")]
                                    if self.cfg_tps_sync_every != 0
                                        && num_chunks % self.cfg_tps_sync_every == 0
                                    {
                                        if let Some(gpu) = self.gpu_deq.as_ref() {
                                            gpu.device.poll(wgpu::Maintain::Poll);
                                        }
                                    }
                                    continue;
                                }

                                let take = remaining.min(seg_len - offset);
                                batch_train_buf
                                    .extend_from_slice(&train_seg[offset..offset + take]);
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
                                let eps = self.progressive_epsilon(epoch, epochs);
                                let loss = self.train_sequence(
                                    &batch_train_buf,
                                    &batch_tgt_buf,
                                    seg_start > 0,
                                    eps,
                                );
                                if is_val {
                                    self.eval_mode = false;
                                    println!(
                                        "    \x1b[93m[VAL] chunk {:>5}  val_loss={:.4}\x1b[0m",
                                        num_chunks, loss
                                    );
                                } else if loss.is_finite() && loss > 0.0 {
                                    epoch_loss += loss;
                                    train_chunks += 1;
                                }
                                num_chunks += 1;
                                total_tokens += batch_train_buf.len();
                                interval_tokens += batch_train_buf.len();
                                batch_train_buf.clear();
                                batch_tgt_buf.clear();
                                #[cfg(feature = "wgpu")]
                                if self.cfg_tps_sync_every != 0
                                    && num_chunks % self.cfg_tps_sync_every == 0
                                {
                                    if let Some(gpu) = self.gpu_deq.as_ref() {
                                        gpu.device.poll(wgpu::Maintain::Poll);
                                    }
                                }
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
                        let loss =
                            self.train_sequence(&batch_train_buf, &batch_tgt_buf, false, eps);
                        if loss.is_finite() && loss > 0.0 {
                            epoch_loss += loss;
                            train_chunks += 1;
                        }
                        num_chunks += 1;
                        total_tokens += batch_train_buf.len();
                        batch_train_buf.clear();
                        batch_tgt_buf.clear();
                        #[cfg(feature = "wgpu")]
                        if self.cfg_tps_sync_every != 0 && num_chunks % self.cfg_tps_sync_every == 0
                        {
                            if let Some(gpu) = self.gpu_deq.as_ref() {
                                gpu.device.poll(wgpu::Maintain::Poll);
                            }
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
                        // Same rationale as the in-memory loop above: visible progress should reflect
                        // finished GPU work, not commands still queued in Metal.
                        gpu.device.poll(wgpu::Maintain::Wait);
                    }
                    let elapsed = t_start.elapsed().as_secs_f32();
                    let tps_run = total_tokens as f32 / elapsed.max(1e-9);
                    let tps_win =
                        interval_tokens as f32 / interval_start.elapsed().as_secs_f32().max(1e-9);
                    let current_loss = self.visible_loss_text(if train_chunks > 0 {
                        Some(epoch_loss / train_chunks as f32)
                    } else {
                        None
                    });

                    println!(
                        "    \x1b[95m[progress]\x1b[0m chunk {:>5}  \x1b[92mloss={}\x1b[0m  \x1b[96mtps_win={:>8.1}\x1b[0m  \x1b[94mtps_run={:>8.1}\x1b[0m  \x1b[90mtime={:.1}s\x1b[0m",
                        num_chunks, current_loss, tps_win, tps_run, elapsed
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
                .and_then(|gpu| self.cached_loss_after_sync(gpu, true))
            {
                self.last_gpu_loss = loss;
            }

            let elapsed = t_start.elapsed().as_secs_f32();
            let tps = if elapsed > 0.001 {
                total_tokens as f32 / elapsed
            } else {
                0.0
            };
            let epoch_loss_avg = if train_chunks > 0 {
                Some(epoch_loss / train_chunks as f32)
            } else {
                None
            };
            let cached_gpu_loss = if self.last_gpu_loss.is_finite() && self.last_gpu_loss > 0.0 {
                Some(self.last_gpu_loss)
            } else {
                None
            };
            let tracked_epoch_loss = cached_gpu_loss.or(epoch_loss_avg);

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

                let display_loss = tracked_epoch_loss
                    .map(|v| format!("{:.4}", v))
                    .unwrap_or_else(|| "n/a".to_string());
                let epoch_loss_text = epoch_loss_avg
                    .map(|v| format!("{:.4}", v))
                    .unwrap_or_else(|| "n/a".to_string());
                let cached_loss_text = cached_gpu_loss
                    .map(|v| format!("{:.4}", v))
                    .unwrap_or_else(|| "n/a".to_string());

                println!(
                    "  epoch {epoch:>4}/{epochs}  loss={}  epoch_loss_avg={}  gpu_loss_cached={}  lr={:.6}  tps_epoch={:>8.1}  time={:.2}s  tokens={} {}",
                    display_loss, epoch_loss_text, cached_loss_text, current_lr, tps, elapsed, total_tokens, gpu_stats
                );
                #[cfg(feature = "wgpu")]
                {
                    let fpm_enabled = Self::effective_fpm_enabled();
                    let debug_fpm = Self::env_flag("AIDEEN_DEBUG_FPM");
                    if fpm_enabled && debug_fpm {
                        if let Some(gpu) = self.gpu_deq.as_ref() {
                            let (wx_stats, wout_stats, alog_stats) =
                                gpu.read_hist_carrier_param_stats();
                            println!(
                                "    [FPM-CARRIER] wx(mean={:.3e},max={:.3e}) wout(mean={:.3e},max={:.3e}) alog(mean={:.3e},max={:.3e})",
                                wx_stats.0,
                                wx_stats.1,
                                wout_stats.0,
                                wout_stats.1,
                                alog_stats.0,
                                alog_stats.1,
                            );
                        }
                    }
                    let wdelta_audit = std::env::var("AIDEEN_WDELTA_AUDIT")
                        .ok()
                        .map(|v| {
                            let vl = v.trim().to_ascii_lowercase();
                            vl == "1" || vl == "true" || vl == "yes"
                        })
                        .unwrap_or(false);
                    if debug_fpm && !self.cached_debug_buf.is_empty() && !Self::valid_debug_snapshot(&self.cached_debug_buf) {
                        if let Some((sig, eps, tokens, slots, aux0, aux1, aux2)) =
                            Self::decode_debug_snapshot_header(&self.cached_debug_buf)
                        {
                            println!(
                                "    [DEBUG-BUF] valid=0 sig={:.3e} eps={:.3e} tokens={:.3e} slots={:.3e} aux=[{:.3e},{:.3e},{:.3e}] len={}",
                                sig,
                                eps,
                                tokens,
                                slots,
                                aux0,
                                aux1,
                                aux2,
                                self.cached_debug_buf.len()
                            );
                        } else {
                            println!(
                                "    [DEBUG-BUF] valid=0 len={} header=short",
                                self.cached_debug_buf.len()
                            );
                        }
                    }
                    if fpm_enabled && Self::valid_debug_snapshot(&self.cached_debug_buf) {
                        let slot_obs = Self::decode_slot_observability_metrics(
                            &self.cached_debug_buf,
                            self.config.h_slots,
                        );
                        let slot_a_delta = Self::decode_slot_max_a_delta_metrics(
                            &self.cached_debug_buf,
                            self.config.h_slots,
                        );
                        let slot_diag = Self::decode_fpm_diag_metrics(
                            &self.cached_debug_buf,
                            self.config.h_slots,
                        );
                        let slot_read = Self::decode_fpm_read_metrics(
                            &self.cached_debug_buf,
                            self.config.h_slots,
                        );
                        let slot_write_branches = Self::decode_fpm_write_branch_metrics(
                            &self.cached_debug_buf,
                            self.config.h_slots,
                        );
                        if slot_obs.iter().any(|(self_w, ent, mov)| {
                            self_w.abs() > 1e-6 || ent.abs() > 1e-6 || mov.abs() > 1e-6
                        }) {
                            let assign = slot_obs
                                .iter()
                                .map(|(self_w, _, _)| format!("{self_w:.3}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let entropy = slot_obs
                                .iter()
                                .map(|(_, ent, _)| format!("{ent:.3}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let movement = slot_obs
                                .iter()
                                .map(|(_, _, mov)| format!("{mov:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let a_delta = slot_a_delta
                                .iter()
                                .map(|v| format!("{v:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            println!(
                                "    [FPM-SLOTS] self_assign=[{}] entropy=[{}] move=[{}] a_delta_max=[{}]",
                                assign, entropy, movement, a_delta
                            );
                        }
                        if !slot_diag.is_empty() {
                            let err_h = slot_diag
                                .iter()
                                .map(|(err_h, _, _, _, _, _, _, _, _, _, _, _)| {
                                    format!("{err_h:.3e}")
                                })
                                .collect::<Vec<_>>()
                                .join(",");
                            let exit_err_h = slot_diag
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, exit_err_h, _, _, _)| {
                                    format!("{exit_err_h:.3e}")
                                })
                                .collect::<Vec<_>>()
                                .join(",");
                            let exit_iter = slot_diag
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, _, exit_iter, _, _)| {
                                    format!("{exit_iter:.2}")
                                })
                                .collect::<Vec<_>>()
                                .join(",");
                            let err_m = slot_diag
                                .iter()
                                .map(|(_, err_m, _, _, _, _, _, _, _, _, _, _)| {
                                    format!("{err_m:.3e}")
                                })
                                .collect::<Vec<_>>()
                                .join(",");
                            let z = slot_diag
                                .iter()
                                .map(|(_, _, z, _, _, _, _, _, _, _, _, _)| format!("{z:.3}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let rescue = slot_diag
                                .iter()
                                .map(
                                    |(_, _, _, rescue, recovered, _, _, _, _, _, entered, pre)| {
                                        format!(
                                            "{:.0}/{:.0}/{:.0}/{:.0}",
                                            rescue, recovered, entered, pre
                                        )
                                    },
                                )
                                .collect::<Vec<_>>()
                                .join(",");
                            let dead = slot_diag
                                .iter()
                                .map(|(_, _, _, _, _, dead, _, _, _, _, _, _)| format!("{dead:.0}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let ratio = slot_diag
                                .iter()
                                .map(|(_, _, _, _, _, _, ratio, _, _, _, _, _)| {
                                    format!("{ratio:.3e}")
                                })
                                .collect::<Vec<_>>()
                                .join(",");
                            let sat = slot_diag
                                .iter()
                                .map(|(_, _, _, _, _, _, _, sat, _, _, _, _)| format!("{sat:.0}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let read_rms = slot_read
                                .iter()
                                .map(|(rms, _, _, _, _, _, _, _, _, _, _, _, _, _)| format!("{rms:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let read_to_sig = slot_read
                                .iter()
                                .map(|(_, ratio, _, _, _, _, _, _, _, _, _, _, _, _)| format!("{ratio:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let retain_max = slot_read
                                .iter()
                                .map(|(_, _, mx, _, _, _, _, _, _, _, _, _, _, _)| format!("{mx:.3}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let retain_avg = slot_read
                                .iter()
                                .map(|(_, _, _, avg, _, _, _, _, _, _, _, _, _, _)| format!("{avg:.3}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let prop_rms = slot_read
                                .iter()
                                .map(|(_, _, _, _, pr, _, _, _, _, _, _, _, _, _)| format!("{pr:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let cand_rms = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, cr, _, _, _, _, _, _, _, _)| format!("{cr:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let q_rms = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, q, _, _, _, _, _, _, _)| format!("{q:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let k_rms = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, _, k, _, _, _, _, _, _)| format!("{k:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let logit_gap = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, gap, _, _, _, _, _)| format!("{gap:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let confidence = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, _, conf, _, _, _, _)| format!("{conf:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let src_m_rms = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, _, _, m, _, _, _)| format!("{m:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let keyed_m_rms = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, _, _, _, km, _, _)| format!("{km:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let peak_w = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, _, _, _, _, peak, _)| format!("{peak:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let raw_gap = slot_read
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, _, _, _, _, _, rg)| format!("{rg:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let write_h_unit = slot_write_branches
                                .iter()
                                .map(|(v, _, _, _, _, _)| format!("{v:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let write_wx = slot_write_branches
                                .iter()
                                .map(|(_, v, _, _, _, _)| format!("{v:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let write_inj = slot_write_branches
                                .iter()
                                .map(|(_, _, v, _, _, _)| format!("{v:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let write_inner = slot_write_branches
                                .iter()
                                .map(|(_, _, _, v, _, _)| format!("{v:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let write_out = slot_write_branches
                                .iter()
                                .map(|(_, _, _, _, v, _)| format!("{v:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let write_cand = slot_write_branches
                                .iter()
                                .map(|(_, _, _, _, _, v)| format!("{v:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            println!(
                                "    [FPM-PATH] read(mem_rms=[{}],mem2sig=[{}],retain_max=[{}],retain_avg=[{}],src_m_rms=[{}],keyed_m_rms=[{}],q_rms=[{}],k_rms=[{}],raw_gap=[{}],gap=[{}],peak=[{}],conf=[{}]) solve(err_h=[{}],exit_err_h=[{}],exit_iter=[{}],dead=[{}],rescue=[{}]) write(err_M=[{}],z=[{}],u2v=[{}],sat=[{}],prop_rms=[{}],cand_rms=[{}]) knobs(stage={},alpha_m={:.3},tau={:.2})",
                                read_rms,
                                read_to_sig,
                                retain_max,
                                retain_avg,
                                src_m_rms,
                                keyed_m_rms,
                                q_rms,
                                k_rms,
                                raw_gap,
                                logit_gap,
                                peak_w,
                                confidence,
                                err_h,
                                exit_err_h,
                                exit_iter,
                                dead,
                                rescue,
                                err_m,
                                z,
                                ratio,
                                sat,
                                prop_rms,
                                cand_rms,
                                Self::fpm_stage_from_env(),
                                self.fpm_alpha_m_current,
                                self.fpm_tau_current
                            );
                            if !slot_write_branches.is_empty() {
                                println!(
                                    "    [FPM-WRITE] h_unit=[{}] wx_h=[{}] write=[{}] m_inner=[{}] wout=[{}] cand=[{}]",
                                    write_h_unit,
                                    write_wx,
                                    write_inj,
                                    write_inner,
                                    write_out,
                                    write_cand
                                );
                            }
                            /*
                             * rescue tuple semantics:
                             *   entered-during-loop / recovered-after-rescue / rescue-entered-count / converged-before-rescue
                             */
                        }
                    }
                    // Slot-coord diagnostics (slot attention, active in DEQ_NO_MAMBA mode)
                    if Self::slot_coord_mode_active() && Self::valid_debug_snapshot(&self.cached_debug_buf) {
                        let slot_obs = Self::decode_slot_observability_metrics(
                            &self.cached_debug_buf,
                            self.config.h_slots,
                        );
                        let slot_diag = Self::decode_fpm_diag_metrics(
                            &self.cached_debug_buf,
                            self.config.h_slots,
                        );
                        let slot_path_audit = if let Some(gpu) = self.gpu_deq.as_ref() {
                            if self.cfg_slot_path_audit {
                                self.slot_path_audit_stats(gpu, self.config.ctx_len as u32)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        if slot_obs.iter().any(|(sw, ent, _)| sw.abs() > 1e-6 || ent.abs() > 1e-6) {
                            let assign = slot_obs
                                .iter()
                                .map(|(sw, _, _)| format!("{sw:.3}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let entropy = slot_obs
                                .iter()
                                .map(|(_, ent, _)| format!("{ent:.3}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let err_h = slot_diag
                                .iter()
                                .map(|(eh, _, _, _, _, _, _, _, _, _, _, _)| format!("{eh:.3e}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let iters = slot_diag
                                .iter()
                                .map(|(_, _, _, _, _, _, _, _, _, ei, _, _)| format!("{ei:.2}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            let dead = slot_diag
                                .iter()
                                .map(|(_, _, _, _, _, dead, _, _, _, _, _, _)| format!("{dead:.0}"))
                                .collect::<Vec<_>>()
                                .join(",");
                            if let Some(incoming) = slot_path_audit
                                .as_deref()
                                .and_then(Self::extract_slot_incoming)
                            {
                                println!(
                                    "    [SLOT-COORD] diag_assign=[{}] incoming=[{}] entropy=[{}] err_h=[{}] iters=[{}] diag_lt_tau=[{}]",
                                    assign, incoming, entropy, err_h, iters, dead
                                );
                            } else {
                                println!(
                                    "    [SLOT-COORD] diag_assign=[{}] entropy=[{}] err_h=[{}] iters=[{}] diag_lt_tau=[{}]",
                                    assign, entropy, err_h, iters, dead
                                );
                            }
                        }
                        if let Some(gpu) = self.gpu_deq.as_ref() {
                            if let Some((dq, dk, dv, do_, din, dbias)) =
                                self.slot_weight_specialization_stats(gpu)
                            {
                                println!(
                                    "    [SLOT-WEIGHTS] q={:.3e} k={:.3e} v={:.3e} o={:.3e} in={:.3e} bias={:.3e}",
                                    dq, dk, dv, do_, din, dbias
                                );
                            }
                            if let Some(audit) = slot_path_audit {
                                println!("    [SLOT-PATH] {audit}");
                            }
                        }
                    }
                    if fpm_enabled && wdelta_audit {
                        self.sync_inference_weights();
                        if let Some(gpu) = self.gpu_deq.as_ref() {
                            let (
                                (w_hist_mean, w_hist_max),
                                (w_delta_mean, w_delta_max),
                                (b_delta_mean, b_delta_max),
                            ) = gpu.read_hist_selective_param_stats();
                            let ((delta_mean, delta_max, delta_nz), (a_mean, a_min, a_max)) =
                                gpu.read_hist_selective_forward_stats(self.config.ctx_len as u32);
                            println!(
                                "    [WDELTA-STATS] w_hist(mean_abs={:.3e},max_abs={:.3e}) w_delta(mean_abs={:.3e},max_abs={:.3e}) b_delta(mean_abs={:.3e},max_abs={:.3e}) delta(mean_abs={:.3e},max_abs={:.3e},nz={}) a_t(mean={:.3e},min={:.3e},max={:.3e})",
                                w_hist_mean,
                                w_hist_max,
                                w_delta_mean,
                                w_delta_max,
                                b_delta_mean,
                                b_delta_max,
                                delta_mean,
                                delta_max,
                                delta_nz,
                                a_mean,
                                a_min,
                                a_max
                            );
                        }
                        if let Some((stable_rank, top8, top32)) = Self::w_delta_rank_summary(
                            &self.reasoning.w_k_write,
                            self.config.h_slots,
                            self.config.d_r,
                        ) {
                            println!(
                                "    [WDELTA-RANK] stable_rank=[{}] top8_energy=[{}] top32_energy=[{}]",
                                stable_rank, top8, top32
                            );
                        }
                    }
                }
            }

            self.completed_file_epochs = global_epoch + 1;

            if let Some(loss_avg) = tracked_epoch_loss {
                if let Err(e) = self.append_epoch_metrics(
                    checkpoint_path,
                    epoch,
                    loss_avg,
                    current_lr,
                    tps,
                    total_tokens,
                ) {
                    eprintln!(
                        "[metrics] Error guardando métricas en '{}': {}",
                        Self::checkpoint_metrics_path(checkpoint_path),
                        e
                    );
                }

                let should_save_best = best_epoch_loss.map_or(true, |best| loss_avg < best);
                if should_save_best && !checkpoint_path.is_empty() {
                    if let Err(e) = self.save_best_loss_checkpoint(
                        checkpoint_path,
                        epoch,
                        loss_avg,
                        current_lr,
                        total_tokens,
                    ) {
                        eprintln!(
                            "[checkpoint] Error guardando best_loss para '{}': {}",
                            checkpoint_path, e
                        );
                    } else {
                        eprintln!(
                            "[checkpoint] Nuevo best_loss {:.6} guardado en '{}' (epoch {})",
                            loss_avg,
                            Self::checkpoint_best_base_path(checkpoint_path),
                            epoch
                        );
                        best_epoch_loss = Some(loss_avg);
                    }
                }

                if let Some(new_lr) = self.update_lr_plateau_controller(loss_avg, current_lr) {
                    eprintln!(
                        "[lr-controller] plateau detectado: lr_cap -> {:.6} (best_loss={:.6}, cooldown={}, patience={})",
                        new_lr,
                        self.plateau_best_loss.unwrap_or(loss_avg),
                        self.cfg_lr_plateau_cooldown,
                        self.cfg_lr_plateau_patience,
                    );
                }
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
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Checkpointing: pesos + estado del optimizador
    // ─────────────────────────────────────────────────────────────────────────

    /// Guarda el modelo completo más el estado del optimizador.
    /// Formato: <path>.aidn (pesos) + <path>.opt (momentos Adam).
    pub fn save_checkpoint(&mut self, base_path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(base_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
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
                if !self.frozen_lm
                    && sum_mw == 0.0
                    && sum_vw == 0.0
                    && self.optimizer.step_count() > 10
                {
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
                if !self.frozen_emb && sum_me == 0.0 && self.optimizer.step_count() > 10 {
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
            "completed_file_epochs".to_string(),
            self.completed_file_epochs.to_string(),
        );
        if let Some(best) = self.plateau_best_loss {
            model
                .metadata
                .insert("plateau_best_loss".to_string(), best.to_string());
        }
        model.metadata.insert(
            "plateau_bad_epochs".to_string(),
            self.plateau_bad_epochs.to_string(),
        );
        model.metadata.insert(
            "plateau_cooldown_left".to_string(),
            self.plateau_cooldown_left.to_string(),
        );
        model.metadata.insert(
            "plateau_lr_cap".to_string(),
            self.plateau_lr_cap.to_string(),
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
            .get("completed_file_epochs")
            .and_then(|s| s.parse().ok())
        {
            trainer.completed_file_epochs = v;
        }
        if let Some(v) = model
            .metadata
            .get("plateau_best_loss")
            .and_then(|s| s.parse().ok())
        {
            trainer.plateau_best_loss = Some(v);
        }
        if let Some(v) = model
            .metadata
            .get("plateau_bad_epochs")
            .and_then(|s| s.parse().ok())
        {
            trainer.plateau_bad_epochs = v;
        }
        if let Some(v) = model
            .metadata
            .get("plateau_cooldown_left")
            .and_then(|s| s.parse().ok())
        {
            trainer.plateau_cooldown_left = v;
        }
        if let Some(v) = model
            .metadata
            .get("plateau_lr_cap")
            .and_then(|s| s.parse().ok())
        {
            trainer.plateau_lr_cap = v;
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
        if Self::slot_coord_mode_active() {
            // Temporary GPU-only guard for slot-coord mode:
            // the active training state contains per-slot Q/K/V/O/W_in matrices on GPU, while the
            // CPU-side reasoning object can only represent shared prototypes plus a few slot-wise
            // terms. Averaging GPU slot matrices back into CPU loses specialization and risks
            // re-uploading flattened weights later in the same process or after a checkpoint round-trip.
            //
            // Removal criteria: replace this guard once checkpoint/export paths can serialize and
            // restore exact per-slot slot-coord weights instead of lossy shared averages.
            return;
        }
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
                        self.reasoning.w_q = nalgebra::DMatrix::from_column_slice(
                            d_r,
                            d_r,
                            &avg_mat(&wq[..mat_total]),
                        );
                        self.reasoning.q_bias =
                            nalgebra::DMatrix::from_row_slice(h_slots, d_r, &wq[mat_total..]);
                        self.reasoning.w_k = nalgebra::DMatrix::from_column_slice(
                            d_r,
                            d_r,
                            &avg_mat(&wk[..mat_total]),
                        );
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
                        self.reasoning.a_log =
                            nalgebra::DMatrix::from_row_slice(h_slots, d_r, &alog);
                    }
                    self.reasoning.norm_scale = nalgebra::DVector::from_column_slice(&nscale);
                    {
                        let d_r = self.reasoning.config.d_r;
                        let h_slots = self.reasoning.config.h_slots;
                        let hist = gpu.read_hist_params_full();
                        const RETAIN_RANK: usize = 32;
                        let hist_mat_base = 0usize;
                        let slot_scale_base = hist_mat_base + d_r * d_r;
                        let hist_bias_base = slot_scale_base + h_slots * d_r;
                        let hist_gate_base = hist_bias_base + h_slots * d_r;
                        let slot_anchor_base = hist_gate_base + h_slots;
                        let w_kv_write_base = slot_anchor_base + h_slots * d_r;
                        let b_delta_base = w_kv_write_base + 2 * h_slots * d_r * RETAIN_RANK;
                        let scalar_base = b_delta_base + d_r;
                        let w_gate_hist_base = scalar_base + 21;
                        let w_write_gate_base = w_gate_hist_base + h_slots * d_r;
                        let b_write_mem_base = w_write_gate_base + h_slots * d_r;
                        let hhist_gamma_base = b_write_mem_base + h_slots;
                        let w_retain_up_base = hhist_gamma_base + h_slots;
                        let w_retain_down_base = w_retain_up_base + h_slots * d_r * 32;
                        let b_retain_base = w_retain_down_base + h_slots * 32 * d_r;
                        let w_q_mem_base = b_retain_base + h_slots * d_r;
                        let w_k_mem_base = w_q_mem_base + h_slots * d_r * 32;
                        let b_read_mem_base = w_k_mem_base + h_slots * d_r * 32;
                        let w_k_assoc_base = b_read_mem_base + h_slots;
                        let w_v_assoc_base = w_k_assoc_base + h_slots * d_r * 32;
                        let w_q_assoc_base = w_v_assoc_base + h_slots * d_r * 32;
                        let alpha_assoc_base = w_q_assoc_base + h_slots * d_r * 32;

                        self.reasoning.w_hist_shared = nalgebra::DMatrix::from_row_slice(
                            d_r,
                            d_r,
                            &hist[hist_mat_base..slot_scale_base],
                        );
                        self.reasoning.hist_slot_scale = nalgebra::DMatrix::from_row_slice(
                            h_slots,
                            d_r,
                            &hist[slot_scale_base..hist_bias_base],
                        );
                        self.reasoning.hist_slot_bias = nalgebra::DMatrix::from_row_slice(
                            h_slots,
                            d_r,
                            &hist[hist_bias_base..hist_gate_base],
                        );
                        self.reasoning.hist_gate_logit = nalgebra::DVector::from_column_slice(
                            &hist[hist_gate_base..slot_anchor_base],
                        );
                        self.reasoning.slot_anchor = nalgebra::DMatrix::from_row_slice(
                            h_slots,
                            d_r,
                            &hist[slot_anchor_base..w_kv_write_base],
                        );
                        self.reasoning.w_k_write = nalgebra::DMatrix::from_row_slice(
                            h_slots * d_r,
                            RETAIN_RANK,
                            &hist[w_kv_write_base..w_kv_write_base + h_slots * d_r * RETAIN_RANK],
                        );
                        self.reasoning.w_v_write = nalgebra::DMatrix::from_row_slice(
                            h_slots * RETAIN_RANK,
                            d_r,
                            &hist[w_kv_write_base + h_slots * d_r * RETAIN_RANK..b_delta_base],
                        );
                        self.reasoning.b_delta =
                            nalgebra::DVector::from_column_slice(&hist[b_delta_base..scalar_base]);
                        self.reasoning.w_gate_hist = nalgebra::DMatrix::from_row_slice(
                            h_slots,
                            d_r,
                            &hist[w_gate_hist_base..w_write_gate_base],
                        );
                        self.reasoning.w_write_gate = nalgebra::DMatrix::from_row_slice(
                            h_slots,
                            d_r,
                            &hist[w_write_gate_base..b_write_mem_base],
                        );
                        self.reasoning.b_write_mem = nalgebra::DVector::from_column_slice(
                            &hist[b_write_mem_base..hhist_gamma_base],
                        );
                        self.reasoning.w_retain_up = hist[w_retain_up_base..w_retain_down_base]
                            .chunks_exact(d_r * 32)
                            .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, 32, chunk))
                            .collect();
                        self.reasoning.w_retain_down = hist[w_retain_down_base..b_retain_base]
                            .chunks_exact(32 * d_r)
                            .map(|chunk| nalgebra::DMatrix::from_row_slice(32, d_r, chunk))
                            .collect();
                        self.reasoning.b_retain = nalgebra::DMatrix::from_row_slice(
                            h_slots,
                            d_r,
                            &hist[b_retain_base..w_q_mem_base],
                        );
                        self.reasoning.w_q_mem = hist[w_q_mem_base..w_k_mem_base]
                            .chunks_exact(d_r * 32)
                            .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, 32, chunk))
                            .collect();
                        self.reasoning.w_k_mem = hist[w_k_mem_base..b_read_mem_base]
                            .chunks_exact(d_r * 32)
                            .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, 32, chunk))
                            .collect();
                        self.reasoning.b_read_mem = nalgebra::DVector::from_column_slice(
                            &hist[b_read_mem_base..b_read_mem_base + h_slots],
                        );
                        self.reasoning.w_k_assoc = hist[w_k_assoc_base..w_v_assoc_base]
                            .chunks_exact(d_r * 32)
                            .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, 32, chunk))
                            .collect();
                        self.reasoning.w_v_assoc = hist[w_v_assoc_base..w_q_assoc_base]
                            .chunks_exact(d_r * 32)
                            .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, 32, chunk))
                            .collect();
                        self.reasoning.w_q_assoc = hist[w_q_assoc_base..alpha_assoc_base]
                            .chunks_exact(d_r * 32)
                            .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, 32, chunk))
                            .collect();
                        self.reasoning.alpha_assoc = nalgebra::DVector::from_column_slice(
                            &hist[alpha_assoc_base..alpha_assoc_base + h_slots],
                        );
                    }
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
