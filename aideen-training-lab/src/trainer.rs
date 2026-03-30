//! Main training loop for AIDEEN.
//!
//! Pipeline per step:
//!   ① tokenizer.embed_context(tokens) → query D_R
//!   ② query → DEQ forward → H*
//!   ③ H* → LmHead → logits
//!   ④ loss = cross_entropy(logits, target)
//!   ⑤ backward LmHead + backward embedding (analytical)
//!   ⑥ backward DEQ (implicit diff via CG)
//!   ⑦ Adam update (embeddings + LmHead + DEQ)
//!   ⑧ renormalize_weights() (spectral norm)

use aideen_core::{
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

/// Training configuration (hyperparameters).
pub struct TrainingConfig {
    pub lr: f32,
    /// Minimum LR at the end of the cosine schedule (default: lr/10).
    pub lr_min: f32,
    pub epochs: usize,
    pub log_every: usize,
    /// Warmup epochs: LR ramps linearly from lr_min to lr.
    pub warmup_epochs: usize,
    /// "Bit-Diet" experiment: Projects weights to ternary values (-1, 0, 1).
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
    // Ablation and validation flags
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
    cfg_wv_debug: bool,            // AIDEEN_DEQ_WV_DEBUG
    cfg_ssm_debug: bool,           // AIDEEN_SSM_DEBUG
    cfg_max_chunks: usize,         // AIDEEN_MAX_CHUNKS
    cfg_adj_iters_override: Option<u32>, // AIDEEN_ADJ_ITERS_OVERRIDE
}

impl Trainer {
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
        let alpha = alpha_env.unwrap_or(0.0); // v14: Mathematically proven to require 0.0
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

    /// Creates a Trainer with a pre-built tokenizer.
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
                .ok().and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(20).max(1),
            cfg_wv_debug: Self::env_flag("AIDEEN_DEQ_WV_DEBUG"),
            cfg_ssm_debug: Self::env_flag("AIDEEN_SSM_DEBUG"),
            cfg_max_chunks: std::env::var("AIDEEN_MAX_CHUNKS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(usize::MAX),
            cfg_adj_iters_override: std::env::var("AIDEEN_ADJ_ITERS_OVERRIDE")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok()),
        };
        trainer.apply_experimental_profile_from_env();
        trainer
    }

    /// Same as `from_tokenizer`, but forcing deterministic initialization
    /// of the reasoning weights (DEQ core) for seed-based reproducibility.
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
                .ok().and_then(|s| s.trim().parse::<u32>().ok()).unwrap_or(20).max(1),
            cfg_wv_debug: Self::env_flag("AIDEEN_DEQ_WV_DEBUG"),
            cfg_ssm_debug: Self::env_flag("AIDEEN_SSM_DEBUG"),
            cfg_max_chunks: std::env::var("AIDEEN_MAX_CHUNKS")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(usize::MAX),
            cfg_adj_iters_override: std::env::var("AIDEEN_ADJ_ITERS_OVERRIDE")
                .ok()
                .and_then(|v| v.trim().parse::<u32>().ok()),
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

    /// Resets cognitive states (slots) on both CPU and GPU.
    pub fn reset_state(&mut self) {
        // MambaSlotReasoning is stateless between calls: the DEQ recomputes h* from scratch
        // on each forward pass, so there is no persistent hidden state to clear.
        // reset_state forces the next sequence to not share context.
        self.m_prev = None;
        #[cfg(feature = "wgpu")]
        if let Some(gpu) = self.gpu_deq.as_ref() {
            gpu.reset_state();
        }
    }

    /// Executes a training step given a slice of tokens.
    /// `context`: context tokens (input)
    /// `target`: token to predict
    /// `reset_state`: if true, clears hidden state before processing.
    pub fn train_step(&mut self, context: &[u32], target: u32, reset_state: bool) -> f32 {
        if reset_state {
            self.reset_state();
        }

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

    /// Training step for a full sequence (Sequence Fusing).
    /// Processes 1..N tokens in a single GPU burst.
    /// `reset_state`: if true, clears hidden state before starting the sequence.
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

                // Defensive fix to avoid underflow if seq_len < ctx_len.
                // For batch > 1, the training loop passes B*ctx_len tokens — do not truncate.
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
                // TEST-ONLY: force one spectral renorm before the first step.
                // Disabled by default; set AIDEEN_DEQ_FORCE_RENORM=1 in tests only.
                if !self.force_renorm_done && Self::env_flag("AIDEEN_DEQ_FORCE_RENORM") {
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
        // CPU fallback: iterate through the sequence with autoregressive steps.
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

        // Sync CPU lm_head only when frozen/debug paths need it — not on every training step.
        #[cfg(feature = "wgpu")]
        if self.frozen_lm
            || Self::env_flag("AIDEEN_LM_FORCE_CPU_DLDH")
            || Self::env_flag("AIDEEN_LM_DLDH_PARITY")
        {
            self.sync_lm_head_from_gpu_if_needed();
        }

        if let (Some(gpu), Some(gpu_lm), Some(gpu_emb)) =
            (gpu_ctx, self.gpu_lm.as_mut(), self.gpu_emb.as_ref())
        {
            self.optimizer.tick();
            let num_tokens = targets.len();
            if num_tokens == 0 {
                return 0.0;
            }
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();

            // 1. Prepare DEQ input on GPU
            let fwd_batch_size = self.cfg_fwd_batch_size;
            // per_seq_len = tokens per sequence (num_tokens / B for batch mode, or 1 for query mode)
            let per_seq_len = (num_tokens as u32) / fwd_batch_size;
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
            }

            // 2. DEQ Forward (GPU-Only) - v13.1 Adaptive
            let debug_every = self.cfg_debug_sample_every;
            let debug_enable = debug_every != 0
                && (self.optimizer.step_count() % debug_every == 0);
            let _ = gpu.run_forward(
                fwd_batch_size,
                per_seq_len,
                self.adaptive_max_iters,
                damping_eff,
                epsilon,
                debug_enable,
            );
            // Use cached debug buffer — refresh deferred to end of step (after GPU is idle).
            // The DEQ-INVALID streak check needs 3 consecutive failures, so 1-step lag is safe.
            let fw = self.cached_debug_buf.clone();
            let heartbeat = if fw.len() > 10 { fw[10] } else { 1.0 };
            let max_delta = if fw.len() > 16 { fw[16] } else { 0.0 };
            let unconverged_count = if fw.len() > 15 { fw[15] } else { 0.0 };
            let contractivity = if fw.len() > 21 { fw[21] } else { 0.0 };
            let seq = heartbeat.max(1.0);
            let unconverged_ratio = unconverged_count.max(0.0) / seq;
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
                    "    [DEQ-INVALID] step={} contr={:.3} unconverged_ratio={:.3} maxΔ={:.3e} seq={:.0}",
                    self.optimizer.step_count(),
                    contractivity,
                    unconverged_ratio,
                    max_delta,
                    seq
                );
                self.invalid_hi_streak = 0;
                self.emergency_left = 3;
                self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(48);
                #[cfg(feature = "wgpu")]
                let _ = gpu.renormalize_spectral();
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
                    "[GPU-LM] Syncing LM Head weights... (abs_sum={:.4})",
                    w_sum
                );

                gpu_lm.upload_weights_only(&gpu.queue, w_raw, b_raw, g_raw);
                self.gpu_lm_weights_uploaded = true;
            }

            // Submit LM forward+backward without blocking for loss readback.
            // For eval_mode (validation) we still read synchronously since there's no adjoint.
            // For train mode, loss is read after apply_gradient_update when GPU is already idle.
            let read_loss_now = self.eval_mode;
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
            if self.eval_mode {
                self.last_gpu_loss = current_loss_sync;
            }
            let mut current_loss = self.last_gpu_loss; // will be updated after GPU idle for train path
            if lm_lr > 0.0 {
                self.lm_head_cpu_stale = true;
            }

            // 4. Embedding Update from GPU dl_dh buffer (Moved to step 6 to avoid duplication)

            // 5. DEQ Reasoning Core Update (Picard Adjoint + Fused GPU Weight Update)
            if self.eval_mode {
                return current_loss;
            }

            if !self.frozen_deq && !invalid_fixed_point {
                // ⑥ Backward DEQ — Picard Adjoint (GPU, always).
                // Skip when DEQ diverged (invalid_fixed_point): gradients from a non-converged
                // forward pass are unreliable (∂L/∂θ via implicit diff requires h* to exist).
                // Staged Picard fills fused_mix_buf with g_comb, then apply_fused_deq_update
                // applies the full weight update on GPU. Single path, always correct.
                let batch_size = fwd_batch_size;
                let _ = gpu.run_staged_adjoint_picard_no_readback(
                    per_seq_len,
                    self.reasoning.damping,
                    self.config.adj_iters as u32,
                    Some(&gpu_lm.dl_dh_buf),
                    true, // clear fused_hist_ctx_buf (rhs_slot) before adjoint — eliminates hist rerun
                    batch_size,
                );
                let grad_accum = self.cfg_grad_accum;
                // Cross-step gradient accumulation:
                // Each train_step() call accumulates gradients from a different sequence.
                // Weight update is applied only every grad_accum steps.
                let mode = if grad_accum == 1 { 0u32 } else { 1u32 };
                let apply_accum = grad_accum == 1 || self.grad_accum_counter + 1 >= grad_accum;
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
                self.grad_accum_counter += 1;
                if self.grad_accum_counter >= grad_accum {
                    self.grad_accum_counter = 0;
                    // GPU is now idle after fused apply_grad pass (if apply_accum=true).
                    // Read deferred results — both calls are near-instant since GPU is idle.
                    if !self.eval_mode {
                        let every = self.cfg_loss_readback_every;
                        let should_read = every != 0
                            && (every == 1 || (self.optimizer.step_count() % every == 0));
                        if should_read {
                            self.last_gpu_loss = gpu_lm.read_cached_loss(&gpu.device);
                        }
                        current_loss = self.last_gpu_loss;
                    }
                    // Refresh debug buffer cache if this is a sample step.
                    let debug_every = self.cfg_debug_sample_every;
                    if debug_every != 0
                        && (self.optimizer.step_count() % debug_every == 0) {
                        self.cached_debug_buf = gpu.read_debug_buffer();
                    }
                } else {
                    // Intermediate grad_accum step: no weight update poll yet.
                    // Read loss now (will cause one poll but GPU is nearly done with fused_update).
                    if !self.eval_mode {
                        let every = self.cfg_loss_readback_every;
                        let should_read = every != 0
                            && (every == 1 || (self.optimizer.step_count() % every == 0));
                        if should_read {
                            self.last_gpu_loss = gpu_lm.read_cached_loss(&gpu.device);
                        }
                        current_loss = self.last_gpu_loss;
                    }
                }
                self.gpu_weights_uploaded = true;
                self.gpu_cg_weights_uploaded = true;

                // =============================================================================
                // [LEGACY — NOT USED] CPU fallback path (CG on CPU with h* read from GPU).
                //
                // Replaced by: run_staged_adjoint_picard_no_readback() + apply_fused_deq_update()
                // (full GPU path, lines above this block).
                //
                // This block never executes (if false). What it does:
                //   1. Reads h* from GPU, recomputes on CPU via reasoning.step()
                //   2. Runs deq_implicit_grad (numerical CG with finite differences)
                //   3. Applies weight update on CPU and re-uploads everything to GPU
                //
                // Issues: readback latency, scalar CG (O(N^2) per iter),
                // does not support W_hist or hist_gate (only W_q/W_k/W_v/W_o/W_in/NormScale).
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

                    let force_cpu_lm_dldh = Self::env_flag("AIDEEN_LM_FORCE_CPU_DLDH");
                    let parity_check = Self::env_flag("AIDEEN_LM_DLDH_PARITY");
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
                    let _ = gpu.renormalize_spectral();
                }
            }

            if !self.frozen_emb {
                let emb_lr = base_lr * self.training_config.emb_lr_mult;
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
                self.gpu_emb_weights_uploaded = true;
            }

            // --- TPS tracking ---
            self.debug_tokens_accum += num_tokens as u32;

            // --- DIAGNÓSTICOS GPU (v13.1 Auto-Healing) ---
            // Reuse cached debug buffer to avoid blocking GPU every diagnostic step.
            let debug_every = self.cfg_debug_sample_every;
            if debug_every != 0
                && (self.optimizer.step_count() % debug_every == 0) {
                self.cached_debug_buf = gpu.read_debug_buffer();
            }
            if !self.cached_debug_buf.is_empty() {
                let fw = &self.cached_debug_buf;

                let rs_cg = 0.0f32;

                let heartbeat = if fw.len() > 10 { fw[10] } else { 0.0 }; // seq
                let max_h = if fw.len() > 11 { fw[11] } else { 0.0 };
                let avg_iters = if fw.len() > 13 { fw[13] } else { 0.0 };
                let unconverged_count = if fw.len() > 15 { fw[15] } else { 0.0 };
                let max_delta = if fw.len() > 16 { fw[16] } else { 0.0 };
                let _last_delta = if fw.len() > 17 { fw[17] } else { 0.0 };
                let trunc_flag = if fw.len() > 18 { fw[18] } else { 0.0 };
                let total_elems = if fw.len() > 19 { fw[19] } else { 0.0 };
                let contractivity = if fw.len() > 21 { fw[21] } else { 0.0 };
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
                let unconverged = unconverged_count.max(0.0);
                let unconverged_ratio = unconverged / seq;

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
                    let _ = gpu.renormalize_spectral();
                    self.contractivity_hi_streak = 0;
                    if contractivity > 1.20 {
                        self.emergency_left = 2; // 2 debug windows (~20 steps)
                    }
                }

                // 3. Forward Iters Hysteresis (v13.3)
                // For very short sequences (e.g. seq=1), unconverged_ratio is not informative
                // and tends to force unnecessary iteration growth.
                if seq >= 8.0 {
                    if unconverged_ratio > 0.08 {
                        self.hit_hi_streak += 1;
                        self.hit_lo_streak = 0;
                    } else if unconverged_ratio < 0.03 {
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

                // Increase iters if consistently hitting the ceiling
                if self.hit_hi_streak >= 2 {
                    self.adaptive_max_iters = (self.adaptive_max_iters + 1).min(12);
                    self.hit_hi_streak = 0;
                }

                // Decrease iters if consistently overprovisioned
                if self.hit_lo_streak >= 10 {
                    self.adaptive_max_iters = self.adaptive_max_iters.saturating_sub(1).max(4);
                    self.hit_lo_streak = 0;
                }

                // Damping BOOST on transient instability or high unconverged ratio
                if unconverged_ratio > 0.20 || max_delta > 1e-3 {
                    self.damping_boost_left = 2; // 2 debug windows (~20 steps)
                }

                // Explosive growth or pure instability detector (Stability Guardrail)
                let growth = if self.last_max_h > 0.0 {
                    max_h / self.last_max_h
                } else {
                    1.0
                };
                self.last_max_h = max_h;

                // Avoid false positives when activations are still tiny.
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

                // EMERGENCY Triggers: rapid growth, NaNs, sustained unacceptable residual
                // or divergence (>1.20).
                if self.max_h_growth_streak >= 3
                    || self.max_delta_hi_streak >= 3
                    || max_h.is_nan()
                    || max_delta.is_nan()
                    || contractivity > 1.20
                {
                    self.emergency_left = 3; // 3 debug windows (~30 steps)
                    self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(48);
                    self.max_h_growth_streak = 0;
                    self.max_delta_hi_streak = 0;
                    // Trigger spectral renorm immediately
                    #[cfg(feature = "wgpu")]
                    let _ = gpu.renormalize_spectral();
                }

                // Countdown de modos temporales
                if self.damping_boost_left > 0 {
                    self.damping_boost_left -= 1;
                }
                if self.emergency_left > 0 {
                    self.emergency_left -= 1;
                }

                // Determine mode and effective damping
                let damping_eff = self.damping_eff();
                let mode_str = if self.emergency_left > 0 {
                    "EMERG"
                } else if self.damping_boost_left > 0 {
                    "BOOST"
                } else {
                    "NORMAL"
                };
                let conv_ok =
                    unconverged_ratio <= 0.05 || max_delta <= (self.config.deq_epsilon * 4.0).max(3e-4);
                let conv_str = if conv_ok { "OK" } else { "FAIL" };
                let unconverged_i = unconverged.round() as i32;
                println!(
                    "    \x1b[90m[GPU-DEBUG] Step {:>2}: unconverged={:>3}/{:.0} ({:>5.1}%) contr={:.3} maxΔ={:.3e} rs_cg={:.1e} iters={:.1} cap={} damp={:.2} mode={} conv={} tps={:.1} max_h={:.6} inj_rms={:.3e} hist_rms={:.3e} hist/inj={:.3e} mamba_rms={:.3e} q/k/v={:.3e}/{:.3e}/{:.3e} mix/attn={:.3e}/{:.3e} attn_max={:.3} attn_ent={:.3} comb_rms={:.3e} hist=[{:.3e},{:.3e},{:.3e}] anchor=[{:.3e},{:.3e}] floors=[{:.3e},{:.3e}] flags=[{:.0},{:.0},{:.0},{:.0},{:.0}] shared={} total={:.0}\x1b[0m",
                    self.optimizer.step_count() % 100,
                    unconverged_i,
                    seq,
                    100.0 * unconverged_ratio,
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

                // GPU-SSM per-slot decay diagnostics (enable with AIDEEN_SSM_DEBUG=1).
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

    /// Cosine LR schedule with warmup.
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

    /// Progressive epsilon schedule for the DEQ solver.
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

    /// Executes the training loop over a tokenized corpus.
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
            let ctx_len = self.config.ctx_len.max(1);
            let batch_size = self.cfg_fwd_batch_size.max(1) as usize;
            let step = ctx_len * batch_size;

            for i in (0..train_tokens.len()).step_by(step) {
                let end = (i + step).min(train_tokens.len());
                let batch_ctx = &train_tokens[i..end];
                let batch_tgt = &targets[i..end];
                if epoch % log_every == 0 && i == 0 {
                    use std::collections::HashSet;
                    let uniq: HashSet<u32> = batch_ctx.iter().copied().collect();
                    eprintln!(
                        "[EMB] uniq_tokens={}/{} ({:.1}%)",
                        uniq.len(),
                        batch_ctx.len(),
                        100.0 * (uniq.len() as f32) / (batch_ctx.len() as f32)
                    );
                }

                // Detect if the first token is a separator/EOS (e.g. 0)
                // In many tokenized datasets, 0 is used to mark document start/end.
                let mut reset_requested = false;
                if i == 0 {
                    reset_requested = true; // Always reset at epoch start
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
                #[cfg(feature = "wgpu")]
                if self.cfg_tps_sync_every != 0
                    && num_chunks % self.cfg_tps_sync_every == 0 {
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        // Progress/TPS sync point only. This is outside the per-step GPU hot path
                        // and exists so reported throughput reflects completed GPU work.
                        gpu.device.poll(wgpu::Maintain::Wait);
                    }
                }

                if num_chunks % 10 == 0 {
                    let interval_elapsed = interval_start.elapsed().as_secs_f32();
                    let instant_tps = interval_tokens as f32 / interval_elapsed.max(1e-9);

                    #[cfg(feature = "wgpu")]
                    let current_loss_disp = if let Some(gpu_lm) = self.gpu_lm.as_ref() {
                        if let Some(gpu) = self.gpu_deq.as_ref() {
                            let every = self.cfg_loss_readback_every;
                            let should_read = every != 0
                                && (every == 1 || (self.optimizer.step_count() % every == 0));
                            if should_read {
                                self.last_gpu_loss = gpu_lm.read_cached_loss(&gpu.device);
                            }
                            self.last_gpu_loss
                        } else {
                            epoch_loss / num_chunks as f32
                        }
                    } else {
                        epoch_loss / num_chunks as f32
                    };
                    #[cfg(not(feature = "wgpu"))]
                    let current_loss_disp = epoch_loss / num_chunks as f32;

                    println!(
                        "    \x1b[95m[progress]\x1b[0m chunk {:>5}  \x1b[92mloss={:.4}\x1b[0m  \x1b[96mtps={:>8.1}\x1b[0m  \x1b[90mtime={:.1}s\x1b[0m",
                        num_chunks, current_loss_disp, instant_tps, t_start.elapsed().as_secs_f32()
                    );

                    // Reset interval timers
                    interval_start = std::time::Instant::now();
                    interval_tokens = 0;
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
            let elapsed = t_start.elapsed().as_secs_f32();
            // Use actual processed tokens, not num_chunks * ctx_len (which over-counts the last chunk).
            let tokens_processed = train_tokens.len();
            let tps = tokens_processed as f32 / elapsed.max(1e-9);

            if epoch % log_every == 0 {
                // GPU already idle (poll above) — read_cached_loss is near-instant here
                #[cfg(feature = "wgpu")]
                let display_loss = if let Some(gpu_lm) = self.gpu_lm.as_ref() {
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        let every = self.cfg_loss_readback_every;
                        let should_read = every != 0
                            && (every == 1 || (self.optimizer.step_count() % every == 0));
                        if should_read {
                            self.last_gpu_loss = gpu_lm.read_cached_loss(&gpu.device);
                        }
                        self.last_gpu_loss
                    } else {
                        total_loss
                    }
                } else {
                    total_loss
                };
                #[cfg(not(feature = "wgpu"))]
                let display_loss = total_loss;
                let mut gpu_suffix = String::new();
                #[cfg(feature = "wgpu")]
                if let Some(gpu) = self.gpu_deq.as_ref() {
                    if let Some(ns) = gpu.read_tps_epoch_ns() {
                        let tps_gpu = (tokens_processed as f64) / (ns / 1e9);
                        gpu_suffix = format!("  tps_gpu={:>8.1}", tps_gpu);
                    }
                }
                println!(
                    "  epoch {epoch:>4}/{epochs}  loss={:.4}  lr={:.6}  tps={:>8.1}  time={:.2}s{}",
                    display_loss, current_lr, tps, elapsed, gpu_suffix
                );
            }
        }
    }

    /// Generates text from a prompt.
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
            // Inference policy: prefer GPU; fall back to CPU if no device available.
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

        for _ in 0..max_tokens {
            let ctx_start = tokens.len().saturating_sub(self.config.ctx_len);
            let context = &tokens[ctx_start..];

            #[cfg(feature = "wgpu")]
            if self.gpu_deq.is_some() {
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
            let query = self.tokenizer.embed_context(context, self.config.ctx_len);
            let mut h = self.reasoning.init(&query);
            for _ in 0..10 {
                h = self.reasoning.step(&h, &query, None);
            }
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
        self.tokenizer.decode(&tokens[prompt_len..])
    }

    /// Streaming version of generate: calls `on_token` with each text fragment
    /// generated in real-time, without waiting for the full generation to finish.
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

        for _ in 0..max_tokens {
            let ctx_start = tokens.len().saturating_sub(self.config.ctx_len);
            let context = &tokens[ctx_start..];

            #[cfg(feature = "wgpu")]
            if self.gpu_deq.is_some() {
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
            let query = self.tokenizer.embed_context(context, self.config.ctx_len);
            let mut h = self.reasoning.init(&query);
            for _ in 0..10 {
                h = self.reasoning.step(&h, &query, None);
            }
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

        self.tokenizer.decode(&tokens[prompt_len..])
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
    }

    /// Computes cross-entropy loss over a sequence without updating weights.
    /// Useful for clean validation (forward-only, no backprop).
    pub fn eval_loss(&self, tokens: &[u32]) -> f32 {
        if tokens.len() < 2 {
            return f32::NAN;
        }
        let inputs = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        for (i, &target) in targets.iter().enumerate() {
            let ctx_start = i.saturating_sub(self.config.ctx_len.saturating_sub(1));
            let context = &inputs[ctx_start..=i];

            let query = self.tokenizer.embed_context(context, self.config.ctx_len);
            let mut h = self.reasoning.init(&query);
            for _ in 0..self.config.max_deq_iters.max(1) {
                h = self.reasoning.step(&h, &query, None);
            }
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

    /// Scales `grad` in-place if its L2 norm exceeds `max_norm`.
    /// Equivalent to PyTorch's `clip_grad_norm_` for a single tensor.
    fn clip_grad_norm(grad: &mut nalgebra::DVector<f32>, max_norm: f32) {
        let norm = grad.norm();
        if norm > max_norm {
            *grad *= max_norm / (norm + 1e-6);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Streaming dataloader — training from binary token file
    // ─────────────────────────────────────────────────────────────────────────

    /// Trains on a binary file of u32 tokens (little-endian).
    ///
    /// The file can be any size: it is read in chunks of `ctx_len + 1` tokens
    /// with an overlapping window of `overlap` tokens to preserve context
    /// between chunks. When the `eos_token` appears, context is reset.
    ///
    /// `save_every`: saves checkpoint every N epochs (0 = never).
    /// `skip_chunks`: number of chunks to skip at start of first epoch (for resuming).
    pub fn train_on_file(
        &mut self,
        path: &str,
        epochs: usize,
        log_every: usize,
        eos_token: u32,
        save_every: usize,
        checkpoint_path: &str,
        skip_chunks: usize,
    ) -> std::io::Result<()> {
        use std::io::{Read, Seek, SeekFrom};

        let file_size = std::fs::metadata(path)?.len() as usize;
        let total_file_tokens = file_size / 4;
        if total_file_tokens < 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File has fewer than 2 tokens",
            ));
        }

        let ctx_len = self.config.ctx_len.max(1);
        // Overlapping window: read ctx_len + 1 tokens, advance ctx_len/2 for continuous context.
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

            // v13.1 Adaptive Epoch Schedule (iteration floor)
            let deq_progress = epoch as f32 / epochs.max(1) as f32;
            let sched_floor = if deq_progress < 0.25 {
                8
            } else if deq_progress < 0.60 {
                10
            } else {
                12
            };
            self.adaptive_max_iters = self.adaptive_max_iters.max(sched_floor);
            self.config.adj_iters = if deq_progress < 0.25 {
                4
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

            // Skip chunks: advance file position past already-trained data (first epoch only).
            // Each chunk advances by `stride` tokens (stride = ctx_len/2).
            let skip_offset = if skip_chunks > 0 && epoch == 0 {
                let skip_bytes = ((skip_chunks * stride * 4) as u64).min(file_size as u64);
                println!(
                    "    [skip] Skipping {skip_chunks} chunks ({} tokens, {:.2} MB)",
                    skip_bytes / 4,
                    skip_bytes as f64 / 1_048_576.0
                );
                skip_bytes
            } else {
                0u64
            };

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
                if skip_offset > 0 {
                    if f.seek(SeekFrom::Start(skip_offset)).is_err() {
                        let _ = tx.send((Vec::new(), 0));
                        return;
                    }
                }
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
            let mut num_chunks = if skip_chunks > 0 && epoch == 0 { skip_chunks } else { 0 };
            let mut total_tokens = 0usize;
            // Buffer of unconsumed tokens from the previous chunk (for overlapping window).
            let mut carry: Vec<u32> = Vec::with_capacity(stride);
            // Pre-allocate token window to avoid per-chunk heap allocations.
            let mut tokens: Vec<u32> = Vec::with_capacity(chunk_tokens + stride);
            let mut last_save_time = std::time::Instant::now();

            loop {
                // Prepend carry from previous chunk + read new bytes (prefetch thread).
                let (read_buf, n) = match rx.recv() {
                    Ok(v) => v,
                    Err(_) => (Vec::new(), 0),
                };
                if n == 0 && carry.is_empty() {
                    break;
                }

                // Strict state reset at document boundaries (optional depending on eos_token)
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

                // Detect EOS: split the chunk into document sub-sequences.
                let mut seg_start = 0;
                while seg_start < tokens.len().saturating_sub(1) {
                    // Find the next EOS within this chunk.
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
                                    let is_val = num_chunks % 20 == 0;
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
                                    } else {
                                        epoch_loss += loss;
                                    }
                                    num_chunks += 1;
                                    total_tokens += batch_train_buf.len();
                                    batch_train_buf.clear();
                                    batch_tgt_buf.clear();
                                    #[cfg(feature = "wgpu")]
                                    if self.cfg_tps_sync_every != 0
                                        && num_chunks % self.cfg_tps_sync_every == 0 {
                                        if let Some(gpu) = self.gpu_deq.as_ref() {
                                            gpu.device.poll(wgpu::Maintain::Wait);
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
                                let is_val = num_chunks % 20 == 0;
                                if is_val {
                                    self.eval_mode = true;
                                }
                                let eps = self.progressive_epsilon(epoch, epochs);
                                let loss =
                                    self.train_sequence(&batch_train_buf, &batch_tgt_buf, seg_start > 0, eps);
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
                                batch_train_buf.clear();
                                batch_tgt_buf.clear();
                                #[cfg(feature = "wgpu")]
                                if self.cfg_tps_sync_every != 0
                                    && num_chunks % self.cfg_tps_sync_every == 0 {
                                    if let Some(gpu) = self.gpu_deq.as_ref() {
                                        // Validation/debug path: CPU reads results immediately after.
                                        gpu.device.poll(wgpu::Maintain::Wait);
                                    }
                                }
                            }
                        }
                    }

                    // Skip the EOS token and continue to the next sub-sequence.
                    seg_start = if seg_end < tokens.len() {
                        seg_end + 1
                    } else {
                        tokens.len()
                    };
                }

                // Carry: overlap the last min(stride, tokens.len()) tokens for context.
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
                                // Validation/debug path: CPU reads results immediately after.
                                gpu.device.poll(wgpu::Maintain::Wait);
                            }
                        }
                    }
                    break; // EOF
                }

                // Mini-log every 10 chunks to show real-time progress
                if num_chunks % 10 == 0 && num_chunks > 0 {
                    let elapsed = t_start.elapsed().as_secs_f32();
                    let tps = total_tokens as f32 / elapsed.max(1e-9);
                    // Main metric: real accumulated training average (not GPU cache).
                    let current_loss = epoch_loss / num_chunks as f32;

                    println!(
                        "    \x1b[95m[progress]\x1b[0m chunk {:>5}  \x1b[92mloss={:.4}\x1b[0m  \x1b[96mtps={:>8.1}\x1b[0m  \x1b[90mtime={:.1}s\x1b[0m",
                        num_chunks, current_loss, tps, elapsed
                    );

                    // Intra-epoch auto-save: time-based only (every 30 min).
                    // The `save_every` checkpoint is already applied per epoch at the end of the loop.
                    let time_save = last_save_time.elapsed().as_secs() > 1800;
                    if time_save && !checkpoint_path.is_empty() {
                        println!(
                            "    \x1b[93m[auto-save]\x1b[0m Saving progress (chunk {})...",
                            num_chunks
                        );
                        if let Err(e) = self.save_checkpoint(checkpoint_path) {
                            eprintln!("    \x1b[31m[error]\x1b[0m Failed to save: {}", e);
                        } else {
                            println!(
                                "    \x1b[32m[success]\x1b[0m Checkpoint '{}' updated.",
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
            // Flush GPU queue at the end of each epoch.
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
                // Validation/end-of-run boundary. We intentionally synchronize before final metrics.
                gpu.device.poll(wgpu::Maintain::Wait);
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
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        // Optimization: Only read the debug_buffer (small) instead of massive weights (36MB)
                        let debug = gpu.read_debug_buffer();
                        let nz_h = debug[9];
                        // We skip wq_sum during the loop for maximum speed,
                        // it can be queried at the end of the epoch or on sync.
                        gpu_stats = format!(" nz_h={:.4}", nz_h);
                    }
                }
                #[cfg(feature = "wgpu")]
                if let Some(gpu) = self.gpu_deq.as_ref() {
                    if let Some(ns) = gpu.read_tps_epoch_ns() {
                        let tps_gpu = (total_tokens as f64) / (ns / 1e9);
                        gpu_stats.push_str(&format!(" tps_gpu={:.1}", tps_gpu));
                    }
                }

                #[cfg(feature = "wgpu")]
                let display_loss = if num_chunks > 0 {
                    epoch_loss / num_chunks as f32
                } else {
                    0.0
                };
                #[cfg(not(feature = "wgpu"))]
                let display_loss = if num_chunks > 0 {
                    epoch_loss / num_chunks as f32
                } else {
                    0.0
                };

                println!(
                    "  epoch {epoch:>4}/{epochs}  loss={:.4}  lr={:.6}  tps={:>8.1}  time={:.2}s  tokens={} {}",
                    display_loss, current_lr, tps, elapsed, total_tokens, gpu_stats
                );
            }

            if save_every > 0 && (epoch + 1) % save_every == 0 && !checkpoint_path.is_empty() {
                if let Err(e) = self.save_checkpoint(checkpoint_path) {
                    eprintln!(
                        "[checkpoint] Error saving to '{}': {}",
                        checkpoint_path, e
                    );
                } else {
                    eprintln!(
                        "[checkpoint] Saved to '{}' (epoch {})",
                        checkpoint_path,
                        epoch + 1
                    );
                }
            }
        }
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Checkpointing: weights + optimizer state
    // ─────────────────────────────────────────────────────────────────────────

    /// Saves the full model plus optimizer state.
    /// Format: <path>.aidn (weights) + <path>.opt (Adam moments).
    pub fn save_checkpoint(&mut self, base_path: &str) -> std::io::Result<()> {
        // Model weights
        self.save_full(&format!("{base_path}.aidn"))?;

        // Sync GPU moments to CPU Adam before saving
        #[cfg(feature = "wgpu")]
        self.sync_gpu_moments_to_cpu();

        // Optimizer state (with moments already populated)
        self.optimizer
            .save_state(&format!("{base_path}.opt"))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Loads a full checkpoint (weights + optimizer state).
    pub fn load_checkpoint(base_path: &str) -> std::io::Result<Self> {
        let mut trainer = Self::load_full(&format!("{base_path}.aidn"))?;
        let opt_path = format!("{base_path}.opt");
        if std::path::Path::new(&opt_path).exists() {
            trainer
                .optimizer
                .load_state(&opt_path)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            // Restore Adam moments on GPU if available
            #[cfg(feature = "wgpu")]
            trainer.sync_cpu_moments_to_gpu();
        }
        Ok(trainer)
    }

    /// Downloads Adam moments from GPU to CPU Adam struct for saving.
    #[cfg(feature = "wgpu")]
    fn sync_gpu_moments_to_cpu(&mut self) {
        use nalgebra::{DMatrix, DVector};
        let gpu = match self.gpu_deq.as_ref() {
            Some(g) => g,
            None => return,
        };

        println!("[GPU-CHECK] Starting VRAM checksum before saving...");
        // Checkpoint checksum path: CPU is about to read GPU moments synchronously.
        gpu.device.poll(wgpu::Maintain::Wait);

        // LM Head moments
        if let Some(gpu_lm) = self.gpu_lm.as_ref() {
            if let Ok((m_w, v_w, m_b, v_b, m_g, v_g)) = gpu_lm.read_moments(&gpu.device, &gpu.queue)
            {
                let sum_mw: f32 = m_w.iter().map(|x| x.abs()).sum();
                let sum_vw: f32 = v_w.iter().map(|x| x.abs()).sum();

                // VRAM Checksum: If moments are exactly zero, something failed in the mapping
                if sum_mw == 0.0 && sum_vw == 0.0 && self.optimizer.step_count() > 10 {
                    panic!("\x1b[31m[CRITICAL ERROR]\x1b[0m VRAM checksum failed (LM moments=0). Aborting to protect checkpoint.");
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
                    panic!("\x1b[31m[CRITICAL ERROR]\x1b[0m VRAM checksum failed (EMB moments=0). Aborting.");
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

    /// Uploads Adam moments from CPU to GPU after loading a checkpoint.
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

    /// Saves the full model (Config + Reasoning + LmHead + Tokenizer) into a single .aidn
    pub fn save_full(&mut self, path: &str) -> std::io::Result<()> {
        use aideen_core::model::AidenModel;

        #[cfg(feature = "wgpu")]
        self.sync_inference_weights();

        let mut model = AidenModel::new(self.config.clone());

        // Pack Reasoning weights
        for (k, v) in self.reasoning.export_weights() {
            model.set_weight(&k, v);
        }

        // Pack LmHead weights
        for (k, v) in self.lm_head.export_weights() {
            model.set_weight(&k, v);
        }

        // Pack Tokenizer embeddings
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

        // Persist Stability Oracle state (v13.2)
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

    /// Loads the full model (Config + Reasoning + LmHead + Tokenizer) from a .aidn
    pub fn load_full(path: &str) -> std::io::Result<Self> {
        use aideen_core::model::AidenModel;

        let model = AidenModel::load(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Reconstruct Tokenizer
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

        // Restore the HF tokenizer (BPE) if available at standard paths.
        // Without this, encode() uses the empty char-level fallback and returns [].
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

        // Reconstruct Trainer
        let mut trainer = Trainer::from_tokenizer(tokenizer, 0.001); // default LR
        trainer.config = model.config.clone();

        // Import weights into Reasoning
        trainer
            .reasoning
            .import_weights(&model.weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Import weights into LmHead
        trainer
            .lm_head
            .import_weights(&model.weights)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        // Restore Stability Oracle state (v13.2)
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
            // Sync DEQ Core only if weights were uploaded/updated on GPU.
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

        // Sync Embeddings
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

    /// Explicitly configures the inference backend.
    /// - `prefer_gpu=true`: tries to activate GPU; falls back to CPU if unavailable.
    /// - `prefer_gpu=false`: forces CPU.
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

    /// In builds without `wgpu`, always uses CPU.
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
        let mut trainer = Trainer::from_tokenizer(tok, 0.01);
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
}
