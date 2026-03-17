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
    pub adaptive_cg_iters: u32,
    pub hit_hi_streak: u32,
    pub hit_lo_streak: u32,
    pub cg_res_hi_streak: u32,
    pub damping_boost_left: u32,
    pub emergency_left: u32,
    pub last_max_h: f32,
    pub max_h_growth_streak: u32,
    pub contractivity_hi_streak: u32,

    // --- v14 Temporal Memory State ---
    pub m_prev: Option<HSlots>,
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

    fn env_f32_default(name: &str, default: f32) -> f32 {
        Self::env_f32(name).unwrap_or(default)
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

    fn env_u32(name: &str) -> Option<u32> {
        std::env::var(name).ok().and_then(|v| v.parse::<u32>().ok())
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
            self.config.cg_iters = self.config.cg_iters.min(8).max(4);
            self.adaptive_max_iters = self.adaptive_max_iters.min(8).max(4);
            self.adaptive_cg_iters = self.adaptive_cg_iters.min(8).max(4);
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
            adaptive_max_iters: 6,
            adaptive_damping: 0.85,
            adaptive_cg_iters: 8,
            hit_hi_streak: 0,
            hit_lo_streak: 0,
            cg_res_hi_streak: 0,
            damping_boost_left: 0,
            emergency_left: 0,
            last_max_h: 0.0,
            max_h_growth_streak: 0,
            contractivity_hi_streak: 0,
            m_prev: None,
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
            adaptive_max_iters: 6,
            adaptive_damping: 0.85,
            adaptive_cg_iters: 8,
            hit_hi_streak: 0,
            hit_lo_streak: 0,
            cg_res_hi_streak: 0,
            damping_boost_left: 0,
            emergency_left: 0,
            last_max_h: 0.0,
            max_h_growth_streak: 0,
            contractivity_hi_streak: 0,
            m_prev: None,
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
            let safe_ctx = self.config.ctx_len.max(1024);
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

        #[cfg(feature = "wgpu")]
        self.sync_lm_head_from_gpu_if_needed();

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
        // CPU fallback real: forward DEQ + backward/update en CPU.
        let query = self.tokenizer.embed_context(context, self.config.ctx_len);
        let damping_eff = self.damping_eff();
        self.reasoning.damping = damping_eff;

        let mut h = if let Some(m) = &self.m_prev {
            m.clone()
        } else {
            self.reasoning.init(&query)
        };

        for _ in 0..self.adaptive_max_iters.max(1) {
            h = self.reasoning.step(&h, &query, None);
        }

        let m_next = self.reasoning.temporal_step(
            self.m_prev.as_ref().unwrap_or(&self.reasoning.init(&query)),
            &h,
        );
        self.m_prev = Some(m_next);

        #[cfg(feature = "wgpu")]
        {
            self.apply_training_update(context, target, &query, &h, None)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            self.apply_training_update(context, target, &query, &h, None)
        }
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

        // hist_gated is default — always enforce min_iters for stable history injection.
        {
            let min_iters = Self::env_u32("AIDEEN_HIST_MIN_ITERS").unwrap_or(20);
            if self.adaptive_max_iters < min_iters {
                self.adaptive_max_iters = min_iters;
            }
        }

        #[cfg(feature = "wgpu")]
        self.sync_lm_head_from_gpu_if_needed();

        #[cfg(feature = "wgpu")]
        if self.gpu_deq.is_some() {
            let gpu = self.take_gpu().expect("gpu_deq checked as Some");

            let out = (|| {
                self.ensure_gpu_trainers(&gpu);

                // Arreglo defensivo para evitar underflow si seq_len < ctx_len
                let seq_len = tokens.len().min(targets.len());
                let actual_ctx_len = seq_len.min(self.config.ctx_len);
                let ctx = &tokens[seq_len - actual_ctx_len..];
                let ctx_targets = &targets[seq_len - actual_ctx_len..];

                let Some(gpu_emb) = self.gpu_emb.as_ref() else {
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
                    ) = self.reasoning.history_params_gpu_layout();
                    gpu.upload_weights(
                        &gpu.queue,
                        self.reasoning.w_q.as_slice(),
                        self.reasoning.w_k.as_slice(),
                        self.reasoning.w_v.as_slice(),
                        self.reasoning.w_o.as_slice(),
                        self.reasoning.w_in.as_slice(),
                        self.reasoning.w_x.as_slice(),
                        self.reasoning.w_out.as_slice(),
                        self.reasoning.a_log.as_slice(),
                        self.reasoning.norm_scale.as_slice(),
                        w_hist_shared_rm.as_slice(),
                        hist_slot_scale_rm.as_slice(),
                        hist_slot_bias_rm.as_slice(),
                        hist_gate_logit.as_slice(),
                        slot_anchor_rm.as_slice(),
                        w_delta_rm.as_slice(),
                        b_delta.as_slice(),
                    );
                    self.gpu_weights_uploaded = true;
                    self.gpu_cg_weights_uploaded = true;
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

    fn apply_training_update(
        &mut self,
        context: &[u32],
        target: u32,
        query: &nalgebra::DVector<f32>,
        h: &HSlots,
        #[cfg(feature = "wgpu")] gpu_ctx: Option<&GpuDeqBackend>,
        #[cfg(not(feature = "wgpu"))] _gpu_ctx: Option<()>,
    ) -> f32 {
        // ③ Forward LmHead: H* → logits
        let logits = self.lm_head.forward(h);

        // ④ Loss
        let current_loss = loss::cross_entropy(&logits, target);

        // ⑤ Backward LmHead
        let dl_dlogits = loss::cross_entropy_grad(&logits, target);
        let h_pooled = self.lm_head.pool_h_star(h);
        let (lm_grads, dl_dh) =
            gradients::lmhead_backward(&dl_dlogits, &h_pooled, &self.lm_head.w, &self.lm_head.g);

        // Actualizar LmHead
        self.optimizer.tick();
        self.optimizer
            .step_matrix("lm_w", &mut self.lm_head.w, &lm_grads.dw);
        self.optimizer
            .step_vector("lm_b", &mut self.lm_head.b, &lm_grads.db);
        self.optimizer
            .step_vector("lm_g", &mut self.lm_head.g, &lm_grads.dg);
        #[cfg(feature = "wgpu")]
        {
            self.gpu_lm_weights_uploaded = false;
            self.lm_head_cpu_stale = false;
        }

        #[cfg(feature = "wgpu")]
        {
            if let (Some(gpu), Some(gpu_emb)) = (gpu_ctx, self.gpu_emb.as_ref()) {
                let emb_lr = self.optimizer.lr * self.training_config.emb_lr_mult;
                if gpu_emb
                    .apply_embedding_update(
                        &gpu.device,
                        &gpu.queue,
                        context,
                        self.config.ctx_len,
                        dl_dh.as_slice(),
                        emb_lr,
                        self.optimizer.beta1,
                        self.optimizer.beta2,
                        self.optimizer.eps,
                        self.optimizer.step_count() as u32,
                        self.training_config.ternary,
                    )
                    .is_ok()
                {
                    self.gpu_emb_weights_uploaded = true;
                } else {
                    self.gpu_emb_weights_uploaded = false;
                    return 0.0;
                }
            } else {
                // Fallback CPU cuando el build tiene `wgpu` pero el backend activo es CPU.
                let ctx_start = context.len().saturating_sub(self.config.ctx_len);
                let ctx = &context[ctx_start..];
                let ctx_len_f = ctx.len().max(1) as f32;
                let emb_lr = self.optimizer.lr * self.training_config.emb_lr_mult;
                for (pos, &tok) in ctx.iter().enumerate() {
                    let pos_weight = (pos + 1) as f32 / ctx_len_f;
                    for d in 0..self.config.d_r {
                        self.tokenizer.embeddings[(tok as usize, d)] -=
                            emb_lr * pos_weight * dl_dh[d];
                    }
                }
            }
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let ctx_start = context.len().saturating_sub(self.config.ctx_len);
            let ctx = &context[ctx_start..];
            let ctx_len_f = ctx.len().max(1) as f32;
            let emb_lr = self.optimizer.lr * self.training_config.emb_lr_mult;
            for (pos, &tok) in ctx.iter().enumerate() {
                let pos_weight = (pos + 1) as f32 / ctx_len_f;
                for d in 0..self.config.d_r {
                    self.tokenizer.embeddings[(tok as usize, d)] -= emb_lr * pos_weight * dl_dh[d];
                }
            }
        }

        // ⑥ Backward DEQ (Picard Adjoint)
        if self.config.train_deq {
            #[cfg(feature = "wgpu")]
            let mut deq_updated_on_gpu = false;
            #[cfg(feature = "wgpu")]
            let mut deq_touched_on_cpu = false;
            let v = {
                #[cfg(feature = "wgpu")]
                {
                    if let Some(gpu) = gpu_ctx {
                        let needs_cg_upload = !self.gpu_cg_weights_uploaded;
                        let h_star_flat = h.to_flat();
                        if let Ok(v_gpu_flat) = gpu.run_backward_deq(
                            1,
                            query.as_slice(),
                            &h_star_flat,
                            dl_dh.as_slice(),
                            self.reasoning.w_q.as_slice(),
                            self.reasoning.w_k.as_slice(),
                            self.reasoning.w_v.as_slice(),
                            self.reasoning.w_o.as_slice(),
                            self.reasoning.w_in.as_slice(),
                            self.reasoning.w_x.as_slice(),
                            self.reasoning.w_out.as_slice(),
                            self.reasoning.a_log.as_slice(),
                            self.reasoning.norm_scale.as_slice(),
                            self.config.cg_iters as u32,
                            needs_cg_upload,
                        ) {
                            self.gpu_cg_weights_uploaded = true;
                            let d_r = self.config.d_r;
                            nalgebra::DVector::from_iterator(
                                d_r,
                                v_gpu_flat.iter().take(d_r).copied(),
                            )
                        } else {
                            self.gpu_cg_weights_uploaded = false;
                            gradients::deq_implicit_grad(
                                &self.reasoning,
                                h,
                                query,
                                &dl_dh,
                                self.config.cg_iters,
                            )
                        }
                    } else {
                        gradients::deq_implicit_grad(
                            &self.reasoning,
                            h,
                            query,
                            &dl_dh,
                            self.config.cg_iters,
                        )
                    }
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    gradients::deq_implicit_grad(
                        &self.reasoning,
                        h,
                        query,
                        &dl_dh,
                        self.config.cg_iters,
                    )
                }
            };

            let grad_mat = v.clone() * query.transpose() * self.config.deq_grad_scale;
            let grad_vec = v * self.config.deq_grad_scale;

            #[cfg(feature = "wgpu")]
            if let Some(gpu) = gpu_ctx {
                let deq_lr_mult = Self::env_f32_default(
                    "AIDEEN_DEQ_LR_MULT",
                    self.training_config.deq_lr_mult,
                );
                let deq_lr = self.optimizer.lr * deq_lr_mult;
                if gpu
                    .apply_deq_sgd_update_and_sync_cg(
                        deq_lr,
                        grad_mat.as_slice(),
                        grad_vec.as_slice(),
                    )
                    .is_ok()
                {
                    self.gpu_weights_uploaded = true;
                    self.gpu_cg_weights_uploaded = true;
                    deq_updated_on_gpu = true;
                } else {
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
                    deq_touched_on_cpu = true;
                }
            } else {
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
                self.optimizer
                    .step_vector("deq_norm", &mut self.reasoning.norm_scale, &grad_vec);
                #[cfg(feature = "wgpu")]
                {
                    deq_touched_on_cpu = true;
                }
            }

            #[cfg(not(feature = "wgpu"))]
            {
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
                self.optimizer
                    .step_vector("deq_norm", &mut self.reasoning.norm_scale, &grad_vec);
            }

            #[cfg(feature = "wgpu")]
            {
                let renorm_every = self.config.renorm_every_steps.max(1);
                if self.optimizer.step_count() % renorm_every == 0 {
                    if deq_updated_on_gpu {
                        // Evita desync CPU<->GPU: no renormalizar en CPU si DEQ vive en VRAM.
                    } else {
                        self.reasoning.renormalize_weights();
                        deq_touched_on_cpu = true;
                    }
                }
                if deq_touched_on_cpu {
                    self.gpu_weights_uploaded = false;
                    self.gpu_cg_weights_uploaded = false;
                }
            }

            #[cfg(not(feature = "wgpu"))]
            {
                let renorm_every = self.config.renorm_every_steps.max(1);
                if self.optimizer.step_count() % renorm_every == 0 {
                    self.reasoning.renormalize_weights();
                }
            }
        }

        current_loss
    }

    fn apply_training_update_from_pooled(
        &mut self,
        context: &[u32],
        targets: &[u32],
        query: &nalgebra::DVector<f32>,
        h_pooled: &[f32],
        #[cfg(feature = "wgpu")] gpu_ctx: Option<&GpuDeqBackend>,
        #[cfg(not(feature = "wgpu"))] _gpu_ctx: Option<()>,
    ) -> f32 {
        self.optimizer.tick();

        let num_tokens = targets.len();
        if num_tokens == 0 {
            return 0.0;
        }
        let d_r = self.config.d_r;
        #[cfg(feature = "wgpu")]
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();

        let (current_loss, dl_dh) = {
            #[cfg(feature = "wgpu")]
            if let (Some(gpu), Some(gpu_lm)) = (gpu_ctx, self.gpu_lm.as_mut()) {
                let upload_w = !self.gpu_lm_weights_uploaded;
                let lm_lr = self.optimizer.lr * self.training_config.lm_lr_mult;
                if let Ok((loss, dl_bytes)) = gpu_lm.train_step_from_buffer(
                    &gpu.device,
                    &gpu.queue,
                    &gpu.bridge.hpooled_buf,
                    0, // We read the whole sequence from the buffer
                    &targets_u32,
                    lm_lr,
                    self.optimizer.step_count() as u32,
                    self.lm_head.w.as_slice(),
                    self.lm_head.b.as_slice(),
                    self.lm_head.g.as_slice(),
                    upload_w,
                    self.training_config.ternary,
                ) {
                    self.gpu_lm_weights_uploaded = true;
                    self.lm_head_cpu_stale = true;
                    let dl: Vec<f32> = bytemuck::cast_slice(&dl_bytes).to_vec();
                    (loss, nalgebra::DVector::from_vec(dl))
                } else {
                    // CPU fallback loop
                    let mut sum_loss = 0.0;
                    let mut accum_dl_dh = nalgebra::DVector::zeros(d_r);
                    let mut accum_dw =
                        nalgebra::DMatrix::zeros(self.lm_head.w.nrows(), self.lm_head.w.ncols());
                    let mut accum_db = nalgebra::DVector::zeros(self.lm_head.b.nrows());
                    let mut accum_dg = nalgebra::DVector::zeros(self.lm_head.g.nrows());

                    for i in 0..num_tokens {
                        let target = targets[i];
                        let h_slice = &h_pooled[i * d_r..(i + 1) * d_r];
                        let h_vec = nalgebra::DVector::from_column_slice(h_slice);
                        let logits = &self.lm_head.w * &h_vec + &self.lm_head.b;
                        sum_loss += loss::cross_entropy(&logits, target);
                        let dl_dlogits = loss::cross_entropy_grad(&logits, target);
                        let (lm_grads, dl_dh_t) = gradients::lmhead_backward(
                            &dl_dlogits,
                            &h_vec,
                            &self.lm_head.w,
                            &self.lm_head.g,
                        );
                        accum_dl_dh += dl_dh_t;
                        accum_dw += lm_grads.dw;
                        accum_db += lm_grads.db;
                        accum_dg += lm_grads.dg;
                    }

                    let norm = 1.0 / (num_tokens as f32);
                    sum_loss *= norm;
                    accum_dl_dh *= norm;
                    accum_dw *= norm;
                    accum_db *= norm;
                    accum_dg *= norm;

                    self.optimizer
                        .step_matrix("lm_w", &mut self.lm_head.w, &accum_dw);
                    self.optimizer
                        .step_vector("lm_b", &mut self.lm_head.b, &accum_db);
                    self.optimizer
                        .step_vector("lm_g", &mut self.lm_head.g, &accum_dg);
                    self.lm_head_cpu_stale = false;
                    (sum_loss, accum_dl_dh)
                }
            } else {
                let mut sum_loss = 0.0;
                let mut accum_dl_dh = nalgebra::DVector::zeros(d_r);
                let mut accum_dw =
                    nalgebra::DMatrix::zeros(self.lm_head.w.nrows(), self.lm_head.w.ncols());
                let mut accum_db = nalgebra::DVector::zeros(self.lm_head.b.nrows());
                let mut accum_dg = nalgebra::DVector::zeros(self.lm_head.g.nrows());

                for i in 0..num_tokens {
                    let target = targets[i];
                    let h_slice = &h_pooled[i * d_r..(i + 1) * d_r];
                    let h_vec = nalgebra::DVector::from_column_slice(h_slice);
                    let logits = &self.lm_head.w * &h_vec + &self.lm_head.b;
                    sum_loss += loss::cross_entropy(&logits, target);
                    let dl_dlogits = loss::cross_entropy_grad(&logits, target);
                    let (lm_grads, dl_dh_t) = gradients::lmhead_backward(
                        &dl_dlogits,
                        &h_vec,
                        &self.lm_head.w,
                        &self.lm_head.g,
                    );
                    accum_dl_dh += dl_dh_t;
                    accum_dw += lm_grads.dw;
                    accum_db += lm_grads.db;
                    accum_dg += lm_grads.dg;
                }

                let norm = 1.0 / (num_tokens as f32);
                sum_loss *= norm;
                accum_dl_dh *= norm;
                accum_dw *= norm;
                accum_db *= norm;
                accum_dg *= norm;

                self.optimizer
                    .step_matrix("lm_w", &mut self.lm_head.w, &accum_dw);
                self.optimizer
                    .step_vector("lm_b", &mut self.lm_head.b, &accum_db);
                self.optimizer
                    .step_vector("lm_g", &mut self.lm_head.g, &accum_dg);
                self.lm_head_cpu_stale = false;
                (sum_loss, accum_dl_dh)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                let mut sum_loss = 0.0;
                let mut accum_dl_dh = nalgebra::DVector::zeros(d_r);
                let mut accum_dw =
                    nalgebra::DMatrix::zeros(self.lm_head.w.nrows(), self.lm_head.w.ncols());
                let mut accum_db = nalgebra::DVector::zeros(self.lm_head.b.nrows());
                let mut accum_dg = nalgebra::DVector::zeros(self.lm_head.g.nrows());

                for i in 0..num_tokens {
                    let target = targets[i];
                    let h_slice = &h_pooled[i * d_r..(i + 1) * d_r];
                    let h_vec = nalgebra::DVector::from_column_slice(h_slice);
                    let logits = &self.lm_head.w * &h_vec + &self.lm_head.b;
                    sum_loss += loss::cross_entropy(&logits, target);
                    let dl_dlogits = loss::cross_entropy_grad(&logits, target);
                    let (lm_grads, dl_dh_t) = gradients::lmhead_backward(
                        &dl_dlogits,
                        &h_vec,
                        &self.lm_head.w,
                        &self.lm_head.g,
                    );
                    accum_dl_dh += dl_dh_t;
                    accum_dw += lm_grads.dw;
                    accum_db += lm_grads.db;
                    accum_dg += lm_grads.dg;
                }

                let norm = 1.0 / (num_tokens as f32);
                sum_loss *= norm;
                accum_dl_dh *= norm;
                accum_dw *= norm;
                accum_db *= norm;
                accum_dg *= norm;

                self.optimizer
                    .step_matrix("lm_w", &mut self.lm_head.w, &accum_dw);
                self.optimizer
                    .step_vector("lm_b", &mut self.lm_head.b, &accum_db);
                self.optimizer
                    .step_vector("lm_g", &mut self.lm_head.g, &accum_dg);
                (sum_loss, accum_dl_dh)
            }
        };

        #[cfg(feature = "wgpu")]
        {
            if let (Some(gpu), Some(gpu_emb)) = (gpu_ctx, self.gpu_emb.as_ref()) {
                let emb_lr = self.optimizer.lr * self.training_config.emb_lr_mult;
                if gpu_emb
                    .apply_embedding_update(
                        &gpu.device,
                        &gpu.queue,
                        context,
                        self.config.ctx_len,
                        dl_dh.as_slice(),
                        emb_lr,
                        self.optimizer.beta1,
                        self.optimizer.beta2,
                        self.optimizer.eps,
                        self.optimizer.step_count() as u32,
                        self.training_config.ternary,
                    )
                    .is_ok()
                {
                    self.gpu_emb_weights_uploaded = true;
                } else {
                    self.gpu_emb_weights_uploaded = false;
                    return 0.0;
                }
            } else {
                return 0.0;
            }
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let ctx_start = context.len().saturating_sub(self.config.ctx_len);
            let ctx = &context[ctx_start..];
            let ctx_len_f = ctx.len().max(1) as f32;
            let emb_lr = self.optimizer.lr * 0.5;
            for (pos, &tok) in ctx.iter().enumerate() {
                let pos_weight = (pos + 1) as f32 / ctx_len_f;
                for d in 0..self.config.d_r {
                    self.tokenizer.embeddings[(tok as usize, d)] -= emb_lr * pos_weight * dl_dh[d];
                }
            }
        }

        // Global Gradient Clipping (Safety Barrier)
        let mut dl_dh = dl_dh;
        Self::clip_grad_norm(&mut dl_dh, 1.0);

        if self.config.train_deq {
            #[cfg(feature = "wgpu")]
            let mut deq_updated_on_gpu = false;
            #[cfg(feature = "wgpu")]
            let mut deq_touched_on_cpu = false;
            #[cfg(feature = "wgpu")]
            let v = if let Some(gpu) = gpu_ctx {
                let needs_cg_upload = !self.gpu_cg_weights_uploaded;
                let num_tokens = context.len();
                let h_offset = (num_tokens.saturating_sub(1) * self.config.d_r * 4) as u64;

                if let Ok(v_gpu_flat) = gpu.run_backward_deq_from_forward_state(
                    1,
                    query.as_slice(),
                    h_offset,
                    dl_dh.as_slice(),
                    self.reasoning.w_q.as_slice(),
                    self.reasoning.w_k.as_slice(),
                    self.reasoning.w_v.as_slice(),
                    self.reasoning.w_o.as_slice(),
                    self.reasoning.w_in.as_slice(),
                    self.reasoning.w_x.as_slice(),
                    self.reasoning.w_out.as_slice(),
                    self.reasoning.a_log.as_slice(),
                    self.reasoning.norm_scale.as_slice(),
                    self.config.cg_iters as u32,
                    needs_cg_upload,
                ) {
                    self.gpu_cg_weights_uploaded = true;
                    nalgebra::DVector::from_iterator(
                        self.config.d_r,
                        v_gpu_flat.iter().take(self.config.d_r).copied(),
                    )
                } else {
                    self.gpu_cg_weights_uploaded = false;
                    dl_dh.clone()
                }
            } else {
                dl_dh.clone()
            };

            #[cfg(feature = "wgpu")]
            let v = {
                let mut v = v;
                Self::clip_grad_norm(&mut v, 1.0);
                v
            };

            #[cfg(not(feature = "wgpu"))]
            let v = {
                let mut v = dl_dh.clone();
                Self::clip_grad_norm(&mut v, 1.0);
                v
            };

            let grad_mat = v.clone() * query.transpose() * self.config.deq_grad_scale;
            let grad_vec = v * self.config.deq_grad_scale;

            #[cfg(feature = "wgpu")]
            if let Some(gpu) = gpu_ctx {
            let deq_lr_mult = Self::env_f32_default(
                "AIDEEN_DEQ_LR_MULT",
                self.training_config.deq_lr_mult,
            );
            let deq_lr = self.optimizer.lr * deq_lr_mult;
                if gpu
                    .apply_deq_sgd_update_and_sync_cg(
                        deq_lr,
                        grad_mat.as_slice(),
                        grad_vec.as_slice(),
                    )
                    .is_ok()
                {
                    self.gpu_weights_uploaded = true;
                    self.gpu_cg_weights_uploaded = true;
                    deq_updated_on_gpu = true;
                } else {
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
                    deq_touched_on_cpu = true;
                }
            } else {
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
                self.optimizer
                    .step_vector("deq_norm", &mut self.reasoning.norm_scale, &grad_vec);
                deq_touched_on_cpu = true;
            }

            #[cfg(not(feature = "wgpu"))]
            {
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
                self.optimizer
                    .step_vector("deq_norm", &mut self.reasoning.norm_scale, &grad_vec);
            }
            #[cfg(feature = "wgpu")]
            {
                let renorm_every = self.config.renorm_every_steps.max(1);
                if self.optimizer.step_count() % renorm_every == 0 {
                    if deq_updated_on_gpu {
                        // Evita desync CPU<->GPU: no renormalizar en CPU si DEQ vive en VRAM.
                    } else {
                        self.reasoning.renormalize_weights();
                        deq_touched_on_cpu = true;
                    }
                }
                if deq_touched_on_cpu {
                    self.gpu_weights_uploaded = false;
                    self.gpu_cg_weights_uploaded = false;
                }
                // LM head ya se actualizó en GPU; mantener cache viva.
            }

            #[cfg(not(feature = "wgpu"))]
            {
                let renorm_every = self.config.renorm_every_steps.max(1);
                if self.optimizer.step_count() % renorm_every == 0 {
                    self.reasoning.renormalize_weights();
                }
            }
        }
        // REMOVED: sync_inference_weights() — This triggered 196MB uploads every chunk!
        current_loss
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
            let single_query_mode = query.len() == self.config.d_r && num_tokens == 1;
            if single_query_mode {
                // train_step path: match CPU semantics by feeding the pooled query vector.
                let q_bytes = bytemuck::cast_slice(query.as_slice());
                gpu.queue.write_buffer(&gpu.bridge.s_buf, 0, q_bytes);
                gpu.queue.write_buffer(&gpu.cg_bridge.b_sin, 0, q_bytes);
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
                let _ = gpu_emb.prepare_sequence_gpu_only(&gpu.device, &gpu.queue, context);
                let mut conn_enc =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Connect Embedding -> DEQ Encoder"),
                        });
                let copy_size = (num_tokens as u64) * (self.config.d_r as u64) * 4;
                conn_enc.copy_buffer_to_buffer(
                    &gpu_emb.seq_buf,
                    0,
                    &gpu.bridge.s_buf,
                    0,
                    copy_size,
                );
                conn_enc.copy_buffer_to_buffer(
                    &gpu_emb.seq_buf,
                    0,
                    &gpu.cg_bridge.b_sin,
                    0,
                    copy_size,
                );
                gpu.queue.submit(Some(conn_enc.finish()));
            }

            // 2. DEQ Forward (GPU-Only) - v13.1 Adaptive
            let _ = gpu.run_forward(
                1,
                num_tokens as u32,
                self.adaptive_max_iters,
                damping_eff,
                epsilon,
            );
            let fw = gpu.read_debug_buffer();
            let heartbeat = if fw.len() > 10 { fw[10] } else { 1.0 };
            let max_delta = if fw.len() > 16 { fw[16] } else { 0.0 };
            let hit_count = if fw.len() > 15 { fw[15] } else { 0.0 };
            let contractivity = if fw.len() > 21 { fw[21] } else { 0.0 };
            let seq = heartbeat.max(1.0);
            let hit_ratio = hit_count.max(0.0) / seq;
            // DEQ-INVALID: only when the system FAILED to converge (maxΔ >> epsilon) while
            // also being non-contractive. Non-monotone convergence (contr transiently > 1
            // but maxΔ ≈ epsilon) is a normal property of non-linear Picard iterations and
            // does NOT indicate an invalid fixed point — the system DID find h*.
            let invalid_fixed_point =
                contractivity > 1.0 && max_delta > self.config.deq_epsilon * 10.0;
            if invalid_fixed_point {
                eprintln!(
                    "    [DEQ-INVALID] step={} contr={:.3} hit_ratio={:.3} maxΔ={:.3e} seq={:.0}",
                    self.optimizer.step_count(),
                    contractivity,
                    hit_ratio,
                    max_delta,
                    seq
                );
                self.emergency_left = 3;
                self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(24);
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
                    "[GPU-LM] Sincronizando pesos del LM Head... (abs_sum={:.4})",
                    w_sum
                );

                gpu_lm.upload_weights_only(&gpu.queue, w_raw, b_raw, g_raw);
                self.gpu_lm_weights_uploaded = true;
            }

            let current_loss = gpu_lm
                .train_step_no_readback(
                    &gpu.device,
                    &gpu.queue,
                    &gpu.bridge.hpooled_buf,
                    0,
                    &targets_u32,
                    lm_lr,
                    self.optimizer.step_count() as u32,
                    self.training_config.ternary,
                    true, // read_loss = true: pérdida síncrona del paso actual
                )
                .unwrap_or(0.0);
            if lm_lr > 0.0 {
                self.lm_head_cpu_stale = true;
            }

            // 4. Embedding Update from GPU dl_dh buffer (Moved to step 6 to avoid duplication)

            // 5. DEQ Reasoning Core Update (Picard Adjoint + Fused GPU Weight Update)
            if self.eval_mode {
                return current_loss;
            }

            if !self.frozen_deq {
                // ⑥ Backward DEQ — Picard Adjoint (GPU, siempre).
                // staged Picard llena fused_mix_buf con g_comb, luego apply_fused_deq_update
                // aplica el weight update completo en GPU. Un solo path, siempre correcto.
                let seq_len = targets_u32.len() as u32;
                let _ = gpu.run_staged_adjoint_picard_no_readback(
                    seq_len,
                    self.reasoning.damping,
                    self.config.cg_iters as u32,
                    Some(&gpu_lm.dl_dh_buf),
                    false,
                );
                let _ = gpu.apply_fused_deq_update(
                    base_lr,
                    self.config.deq_grad_scale,
                    self.training_config.ternary,
                    self.config.weight_decay,
                    seq_len,
                    self.reasoning.damping,
                );
                self.gpu_weights_uploaded = true;
                self.gpu_cg_weights_uploaded = true;

                // CPU PATH — por las dudas, si GPU no disponible.
                // Para activar: comentar el bloque GPU de arriba y descomentar este bloque.
                // Recalcula h* en CPU, corre CG, aplica weight update en CPU y re-sube a GPU.
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
                        let dl_dh_cpu = dl_dh_cpu_parity_opt
                            .as_ref()
                            .or(dl_dh_cpu_opt.as_ref());
                        if let Some(dl_dh_cpu) = dl_dh_cpu {
                            let dl_dh_gpu = if self.frozen_lm || force_cpu_lm_dldh {
                                match gpu_lm.read_dl_dh(
                                    &gpu.device,
                                    &gpu.queue,
                                    self.config.d_r,
                                ) {
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
                        self.config.cg_iters,
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
                    ) = self.reasoning.history_params_gpu_layout();
                    gpu.upload_weights(
                        &gpu.queue,
                        self.reasoning.w_q.as_slice(),
                        self.reasoning.w_k.as_slice(),
                        self.reasoning.w_v.as_slice(),
                        self.reasoning.w_o.as_slice(),
                        self.reasoning.w_in.as_slice(),
                        self.reasoning.w_x.as_slice(),
                        self.reasoning.w_out.as_slice(),
                        self.reasoning.a_log.as_slice(),
                        self.reasoning.norm_scale.as_slice(),
                        w_hist_shared_rm.as_slice(),
                        hist_slot_scale_rm.as_slice(),
                        hist_slot_bias_rm.as_slice(),
                        hist_gate_logit.as_slice(),
                        slot_anchor_rm.as_slice(),
                        w_delta_rm.as_slice(),
                        b_delta.as_slice(),
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

            // --- DIAGNÓSTICOS GPU (v13.1 Auto-Healing) ---
            if self.optimizer.step_count() % 10 == 0 {
                let d = gpu.read_cg_debug_buffer();
                let fw = gpu.read_debug_buffer();

                let _b0 = if d.len() > 1 { d[0] } else { 0.0 };
                let rs_cg = if d.len() > 2 { d[2] } else { 0.0 };

                let heartbeat = if fw.len() > 10 { fw[10] } else { 0.0 }; // seq
                let max_h = if fw.len() > 11 { fw[11] } else { 0.0 };
                let avg_iters = if fw.len() > 13 { fw[13] } else { 0.0 };
                let hit_count = if fw.len() > 15 { fw[15] } else { 0.0 };
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
                let hist0 = if fw.len() > 40 { fw[40] } else { 0.0 };
                let hist1 = if fw.len() > 41 { fw[41] } else { 0.0 };
                let hist2 = if fw.len() > 42 { fw[42] } else { 0.0 };
                let hist_anchor0 = if fw.len() > 43 { fw[43] } else { 0.0 };
                let hist_anchor1 = if fw.len() > 44 { fw[44] } else { 0.0 };
                let hist_rms_floor = if fw.len() > 45 { fw[45] } else { 0.0 };
                let hist_contr_floor = if fw.len() > 46 { fw[46] } else { 0.0 };
                let hist_inject = if fw.len() > 47 { fw[47] } else { 0.0 };
                let hist_minner_zero = if fw.len() > 48 { fw[48] } else { 0.0 };
                let hist_force_nomamba = if fw.len() > 49 { fw[49] } else { 0.0 };
                let hist_prelude_skip = if fw.len() > 50 { fw[50] } else { 0.0 };
                let hist_loop_force_nomamba = if fw.len() > 51 { fw[51] } else { 0.0 };

                let trunc_str = if trunc_flag >= 0.5 { "TRUNC" } else { "OK" };

                let seq = heartbeat.max(1.0);
                let hit = hit_count.max(0.0);
                let hit_ratio = hit / seq;

                // ---------- v13.2 Stability Oracle Logic ----------

                // 1. CG Adaptation
                if rs_cg > 1e-4 {
                    self.cg_res_hi_streak += 1;
                } else {
                    self.cg_res_hi_streak = 0;
                }
                if self.cg_res_hi_streak >= 3 {
                    self.adaptive_cg_iters = (self.adaptive_cg_iters + 4).min(20);
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

                // EMERGENCY Triggers: crecimiento rápido, NaNs, residuo inaceptable (>1e-2) o Divergencia (>1.05)
                if self.max_h_growth_streak >= 3
                    || max_h.is_nan()
                    || max_delta.is_nan()
                    || max_delta > 5e-1
                    || contractivity > 1.20
                {
                    self.emergency_left = 3; // 3 windows de debug (~30 steps)
                    self.adaptive_max_iters = (self.adaptive_max_iters + 2).min(24);
                    self.max_h_growth_streak = 0;
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
                    "    \x1b[90m[GPU-DEBUG] Step {:>2}: hit={:>3}/{:.0} ({:>5.1}%) contr={:.3} maxΔ={:.3e} rs_cg={:.1e} iters={:.1} cap={} damp={:.2} mode={} conv={} max_h={:.6} inj_rms={:.3e} hist_rms={:.3e} hist/inj={:.3e} mamba_rms={:.3e} q/k/v={:.3e}/{:.3e}/{:.3e} mix/attn={:.3e}/{:.3e} attn_max={:.3} attn_ent={:.3} comb_rms={:.3e} hist=[{:.3e},{:.3e},{:.3e}] anchor=[{:.3e},{:.3e}] floors=[{:.3e},{:.3e}] flags=[{:.0},{:.0},{:.0},{:.0},{:.0}] shared={} total={:.0}\x1b[0m",
                    self.optimizer.step_count() % 100,
                    hit_i,
                    seq,
                    100.0 * hit_ratio,
                    contractivity,
                    max_delta,
                    rs_cg,
                    avg_iters,
                    self.adaptive_max_iters,
                    damping_eff,
                    mode_str,
                    conv_str,
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
            self.config.cg_iters = if deq_progress < 0.25 {
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

            for i in (0..train_tokens.len()).step_by(ctx_len) {
                let end = (i + ctx_len).min(train_tokens.len());
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

                if num_chunks % 10 == 0 {
                    let interval_elapsed = interval_start.elapsed().as_secs_f32();
                    let instant_tps = interval_tokens as f32 / interval_elapsed.max(1e-9);

                    #[cfg(feature = "wgpu")]
                    let current_loss_disp = if let Some(gpu_lm) = self.gpu_lm.as_ref() {
                        if let Some(gpu) = self.gpu_deq.as_ref() {
                            gpu_lm.read_cached_loss(&gpu.device)
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

            // Drain GPU queue every epoch — prevents Metal command queue overflow and
            // makes elapsed/TPS measurements reflect actual GPU execution time.
            #[cfg(feature = "wgpu")]
            {
                if let Some(gpu) = self.gpu_deq.as_ref() {
                    gpu.device.poll(wgpu::Maintain::Wait);
                }
            }
            let elapsed = t_start.elapsed().as_secs_f32();
            // Usar tokens reales procesados, no num_chunks * ctx_len (que sobre-cuenta el último chunk).
            let tokens_processed = train_tokens.len();
            let tps = tokens_processed as f32 / elapsed.max(1e-9);

            if epoch % log_every == 0 {
                // GPU already idle (poll above) — read_cached_loss is near-instant here
                #[cfg(feature = "wgpu")]
                let display_loss = if let Some(gpu_lm) = self.gpu_lm.as_ref() {
                    if let Some(gpu) = self.gpu_deq.as_ref() {
                        gpu_lm.read_cached_loss(&gpu.device)
                    } else {
                        total_loss
                    }
                } else {
                    total_loss
                };
                #[cfg(not(feature = "wgpu"))]
                let display_loss = total_loss;
                println!(
                    "  epoch {epoch:>4}/{epochs}  loss={:.4}  lr={:.6}  tps={:>8.1}  time={:.2}s",
                    display_loss, current_lr, tps, elapsed
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
                    self.reasoning.w_q.as_slice(),
                    self.reasoning.w_k.as_slice(),
                    self.reasoning.w_v.as_slice(),
                    self.reasoning.w_o.as_slice(),
                    self.reasoning.w_in.as_slice(),
                    self.reasoning.w_x.as_slice(),
                    self.reasoning.w_out.as_slice(),
                    self.reasoning.a_log.as_slice(),
                    self.reasoning.norm_scale.as_slice(),
                    needs_upload,
                ) {
                    self.gpu_weights_uploaded = true;
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
                    self.reasoning.w_q.as_slice(),
                    self.reasoning.w_k.as_slice(),
                    self.reasoning.w_v.as_slice(),
                    self.reasoning.w_o.as_slice(),
                    self.reasoning.w_in.as_slice(),
                    self.reasoning.w_x.as_slice(),
                    self.reasoning.w_out.as_slice(),
                    self.reasoning.a_log.as_slice(),
                    self.reasoning.norm_scale.as_slice(),
                    needs_upload,
                ) {
                    self.gpu_weights_uploaded = true;
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

        for epoch in 0..epochs {
            let t_start = std::time::Instant::now();
            let current_lr = self.cosine_lr(epoch, epochs);
            self.optimizer.lr = current_lr;

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
            self.config.cg_iters = if deq_progress < 0.25 {
                4
            } else if deq_progress < 0.60 {
                6
            } else {
                8
            };

            let mut file = std::fs::File::open(path)?;
            let mut epoch_loss = 0.0f32;
            let mut num_chunks = 0usize;
            let mut total_tokens = 0usize;
            // Buffer de tokens no consumidos del chunk anterior (para ventana solapada).
            let mut carry: Vec<u32> = Vec::new();
            let mut last_save_time = std::time::Instant::now();

            loop {
                // Prepend carry del chunk anterior + leer nuevos bytes
                let carry_bytes = carry.len() * 4;
                let need_bytes = chunk_bytes.saturating_sub(carry_bytes);
                let mut buf = vec![0u8; need_bytes];
                let n = file.read(&mut buf)?;
                if n == 0 && carry.is_empty() {
                    break;
                }

                // Reset de estado estricto en límites de documento (opcional según eos_token)
                if eos_token != 0 && carry.len() > 0 && carry[0] == eos_token {
                    self.reset_state();
                }

                let new_tokens: Vec<u32> = if n > 0 {
                    bytemuck::cast_slice(&buf[..n & !3]).to_vec()
                } else {
                    Vec::new()
                };

                let mut tokens: Vec<u32> = carry.drain(..).collect();
                tokens.extend_from_slice(&new_tokens);

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
                            // --- VALIDATION PATH (Every 20 segments) ---
                            let is_val = num_chunks % 20 == 0;
                            if is_val {
                                self.eval_mode = true;
                            }

                            // reset_state=true en sub-secuencias después de EOS (seg_start>0).
                            let eps = self.progressive_epsilon(epoch, epochs);
                            let loss = self.train_sequence(train_seg, tgt_seg, seg_start > 0, eps);

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
                            total_tokens += train_seg.len();
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
                carry = tokens[overlap_start..].to_vec();

                if n == 0 {
                    break; // EOF
                }

                // Mini-log cada 10 chunks para ver progreso en tiempo real
                if num_chunks % 10 == 0 && num_chunks > 0 {
                    let elapsed = t_start.elapsed().as_secs_f32();
                    let tps = total_tokens as f32 / elapsed.max(1e-9);
                    // Métrica principal: promedio acumulado real de train (no cache GPU).
                    let current_loss = epoch_loss / num_chunks as f32;

                    println!(
                        "    \x1b[95m[progress]\x1b[0m chunk {:>5}  \x1b[92mloss={:.4}\x1b[0m  \x1b[96mtps={:>8.1}\x1b[0m  \x1b[90mtime={:.1}s\x1b[0m",
                        num_chunks, current_loss, tps, elapsed
                    );

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

            // Vaciar cola GPU al final de cada epoch.
            #[cfg(feature = "wgpu")]
            if let Some(gpu) = self.gpu_deq.as_ref() {
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
                if let Some(gpu) = self.gpu_deq.as_ref() {
                    // Optimizamos: Solo leemos el debug_buffer (pequeño) en lugar de pesos masivos (36MB)
                    let debug = gpu.read_debug_buffer();
                    let nz_h = debug[9];
                    // El wq_sum lo omitimos durante el loop para máxima velocidad,
                    // se puede consultar al final del epoch o en sync.
                    gpu_stats = format!(" nz_h={:.4}", nz_h);
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
            "adaptive_cg_iters".to_string(),
            self.adaptive_cg_iters.to_string(),
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
            .get("adaptive_cg_iters")
            .and_then(|s| s.parse().ok())
        {
            trainer.adaptive_cg_iters = v;
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
            // Sincronizar DEQ Core si se entrenó
            if let Ok((wq, wk, wv, wo, win, wx, wout, alog, nscale)) = gpu.read_weights() {
                let to_mat = |vec: Vec<f32>| {
                    let d_r = self.config.d_r;
                    nalgebra::DMatrix::from_column_slice(d_r, d_r, &vec)
                };
                self.reasoning.w_q = to_mat(wq);
                self.reasoning.w_k = to_mat(wk);
                self.reasoning.w_v = to_mat(wv);
                self.reasoning.w_o = to_mat(wo);
                self.reasoning.w_in = to_mat(win);
                self.reasoning.w_x = to_mat(wx);
                self.reasoning.w_out = to_mat(wout);
                self.reasoning.a_log = nalgebra::DVector::from_column_slice(&alog);
                self.reasoning.norm_scale = nalgebra::DVector::from_column_slice(&nscale);
                self.gpu_weights_uploaded = true; // Weights are still on GPU, just synced to CPU
                self.gpu_cg_weights_uploaded = true;
            }
        }

        // Sincronizar Embeddings
        if let Some(gpu_emb) = self.gpu_emb.as_ref() {
            if let Some(gpu) = self.gpu_deq.as_ref() {
                if let Ok(emb_data) = gpu_emb.read_weights(&gpu.device, &gpu.queue) {
                    if emb_data.len() == self.tokenizer.embeddings.len() {
                        self.tokenizer.embeddings.copy_from_slice(&emb_data);
                        self.gpu_emb_weights_uploaded = true; // Still on GPU
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
        let mut trainer = Trainer::from_tokenizer(tok, 0.01);
        trainer.config.max_deq_iters = 20;
        trainer.config.cg_iters = 5;
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
