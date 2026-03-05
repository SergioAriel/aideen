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

use crate::{
    gradients, lm_head::LmHead, loss, mamba_slot_reasoning::MambaSlotReasoning, optimizer::Adam,
    tokenizer::Tokenizer,
};

#[cfg(feature = "wgpu")]
use crate::gpu_deq::GpuDeqBackend;
#[cfg(feature = "wgpu")]
use crate::gpu_embedding::GpuEmbeddingTrainer;
#[cfg(feature = "wgpu")]
use crate::gpu_lm_head::GpuLmHeadTrainer;

/// Configuración del training (hiperparámetros).
pub struct TrainingConfig {
    pub lr: f32,
    /// LR mínimo al final del cosine schedule (default: lr/10).
    pub lr_min: f32,
    pub epochs: usize,
    pub log_every: usize,
    /// Warmup epochs: LR sube linealmente de lr_min a lr.
    pub warmup_epochs: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            lr_min: 0.0001,
            epochs: 100,
            log_every: 10,
            warmup_epochs: 3,
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
}

impl Trainer {
    /// Crea un Trainer con un tokenizer pre-construido.
    pub fn from_tokenizer(tokenizer: Tokenizer, lr: f32) -> Self {
        let config = tokenizer.config.clone();

        #[cfg(feature = "wgpu")]
        let gpu_deq = GpuDeqBackend::new_blocking(config.clone());

        Self {
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
        }
    }

    /// Ejecuta un paso de entrenamiento dado un slice de tokens.
    /// `context`: tokens de contexto (input)
    /// `target`: token a predecir
    pub fn train_step(&mut self, context: &[u32], target: u32) -> f32 {
        #[cfg(feature = "wgpu")]
        self.sync_lm_head_from_gpu_if_needed();

        #[cfg(feature = "wgpu")]
        if self.gpu_deq.is_some() {
            let gpu = self.gpu_deq.take().expect("gpu_deq checked as Some");
            if self.gpu_lm.is_none() {
                self.gpu_lm = Some(GpuLmHeadTrainer::new(
                    &gpu.device,
                    self.lm_head.b.len(),
                    self.config.clone(),
                ));
            }
            if self.gpu_emb.is_none() {
                self.gpu_emb = Some(GpuEmbeddingTrainer::new(
                    &gpu.device,
                    self.tokenizer.vocab_size(),
                    self.config.ctx_len.max(1),
                    self.config.clone(),
                ));
            }
            let Some(gpu_emb) = self.gpu_emb.as_ref() else {
                self.gpu_deq = Some(gpu);
                return 0.0;
            };
            let emb_needs_upload = !self.gpu_emb_weights_uploaded;
            let (s_sequence, query_vec) = match gpu_emb.prepare_sequence_and_query(
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
                    return 0.0;
                }
            };
            let query = nalgebra::DVector::from_vec(query_vec);
            let needs_upload = !self.gpu_weights_uploaded;
            let step_loss = match gpu.run_forward_deq_no_readback(
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
                Ok(()) => {
                    self.gpu_weights_uploaded = true;
                    Some(self.apply_training_update_from_gpu_buffers(
                        context,
                        &[target],
                        &query,
                        #[cfg(feature = "wgpu")]
                        Some(&gpu),
                        #[cfg(not(feature = "wgpu"))]
                        None,
                    ))
                }
                Err(_) => {
                    self.gpu_weights_uploaded = false;
                    None
                }
            };
            self.gpu_deq = Some(gpu);
            if let Some(loss) = step_loss {
                return loss;
            }
            return 0.0;
        }
        0.0
    }

    /// Paso de entrenamiento para una secuencia completa (Sequence Fusing).
    /// Procesa 1..N tokens en una única ráfaga GPU.
    pub fn train_sequence(&mut self, tokens: &[u32], targets: &[u32]) -> f32 {
        let steps = tokens.len().min(targets.len());
        if steps == 0 {
            return 0.0;
        }

        #[cfg(feature = "wgpu")]
        if self.gpu_deq.is_some() {
            let gpu = self.gpu_deq.take().expect("gpu_deq checked as Some");
            if self.gpu_lm.is_none() {
                self.gpu_lm = Some(GpuLmHeadTrainer::new(
                    &gpu.device,
                    self.lm_head.b.len(),
                    self.config.clone(),
                ));
            }
            if self.gpu_emb.is_none() {
                self.gpu_emb = Some(GpuEmbeddingTrainer::new(
                    &gpu.device,
                    self.tokenizer.vocab_size(),
                    self.config.ctx_len.max(1),
                    self.config.clone(),
                ));
            }
            let mut _loss_acc = 0.0f32;
            let mut _gpu_ok = true;
            let mut _gpu_steps = 0usize;
            let ctx_len = tokens.len().min(self.config.ctx_len);
            let ctx = &tokens[tokens.len() - ctx_len..];
            let ctx_targets = &targets[targets.len() - ctx_len..];

            let Some(gpu_emb) = self.gpu_emb.as_ref() else {
                self.gpu_deq = Some(gpu);
                return 0.0;
            };

            let emb_needs_upload = !self.gpu_emb_weights_uploaded;
            let (s_sequence, query_vec) = match gpu_emb.prepare_sequence_and_query(
                &gpu.device,
                &gpu.queue,
                ctx,
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
                    return 0.0;
                }
            };

            let needs_upload = !self.gpu_weights_uploaded;
            let step_loss = match gpu.run_forward_deq_no_readback(
                1,
                ctx.len() as u32,
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
                Ok(()) => {
                    self.gpu_weights_uploaded = true;
                    // Intentamos usar la ruta de alto rendimiento (Zero Readback)
                    if let Ok(()) = self.apply_training_update_high_throughput(
                        ctx,
                        ctx_targets,
                        #[cfg(feature = "wgpu")]
                        Some(&gpu),
                        #[cfg(not(feature = "wgpu"))]
                        None,
                    ) {
                        // En modo de alto rendimiento devolvemos 0.0 para no bloquear con readbacks.
                        Some(0.0)
                    } else {
                        // Fallback a la ruta con readback si algo falla
                        let query = nalgebra::DVector::from_vec(query_vec);
                        let loss = self.apply_training_update_from_gpu_buffers(
                            ctx,
                            ctx_targets,
                            &query,
                            #[cfg(feature = "wgpu")]
                            Some(&gpu),
                            #[cfg(not(feature = "wgpu"))]
                            None,
                        );
                        Some(loss)
                    }
                }
                _ => {
                    self.gpu_weights_uploaded = false;
                    None
                }
            };

            self.gpu_deq = Some(gpu);
            if let Some(loss) = step_loss {
                return loss;
            }
        }
        0.0
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
        let (lm_grads, dl_dh) = gradients::lmhead_backward(&dl_dlogits, &h_pooled, &self.lm_head.w);

        // Actualizar LmHead
        self.optimizer.tick();
        self.optimizer
            .step_matrix("lm_w", &mut self.lm_head.w, &lm_grads.dw);
        self.optimizer
            .step_vector("lm_b", &mut self.lm_head.b, &lm_grads.db);
        #[cfg(feature = "wgpu")]
        {
            self.gpu_lm_weights_uploaded = false;
            self.lm_head_cpu_stale = false;
        }

        #[cfg(feature = "wgpu")]
        {
            if let (Some(gpu), Some(gpu_emb)) = (gpu_ctx, self.gpu_emb.as_ref()) {
                if gpu_emb
                    .apply_embedding_update(
                        &gpu.device,
                        &gpu.queue,
                        context,
                        dl_dh.as_slice(),
                        self.optimizer.lr,
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

        // ⑥ Backward DEQ (Implicit Differentiation via CG)
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
                if gpu
                    .apply_deq_sgd_update_and_sync_cg(
                        self.optimizer.lr,
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
                    self.optimizer
                        .step_matrix("deq_wx", &mut self.reasoning.w_x, &grad_mat);
                    self.optimizer
                        .step_matrix("deq_wout", &mut self.reasoning.w_out, &grad_mat);
                    self.optimizer
                        .step_vector("deq_alog", &mut self.reasoning.a_log, &grad_vec);
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
                    .step_matrix("deq_wx", &mut self.reasoning.w_x, &grad_mat);
                self.optimizer
                    .step_matrix("deq_wout", &mut self.reasoning.w_out, &grad_mat);
                self.optimizer
                    .step_vector("deq_alog", &mut self.reasoning.a_log, &grad_vec);
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
                    .step_matrix("deq_wx", &mut self.reasoning.w_x, &grad_mat);
                self.optimizer
                    .step_matrix("deq_wout", &mut self.reasoning.w_out, &grad_mat);
                self.optimizer
                    .step_vector("deq_alog", &mut self.reasoning.a_log, &grad_vec);
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
            if let (Some(gpu), Some(gpu_lm)) = (gpu_ctx, self.gpu_lm.as_ref()) {
                let upload_w = !self.gpu_lm_weights_uploaded;
                if let Ok((loss, dl_bytes)) = gpu_lm.train_step_from_buffer(
                    &gpu.device,
                    &gpu.queue,
                    &gpu.bridge.hnext_buf,
                    0, // We read the whole sequence from the buffer
                    &targets_u32,
                    self.optimizer.lr,
                    self.optimizer.step_count() as u32,
                    self.lm_head.w.as_slice(),
                    self.lm_head.b.as_slice(),
                    upload_w,
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

                    for i in 0..num_tokens {
                        let target = targets[i];
                        let h_slice = &h_pooled[i * d_r..(i + 1) * d_r];
                        let h_vec = nalgebra::DVector::from_column_slice(h_slice);
                        let logits = &self.lm_head.w * &h_vec + &self.lm_head.b;
                        sum_loss += loss::cross_entropy(&logits, target);
                        let dl_dlogits = loss::cross_entropy_grad(&logits, target);
                        let (lm_grads, dl_dh_t) =
                            gradients::lmhead_backward(&dl_dlogits, &h_vec, &self.lm_head.w);
                        accum_dl_dh += dl_dh_t;
                        accum_dw += lm_grads.dw;
                        accum_db += lm_grads.db;
                    }

                    let norm = 1.0 / (num_tokens as f32);
                    sum_loss *= norm;
                    accum_dl_dh *= norm;
                    accum_dw *= norm;
                    accum_db *= norm;

                    self.optimizer
                        .step_matrix("lm_w", &mut self.lm_head.w, &accum_dw);
                    self.optimizer
                        .step_vector("lm_b", &mut self.lm_head.b, &accum_db);
                    self.lm_head_cpu_stale = false;
                    (sum_loss, accum_dl_dh)
                }
            } else {
                let mut sum_loss = 0.0;
                let mut accum_dl_dh = nalgebra::DVector::zeros(d_r);
                let mut accum_dw =
                    nalgebra::DMatrix::zeros(self.lm_head.w.nrows(), self.lm_head.w.ncols());
                let mut accum_db = nalgebra::DVector::zeros(self.lm_head.b.nrows());

                for i in 0..num_tokens {
                    let target = targets[i];
                    let h_slice = &h_pooled[i * d_r..(i + 1) * d_r];
                    let h_vec = nalgebra::DVector::from_column_slice(h_slice);
                    let logits = &self.lm_head.w * &h_vec + &self.lm_head.b;
                    sum_loss += loss::cross_entropy(&logits, target);
                    let dl_dlogits = loss::cross_entropy_grad(&logits, target);
                    let (lm_grads, dl_dh_t) =
                        gradients::lmhead_backward(&dl_dlogits, &h_vec, &self.lm_head.w);
                    accum_dl_dh += dl_dh_t;
                    accum_dw += lm_grads.dw;
                    accum_db += lm_grads.db;
                }

                let norm = 1.0 / (num_tokens as f32);
                sum_loss *= norm;
                accum_dl_dh *= norm;
                accum_dw *= norm;
                accum_db *= norm;

                self.optimizer
                    .step_matrix("lm_w", &mut self.lm_head.w, &accum_dw);
                self.optimizer
                    .step_vector("lm_b", &mut self.lm_head.b, &accum_db);
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

                for i in 0..num_tokens {
                    let target = targets[i];
                    let h_slice = &h_pooled[i * d_r..(i + 1) * d_r];
                    let h_vec = nalgebra::DVector::from_column_slice(h_slice);
                    let logits = &self.lm_head.w * &h_vec + &self.lm_head.b;
                    sum_loss += loss::cross_entropy(&logits, target);
                    let dl_dlogits = loss::cross_entropy_grad(&logits, target);
                    let (lm_grads, dl_dh_t) =
                        gradients::lmhead_backward(&dl_dlogits, &h_vec, &self.lm_head.w);
                    accum_dl_dh += dl_dh_t;
                    accum_dw += lm_grads.dw;
                    accum_db += lm_grads.db;
                }

                let norm = 1.0 / (num_tokens as f32);
                sum_loss *= norm;
                accum_dl_dh *= norm;
                accum_dw *= norm;
                accum_db *= norm;

                self.optimizer
                    .step_matrix("lm_w", &mut self.lm_head.w, &accum_dw);
                self.optimizer
                    .step_vector("lm_b", &mut self.lm_head.b, &accum_db);
                (sum_loss, accum_dl_dh)
            }
        };

        #[cfg(feature = "wgpu")]
        {
            if let (Some(gpu), Some(gpu_emb)) = (gpu_ctx, self.gpu_emb.as_ref()) {
                if gpu_emb
                    .apply_embedding_update(
                        &gpu.device,
                        &gpu.queue,
                        context,
                        dl_dh.as_slice(),
                        self.optimizer.lr,
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

            #[cfg(not(feature = "wgpu"))]
            let v = dl_dh.clone();

            let grad_mat = v.clone() * query.transpose() * self.config.deq_grad_scale;
            let grad_vec = v * self.config.deq_grad_scale;

            #[cfg(feature = "wgpu")]
            if let Some(gpu) = gpu_ctx {
                if gpu
                    .apply_deq_sgd_update_and_sync_cg(
                        self.optimizer.lr,
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
                    self.optimizer
                        .step_matrix("deq_wx", &mut self.reasoning.w_x, &grad_mat);
                    self.optimizer
                        .step_matrix("deq_wout", &mut self.reasoning.w_out, &grad_mat);
                    self.optimizer
                        .step_vector("deq_alog", &mut self.reasoning.a_log, &grad_vec);
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
                    .step_matrix("deq_wx", &mut self.reasoning.w_x, &grad_mat);
                self.optimizer
                    .step_matrix("deq_wout", &mut self.reasoning.w_out, &grad_mat);
                self.optimizer
                    .step_vector("deq_alog", &mut self.reasoning.a_log, &grad_vec);
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
                    .step_matrix("deq_wx", &mut self.reasoning.w_x, &grad_mat);
                self.optimizer
                    .step_matrix("deq_wout", &mut self.reasoning.w_out, &grad_mat);
                self.optimizer
                    .step_vector("deq_alog", &mut self.reasoning.a_log, &grad_vec);
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
        current_loss
    }

    fn apply_training_update_from_gpu_buffers(
        &mut self,
        context: &[u32],
        targets: &[u32],
        query: &nalgebra::DVector<f32>,
        #[cfg(feature = "wgpu")] gpu_ctx: Option<&GpuDeqBackend>,
        #[cfg(not(feature = "wgpu"))] _gpu_ctx: Option<()>,
    ) -> f32 {
        #[cfg(feature = "wgpu")]
        if let (Some(gpu), Some(gpu_lm)) = (gpu_ctx, self.gpu_lm.as_ref()) {
            self.optimizer.tick();
            let num_tokens = targets.len();
            if num_tokens == 0 {
                return 0.0;
            }
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
            let upload_w = !self.gpu_lm_weights_uploaded;

            if let Ok((loss, dl_bytes)) = gpu_lm.train_step_from_buffer(
                &gpu.device,
                &gpu.queue,
                &gpu.bridge.hnext_buf,
                0, // Sequential read from start
                &targets_u32,
                self.optimizer.lr,
                self.optimizer.step_count() as u32,
                self.lm_head.w.as_slice(),
                self.lm_head.b.as_slice(),
                upload_w,
            ) {
                self.gpu_lm_weights_uploaded = true;
                self.lm_head_cpu_stale = true;

                let dl_dh = nalgebra::DVector::from_vec(bytemuck::cast_slice(&dl_bytes).to_vec());
                let current_loss = loss;

                // 1. Update Embeddings
                if let Some(gpu_emb) = self.gpu_emb.as_ref() {
                    if gpu_emb
                        .apply_embedding_update(
                            &gpu.device,
                            &gpu.queue,
                            context,
                            dl_dh.as_slice(),
                            self.optimizer.lr,
                        )
                        .is_ok()
                    {
                        self.gpu_emb_weights_uploaded = true;
                    }
                }

                // 2. Update DEQ reasoning core
                if self.config.train_deq {
                    let mut deq_updated_on_gpu = false;
                    let mut deq_touched_on_cpu = false;
                    let needs_cg_upload = !self.gpu_cg_weights_uploaded;

                    // Solo usamos el gradiente del último token para el retro-propagación implícita
                    // (aproximación simplificada para secuencias o usar el promedio dl_dh)
                    let h_offset_last = (num_tokens.saturating_sub(1) * self.config.d_r * 4) as u64;

                    if let Ok(v_gpu_flat) = gpu.run_backward_deq_from_forward_state(
                        1,
                        query.as_slice(),
                        h_offset_last,
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
                        let v = nalgebra::DVector::from_iterator(
                            d_r,
                            v_gpu_flat.iter().take(d_r).copied(),
                        );

                        // Extraer solo la query correspondiente al último token
                        let query_last = nalgebra::DVector::from_iterator(
                            d_r,
                            query.as_slice().chunks(d_r).last().unwrap().iter().copied(),
                        );

                        let grad_mat =
                            v.clone() * query_last.transpose() * self.config.deq_grad_scale;
                        let grad_vec = v * self.config.deq_grad_scale;

                        if gpu
                            .apply_deq_sgd_update_and_sync_cg(
                                self.optimizer.lr,
                                grad_mat.as_slice(),
                                grad_vec.as_slice(),
                            )
                            .is_ok()
                        {
                            self.gpu_weights_uploaded = true;
                            self.gpu_cg_weights_uploaded = true;
                            deq_updated_on_gpu = true;
                        } else {
                            // Fallback CPU Step
                            self.optimizer.step_matrix(
                                "deq_wq",
                                &mut self.reasoning.w_q,
                                &grad_mat,
                            );
                            self.optimizer.step_matrix(
                                "deq_wk",
                                &mut self.reasoning.w_k,
                                &grad_mat,
                            );
                            self.optimizer.step_matrix(
                                "deq_wv",
                                &mut self.reasoning.w_v,
                                &grad_mat,
                            );
                            self.optimizer.step_matrix(
                                "deq_wo",
                                &mut self.reasoning.w_o,
                                &grad_mat,
                            );
                            self.optimizer.step_matrix(
                                "deq_win",
                                &mut self.reasoning.w_in,
                                &grad_mat,
                            );
                            self.optimizer.step_matrix(
                                "deq_wx",
                                &mut self.reasoning.w_x,
                                &grad_mat,
                            );
                            self.optimizer.step_matrix(
                                "deq_wout",
                                &mut self.reasoning.w_out,
                                &grad_mat,
                            );
                            self.optimizer.step_vector(
                                "deq_alog",
                                &mut self.reasoning.a_log,
                                &grad_vec,
                            );
                            self.optimizer.step_vector(
                                "deq_norm",
                                &mut self.reasoning.norm_scale,
                                &grad_vec,
                            );
                            deq_touched_on_cpu = true;
                        }
                    }

                    // Renormalización opcional
                    let renorm_every = self.config.renorm_every_steps.max(1);
                    if self.optimizer.step_count() % renorm_every == 0 {
                        if !deq_updated_on_gpu {
                            self.reasoning.renormalize_weights();
                            deq_touched_on_cpu = true;
                        }
                    }
                    if deq_touched_on_cpu {
                        self.gpu_weights_uploaded = false;
                        self.gpu_cg_weights_uploaded = false;
                    }
                }

                return current_loss;
            }
        }
        0.0
    }

    /// La "Ruta Crítica" de alto rendimiento.
    /// Ejecuta todo el paso de entrenamiento (Forward, Backward, SGD) en la GPU sin readbacks.
    /// Ideal para superar los 15,000 tokens por segundo al eliminar la latencia CPU-GPU.
    pub fn apply_training_update_high_throughput(
        &mut self,
        context: &[u32],
        targets: &[u32],
        #[cfg(feature = "wgpu")] gpu_ctx: Option<&GpuDeqBackend>,
        #[cfg(not(feature = "wgpu"))] _gpu_ctx: Option<()>,
    ) -> Result<(), String> {
        #[cfg(feature = "wgpu")]
        if let (Some(gpu), Some(gpu_lm), Some(gpu_emb)) =
            (gpu_ctx, self.gpu_lm.as_ref(), self.gpu_emb.as_ref())
        {
            // self.optimizer.tick(); // Quitamos para evitar doble tick si ya se hizo fuera
            let num_tokens = targets.len();
            if num_tokens == 0 {
                return Ok(());
            }
            let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();

            // 1. LM Head Train (Compute dL/dh on GPU)
            gpu_lm.train_step_no_readback(
                &gpu.device,
                &gpu.queue,
                &gpu.bridge.hnext_buf,
                0,
                &targets_u32,
                self.optimizer.lr,
                self.optimizer.step_count() as u32,
            )?;
            self.gpu_lm_weights_uploaded = true;

            // 2. Embedding update directly from GPU dl_dh buffer
            gpu_emb.apply_embedding_update_from_buffer(
                &gpu.device,
                &gpu.queue,
                context,
                &gpu_lm.dl_dh_buf,
                self.optimizer.lr,
            )?;
            self.gpu_emb_weights_uploaded = true;

            // 3. DEQ Backward & Fused Update
            if self.config.train_deq {
                // Backprop (CG Solver)
                gpu.run_backward_no_readback(
                    1, // batch
                    num_tokens as u32,
                    &gpu_lm.dl_dh_buf,
                    self.config.cg_iters as u32,
                )?;

                // Rank-1 Spectral update
                gpu.apply_fused_deq_update(self.optimizer.lr, self.config.deq_grad_scale)?;

                self.gpu_weights_uploaded = true;
                self.gpu_cg_weights_uploaded = true;
            }

            return Ok(());
        }
        Err("High Throughput mode requires WGPU".to_string())
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

    /// Ejecuta el training loop sobre un corpus tokenizado.
    pub fn train_on_tokens(&mut self, tokens: &[u32], epochs: usize, log_every: usize) {
        if tokens.len() < 2 {
            return;
        }
        let train_tokens = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];

        for epoch in 0..epochs {
            let current_lr = self.cosine_lr(epoch, epochs);
            self.optimizer.lr = current_lr;

            // Training GPU estricto: si no hay backend GPU activo, devuelve 0.0.
            let total_loss = self.train_sequence(train_tokens, targets);
            if epoch % log_every == 0 {
                println!("  epoch {epoch:>4}/{epochs}  loss={:.4}", total_loss);
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

        for _ in 0..max_tokens {
            let ctx_start = tokens.len().saturating_sub(self.config.ctx_len);
            let context = &tokens[ctx_start..];

            #[cfg(feature = "wgpu")]
            if self.gpu_deq.is_some() {
                let gpu = self.gpu_deq.take().expect("gpu_deq checked as Some");
                if self.gpu_emb.is_none() {
                    self.gpu_emb = Some(GpuEmbeddingTrainer::new(
                        &gpu.device,
                        self.tokenizer.vocab_size(),
                        self.config.ctx_len.max(1),
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
        self.tokenizer.decode(&tokens)
    }

    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens)
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
                self.gpu_weights_uploaded = false;
                self.gpu_cg_weights_uploaded = false;
            }
        }

        // Sincronizar Embeddings
        if let Some(gpu_emb) = self.gpu_emb.as_ref() {
            if let Some(gpu) = self.gpu_deq.as_ref() {
                if let Ok(emb_data) = gpu_emb.read_weights(&gpu.device, &gpu.queue) {
                    if emb_data.len() == self.tokenizer.embeddings.len() {
                        self.tokenizer.embeddings.copy_from_slice(&emb_data);
                        self.gpu_emb_weights_uploaded = false;
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
        if let Ok((w, b)) = gpu_lm.read_weights(&gpu.device, &gpu.queue) {
            if w.len() == self.lm_head.w.len() && b.len() == self.lm_head.b.len() {
                self.lm_head.w = nalgebra::DMatrix::from_column_slice(
                    self.lm_head.w.nrows(),
                    self.config.d_r,
                    &w,
                );
                self.lm_head.b = nalgebra::DVector::from_column_slice(&b);
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

        if let Some(gpu) = crate::gpu_backend::WgpuBlockBackend::new_blocking() {
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
        trainer.config.max_deq_iters = 8;
        trainer.config.cg_iters = 5;
        (trainer, tokens)
    }

    #[test]
    fn loss_decreases_with_embeddings() {
        let (mut trainer, tokens) = make_trainer();
        let ctx = &tokens[0..5];
        let target = tokens[5];
        let loss_0 = trainer.train_step(ctx, target);
        for _ in 0..15 {
            trainer.train_step(ctx, target);
        }
        let loss_15 = trainer.train_step(ctx, target);
        assert!(loss_15 < loss_0);
    }
}
