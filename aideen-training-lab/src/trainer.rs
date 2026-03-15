use aideen_backbone::lm_head::LmHead;
use aideen_backbone::mamba_slot_reasoning::MambaSlotReasoning;
use aideen_backbone::spectral_norm;
use aideen_backbone::tokenizer::Tokenizer;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::ArchitectureConfig;
use nalgebra::{DMatrix, DVector};

use crate::backward::{unrolled_backward, BackwardResult};
use crate::optimizer::AdamW;

/// Training configuration.
pub struct TrainingConfig {
    pub lr: f32,
}

/// Main trainer struct holding model components and training state.
pub struct Trainer {
    pub config: ArchitectureConfig,
    pub training_config: TrainingConfig,
    pub gpu_deq: Option<()>,
    pub gpu_lm: Option<()>,
    pub gpu_emb: Option<()>,
    pub tokenizer: Tokenizer,
    pub reasoning: MambaSlotReasoning,
    pub lm_head: LmHead,
    pub frozen_deq: bool,
    pub frozen_emb: bool,
    pub frozen_lm: bool,
    pub adaptive_max_iters: usize,
    pub adaptive_damping: f32,
    pub optimizer: AdamW,
    pub step_count: usize,
}

impl Trainer {
    /// Create a Trainer from a tokenizer with a given learning rate and seed.
    pub fn from_tokenizer_seeded(tok: Tokenizer, lr: f32, seed: u64) -> Self {
        let config = tok.config.clone();
        let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
        let lm_head = LmHead::new(config.clone());
        let optimizer = AdamW::default_with_lr(lr);
        Self {
            config: config.clone(),
            training_config: TrainingConfig { lr },
            gpu_deq: None,
            gpu_lm: None,
            gpu_emb: None,
            tokenizer: tok,
            reasoning,
            lm_head,
            frozen_deq: false,
            frozen_emb: false,
            frozen_lm: false,
            adaptive_max_iters: config.max_deq_iters,
            adaptive_damping: 0.70,
            optimizer,
            step_count: 0,
        }
    }

    /// Train on a sequence of tokens. Returns the average loss.
    ///
    /// `tokens_in` and `targets` must have the same length.
    /// For each pair (tokens_in[i], targets[i]):
    ///   1. Embed tokens_in[i]
    ///   2. Run unrolled_backward to get loss + gradients
    ///   3. If backward_enabled, apply gradients via AdamW
    ///   4. Periodically renormalize weights
    pub fn train_sequence(
        &mut self,
        tokens_in: &[u32],
        targets: &[u32],
        backward_enabled: bool,
        _epsilon: f32,
    ) -> f32 {
        assert_eq!(tokens_in.len(), targets.len());
        let n = tokens_in.len();
        if n == 0 {
            return 0.0;
        }

        let max_iters = self.adaptive_max_iters;
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0usize;

        for i in 0..n {
            let token = tokens_in[i];
            let target = targets[i];

            // 1. Embed the input token
            let s = self.tokenizer.embed(token);

            // 2. Unrolled backward: forward + backward through N Picard iterations
            let result = unrolled_backward(
                &self.reasoning,
                &self.lm_head,
                &s,
                target,
                max_iters,
            );

            if result.loss.is_finite() {
                loss_sum += result.loss;
                loss_count += 1;
            }

            // 3. Apply gradients if backward is enabled
            if backward_enabled && result.loss.is_finite() {
                self.apply_gradients(&result, token);
                self.step_count += 1;

                // 4. Periodically apply spectral renormalization
                let renorm_every = self.config.renorm_every_steps;
                if renorm_every > 0 && self.step_count % renorm_every == 0 {
                    self.renorm_weights();
                }
            }
        }

        if loss_count > 0 {
            loss_sum / loss_count as f32
        } else {
            f32::NAN
        }
    }

    /// Apply gradients from a backward pass to all trainable parameters.
    fn apply_gradients(&mut self, result: &BackwardResult, token: u32) {
        let d_r = self.config.d_r;

        // Ensure optimizer LR is in sync with training_config
        self.optimizer.lr = self.training_config.lr;

        // --- DEQ weight matrices ---
        if !self.frozen_deq {
            apply_matrix_grad_to(&mut self.optimizer, "w_q", &mut self.reasoning.w_q, &result.grad_w_q);
            apply_matrix_grad_to(&mut self.optimizer, "w_k", &mut self.reasoning.w_k, &result.grad_w_k);
            apply_matrix_grad_to(&mut self.optimizer, "w_v", &mut self.reasoning.w_v, &result.grad_w_v);
            apply_matrix_grad_to(&mut self.optimizer, "w_o", &mut self.reasoning.w_o, &result.grad_w_o);
            apply_matrix_grad_to(&mut self.optimizer, "w_in", &mut self.reasoning.w_in, &result.grad_w_in);

            // norm_scale (vector)
            {
                let mut flat: Vec<f32> = self.reasoning.norm_scale.as_slice().to_vec();
                let grad_flat: Vec<f32> = result.grad_norm_scale.as_slice().to_vec();
                self.optimizer.step("norm_scale", &mut flat, &grad_flat);
                self.reasoning.norm_scale = DVector::from_vec(flat);
            }

            // slot_anchor (h_slots x d_r matrix)
            {
                let h_slots = self.config.h_slots;
                let mut flat = matrix_to_flat(&self.reasoning.slot_anchor);
                let grad_flat = matrix_to_flat(&result.grad_slot_anchor);
                self.optimizer.step("slot_anchor", &mut flat, &grad_flat);
                self.reasoning.slot_anchor = DMatrix::from_row_slice(h_slots, d_r, &flat);
            }
        }

        // --- LmHead weights ---
        if !self.frozen_lm {
            // W: [vocab_size x d_r]
            {
                let vocab_size = self.config.vocab_size;
                let mut flat = matrix_to_flat(&self.lm_head.w);
                let grad_flat = matrix_to_flat(&result.grad_lm.grad_w);
                self.optimizer.step("lm_w", &mut flat, &grad_flat);
                self.lm_head.w = DMatrix::from_row_slice(vocab_size, d_r, &flat);
            }
            // b: [vocab_size]
            {
                let mut flat: Vec<f32> = self.lm_head.b.as_slice().to_vec();
                let grad_flat: Vec<f32> = result.grad_lm.grad_b.as_slice().to_vec();
                self.optimizer.step("lm_b", &mut flat, &grad_flat);
                self.lm_head.b = DVector::from_vec(flat);
            }
            // g: [d_r]
            {
                let mut flat: Vec<f32> = self.lm_head.g.as_slice().to_vec();
                let grad_flat: Vec<f32> = result.grad_lm.grad_g.as_slice().to_vec();
                self.optimizer.step("lm_g", &mut flat, &grad_flat);
                self.lm_head.g = DVector::from_vec(flat);
            }
        }

        // --- Embedding for the current token ---
        if !self.frozen_emb {
            let tok_idx = token as usize;
            if tok_idx < self.tokenizer.embeddings.nrows() {
                let mut row: Vec<f32> = self.tokenizer.embeddings.row(tok_idx).iter().copied().collect();
                let grad_row: Vec<f32> = result.grad_s.iter().copied().take(d_r).collect();
                if row.len() == grad_row.len() {
                    let name = format!("emb_{}", tok_idx);
                    self.optimizer.step(&name, &mut row, &grad_row);
                    for (j, val) in row.iter().enumerate() {
                        self.tokenizer.embeddings[(tok_idx, j)] = *val;
                    }
                }
            }
        }
    }

    /// Evaluate loss on a token sequence (forward only, no weight update).
    ///
    /// For each consecutive pair (tokens[i], tokens[i+1]):
    ///   - Embed tokens[i], run DEQ forward (init + N steps), compute forward_loss
    /// Returns average loss.
    pub fn eval_loss(&self, tokens: &[u32]) -> f32 {
        if tokens.len() < 2 {
            return f32::NAN;
        }

        let max_iters = self.adaptive_max_iters;
        let mut loss_sum = 0.0f32;
        let mut loss_count = 0usize;

        for i in 0..(tokens.len() - 1) {
            let input_token = tokens[i];
            let target_token = tokens[i + 1];

            // Embed the input token
            let s = self.tokenizer.embed(input_token);

            // Run DEQ forward: init + N Picard steps
            let mut h = self.reasoning.init(&s);
            for _ in 0..max_iters {
                h = self.reasoning.step(&h, &s, None);
            }

            // Compute loss via LmHead
            let loss = self.lm_head.forward_loss(&h, target_token);
            if loss.is_finite() {
                loss_sum += loss;
                loss_count += 1;
            }
        }

        if loss_count > 0 {
            loss_sum / loss_count as f32
        } else {
            f32::NAN
        }
    }

    /// Single training step: process context and predict target token.
    pub fn train_step(&mut self, ctx: &[u32], tgt: u32, backward: bool) -> f32 {
        if ctx.is_empty() {
            return f32::NAN;
        }
        let last_token = ctx[ctx.len() - 1];
        self.train_sequence(&[last_token], &[tgt], backward, self.config.deq_epsilon)
    }

    /// Apply spectral renormalization to attention weight matrices.
    fn renorm_weights(&mut self) {
        let threshold = 0.10_f32;
        let n_iter = 20;
        spectral_norm::normalize_if_needed(&mut self.reasoning.w_q, threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut self.reasoning.w_k, threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut self.reasoning.w_v, threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut self.reasoning.w_o, threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut self.reasoning.w_in, threshold, n_iter);
    }

    /// Synchronize inference weights (e.g., from GPU buffers back to CPU).
    pub fn sync_inference_weights(&mut self) {
        // No-op stub for CPU-only path
    }
}

/// Apply optimizer step to a weight matrix, given the optimizer and gradient matrix.
fn apply_matrix_grad_to(
    optimizer: &mut AdamW,
    name: &str,
    matrix: &mut DMatrix<f32>,
    grad: &DMatrix<f32>,
) {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut flat = matrix_to_flat(matrix);
    let grad_flat = matrix_to_flat(grad);
    optimizer.step(name, &mut flat, &grad_flat);
    *matrix = DMatrix::from_row_slice(rows, cols, &flat);
}

/// Convert a nalgebra DMatrix to a row-major flat Vec<f32>.
fn matrix_to_flat(m: &DMatrix<f32>) -> Vec<f32> {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            flat.push(m[(i, j)]);
        }
    }
    flat
}
