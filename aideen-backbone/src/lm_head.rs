use aideen_core::{
    model_head::ModelHead,
    state::{ArchitectureConfig, HSlots},
};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::collections::HashSet;

/// Gradients for LmHead parameters.
pub struct LmHeadGrads {
    /// Gradient w.r.t. W: [vocab_size × d_r]
    pub grad_w: DMatrix<f32>,
    /// Gradient w.r.t. b: [vocab_size]
    pub grad_b: DVector<f32>,
    /// Gradient w.r.t. g (RMSNorm scale): [d_r]
    pub grad_g: DVector<f32>,
}

/// LmHead — proyección del H* del DEQ al espacio de vocabulario.
pub struct LmHead {
    pub config: ArchitectureConfig,
    /// W: [vocab_size × D_R]
    pub w: DMatrix<f32>,
    /// b: [vocab_size]
    pub b: DVector<f32>,
    /// g (RMSNorm scale): [D_R]
    pub g: DVector<f32>,
}

impl LmHead {
    pub fn new(config: ArchitectureConfig) -> Self {
        let vocab_size = config.vocab_size;
        let d_r = config.d_r;
        let scale = (d_r as f32).sqrt().recip() * 0.1;

        let w = DMatrix::from_fn(vocab_size, d_r, |i, j| {
            let v = ((i * d_r + j) as f32 * 1.6180339) % 1.0;
            (v - 0.5) * scale
        });
        let b = DVector::zeros(vocab_size);
        let g = DVector::from_element(d_r, 1.0);
        Self { w, b, g, config }
    }

    pub fn pool_h_star(&self, h_star: &HSlots) -> DVector<f32> {
        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;
        (0..h_slots)
            .map(|k| h_star.slot(k))
            .fold(DVector::zeros(d_r), |acc, s| acc + s)
            / h_slots as f32
    }

    pub fn rmsnorm(x: &DVector<f32>, g: &DVector<f32>, eps: f32) -> DVector<f32> {
        let mean_sq = x.map(|v| v * v).mean();
        let rms = (mean_sq + eps).sqrt();
        x.map(|v| v / rms).component_mul(g)
    }

    pub fn forward(&self, h_star: &HSlots) -> DVector<f32> {
        let pooled = self.pool_h_star(h_star);
        let h_norm = Self::rmsnorm(&pooled, &self.g, 1e-5);
        &self.w * h_norm + &self.b
    }

    pub fn forward_on_flat(&self, flat_slot: &[f32]) -> DVector<f32> {
        let v = DVector::from_column_slice(flat_slot);
        let v_norm = Self::rmsnorm(&v, &self.g, 1e-5);
        &self.w * v_norm + &self.b
    }

    /// Run forward pass and compute cross-entropy loss against the target token.
    pub fn forward_loss(&self, h_star: &HSlots, target: u32) -> f32 {
        let logits = self.forward(h_star);
        let probs = Self::softmax(&logits);
        let p = probs[target as usize].max(1e-12);
        -p.ln()
    }

    /// Backward pass of RMSNorm.
    ///
    /// Given y = (x / rms) * g where rms = sqrt(mean(x^2) + eps),
    /// returns (dL/dx, dL/dg).
    pub fn rmsnorm_backward(
        x: &DVector<f32>,
        g: &DVector<f32>,
        dl_dy: &DVector<f32>,
        eps: f32,
    ) -> (DVector<f32>, DVector<f32>) {
        let n = x.len() as f32;
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
        let rms = (mean_sq + eps).sqrt();
        let x_hat = x / rms;

        // dL/dg = dL/dy * x_hat (element-wise)
        let dl_dg = dl_dy.component_mul(&x_hat);

        // dl_dy_g = dL/dy . g (element-wise)
        let dl_dy_g = dl_dy.component_mul(g);

        // dL/dx = (1/rms) * (dl_dy_g - x_hat * mean(dl_dy_g . x_hat))
        let mean_term: f32 = dl_dy_g.component_mul(&x_hat).sum() / n;
        let dl_dx = (&dl_dy_g - &x_hat * mean_term) / rms;

        (dl_dx, dl_dg)
    }

    /// Full backward pass through LmHead.
    ///
    /// Returns (loss, grad_h_star, LmHeadGrads).
    ///
    /// The chain is:
    /// 1. pooled = mean(h_star slots)
    /// 2. h_norm = RMSNorm(pooled, g, eps)
    /// 3. logits = W @ h_norm + b
    /// 4. loss = cross_entropy(logits, target)
    ///
    /// Backward:
    /// - dL/d_logits = softmax(logits) - one_hot(target)
    /// - dL/d_b = dL/d_logits
    /// - dL/d_h_norm = W^T @ dL/d_logits
    /// - dL/d_W = dL/d_logits @ h_norm^T (outer product)
    /// - dL/d_pooled, dL/d_g = rmsnorm_backward(pooled, g, dL/d_h_norm)
    /// - dL/d_h_star[k] = dL/d_pooled / h_slots for each slot k
    pub fn backward(&self, h_star: &HSlots, target: u32) -> (f32, HSlots, LmHeadGrads) {
        let h_slots = self.config.h_slots;
        let eps = 1e-5_f32;

        // --- Forward ---
        let pooled = self.pool_h_star(h_star);
        let h_norm = Self::rmsnorm(&pooled, &self.g, eps);
        let logits = &self.w * &h_norm + &self.b;

        // --- Loss ---
        let probs = Self::softmax(&logits);
        let p = probs[target as usize].max(1e-12);
        let loss = -p.ln();

        // --- dL/d_logits = softmax - one_hot ---
        let mut dl_dlogits = probs;
        dl_dlogits[target as usize] -= 1.0;

        // --- dL/d_b = dL/d_logits ---
        let grad_b = dl_dlogits.clone();

        // --- dL/d_W = dL/d_logits @ h_norm^T (outer product) ---
        let grad_w = &dl_dlogits * h_norm.transpose();

        // --- dL/d_h_norm = W^T @ dL/d_logits ---
        let dl_dh_norm = self.w.transpose() * &dl_dlogits;

        // --- RMSNorm backward ---
        let (dl_dpooled, grad_g) = Self::rmsnorm_backward(&pooled, &self.g, &dl_dh_norm, eps);

        // --- Broadcast gradient to each slot: dL/d_h_star[k] = dL/d_pooled / h_slots ---
        let mut grad_h_star = HSlots::zeros(&self.config);
        let scale = 1.0 / h_slots as f32;
        let slot_grad = &dl_dpooled * scale;
        for k in 0..h_slots {
            grad_h_star.set_slot(k, &slot_grad);
        }

        let grads = LmHeadGrads {
            grad_w,
            grad_b,
            grad_g,
        };

        (loss, grad_h_star, grads)
    }

    pub fn argmax(logits: &DVector<f32>) -> u32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    pub fn softmax(logits: &DVector<f32>) -> DVector<f32> {
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: DVector<f32> = logits.map(|l| (l - max_l).exp());
        let sum_exp: f32 = exps.sum();
        exps / sum_exp
    }

    pub fn sample(
        logits_in: &DVector<f32>,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        context: &[u32],
    ) -> u32 {
        let mut logits = logits_in.clone();

        // 1. Repetition Penalty
        if (repetition_penalty - 1.0).abs() > 1e-4 && !context.is_empty() {
            let context_set: HashSet<u32> = context.iter().copied().collect();
            for &tok in &context_set {
                let tok_idx = tok as usize;
                if tok_idx < logits.len() {
                    let score = logits[tok_idx];
                    if score < 0.0 {
                        logits[tok_idx] = score * repetition_penalty;
                    } else {
                        logits[tok_idx] = score / repetition_penalty;
                    }
                }
            }
        }

        // 2. Temperature
        if temperature > 0.0 && (temperature - 1.0).abs() > 1e-4 {
            for v in logits.iter_mut() {
                *v /= temperature;
            }
        } else if temperature <= 1e-4 {
            // Greedy
            return Self::argmax(&logits);
        }

        // 3. Softmax
        let probs = Self::softmax(&logits).as_slice().to_vec();

        // Pair up items with their indices and sort descending
        let mut prob_items: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
        prob_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 4. Top-K
        let k = top_k.min(prob_items.len());
        if k > 0 && k < prob_items.len() {
            prob_items.truncate(k);
        }

        // 5. Top-P (Nucleus)
        if top_p > 0.0 && top_p < 1.0 {
            let mut cumulative_prob = 0.0;
            let mut cutoff_idx = prob_items.len();
            for (i, &(_, p)) in prob_items.iter().enumerate() {
                cumulative_prob += p;
                if cumulative_prob > top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            // keep at least 1
            prob_items.truncate(cutoff_idx.max(1));
        }

        // 6. Normalize
        let sum: f32 = prob_items.iter().map(|&(_, p)| p).sum();
        if sum > 1e-6 {
            for item in prob_items.iter_mut() {
                item.1 /= sum;
            }
        } else {
            return Self::argmax(logits_in);
        }

        // 7. Sample
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cumulative = 0.0;
        for &(idx, p) in &prob_items {
            cumulative += p;
            if r <= cumulative {
                return idx as u32;
            }
        }

        // Fallback
        prob_items
            .first()
            .map(|&(idx, _)| idx as u32)
            .unwrap_or_else(|| Self::argmax(logits_in))
    }
    pub fn export_weights(&self) -> std::collections::HashMap<String, Vec<f32>> {
        let mut weights = std::collections::HashMap::new();

        // nalgebra es Col-Major. Para la GPU y persistencia preferimos Row-Major (v * d_r + d)
        // para asegurar accesos contiguos en los bucles internos de los shaders.
        let rows = self.w.nrows();
        let cols = self.w.ncols();
        let mut row_major = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                row_major[i * cols + j] = self.w[(i, j)];
            }
        }

        weights.insert("head.w".to_string(), row_major);
        weights.insert("head.b".to_string(), self.b.as_slice().to_vec());
        weights.insert("head.g".to_string(), self.g.as_slice().to_vec());

        weights
    }

    pub fn import_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<(), String> {
        let vocab_size = self.config.vocab_size;
        let d_r = self.config.d_r;

        let data_w = weights.get("head.w").ok_or("head.w not found")?;
        if data_w.len() != vocab_size * d_r {
            return Err(format!(
                "head.w size mismatch: expected {}, got {}",
                vocab_size * d_r,
                data_w.len()
            ));
        }
        // Convertir de Row-Major (del archivo/GPU) a Col-Major (nalgebra)
        let mut matrix_w = nalgebra::DMatrix::zeros(vocab_size, d_r);
        for i in 0..vocab_size {
            for j in 0..d_r {
                matrix_w[(i, j)] = data_w[i * d_r + j];
            }
        }
        self.w = matrix_w;

        let data_b = weights.get("head.b").ok_or("head.b not found")?;
        if data_b.len() != vocab_size {
            return Err(format!(
                "head.b size mismatch: expected {}, got {}",
                vocab_size,
                data_b.len()
            ));
        }
        self.b = nalgebra::DVector::from_vec(data_b.clone());

        if let Some(data_g) = weights.get("head.g") {
            if data_g.len() != d_r {
                return Err(format!(
                    "head.g size mismatch: expected {}, got {}",
                    d_r,
                    data_g.len()
                ));
            }
            self.g = nalgebra::DVector::from_vec(data_g.clone());
        } else {
            // Backward compatibility
            self.g = nalgebra::DVector::from_element(d_r, 1.0);
        }

        Ok(())
    }
}

impl ModelHead for LmHead {
    type Output = DVector<f32>;
    fn forward(&self, h_star: &HSlots) -> DVector<f32> {
        self.forward(h_star)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_config(vocab: usize) -> ArchitectureConfig {
        let mut config = ArchitectureConfig::default();
        config.vocab_size = vocab;
        config
    }

    #[test]
    fn lm_head_output_shape() {
        let vocab = 1024;
        let config = make_test_config(vocab);
        let head = LmHead::new(config.clone());
        let h = HSlots::zeros(&config);
        let logits = head.forward(&h);
        assert_eq!(logits.len(), vocab);
    }

    #[test]
    fn argmax_returns_valid_index() {
        let vocab = 512;
        let config = make_test_config(vocab);
        let head = LmHead::new(config.clone());
        let h = HSlots::zeros(&config);
        let logits = head.forward(&h);
        let token = LmHead::argmax(&logits);
        assert!((token as usize) < vocab);
    }

    #[test]
    fn softmax_sums_to_one() {
        let vocab = 256;
        let config = make_test_config(vocab);
        let head = LmHead::new(config.clone());
        let h = HSlots::zeros(&config);
        let logits = head.forward(&h);
        let probs = LmHead::softmax(&logits);
        let total: f32 = probs.sum();
        assert!((total - 1.0).abs() < 1e-4, "softmax sum = {total}");
    }

    #[test]
    fn different_h_star_different_logits() {
        let vocab = 256;
        let config = make_test_config(vocab);
        let head = LmHead::new(config.clone());
        let h_a = HSlots::zeros(&config);
        let mut h_b = HSlots::zeros(&config);
        // Modificar un slot
        let mut slot = h_b.slot(0);
        slot[0] = 1.0;
        h_b.set_slot(0, &slot);

        let logits_a = head.forward(&h_a);
        let logits_b = head.forward(&h_b);
        assert_ne!(
            logits_a, logits_b,
            "H* distintos deben producir logits distintos"
        );
    }
}
