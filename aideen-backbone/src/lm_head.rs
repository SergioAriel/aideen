use aideen_core::{
    model_head::ModelHead,
    state::{ArchitectureConfig, HSlots},
};
use nalgebra::{DMatrix, DVector};
use rand::Rng;
use std::collections::HashSet;

/// LmHead — proyección del H* del DEQ al espacio de vocabulario.
pub struct LmHead {
    pub config: ArchitectureConfig,
    /// W: [vocab_size × D_R]
    pub w: DMatrix<f32>,
    /// b: [vocab_size]
    pub b: DVector<f32>,
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
        Self { w, b, config }
    }

    pub fn pool_h_star(&self, h_star: &HSlots) -> DVector<f32> {
        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;
        (0..h_slots)
            .map(|k| h_star.slot(k))
            .fold(DVector::zeros(d_r), |acc, s| acc + s)
            / h_slots as f32
    }

    pub fn forward(&self, h_star: &HSlots) -> DVector<f32> {
        let pooled = self.pool_h_star(h_star);
        &self.w * pooled + &self.b
    }

    pub fn forward_on_flat(&self, flat_slot: &[f32]) -> DVector<f32> {
        let v = DVector::from_column_slice(flat_slot);
        &self.w * v + &self.b
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
        let mut probs = Self::softmax(&logits).as_slice().to_vec();

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

        weights.insert("head.w".to_string(), self.w.as_slice().to_vec());
        weights.insert("head.b".to_string(), self.b.as_slice().to_vec());

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
        self.w = nalgebra::DMatrix::from_vec(vocab_size, d_r, data_w.clone());

        let data_b = weights.get("head.b").ok_or("head.b not found")?;
        if data_b.len() != vocab_size {
            return Err(format!(
                "head.b size mismatch: expected {}, got {}",
                vocab_size,
                data_b.len()
            ));
        }
        self.b = nalgebra::DVector::from_vec(data_b.clone());

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
