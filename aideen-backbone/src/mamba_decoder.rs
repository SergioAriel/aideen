use aideen_core::{
    model_head::ModelHead,
    state::{ArchitectureConfig, HSlots},
};
use nalgebra::{DMatrix, DVector};

/// MambaDecoder — lightweight autoregressive decoder conditioned on H*.
pub struct MambaDecoder {
    pub config: ArchitectureConfig,
    /// d_model of the decoder (may differ from config.d_r of the DEQ)
    pub d_model: usize,
    pub n_layers: usize,
    pub embedding: DMatrix<f32>,
    pub a_logs: Vec<DVector<f32>>,
    pub w_ins: Vec<DMatrix<f32>>,
    pub w_outs: Vec<DMatrix<f32>>,
    pub film_projs: Vec<DMatrix<f32>>,
    pub lm_head: DMatrix<f32>,
    pub bos_token: u32,
    pub eos_token: u32,
}

impl MambaDecoder {
    /// Builds a decoder with small random weights.
    pub fn new(n_layers: usize, config: ArchitectureConfig) -> Self {
        let d_model = config.d_r;
        let vocab_size = config.vocab_size;
        let scale = (d_model as f32).sqrt().recip() * 0.1;

        let embedding = DMatrix::from_fn(vocab_size, d_model, |i, j| {
            ((i * d_model + j) as f32 * 1.6180339 % 1.0 - 0.5) * scale
        });

        let lm_head = DMatrix::from_fn(vocab_size, d_model, |i, j| {
            ((i * d_model + j + 1) as f32 * std::f32::consts::E % 1.0 - 0.5) * scale
        });

        let make_mat = |rows: usize, cols: usize, seed: usize| -> DMatrix<f32> {
            DMatrix::from_fn(rows, cols, |i, j| {
                ((i * cols + j + seed) as f32 * std::f32::consts::SQRT_2 % 1.0 - 0.5) * scale
            })
        };

        let a_logs = (0..n_layers)
            .map(|_| DVector::from_element(d_model, -0.5))
            .collect();
        let w_ins = (0..n_layers)
            .map(|l| make_mat(d_model, d_model, l * 1000))
            .collect();
        let w_outs = (0..n_layers)
            .map(|l| {
                DMatrix::identity(d_model, d_model) * 0.9
                    + make_mat(d_model, d_model, l * 2000) * 0.01
            })
            .collect();
        // FiLM: config.d_r → [scale(d_model) | bias(d_model)]
        let film_projs = (0..n_layers)
            .map(|l| make_mat(2 * d_model, config.d_r, l * 3000 + 77))
            .collect();

        Self {
            config,
            n_layers,
            d_model,
            embedding,
            a_logs,
            w_ins,
            w_outs,
            film_projs,
            lm_head,
            bos_token: 1,
            eos_token: 2,
        }
    }

    // ── H* Pooling ─────────────────────────────────────────────────────────

    fn pool_h_star(&self, h_star: &HSlots) -> DVector<f32> {
        let slots = h_star.slots;
        let d_r = h_star.d_r;
        (0..slots)
            .map(|k| h_star.slot(k))
            .fold(DVector::zeros(d_r), |acc, s| acc + s)
            / slots as f32
    }

    // ── FiLM conditioning for a single layer ──────────────────────────────────────

    fn film_params(&self, layer: usize, h_pooled: &DVector<f32>) -> (DVector<f32>, DVector<f32>) {
        let film = &self.film_projs[layer] * h_pooled;
        let scale = film.rows(0, self.d_model).into_owned().map(|x| 1.0 + x); // centered at 1
        let bias = film.rows(self.d_model, self.d_model).into_owned();
        (scale, bias)
    }

    // ── Mamba step ───────────────────────────────────────────────────────────

    fn mamba_layer_step(
        &self,
        layer: usize,
        h: &DVector<f32>,
        scale: &DVector<f32>,
        bias: &DVector<f32>,
    ) -> DVector<f32> {
        let a_bar = self.a_logs[layer].map(|a| a.exp());
        let b_bar = a_bar.map(|a| 1.0 - a);
        let x_proj = &self.w_ins[layer] * h;
        let h_new: DVector<f32> =
            a_bar.zip_map(h, |a, h| a * h) + b_bar.zip_map(&x_proj, |b, x| b * x);
        let out = &self.w_outs[layer] * h_new;
        // FiLM: scale ⊙ out + bias
        out.zip_map(scale, |o, s| o * s) + bias
    }

    /// Generates autoregressive sequences.
    pub fn generate(&self, h_star: &HSlots, prompt_tokens: &[u32], max_tokens: usize) -> Vec<u32> {
        let h_pooled = self.pool_h_star(h_star);

        let film: Vec<(DVector<f32>, DVector<f32>)> = (0..self.n_layers)
            .map(|l| self.film_params(l, &h_pooled))
            .collect();

        let mut tokens: Vec<u32> = if prompt_tokens.is_empty() {
            vec![self.bos_token]
        } else {
            prompt_tokens.to_vec()
        };

        let d_r = self.config.d_r;
        let mut hidden = if d_r >= self.d_model {
            h_pooled.rows(0, self.d_model).into_owned()
        } else {
            let mut v = DVector::zeros(self.d_model);
            v.rows_mut(0, d_r).copy_from(&h_pooled);
            v
        };

        let mut generated: Vec<u32> = vec![];

        for &tok in &tokens {
            if (tok as usize) < self.config.vocab_size {
                let embed = self.embedding.row(tok as usize).transpose();
                hidden += embed * 0.1;
                for l in 0..self.n_layers {
                    hidden = self.mamba_layer_step(l, &hidden, &film[l].0, &film[l].1);
                }
            }
        }

        for _ in 0..max_tokens {
            let logits = &self.lm_head * &hidden;
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eos_token);

            if next_token == self.eos_token {
                break;
            }

            generated.push(next_token);
            tokens.push(next_token);

            if (next_token as usize) < self.config.vocab_size {
                let embed = self.embedding.row(next_token as usize).transpose();
                hidden += embed * 0.1;
                for l in 0..self.n_layers {
                    hidden = self.mamba_layer_step(l, &hidden, &film[l].0, &film[l].1);
                }
            }
        }

        generated
    }
}

pub struct EmbedHead;

impl ModelHead for EmbedHead {
    type Output = DVector<f32>;
    fn forward(&self, h_star: &HSlots) -> DVector<f32> {
        let (slots, d_r) = (h_star.slots, h_star.d_r);
        (0..slots)
            .map(|k| h_star.slot(k))
            .fold(DVector::zeros(d_r), |acc, s| acc + s)
            / slots as f32
    }
}

pub struct ClassHead {
    pub config: ArchitectureConfig,
    pub w: DMatrix<f32>,
}

impl ClassHead {
    pub fn new(num_classes: usize, config: ArchitectureConfig) -> Self {
        let d_r = config.d_r;
        let scale = (d_r as f32).sqrt().recip() * 0.1;
        let w = DMatrix::from_fn(num_classes, d_r, |i, j| {
            ((i * d_r + j) as f32 * std::f32::consts::SQRT_2 % 1.0 - 0.5) * scale
        });
        Self { config, w }
    }
}

impl ModelHead for ClassHead {
    type Output = usize;
    fn forward(&self, h_star: &HSlots) -> usize {
        let pooled = EmbedHead.forward(h_star);
        let logits = &self.w * &pooled;
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_h_star(config: &ArchitectureConfig, seed: f32) -> HSlots {
        let mut h = HSlots::zeros(config);
        let h_slots = config.h_slots;
        let d_r = config.d_r;
        for k in 0..h_slots {
            let mut slot = h.slot(k);
            for i in 0..d_r {
                slot[i] = (seed * (k * d_r + i + 1) as f32).sin() * 0.3;
            }
            h.set_slot(k, &slot);
        }
        h
    }

    #[test]
    fn generate_produces_tokens() {
        let config = ArchitectureConfig::default();
        let decoder = MambaDecoder::new(4, config.clone());
        let h_star = make_h_star(&config, 1.0);
        let tokens = decoder.generate(&h_star, &[], 10);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn class_head_valid_class() {
        let config = ArchitectureConfig::default();
        let head = ClassHead::new(10, config.clone());
        let h = make_h_star(&config, 1.0);
        let class = head.forward(&h);
        assert!(class < 10);
    }
}
