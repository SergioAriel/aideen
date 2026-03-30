/// Simple GPT-style Transformer with manual backprop in nalgebra.
///
/// Architecture: Token+Pos embeddings → N × (CausalSelfAttention + FFN) → LM head
/// No autograd dependencies — all gradients are analytical.

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ─── Configuration ───────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub ctx_len: usize,
    pub n_layers: usize,
}

impl TransformerConfig {
    /// Default config: ~288K parameters with vocab=65.
    pub fn default_small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 128,
            n_heads: 4,
            d_ff: 256,
            ctx_len: 64,
            n_layers: 2,
        }
    }

    pub fn param_count(&self) -> usize {
        let d = self.d_model;
        let v = self.vocab_size;
        let embed = v * d + self.ctx_len * d;
        let per_layer = d * (3 * d) + d * d   // QKV + O
            + d * self.d_ff + self.d_ff * d    // FFN
            + 4 * d;                           // 2 × LayerNorm (scale+bias)
        let lm_head = d * v + v;
        let ln_f = 2 * d;
        embed + self.n_layers * per_layer + lm_head + ln_f
    }
}

// ─── LayerNorm ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct LayerNorm {
    pub scale: DVector<f32>,
    pub bias: DVector<f32>,
}

impl LayerNorm {
    fn new(d: usize) -> Self {
        Self {
            scale: DVector::from_element(d, 1.0),
            bias: DVector::zeros(d),
        }
    }

    /// Forward: returns (normalized, mean, inv_std, x_centered) for backward.
    fn forward(&self, x: &DVector<f32>) -> (DVector<f32>, f32, f32, DVector<f32>) {
        let n = x.len() as f32;
        let mean = x.sum() / n;
        let x_c = x - DVector::from_element(x.len(), mean);
        let var = x_c.dot(&x_c) / n;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        let x_norm = x_c.scale(inv_std);
        let out = x_norm.component_mul(&self.scale) + &self.bias;
        (out, mean, inv_std, x_c)
    }

    /// Backward: retorna (dx, dscale, dbias).
    fn backward(
        &self,
        dout: &DVector<f32>,
        x_c: &DVector<f32>,
        inv_std: f32,
    ) -> (DVector<f32>, DVector<f32>, DVector<f32>) {
        let n = dout.len() as f32;
        let x_norm = x_c.scale(inv_std);
        let dscale = dout.component_mul(&x_norm);
        let dbias = dout.clone();

        // dx via chain rule through normalization
        let dx_norm = dout.component_mul(&self.scale);
        let dvar = dx_norm.dot(&x_c) * (-0.5) * inv_std.powi(3);
        let dmean = -dx_norm.scale(inv_std).sum() - 2.0 * dvar * x_c.sum() / n;
        let dx = dx_norm.scale(inv_std)
            + x_c.scale(2.0 * dvar / n)
            + DVector::from_element(dout.len(), dmean / n);

        (dx, dscale, dbias)
    }
}

// ─── Mini Adam local ──────────────────────────────────────────────────────────

pub struct MiniAdam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    // Momenta: guardados como Vec<f32> aplanados
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl MiniAdam {
    pub fn new(lr: f32, n_params: usize) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: vec![Vec::new(); n_params],
            v: vec![Vec::new(); n_params],
        }
    }

    pub fn step(&mut self, idx: usize, param: &mut [f32], grad: &[f32]) {
        self.t += 1;
        if self.m[idx].is_empty() {
            self.m[idx] = vec![0.0; param.len()];
            self.v[idx] = vec![0.0; param.len()];
        }
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..param.len() {
            self.m[idx][i] = self.beta1 * self.m[idx][i] + (1.0 - self.beta1) * grad[i];
            self.v[idx][i] = self.beta2 * self.v[idx][i] + (1.0 - self.beta2) * grad[i] * grad[i];
            let m_hat = self.m[idx][i] / bc1;
            let v_hat = self.v[idx][i] / bc2;
            param[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ─── Capa Transformer ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct TransformerLayer {
    ln1: LayerNorm,
    w_qkv: DMatrix<f32>, // [d_model × 3*d_model]
    w_o: DMatrix<f32>,   // [d_model × d_model]
    ln2: LayerNorm,
    w_fc: DMatrix<f32>,   // [d_model × d_ff]
    w_proj: DMatrix<f32>, // [d_ff × d_model]
}

impl TransformerLayer {
    fn new<R: Rng>(cfg: &TransformerConfig, rng: &mut R) -> Self {
        let d = cfg.d_model;
        let ff = cfg.d_ff;
        let scale_qkv = (2.0 / (d as f32)).sqrt();
        let scale_ff = (2.0 / (d as f32)).sqrt();
        Self {
            ln1: LayerNorm::new(d),
            w_qkv: random_mat(d, 3 * d, scale_qkv, rng),
            w_o: random_mat(d, d, scale_qkv * 0.5, rng),
            ln2: LayerNorm::new(d),
            w_fc: random_mat(d, ff, scale_ff, rng),
            w_proj: random_mat(ff, d, scale_ff * 0.5, rng),
        }
    }
}

fn random_mat<R: Rng>(rows: usize, cols: usize, scale: f32, rng: &mut R) -> DMatrix<f32> {
    DMatrix::from_fn(rows, cols, |_, _| (rng.gen::<f32>() - 0.5) * scale)
}

fn gelu(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + ((c * (x + 0.044715 * x.powi(3))).tanh()))
}

fn gelu_grad(x: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    let t = (c * (x + 0.044715 * x.powi(3))).tanh();
    let sech2 = 1.0 - t * t;
    0.5 * (1.0 + t) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x)
}

// ─── Transformer principal ────────────────────────────────────────────────────

pub struct Transformer {
    cfg: TransformerConfig,
    token_embed: DMatrix<f32>, // [vocab × d_model]
    pos_embed: DMatrix<f32>,   // [ctx_len × d_model]
    layers: Vec<TransformerLayer>,
    ln_f: LayerNorm,
    lm_head: DMatrix<f32>, // [d_model × vocab]
    lm_bias: DVector<f32>, // [vocab]
}

impl Transformer {
    pub fn new(cfg: TransformerConfig) -> Self {
        let mut rng = rand::thread_rng();
        Self::new_with_rng(cfg, &mut rng)
    }

    pub fn new_with_seed(cfg: TransformerConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new_with_rng(cfg, &mut rng)
    }

    fn new_with_rng<R: Rng>(cfg: TransformerConfig, rng: &mut R) -> Self {
        let d = cfg.d_model;
        let v = cfg.vocab_size;
        let scale_emb = 0.02f32;

        let layers = (0..cfg.n_layers)
            .map(|_| TransformerLayer::new(&cfg, rng))
            .collect();

        Self {
            token_embed: random_mat(v, d, scale_emb, rng),
            pos_embed: random_mat(cfg.ctx_len, d, scale_emb, rng),
            layers,
            ln_f: LayerNorm::new(d),
            lm_head: random_mat(d, v, scale_emb, rng),
            lm_bias: DVector::zeros(v),
            cfg,
        }
    }

    pub fn param_count(&self) -> usize {
        self.cfg.param_count()
    }

    /// Trains a batch of tokens, returns loss.
    /// tokens: secuencia de longitud T, targets = tokens[1..]
    pub fn train_step(&mut self, tokens: &[u32], opt: &mut MiniAdam) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }
        let t = tokens.len().min(self.cfg.ctx_len + 1);
        let inputs = &tokens[..t - 1];
        let targets = &tokens[1..t];

        let (loss, grads) = self.forward_backward(inputs, targets);
        self.apply_grads(grads, opt);
        loss
    }

    /// Computes validation loss without updating weights.
    pub fn val_loss(&self, tokens: &[u32]) -> f32 {
        if tokens.len() < 2 {
            return 0.0;
        }
        let t = tokens.len().min(self.cfg.ctx_len + 1);
        let inputs = &tokens[..t - 1];
        let targets = &tokens[1..t];
        let (logits_rows, _) = self.forward_only(inputs);
        cross_entropy_loss(&logits_rows, targets)
    }

    /// Genera texto caracter a caracter.
    pub fn generate(&self, seed: &[u32], max_new: usize, temperature: f32) -> Vec<u32> {
        let mut tokens = seed.to_vec();
        let mut rng = rand::thread_rng();
        for _ in 0..max_new {
            let ctx_start = tokens.len().saturating_sub(self.cfg.ctx_len);
            let ctx = &tokens[ctx_start..];
            let (logits_rows, _) = self.forward_only(ctx);
            let logits = logits_rows.last().unwrap();
            let next = sample_logits(logits, temperature, &mut rng);
            tokens.push(next);
        }
        tokens[seed.len()..].to_vec()
    }
}

// ─── Forward + Backward ───────────────────────────────────────────────────────

// Cache por layer para backward
struct LayerCache {
    x_in: Vec<DVector<f32>>,         // input de la capa
    x_c1: Vec<DVector<f32>>,         // x_centered de ln1
    inv_std1: Vec<f32>,
    qkv: Vec<DMatrix<f32>>,          // [T × 3d] concatenado
    attn_weights: DMatrix<f32>,      // [T × T]
    attn_out: Vec<DVector<f32>>,     // [T × d]
    x_after_attn: Vec<DVector<f32>>, // residual post-attn
    x_c2: Vec<DVector<f32>>,         // x_centered de ln2
    inv_std2: Vec<f32>,
    pre_gelu: Vec<DVector<f32>>,     // [T × d_ff] antes de GELU
    ffn_out: Vec<DVector<f32>>,      // [T × d_ff] después de GELU
}

struct Grads {
    d_token_embed: DMatrix<f32>,
    d_pos_embed: DMatrix<f32>,
    layers: Vec<LayerGrads>,
    d_ln_f_scale: DVector<f32>,
    d_ln_f_bias: DVector<f32>,
    d_lm_head: DMatrix<f32>,
    d_lm_bias: DVector<f32>,
}

struct LayerGrads {
    d_ln1_scale: DVector<f32>,
    d_ln1_bias: DVector<f32>,
    d_w_qkv: DMatrix<f32>,
    d_w_o: DMatrix<f32>,
    d_ln2_scale: DVector<f32>,
    d_ln2_bias: DVector<f32>,
    d_w_fc: DMatrix<f32>,
    d_w_proj: DMatrix<f32>,
}

impl Transformer {
    fn forward_only(&self, tokens: &[u32]) -> (Vec<DVector<f32>>, ()) {
        let d = self.cfg.d_model;
        let t_len = tokens.len();

        // Embeddings
        let mut xs: Vec<DVector<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &tok)| {
                let te = self.token_embed.row(tok as usize).transpose();
                let pe = self.pos_embed.row(pos.min(self.cfg.ctx_len - 1)).transpose();
                te + pe
            })
            .collect();

        // Layers (forward only, no cache)
        for layer in &self.layers {
            let mut new_xs = Vec::with_capacity(t_len);
            // Per-token LN1 + collect for attention
            let mut q_all = DMatrix::zeros(t_len, d);
            let mut k_all = DMatrix::zeros(t_len, d);
            let mut v_all = DMatrix::zeros(t_len, d);

            for (i, x) in xs.iter().enumerate() {
                let (x_norm, _, _, _) = layer.ln1.forward(x);
                let qkv = &layer.w_qkv.transpose() * &x_norm;
                q_all.set_row(i, &qkv.rows(0, d).transpose());
                k_all.set_row(i, &qkv.rows(d, d).transpose());
                v_all.set_row(i, &qkv.rows(2 * d, d).transpose());
            }

            let scale = (d as f32).sqrt().recip();
            let scores = &q_all * k_all.transpose() * scale;
            let attn = causal_softmax(&scores);
            let attn_out = &attn * &v_all; // [T × d]

            for (i, x) in xs.iter().enumerate() {
                let ao = attn_out.row(i).transpose();
                let x_attn = x + &layer.w_o.transpose() * &ao;

                let (x_norm2, _, _, _) = layer.ln2.forward(&x_attn);
                let pre_g = &layer.w_fc.transpose() * &x_norm2;
                let ffn: DVector<f32> = pre_g.map(|v| gelu(v));
                let out = x_attn + &layer.w_proj.transpose() * &ffn;
                new_xs.push(out);
            }
            xs = new_xs;
        }

        // Final LN + LM head
        let logits: Vec<DVector<f32>> = xs
            .iter()
            .map(|x| {
                let (x_norm, _, _, _) = self.ln_f.forward(x);
                &self.lm_head.transpose() * &x_norm + &self.lm_bias
            })
            .collect();

        (logits, ())
    }

    fn forward_backward(&self, tokens: &[u32], targets: &[u32]) -> (f32, Grads) {
        let d = self.cfg.d_model;
        let ff = self.cfg.d_ff;
        let t_len = tokens.len();
        let v = self.cfg.vocab_size;

        // ── Forward ──────────────────────────────────────────────────────────

        // Embeddings
        let mut xs: Vec<DVector<f32>> = tokens
            .iter()
            .enumerate()
            .map(|(pos, &tok)| {
                let te = self.token_embed.row(tok as usize).transpose();
                let pe = self.pos_embed.row(pos.min(self.cfg.ctx_len - 1)).transpose();
                te + pe
            })
            .collect();

        let mut caches: Vec<LayerCache> = Vec::new();

        for layer in &self.layers {
            let mut q_all = DMatrix::zeros(t_len, d);
            let mut k_all = DMatrix::zeros(t_len, d);
            let mut v_all = DMatrix::zeros(t_len, d);
            let mut x_c1_all = Vec::new();
            let mut inv_std1_all = Vec::new();
            let x_in_all = xs.clone();

            for (i, x) in xs.iter().enumerate() {
                let (x_norm, _, inv_std, x_c) = layer.ln1.forward(x);
                x_c1_all.push(x_c);
                inv_std1_all.push(inv_std);
                let qkv = &layer.w_qkv.transpose() * &x_norm;
                q_all.set_row(i, &qkv.rows(0, d).transpose());
                k_all.set_row(i, &qkv.rows(d, d).transpose());
                v_all.set_row(i, &qkv.rows(2 * d, d).transpose());
            }

            let scale = (d as f32).sqrt().recip();
            let scores = &q_all * k_all.transpose() * scale;
            let attn_w = causal_softmax(&scores);
            let attn_out_mat = &attn_w * &v_all;

            let mut x_after_attn_all = Vec::new();
            let mut x_c2_all = Vec::new();
            let mut inv_std2_all = Vec::new();
            let mut pre_gelu_all = Vec::new();
            let mut ffn_out_all = Vec::new();
            let mut new_xs = Vec::new();

            for (i, x) in xs.iter().enumerate() {
                let ao = attn_out_mat.row(i).transpose();
                let x_attn = x + &layer.w_o.transpose() * &ao;
                x_after_attn_all.push(x_attn.clone());

                let (x_norm2, _, inv_std2, x_c2) = layer.ln2.forward(&x_attn);
                x_c2_all.push(x_c2);
                inv_std2_all.push(inv_std2);

                let pre_g = &layer.w_fc.transpose() * &x_norm2;
                let ffn: DVector<f32> = pre_g.map(|val| gelu(val));
                pre_gelu_all.push(pre_g);
                ffn_out_all.push(ffn.clone());

                let out = x_attn + &layer.w_proj.transpose() * &ffn;
                new_xs.push(out);
            }

            // Build qkv cache
            let mut qkv_cache = Vec::new();
            for (i, x_c) in x_c1_all.iter().enumerate() {
                let x_norm = x_c.scale(inv_std1_all[i]);
                let x_norm_w = x_norm.component_mul(&layer.ln1.scale) + &layer.ln1.bias;
                let qkv = layer.w_qkv.transpose() * &x_norm_w;
                let mut m = DMatrix::zeros(1, 3 * d);
                m.set_row(0, &qkv.transpose());
                qkv_cache.push(m);
            }

            caches.push(LayerCache {
                x_in: x_in_all,
                x_c1: x_c1_all,
                inv_std1: inv_std1_all,
                qkv: qkv_cache,
                attn_weights: attn_w,
                attn_out: (0..t_len).map(|i| attn_out_mat.row(i).transpose()).collect(),
                x_after_attn: x_after_attn_all,
                x_c2: x_c2_all,
                inv_std2: inv_std2_all,
                pre_gelu: pre_gelu_all,
                ffn_out: ffn_out_all,
            });

            xs = new_xs;
        }

        // Final LN + logits
        let mut ln_f_x_c = Vec::new();
        let mut ln_f_inv_std = Vec::new();
        let logits: Vec<DVector<f32>> = xs
            .iter()
            .map(|x| {
                let (x_norm, _, inv_std, x_c) = self.ln_f.forward(x);
                ln_f_x_c.push(x_c);
                ln_f_inv_std.push(inv_std);
                &self.lm_head.transpose() * &x_norm + &self.lm_bias
            })
            .collect();

        // ── Loss ─────────────────────────────────────────────────────────────

        let loss = cross_entropy_loss(&logits, targets);

        // ── Backward ─────────────────────────────────────────────────────────

        // dL/dlogits = softmax(logits) - one_hot(target)
        let mut dlogits: Vec<DVector<f32>> = logits
            .iter()
            .zip(targets.iter())
            .map(|(lg, &tgt)| {
                let mut p = softmax_vec(lg);
                p[tgt as usize] -= 1.0;
                p.scale(1.0 / targets.len() as f32)
            })
            .collect();

        // LM head backward
        let mut d_lm_head = DMatrix::zeros(d, v);
        let mut d_lm_bias = DVector::zeros(v);
        let mut d_ln_f_scale = DVector::zeros(d);
        let mut d_ln_f_bias = DVector::zeros(d);
        let mut dxs: Vec<DVector<f32>> = vec![DVector::zeros(d); t_len];

        for i in 0..t_len {
            // dlogits[i] → d_lm_head, d_lm_bias, d_x_norm
            let x_c = &ln_f_x_c[i];
            let x_norm_i = x_c.scale(ln_f_inv_std[i]).component_mul(&self.ln_f.scale) + &self.ln_f.bias;

            d_lm_head += &x_norm_i * dlogits[i].transpose();
            d_lm_bias += &dlogits[i];

            // Through LM head: d_x_norm = lm_head @ dlogits[i]
            let d_xnorm = &self.lm_head * &dlogits[i];

            // Through final LN
            let (dx, dscale, dbias) = self.ln_f.backward(&d_xnorm, x_c, ln_f_inv_std[i]);
            d_ln_f_scale += dscale;
            d_ln_f_bias += dbias;
            dxs[i] = dx;
        }

        // Layers backward (reverse)
        let mut layer_grads = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate().rev() {
            let cache = &caches[layer_idx];
            let mut d_w_proj = DMatrix::zeros(ff, d);
            let mut d_w_fc = DMatrix::zeros(d, ff);
            let mut d_ln2_scale = DVector::zeros(d);
            let mut d_ln2_bias = DVector::zeros(d);
            let mut d_w_o = DMatrix::zeros(d, d);
            let mut d_w_qkv = DMatrix::zeros(d, 3 * d);
            let mut d_ln1_scale = DVector::zeros(d);
            let mut d_ln1_bias = DVector::zeros(d);

            // Per-token FFN backward + accumulate dxs_attn
            let mut dxs_attn: Vec<DVector<f32>> = vec![DVector::zeros(d); t_len];

            for i in 0..t_len {
                let dx_out = &dxs[i]; // gradient into output of this layer

                // FFN residual: dx_after_attn += dx_out (residual branch)
                dxs_attn[i] += dx_out;

                // FFN: x_out = x_after_attn + w_proj.T @ ffn_out
                // d_w_proj += ffn_out @ dx_out.T
                let ffn = &cache.ffn_out[i];
                d_w_proj += ffn * dx_out.transpose();

                // d_ffn = w_proj @ dx_out
                let d_ffn = &layer.w_proj * dx_out;

                // GELU backward
                let d_pre_gelu: DVector<f32> = d_ffn.zip_map(&cache.pre_gelu[i], |df, pg| df * gelu_grad(pg));

                // LN2 input: x_norm2 = ln2(x_after_attn)
                let x_norm2 = cache.x_c2[i].scale(cache.inv_std2[i]).component_mul(&layer.ln2.scale) + &layer.ln2.bias;
                // d_w_fc += x_norm2 @ d_pre_gelu.T
                d_w_fc += &x_norm2 * d_pre_gelu.transpose();

                // d_x_norm2 = w_fc @ d_pre_gelu
                let d_xnorm2 = &layer.w_fc * &d_pre_gelu;

                // LN2 backward
                let (dx_attn_from_ffn, dsc2, db2) = layer.ln2.backward(&d_xnorm2, &cache.x_c2[i], cache.inv_std2[i]);
                d_ln2_scale += dsc2;
                d_ln2_bias += db2;
                dxs_attn[i] += dx_attn_from_ffn;
            }

            // Attention backward
            // Build attn_out matrix for grad computation
            let mut d_attn_out = DMatrix::zeros(t_len, d);
            let mut dxs_pre_attn: Vec<DVector<f32>> = vec![DVector::zeros(d); t_len];

            for i in 0..t_len {
                // x_after_attn = x_in + w_o.T @ attn_out[i]
                // residual: dxs_pre_attn += dxs_attn[i]
                dxs_pre_attn[i] += &dxs_attn[i];

                // w_o backward: d_w_o += attn_out[i] @ dxs_attn[i].T
                let ao = &cache.attn_out[i];
                d_w_o += ao * dxs_attn[i].transpose();

                // d_attn_out[i] = w_o @ dxs_attn[i]
                let d_ao = &layer.w_o * &dxs_attn[i];
                d_attn_out.set_row(i, &d_ao.transpose());
            }

            // Attention weights backward: d_V, d_scores
            // attn_out = attn_w @ V
            // Rebuild Q, K, V from cache
            let mut q_all = DMatrix::zeros(t_len, d);
            let mut k_all = DMatrix::zeros(t_len, d);
            let mut v_all = DMatrix::zeros(t_len, d);
            for i in 0..t_len {
                let qkv_row = cache.qkv[i].row(0);
                q_all.set_row(i, &qkv_row.columns(0, d));
                k_all.set_row(i, &qkv_row.columns(d, d));
                v_all.set_row(i, &qkv_row.columns(2 * d, d));
            }

            // d_V = attn_w.T @ d_attn_out
            let d_v = cache.attn_weights.transpose() * &d_attn_out;
            // d_attn_w = d_attn_out @ V.T
            let d_attn_w = &d_attn_out * v_all.transpose();

            // Softmax backward (with causal mask)
            let d_scores = softmax_backward(&cache.attn_weights, &d_attn_w);
            let scale = (d as f32).sqrt().recip();

            // d_Q = d_scores @ K * scale
            let d_q = &d_scores * &k_all * scale;
            // d_K = d_scores.T @ Q * scale
            let d_k = d_scores.transpose() * &q_all * scale;

            // QKV backward per token
            for i in 0..t_len {
                let d_qkv_i: DVector<f32> = DVector::from_iterator(
                    3 * d,
                    d_q.row(i).iter()
                        .chain(d_k.row(i).iter())
                        .chain(d_v.row(i).iter())
                        .copied(),
                );

                // x_norm1 from cache
                let x_norm1 = cache.x_c1[i].scale(cache.inv_std1[i]).component_mul(&layer.ln1.scale) + &layer.ln1.bias;

                // d_w_qkv += x_norm1 @ d_qkv.T
                d_w_qkv += &x_norm1 * d_qkv_i.transpose();

                // d_x_norm1 = w_qkv @ d_qkv
                let d_xnorm1 = &layer.w_qkv * &d_qkv_i;

                // LN1 backward
                let (dx_from_attn, dsc1, db1) = layer.ln1.backward(&d_xnorm1, &cache.x_c1[i], cache.inv_std1[i]);
                d_ln1_scale += dsc1;
                d_ln1_bias += db1;
                dxs_pre_attn[i] += dx_from_attn;
            }

            dxs = dxs_pre_attn;

            layer_grads.push(LayerGrads {
                d_ln1_scale,
                d_ln1_bias,
                d_w_qkv,
                d_w_o,
                d_ln2_scale,
                d_ln2_bias,
                d_w_fc,
                d_w_proj,
            });
        }
        layer_grads.reverse(); // restore layer order

        // Embedding backward
        let mut d_token_embed = DMatrix::zeros(v, d);
        let mut d_pos_embed = DMatrix::zeros(self.cfg.ctx_len, d);
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = i.min(self.cfg.ctx_len - 1);
            let row_grad = dxs[i].transpose();
            // Accumulate into token embed row
            let mut row = d_token_embed.row_mut(tok as usize);
            row += row_grad.clone();
            let mut prow = d_pos_embed.row_mut(pos);
            prow += row_grad;
        }

        let grads = Grads {
            d_token_embed,
            d_pos_embed,
            layers: layer_grads,
            d_ln_f_scale,
            d_ln_f_bias,
            d_lm_head,
            d_lm_bias,
        };

        (loss, grads)
    }

    fn apply_grads(&mut self, grads: Grads, opt: &mut MiniAdam) {
        let mut idx = 0;

        // Token embed
        opt.step(idx, self.token_embed.as_mut_slice(), grads.d_token_embed.as_slice());
        idx += 1;
        // Pos embed
        opt.step(idx, self.pos_embed.as_mut_slice(), grads.d_pos_embed.as_slice());
        idx += 1;

        for (layer, lg) in self.layers.iter_mut().zip(grads.layers.iter()) {
            opt.step(idx, layer.ln1.scale.as_mut_slice(), lg.d_ln1_scale.as_slice()); idx += 1;
            opt.step(idx, layer.ln1.bias.as_mut_slice(), lg.d_ln1_bias.as_slice()); idx += 1;
            opt.step(idx, layer.w_qkv.as_mut_slice(), lg.d_w_qkv.as_slice()); idx += 1;
            opt.step(idx, layer.w_o.as_mut_slice(), lg.d_w_o.as_slice()); idx += 1;
            opt.step(idx, layer.ln2.scale.as_mut_slice(), lg.d_ln2_scale.as_slice()); idx += 1;
            opt.step(idx, layer.ln2.bias.as_mut_slice(), lg.d_ln2_bias.as_slice()); idx += 1;
            opt.step(idx, layer.w_fc.as_mut_slice(), lg.d_w_fc.as_slice()); idx += 1;
            opt.step(idx, layer.w_proj.as_mut_slice(), lg.d_w_proj.as_slice()); idx += 1;
        }

        opt.step(idx, self.ln_f.scale.as_mut_slice(), grads.d_ln_f_scale.as_slice()); idx += 1;
        opt.step(idx, self.ln_f.bias.as_mut_slice(), grads.d_ln_f_bias.as_slice()); idx += 1;
        opt.step(idx, self.lm_head.as_mut_slice(), grads.d_lm_head.as_slice()); idx += 1;
        opt.step(idx, self.lm_bias.as_mut_slice(), grads.d_lm_bias.as_slice());
    }
}

// ─── Utilities ───────────────────────────────────────────────────────────────

fn causal_softmax(scores: &DMatrix<f32>) -> DMatrix<f32> {
    let t = scores.nrows();
    let mut result = DMatrix::zeros(t, t);
    for i in 0..t {
        let row_max = (0..=i).map(|j| scores[(i, j)]).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..=i {
            let v = (scores[(i, j)] - row_max).exp();
            result[(i, j)] = v;
            sum += v;
        }
        for j in 0..=i {
            result[(i, j)] /= sum;
        }
    }
    result
}

fn softmax_backward(attn_w: &DMatrix<f32>, d_attn_w: &DMatrix<f32>) -> DMatrix<f32> {
    let t = attn_w.nrows();
    let mut d_scores = DMatrix::zeros(t, t);
    for i in 0..t {
        // Only the causal (lower triangle) part is non-zero
        let p: Vec<f32> = (0..=i).map(|j| attn_w[(i, j)]).collect();
        let dp: Vec<f32> = (0..=i).map(|j| d_attn_w[(i, j)]).collect();
        let dot: f32 = p.iter().zip(dp.iter()).map(|(a, b)| a * b).sum();
        for j in 0..=i {
            d_scores[(i, j)] = p[j] * (dp[j] - dot);
        }
    }
    d_scores
}

fn softmax_vec(v: &DVector<f32>) -> DVector<f32> {
    let max = v.max();
    let exp: DVector<f32> = v.map(|x| (x - max).exp());
    let sum = exp.sum();
    exp / sum
}

fn cross_entropy_loss(logits: &[DVector<f32>], targets: &[u32]) -> f32 {
    let n = targets.len() as f32;
    logits
        .iter()
        .zip(targets.iter())
        .map(|(lg, &tgt)| {
            let p = softmax_vec(lg);
            -p[tgt as usize].max(1e-10).ln()
        })
        .sum::<f32>()
        / n
}

fn sample_logits(logits: &DVector<f32>, temperature: f32, rng: &mut impl Rng) -> u32 {
    let scaled = logits.map(|x| x / temperature);
    let probs = softmax_vec(&scaled);
    let mut cum = 0.0f32;
    let r: f32 = rng.gen();
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if r <= cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
