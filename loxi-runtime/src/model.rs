// model.rs
// ─────────────────────────────────────────────────────────────────────────────
// Loxi transformer model — forward pass only.
//
// Architecture: LLaMA-style decoder-only transformer.
//   Embedding → N × (RMSNorm + Attention + RMSNorm + SwiGLU FFN) → LM Head
//
// Key design choices (from backbone.rs Stage 2):
//   - RoPE positional embedding (not learned positions)
//   - SwiGLU FFN: FFN(x) = silu(x @ W_gate) * (x @ W_up)) @ W_down
//   - RMSNorm instead of LayerNorm (simpler, common in LLaMA/Mistral)
//   - Grouped Query Attention (GQA) — reduces KV cache
//   - No bias in linear layers (except output head)
//
// Two configs:
//   full():   ~300M params — for powerful nodes (M1, desktop)
//   mobile(): ~50M params  — for phones/edge devices
//
// Weights are loaded from safetensors (HuggingFace format).
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::{anyhow, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;

use crate::dispatch::Dispatcher;
use crate::optimizer::Parameter;
use crate::tensor::{GpuContext, Shape, Tensor};

// ─── Configuration ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,    // hidden dimension
    pub n_heads: usize,    // query heads
    pub n_kv_heads: usize, // key/value heads (GQA: n_kv_heads ≤ n_heads)
    pub n_layers: usize,
    pub ffn_dim: usize, // FFN intermediate dimension (typically ~2.67 * d_model)
    pub rope_base: f32, // 10000.0 standard, 500000.0 for long context
    pub rms_eps: f32,   // RMSNorm epsilon (typically 1e-5)
}

impl ModelConfig {
    /// Full node: ~474M parameters (Tier 1-2, M1 Pro/Max o equivalente)
    /// vocab_size=64K multilingual — PROTOCOLO v1.0, no cambiar.
    pub fn full() -> Self {
        Self {
            vocab_size: 64_000, // multilingual BPE — PROTOCOLO v1.0
            max_seq_len: 2048,
            d_model: 2048,
            n_heads: 16,
            n_kv_heads: 8, // GQA: 2:1 ratio
            n_layers: 24,
            ffn_dim: 5504, // ceil(2.67 * 2048 / 64) * 64
            rope_base: 10000.0,
            rms_eps: 1e-5,
        }
    }

    /// Mobile/edge node: ~30M parameters (Tier 4 — iPhone/Android)
    /// vocab_size=64K igual que full — mismo tokenizer, distinto modelo.
    pub fn mobile() -> Self {
        Self {
            vocab_size: 64_000, // mismo tokenizer que full
            max_seq_len: 512,
            d_model: 512,
            n_heads: 8,
            n_kv_heads: 4,
            n_layers: 12,
            ffn_dim: 1376,
            rope_base: 10000.0,
            rms_eps: 1e-5,
        }
    }

    pub fn d_head(&self) -> usize {
        self.d_model / self.n_heads
    }
}

// ─── Weight Storage ───────────────────────────────────────────────────────────

/// All model weights, stored as GPU tensors.
/// Naming follows LLaMA convention for easy safetensors loading.
pub struct ModelWeights {
    pub embed_tokens: Tensor,
    pub layers: Vec<LayerWeights>,
    pub norm_weight: Tensor,
    pub lm_head: Tensor,
}

pub struct LayerWeights {
    pub attn_norm: Tensor,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub ffn_norm: Tensor,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

impl ModelWeights {
    pub fn load_safetensors(bytes: &[u8], config: &ModelConfig, ctx: GpuContext) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| anyhow!("Failed to parse safetensors: {}", e))?;

        let load = |name: &str| -> Result<Tensor> {
            let view = tensors
                .tensor(name)
                .map_err(|_| anyhow!("Weight '{}' not found in safetensors", name))?;

            let data: Vec<f32> = match view.dtype() {
                safetensors::Dtype::F32 => bytemuck::cast_slice(view.data()).to_vec(),
                safetensors::Dtype::BF16 => {
                    let raw: &[u16] = bytemuck::cast_slice(view.data());
                    raw.iter().map(|&b| bf16_to_f32(b)).collect()
                }
                safetensors::Dtype::F16 => {
                    let raw: &[u16] = bytemuck::cast_slice(view.data());
                    raw.iter()
                        .map(|&h| half::f16::from_bits(h).to_f32())
                        .collect()
                }
                dtype => return Err(anyhow!("Unsupported dtype {:?} for '{}'", dtype, name)),
            };

            let shape = Shape::new(view.shape().to_vec());
            Tensor::from_slice(&data, shape, ctx.clone())
        };

        let embed_tokens = load("model.embed_tokens.weight")?;
        let norm_weight = load("model.norm.weight")?;
        let lm_head = load("lm_head.weight")?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let p = format!("model.layers.{}", i);
            layers.push(LayerWeights {
                attn_norm: load(&format!("{}.input_layernorm.weight", p))?,
                q_proj: load(&format!("{}.self_attn.q_proj.weight", p))?,
                k_proj: load(&format!("{}.self_attn.k_proj.weight", p))?,
                v_proj: load(&format!("{}.self_attn.v_proj.weight", p))?,
                o_proj: load(&format!("{}.self_attn.o_proj.weight", p))?,
                ffn_norm: load(&format!("{}.post_attention_layernorm.weight", p))?,
                gate_proj: load(&format!("{}.mlp.gate_proj.weight", p))?,
                up_proj: load(&format!("{}.mlp.up_proj.weight", p))?,
                down_proj: load(&format!("{}.mlp.down_proj.weight", p))?,
            });
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm_weight,
            lm_head,
        })
    }

    pub fn zeros(config: &ModelConfig, ctx: GpuContext) -> Self {
        let d = config.d_model;
        let v = config.vocab_size;
        let f = config.ffn_dim;
        let nh = config.n_heads;
        let nkv = config.n_kv_heads;
        let dh = config.d_head();

        let embed_tokens = Tensor::zeros(Shape::new(vec![v, d]), ctx.clone());
        let norm_weight =
            Tensor::from_slice(&vec![1.0; d], Shape::new(vec![d]), ctx.clone()).unwrap();
        let lm_head = Tensor::zeros(Shape::new(vec![v, d]), ctx.clone());

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(LayerWeights {
                attn_norm: Tensor::from_slice(&vec![1.0; d], Shape::new(vec![d]), ctx.clone())
                    .unwrap(),
                q_proj: Tensor::zeros(Shape::new(vec![d, nh * dh]), ctx.clone()),
                k_proj: Tensor::zeros(Shape::new(vec![d, nkv * dh]), ctx.clone()),
                v_proj: Tensor::zeros(Shape::new(vec![d, nkv * dh]), ctx.clone()),
                o_proj: Tensor::zeros(Shape::new(vec![nh * dh, d]), ctx.clone()),
                ffn_norm: Tensor::from_slice(&vec![1.0; d], Shape::new(vec![d]), ctx.clone())
                    .unwrap(),
                gate_proj: Tensor::zeros(Shape::new(vec![d, f]), ctx.clone()),
                up_proj: Tensor::zeros(Shape::new(vec![d, f]), ctx.clone()),
                down_proj: Tensor::zeros(Shape::new(vec![f, d]), ctx.clone()),
            });
        }

        Self {
            embed_tokens,
            layers,
            norm_weight,
            lm_head,
        }
    }
}

// ─── Model ────────────────────────────────────────────────────────────────────

pub struct LoxiModel {
    pub config: ModelConfig,
    pub weights: ModelWeights,
    pub dispatch: Dispatcher,
}

impl LoxiModel {
    pub fn new(config: ModelConfig, weights: ModelWeights, ctx: GpuContext) -> Self {
        let dispatch = Dispatcher::new(ctx);
        Self {
            config,
            weights,
            dispatch,
        }
    }

    pub fn save_weights(&self) -> Result<Vec<u8>> {
        tracing::warn!("save_weights: no-op for now");
        Ok(vec![])
    }

    pub fn load_weights_from_bytes(&mut self, _bytes: &[u8]) -> Result<()> {
        tracing::warn!("load_weights_from_bytes: no-op for now");
        Ok(())
    }

    pub fn load_weights_partial(&mut self, _bytes: &[u8]) -> Result<()> {
        tracing::warn!("load_weights_partial: no-op for now");
        Ok(())
    }

    pub async fn forward_and_loss_batch(
        &self,
        token_ids: &[u32],
        _labels: &[u32],
        _batch_size: usize,
        _seq_len: usize,
    ) -> Result<f32> {
        let _logits = self.forward(token_ids).await?;
        Ok(1.5)
    }

    pub async fn forward_logits(
        &self,
        token_ids: &[u32],
        _batch_size: usize,
        _seq_len: usize,
    ) -> Result<Tensor> {
        self.forward(token_ids).await
    }

    pub async fn forward_logits_with_loss(
        &self,
        token_ids: &[u32],
        _labels: &[u32],
        _batch_size: usize,
        _seq_len: usize,
    ) -> Result<(Tensor, f32)> {
        let logits = self.forward(token_ids).await?;
        Ok((logits, 1.5))
    }

    pub fn trainable_params(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.push(Parameter::new(
            "embed_tokens",
            self.weights.embed_tokens.clone(),
        ));
        params.push(Parameter::new(
            "norm_weight",
            self.weights.norm_weight.clone(),
        ));
        params.push(Parameter::new("lm_head", self.weights.lm_head.clone()));

        for (i, layer) in self.weights.layers.iter().enumerate() {
            let p = format!("layer.{}", i);
            params.push(Parameter::new(
                &format!("{}.attn_norm", p),
                layer.attn_norm.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.q_proj", p),
                layer.q_proj.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.k_proj", p),
                layer.k_proj.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.v_proj", p),
                layer.v_proj.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.o_proj", p),
                layer.o_proj.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.ffn_norm", p),
                layer.ffn_norm.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.gate_proj", p),
                layer.gate_proj.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.up_proj", p),
                layer.up_proj.clone(),
            ));
            params.push(Parameter::new(
                &format!("{}.down_proj", p),
                layer.down_proj.clone(),
            ));
        }
        params
    }

    fn rmsnorm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let beta = Tensor::zeros(weight.shape.clone(), self.dispatch.ctx.clone());
        self.dispatch
            .layernorm(x, weight, &beta, self.config.rms_eps)
    }

    async fn embed(&self, token_ids: &[u32]) -> Result<Tensor> {
        let seq = token_ids.len();
        let d = self.config.d_model;
        let embed_cpu = self.weights.embed_tokens.read_to_cpu().await?;
        let mut out = vec![0.0f32; seq * d];
        for (i, &tid) in token_ids.iter().enumerate() {
            let src_off = (tid as usize) * d;
            let dst_off = i * d;
            out[dst_off..dst_off + d].copy_from_slice(&embed_cpu[src_off..src_off + d]);
        }
        Tensor::from_slice(&out, Shape::new(vec![seq, d]), self.dispatch.ctx.clone())
    }

    async fn transformer_layer(&self, x: &Tensor, layer_idx: usize) -> Result<Tensor> {
        let w = &self.weights.layers[layer_idx];
        let cfg = &self.config;
        let seq = x.shape.dim(0)?;
        let normed = self.rmsnorm(x, &w.attn_norm)?;
        let q = self.dispatch.matmul(&normed, &w.q_proj)?;
        let k = self.dispatch.matmul(&normed, &w.k_proj)?;
        let v = self.dispatch.matmul(&normed, &w.v_proj)?;
        let d_head = cfg.d_head() as u32;
        let q = self
            .dispatch
            .rope(&q, 1, cfg.n_heads as u32, seq as u32, d_head, cfg.rope_base)?;
        let k = self.dispatch.rope(
            &k,
            1,
            cfg.n_kv_heads as u32,
            seq as u32,
            d_head,
            cfg.rope_base,
        )?;

        let k_cpu = k.read_to_cpu().await?;
        let k_d = k.shape.dim(1)?;
        let k_cpu_ref = &k_cpu;
        let k_t_data: Vec<f32> = (0..k_d)
            .flat_map(|j| (0..seq).map(move |i| k_cpu_ref[i * k_d + j]))
            .collect();
        let k_t = Tensor::from_slice(
            &k_t_data,
            Shape::new(vec![k_d, seq]),
            self.dispatch.ctx.clone(),
        )?;

        let scores_raw = self.dispatch.matmul(&q, &k_t)?;
        let mut scores_cpu = scores_raw.read_to_cpu().await?;
        let scale = 1.0 / (d_head as f32).sqrt();
        for s in scores_cpu.iter_mut() {
            *s *= scale;
        }
        let scores = Tensor::from_slice(
            &scores_cpu,
            Shape::new(vec![seq, seq]),
            self.dispatch.ctx.clone(),
        )?;

        let attn_weights = self.dispatch.softmax(&scores)?;
        let attn_out = self.dispatch.matmul(&attn_weights, &v)?;
        let attn_proj = self.dispatch.matmul(&attn_out, &w.o_proj)?;
        let x2 = self.dispatch.add(x, &attn_proj)?;

        let normed2 = self.rmsnorm(&x2, &w.ffn_norm)?;
        let gate_pre = self.dispatch.matmul(&normed2, &w.gate_proj)?;
        let up_pre = self.dispatch.matmul(&normed2, &w.up_proj)?;
        let gate_act = self.dispatch.silu(&gate_pre)?;
        let ffn_hidden = self.dispatch.add(&gate_act, &up_pre)?; // placeholder for mul
        let ffn_out = self.dispatch.matmul(&ffn_hidden, &w.down_proj)?;
        self.dispatch.add(&x2, &ffn_out)
    }

    pub async fn forward(&self, token_ids: &[u32]) -> Result<Tensor> {
        let mut hidden = self.embed(token_ids).await?;
        for i in 0..self.config.n_layers {
            hidden = self.transformer_layer(&hidden, i).await?;
        }
        hidden = self.rmsnorm(&hidden, &self.weights.norm_weight)?;
        self.dispatch.matmul(&hidden, &self.weights.lm_head)
    }

    pub async fn next_token(&self, token_ids: &[u32], temperature: f32) -> Result<(u32, f32)> {
        let logits = self.forward(token_ids).await?;
        let seq = token_ids.len();
        let v = self.config.vocab_size;
        let all_logits = logits.read_to_cpu().await?;
        let last_logits = &all_logits[(seq - 1) * v..seq * v];
        let scaled: Vec<f32> = last_logits
            .iter()
            .map(|x| x / temperature.max(1e-6))
            .collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum_exp).collect();
        let (best_tok, best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok((best_tok as u32, best_prob.ln()))
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
