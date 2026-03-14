use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, loss, AdamW, Embedding, LayerNorm, Linear, Module, Optimizer,
    ParamsAdamW, VarBuilder, VarMap,
};

#[derive(Clone)]
pub struct CandleTransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub ctx_len: usize,
    pub n_layers: usize,
}

impl CandleTransformerConfig {
    pub fn param_count(&self) -> usize {
        let d = self.d_model;
        let v = self.vocab_size;
        let embed = v * d + self.ctx_len * d;
        let per_layer = d * (3 * d) + d * d + d * self.d_ff + self.d_ff * d + 4 * d;
        let lm_head = d * v + v;
        let ln_f = 2 * d;
        embed + self.n_layers * per_layer + lm_head + ln_f
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CandleBackend {
    Cpu,
    Metal,
}

struct CandleBlock {
    ln1: LayerNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ln2: LayerNorm,
    ff1: Linear,
    ff2: Linear,
}

pub struct CandleTransformer {
    cfg: CandleTransformerConfig,
    _vars: VarMap,
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks: Vec<CandleBlock>,
    ln_f: LayerNorm,
    lm_head: Linear,
    opt: AdamW,
    device: Device,
}

impl CandleTransformer {
    pub fn new(cfg: CandleTransformerConfig, lr: f64, backend: CandleBackend) -> Result<Self> {
        let device = match backend {
            CandleBackend::Cpu => Device::Cpu,
            CandleBackend::Metal => Device::new_metal(0)?,
        };
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        let tok_emb = embedding(cfg.vocab_size, cfg.d_model, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.ctx_len, cfg.d_model, vb.pp("pos_emb"))?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = vb.pp(format!("blocks.{i}"));
            blocks.push(CandleBlock {
                ln1: layer_norm(cfg.d_model, 1e-5, p.pp("ln1"))?,
                q_proj: linear(cfg.d_model, cfg.d_model, p.pp("q"))?,
                k_proj: linear(cfg.d_model, cfg.d_model, p.pp("k"))?,
                v_proj: linear(cfg.d_model, cfg.d_model, p.pp("v"))?,
                o_proj: linear(cfg.d_model, cfg.d_model, p.pp("o"))?,
                ln2: layer_norm(cfg.d_model, 1e-5, p.pp("ln2"))?,
                ff1: linear(cfg.d_model, cfg.d_ff, p.pp("ff1"))?,
                ff2: linear(cfg.d_ff, cfg.d_model, p.pp("ff2"))?,
            });
        }
        let ln_f = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_f"))?;
        let lm_head = linear(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))?;
        let opt = AdamW::new(
            vars.all_vars(),
            ParamsAdamW {
                lr,
                ..Default::default()
            },
        )?;

        Ok(Self {
            cfg,
            _vars: vars,
            tok_emb,
            pos_emb,
            blocks,
            ln_f,
            lm_head,
            opt,
            device,
        })
    }

    pub fn param_count(&self) -> usize {
        self.cfg.param_count()
    }

    fn causal_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let t = q.dims()[0];
        let h = self.cfg.n_heads;
        let dh = self.cfg.d_model / h;
        let scale = 1.0f64 / (dh as f64).sqrt();

        let mut heads = Vec::with_capacity(h);
        for head in 0..h {
            let off = head * dh;
            let qh = q.narrow(1, off, dh)?;
            let kh = k.narrow(1, off, dh)?;
            let vh = v.narrow(1, off, dh)?;

            let mut mask = vec![0f32; t * t];
            for i in 0..t {
                for j in (i + 1)..t {
                    mask[i * t + j] = -1e9;
                }
            }
            let mask = Tensor::from_slice(mask.as_slice(), (t, t), &self.device)?;

            let scores = qh.matmul(&kh.t()?)?.affine(scale, 0.)?.broadcast_add(&mask)?;
            let probs = candle_nn::ops::softmax(&scores, 1)?;
            let out = probs.matmul(&vh)?;
            heads.push(out);
        }
        let refs: Vec<&Tensor> = heads.iter().collect();
        Tensor::cat(&refs, 1)
    }

    fn forward_hidden(&self, inputs: &[u32]) -> Result<Tensor> {
        let t = inputs.len();
        let x_tok = Tensor::from_slice(inputs, t, &self.device)?;
        let mut x = self.tok_emb.forward(&x_tok)?;

        let pos_ids: Vec<u32> = (0..t as u32).collect();
        let pos = Tensor::from_slice(pos_ids.as_slice(), t, &self.device)?;
        let p = self.pos_emb.forward(&pos)?;
        x = x.broadcast_add(&p)?;

        for b in &self.blocks {
            let x1 = b.ln1.forward(&x)?;
            let q = b.q_proj.forward(&x1)?;
            let k = b.k_proj.forward(&x1)?;
            let v = b.v_proj.forward(&x1)?;
            let attn = self.causal_attention(&q, &k, &v)?;
            let attn = b.o_proj.forward(&attn)?;
            x = x.broadcast_add(&attn)?;

            let x2 = b.ln2.forward(&x)?;
            let ff = b.ff2.forward(&b.ff1.forward(&x2)?.gelu()?)?;
            x = x.broadcast_add(&ff)?;
        }
        self.ln_f.forward(&x)
    }

    pub fn train_step(&mut self, batch: &[u32]) -> Result<f32> {
        if batch.len() < 2 {
            return Ok(f32::NAN);
        }
        let inputs = &batch[..batch.len() - 1];
        let targets = &batch[1..];

        let h = self.forward_hidden(inputs)?;
        let logits = self.lm_head.forward(&h)?;
        let logits2 = logits.reshape((inputs.len(), self.cfg.vocab_size))?;
        let tgt = Tensor::from_slice(targets, targets.len(), &self.device)?.to_dtype(DType::U32)?;
        let ce = loss::cross_entropy(&logits2, &tgt)?;
        self.opt.backward_step(&ce)?;
        ce.to_vec0::<f32>()
    }

    pub fn val_loss(&self, tokens: &[u32]) -> Result<f32> {
        if tokens.len() < 2 {
            return Ok(f32::NAN);
        }
        let inputs = &tokens[..tokens.len() - 1];
        let targets = &tokens[1..];
        let h = self.forward_hidden(inputs)?;
        let logits = self.lm_head.forward(&h)?;
        let logits2 = logits.reshape((inputs.len(), self.cfg.vocab_size))?;
        let tgt = Tensor::from_slice(targets, targets.len(), &self.device)?.to_dtype(DType::U32)?;
        let ce = loss::cross_entropy(&logits2, &tgt)?;
        ce.to_vec0::<f32>()
    }
}
