/// Aideen V8 — Architecture Configuration
///
/// Two configs available:
///   mini_v8 — 8 layers, D=64→128→256→512, for development & validation
///   full_v8 — 40 layers, D=256→512→1024→2048, production scale
///
/// Progressive dimensionality across 4 blocks:
///   Block A: Fixed-Point Memory only
///   Block B: Fixed-Point Memory + 2D-Attn (light+heavy)
///   Block C: Fixed-Point Memory + 2D-Attn (medium)
///   Block D: Fixed-Point Memory + 2D-Attn (full) + MoE

/// Block-level configuration.
#[derive(Debug, Clone)]
pub struct BlockConfig {
    pub n_layers: usize,
    pub d_model: usize,
    pub attn_dims: Option<(usize, usize)>, // (light_head, heavy_head) or None
    pub has_moe: bool,
}

/// Model-wide hyperparameters. Constructed once at engine init.
#[derive(Debug, Clone)]
pub struct AideenConfig {
    pub vocab_size: usize,
    pub blocks: Vec<BlockConfig>, // A, B, C, D in order
    pub d_state: usize,           // Fixed-Point Memory SSM state dimension
    pub d_conv: usize,            // Fixed-Point Memory 1D conv width
    pub expand: usize,            // Fixed-Point Memory inner expand factor
    pub num_local_experts: usize,
    pub num_total_experts: usize,
    pub k_min: usize,
    pub k_max: usize,
    pub expert_ffn_expand: usize,
}

impl AideenConfig {
    /// Mini V8 — 8 layers, fits CPU RAM (~271 MB), matches aideen_mini_weights.safetensors.
    pub fn mini_v8() -> Self {
        Self {
            vocab_size: 50257,
            blocks: vec![
                BlockConfig {
                    n_layers: 2,
                    d_model: 64,
                    attn_dims: None,
                    has_moe: false,
                },
                BlockConfig {
                    n_layers: 2,
                    d_model: 128,
                    attn_dims: Some((8, 16)),
                    has_moe: false,
                },
                BlockConfig {
                    n_layers: 2,
                    d_model: 256,
                    attn_dims: Some((16, 32)),
                    has_moe: false,
                },
                BlockConfig {
                    n_layers: 2,
                    d_model: 512,
                    attn_dims: Some((32, 64)),
                    has_moe: true,
                },
            ],
            d_state: 16,
            d_conv: 4,
            expand: 2,
            num_local_experts: 2,
            num_total_experts: 8,
            k_min: 1,
            k_max: 2,
            expert_ffn_expand: 4,
        }
    }

    /// Full V8 — 40 layers, production scale.
    pub fn full_v8() -> Self {
        Self {
            vocab_size: 50257,
            blocks: vec![
                BlockConfig {
                    n_layers: 10,
                    d_model: 256,
                    attn_dims: None,
                    has_moe: false,
                },
                BlockConfig {
                    n_layers: 10,
                    d_model: 512,
                    attn_dims: Some((32, 64)),
                    has_moe: false,
                },
                BlockConfig {
                    n_layers: 14,
                    d_model: 1024,
                    attn_dims: Some((64, 128)),
                    has_moe: false,
                },
                BlockConfig {
                    n_layers: 6,
                    d_model: 2048,
                    attn_dims: Some((128, 256)),
                    has_moe: true,
                },
            ],
            d_state: 64,
            d_conv: 4,
            expand: 2,
            num_local_experts: 8,
            num_total_experts: 64,
            k_min: 1,
            k_max: 8,
            expert_ffn_expand: 4,
        }
    }

    /// Total number of layers across all blocks.
    pub fn num_layers(&self) -> usize {
        self.blocks.iter().map(|b| b.n_layers).sum()
    }

    /// Which block a layer index belongs to (0-indexed).
    fn block_of_layer(&self, layer_idx: usize) -> &BlockConfig {
        let mut offset = 0;
        for block in &self.blocks {
            if layer_idx < offset + block.n_layers {
                return block;
            }
            offset += block.n_layers;
        }
        self.blocks.last().expect("at least one block")
    }

    /// (d_in, d_out) for a layer — residual within blocks.
    pub fn layer_dims(&self, layer_idx: usize) -> (usize, usize) {
        let d = self.block_of_layer(layer_idx).d_model;
        (d, d)
    }

    /// D_model at a given layer (input dimension).
    pub fn d_in(&self, layer_idx: usize) -> usize {
        self.block_of_layer(layer_idx).d_model
    }

    /// D_model at the output of the model (final block).
    pub fn d_final(&self) -> usize {
        self.blocks.last().expect("at least one block").d_model
    }

    /// Inter-block projection: returns (d_from, d_to) if D changes before this layer.
    pub fn inter_block_projection(&self, layer_idx: usize) -> Option<(usize, usize)> {
        let mut offset = 0;
        let mut prev_d: Option<usize> = None;
        for block in &self.blocks {
            if layer_idx == offset {
                if let Some(pd) = prev_d {
                    if pd != block.d_model {
                        return Some((pd, block.d_model));
                    }
                }
            }
            if layer_idx < offset + block.n_layers {
                return None;
            }
            prev_d = Some(block.d_model);
            offset += block.n_layers;
        }
        None
    }

    /// Attention head dimensions for the two streams at a given layer.
    pub fn attn_dims(&self, layer_idx: usize) -> Option<(usize, usize)> {
        self.block_of_layer(layer_idx).attn_dims
    }

    /// Whether this layer includes MoE routing.
    pub fn has_moe(&self, layer_idx: usize) -> bool {
        self.block_of_layer(layer_idx).has_moe
    }
}
