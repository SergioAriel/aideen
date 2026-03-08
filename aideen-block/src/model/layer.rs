use super::config::AideenConfig;

/// A single Aideen layer.
/// Holds the configuration context needed to execute a forward step.
pub struct AideenLayer {
    pub idx: usize,
    pub d_in: usize,
    pub d_out: usize,
    /// Attention stream head dims — None for Block A.
    pub attn_dims: Option<(usize, usize)>,
    pub has_moe: bool,
}

impl AideenLayer {
    pub fn new(idx: usize, config: &AideenConfig) -> Self {
        let (d_in, d_out) = config.layer_dims(idx);
        Self {
            idx,
            d_in,
            d_out,
            attn_dims: config.attn_dims(idx),
            has_moe: config.has_moe(idx),
        }
    }
}

/// Build the complete layer stack for an AideenConfig.
pub fn build_layer_stack(config: &AideenConfig) -> Vec<AideenLayer> {
    (0..config.num_layers())
        .map(|i| AideenLayer::new(i, config))
        .collect()
}
