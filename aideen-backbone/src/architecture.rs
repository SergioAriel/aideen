use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Generic linear layer (pure Weights and Biases)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// Framework-agnostic Transformer layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerLayer {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub ffn_up: LinearLayer,
    pub ffn_down: LinearLayer,
}

/// Distributed Mixture of Experts (MoE) definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoE {
    pub num_experts: usize,
    /// In a P2P environment, the actual experts could reside on other nodes.
    /// Here we only define the routing structure (e.g. gating layer).
    pub router: LinearLayer,
}
