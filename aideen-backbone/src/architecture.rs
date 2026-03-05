use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Capa lineal genérica (Pesos y Biases puros)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// Capa Transformer agnóstica
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerLayer {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,
    pub ffn_up: LinearLayer,
    pub ffn_down: LinearLayer,
}

/// Definición de Mixture of Experts (MoE) distribuido
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoE {
    pub num_experts: usize,
    /// En un entorno P2P, los expertos reales podrían estar en otros nodos.
    /// Aquí solo definimos la estructura de enrutamiento (ej. capa de gating).
    pub router: LinearLayer,
}
