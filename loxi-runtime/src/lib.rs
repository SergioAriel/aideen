// lib.rs
// ─────────────────────────────────────────────────────────────────────────────
// loxi-runtime: GPU-agnostic inference and training runtime.
//
// This crate provides:
//   - GPU compute via wgpu (Metal on M1, Vulkan/DX12 on PC, WebGPU on browser)
//   - Tensor operations (matmul, layernorm, softmax, silu, rope, adamw)
//   - C+D dynamics kernels for federated expert training
//   - Transformer model forward pass (LLaMA-style decoder-only)
//   - AdamW optimizer with LR schedule and gradient clipping
//   - Hybrid long-term memory (private HNSW + shared opt-in sync)
//   - Analytical gradient backward passes
//
// Dependency on candle: NONE.
// Dependency on CUDA:   NONE.
// Runs on: M1 Metal, Vulkan (AMD/Intel/NVIDIA), DX12 (Windows), WebGPU.
//
// Usage:
//   ```rust
//   use loxi_runtime::{GpuContext, LoxiModel, ModelConfig, MemoryStore};
//
//   let ctx   = GpuContext::new().await?;
//   let cfg   = ModelConfig::full();
//   let ws    = ModelWeights::load_safetensors(&bytes, &cfg, ctx.clone())?;
//   let model = LoxiModel::new(cfg, ws, ctx.clone());
//   let logits = model.forward(&token_ids).await?;
//   ```
// ─────────────────────────────────────────────────────────────────────────────

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(dead_code)]  // Many items used conditionally

pub mod tensor;
pub mod dispatch;
pub mod autograd;
pub mod model;
pub mod optimizer;
pub mod memory;

// ── Re-exports for convenient use ─────────────────────────────────────────────

// Tensor primitives
pub use tensor::{GpuContext, Shape, Tensor};

// Kernel dispatch
pub use dispatch::{
    Dispatcher, KernelId, KernelCache,
    MatMulDims, LayerNormParams, SoftmaxParams,
    ElemParams, AdamWParams, RopeParams, CdUpdateParams,
};

// Gradient functions
pub use autograd::{
    matmul_backward, layernorm_backward,
    cross_entropy_forward, cross_entropy_backward,
    silu_backward, CrossEntropyResult, MatMulGrad, LayerNormGrad,
};

// Model
pub use model::{LoxiModel, ModelConfig, ModelWeights, LayerWeights};

// Optimizer
pub use optimizer::{AdamW, Parameter, LrSchedule};

// Memory
pub use memory::{
    MemoryStore, MemoryEntry, MemoryPayload, MemoryScope, RetrievalResult,
    MemorySyncClient, MemorySyncMessage,
};

// ── Version ────────────────────────────────────────────────────────────────────
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
