// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Sergio Ariel Solis and Juan Patricio Marchetto

//! aideen-backbone
//! ─────────────────────────────────────────────
//! Model Architecture and base Weights.
//! ML framework-agnostic (No Candle, No PyTorch) for P2P streaming.

pub mod architecture;
pub mod deq_mode;
pub mod ffn_reasoning;
pub mod fixed_point_memory_decoder;
pub mod fixed_point_memory_reasoning;
pub mod generation_strategy;
pub mod gpu_backend;
#[cfg(feature = "wgpu")]
pub mod gpu_deq;
#[cfg(feature = "wgpu")]
pub mod gpu_embedding;
#[cfg(feature = "wgpu")]
pub mod gpu_lm_head;
pub mod linear_reasoning;
pub mod lm_head;
pub mod readout;
pub mod spectral_norm;
pub mod tensor;
pub mod tokenizer;

pub use fixed_point_memory_decoder::{ClassHead, EmbedHead, FixedPointMemoryDecoder};
pub use fixed_point_memory_reasoning::FixedPointMemoryReasoning;
pub use gpu_backend::CpuBlockBackend;
#[cfg(feature = "wgpu")]
pub use gpu_backend::WgpuBlockBackend;
pub use lm_head::LmHead;
