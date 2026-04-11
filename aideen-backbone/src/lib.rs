//! aideen-backbone
//! ─────────────────────────────────────────────
//! Arquitectura del Modelo y Pesos base.
//! Agnóstico de ML frameworks (No Candle, No PyTorch) para P2P streaming.

pub mod architecture;
pub mod deq_mode;
pub mod ffn_reasoning;
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
pub mod fixed_point_memory_decoder;
pub mod fixed_point_memory_reasoning;
pub mod readout;
pub mod spectral_norm;
pub mod tensor;
pub mod tokenizer;

pub use gpu_backend::CpuBlockBackend;
#[cfg(feature = "wgpu")]
pub use gpu_backend::WgpuBlockBackend;
pub use lm_head::LmHead;
pub use fixed_point_memory_decoder::{ClassHead, EmbedHead, FixedPointMemoryDecoder};
pub use fixed_point_memory_reasoning::FixedPointMemoryReasoning;
