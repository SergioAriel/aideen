//! aideen-training
//! ─────────────────────────────────────────────
//! Infraestructura de entrenamiento para AIDEEN.
//! Separado del backbone para que backbone pueda compilarse a WASM.

pub mod checkpoint;
pub mod dataset;
pub mod gradients;
pub mod loss;
pub mod optimizer;
pub mod signer;
pub mod trainer;

pub use trainer::Trainer;
