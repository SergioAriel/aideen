// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Sergio Ariel Solis and Juan Patricio Marchetto

//! aideen-training
//! ─────────────────────────────────────────────
//! Training infrastructure for AIDEEN.
//! Separated from the backbone so backbone can compile to WASM.

pub mod checkpoint;
pub mod dataset;
pub mod gradients;
pub mod loss;
pub mod optimizer;
pub mod signer;
pub mod trainer;

pub use trainer::Trainer;
