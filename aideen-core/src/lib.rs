// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Juan Marchetto & Sergio Solis

//! aideen-core
//! ─────────────────────────────────────────────
//! Mathematical contracts and base types (MRCE + S).
//! This crate does NOT execute logic.
//! This crate does NOT train.
//!
//! It is auditable, stable, and geometrically sealed.

pub mod agent;
pub mod artifacts;
pub mod block_backend;
pub mod capabilities;
pub mod compute;
pub mod control;
pub mod doc_memory;
pub mod ethics;
pub mod integrator;
pub mod memory;
pub mod model;
pub mod model_head;
pub mod protocol;
pub mod quality;
pub mod readout;
pub mod reasoning;
pub mod state;
pub mod types;
