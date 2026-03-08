pub mod agent;
pub mod doc_memory;
pub mod memory;

// Módulos nativos: dependen de tokio, QUIC y backbone, no compilan en wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub mod critic;
#[cfg(not(target_arch = "wasm32"))]
pub mod expert;
#[cfg(not(target_arch = "wasm32"))]
pub mod inference;
#[cfg(not(target_arch = "wasm32"))]
pub mod network;
#[cfg(not(target_arch = "wasm32"))]
pub mod runner;
#[cfg(not(target_arch = "wasm32"))]
pub mod system;
#[cfg(not(target_arch = "wasm32"))]
pub mod update;
// capabilities: sin cfg — tiene implementación native y WASM
pub mod capabilities;
// artifacts: native-only (store/policy local, no necesario en browser)
#[cfg(not(target_arch = "wasm32"))]
pub mod artifacts;
// peers: directorio indexado de peers (native-only, no dial en wasm)
#[cfg(not(target_arch = "wasm32"))]
pub mod peers;
// security: TOFU + pinning de fingerprints TLS (native-only)
#[cfg(not(target_arch = "wasm32"))]
pub mod security;
