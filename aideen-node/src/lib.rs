pub mod agent;
pub mod doc_memory;
pub mod memory;

// Native modules: depend on tokio, QUIC and backbone, do not compile on wasm32.
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
// capabilities: no cfg — has both native and WASM implementations
pub mod capabilities;
// artifacts: native-only (local store/policy, not needed in the browser)
#[cfg(not(target_arch = "wasm32"))]
pub mod artifacts;
// peers: indexed peer directory (native-only, no dialing on wasm)
#[cfg(not(target_arch = "wasm32"))]
pub mod peers;
// security: TOFU + TLS fingerprint pinning (native-only)
#[cfg(not(target_arch = "wasm32"))]
pub mod security;
