// FsAgentStore solo disponible en native (std::fs no existe en wasm32)
#[cfg(not(target_arch = "wasm32"))]
pub mod fs;

// OpfsAgentStore solo en WASM (OPFS API es browser-only)
#[cfg(target_arch = "wasm32")]
pub mod opfs;

#[cfg(not(target_arch = "wasm32"))]
pub use fs::FsAgentStore;

#[cfg(target_arch = "wasm32")]
pub use opfs::OpfsAgentStore;
