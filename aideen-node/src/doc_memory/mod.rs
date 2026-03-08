#[cfg(not(target_arch = "wasm32"))]
pub mod fs;

#[cfg(target_arch = "wasm32")]
pub mod opfs;

#[cfg(not(target_arch = "wasm32"))]
pub use fs::FsDocMemory;

#[cfg(target_arch = "wasm32")]
pub use opfs::OpfsDocMemory;
