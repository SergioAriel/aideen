#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use aideen_core::capabilities::NodeCapabilities;

pub fn detect() -> NodeCapabilities {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native::detect_native()
    }
    #[cfg(target_arch = "wasm32")]
    {
        wasm::detect_wasm()
    }
}
