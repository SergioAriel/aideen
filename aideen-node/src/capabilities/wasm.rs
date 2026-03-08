use aideen_core::capabilities::{Accel, DeviceClass, NodeCapabilities, QuantLevel};

pub fn detect_wasm() -> NodeCapabilities {
    let threads = js_sys::Reflect::get(
        &web_sys::window().expect("no window").navigator().into(),
        &"hardwareConcurrency".into(),
    )
    .ok()
    .and_then(|v| v.as_f64())
    .unwrap_or(1.0) as u32;

    NodeCapabilities {
        device_class: DeviceClass::Unknown,
        cpu_threads: threads.max(1),
        ram_mb: 0,
        accel: Accel::Unknown,
        quant_support: vec![QuantLevel::F32],
    }
}
