#![cfg(not(target_arch = "wasm32"))]

use aideen_core::capabilities::{Accel, QuantLevel};

#[test]
fn test_detect_native_caps() {
    let caps = aideen_node::capabilities::detect();
    assert!(caps.cpu_threads >= 1, "cpu_threads must be >= 1");
    assert!(
        caps.quant_support.contains(&QuantLevel::F32),
        "F32 must be supported"
    );
    assert!(
        matches!(caps.accel, Accel::Unknown),
        "Accel must be Unknown in sync detect()"
    );
    // ram_mb can be 0 only on WASM; on native it must be > 0
    assert!(caps.ram_mb > 0, "ram_mb must be > 0 on native");
}
