#![cfg(not(target_arch = "wasm32"))]

use aideen_core::capabilities::{Accel, QuantLevel};

#[test]
fn test_detect_native_caps() {
    let caps = aideen_node::capabilities::detect();
    assert!(caps.cpu_threads >= 1, "cpu_threads debe ser >= 1");
    assert!(
        caps.quant_support.contains(&QuantLevel::F32),
        "F32 debe estar soportado"
    );
    assert!(
        matches!(caps.accel, Accel::Unknown),
        "Accel debe ser Unknown en detect() sync"
    );
    // ram_mb puede ser 0 solo en WASM; en native debe ser > 0
    assert!(caps.ram_mb > 0, "ram_mb debe ser > 0 en native");
}
