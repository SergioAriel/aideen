use aideen_core::capabilities::{Accel, DeviceClass, NodeCapabilities, QuantLevel};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

pub fn detect_native() -> NodeCapabilities {
    let mut sys =
        System::new_with_specifics(RefreshKind::new().with_memory(MemoryRefreshKind::everything()));
    sys.refresh_memory();

    // sysinfo devuelve bytes (en sysinfo 0.30+); convertir a MB
    let ram_mb = sys.total_memory() / 1024 / 1024;

    // std::thread::available_parallelism evita un refresh_cpu adicional
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1);

    NodeCapabilities {
        device_class: DeviceClass::Unknown,
        cpu_threads: threads,
        ram_mb,
        accel: Accel::Unknown,
        quant_support: vec![QuantLevel::F32],
    }
}
