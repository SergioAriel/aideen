use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DeviceClass {
    Mobile,
    Desktop,
    Server,
    Unknown,
}

/// Siempre Unknown en detect() sync.
/// Sprint 5K añadirá probe async para detectar GPU real via wgpu.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Accel {
    Gpu { vendor: String },
    Npu,
    Unknown,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantLevel {
    F32,
    F16,
    Int8,
    Int4,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeCapabilities {
    pub device_class: DeviceClass,
    pub cpu_threads: u32,
    /// RAM en MB. 0 si desconocido (WASM sin DeviceMemory API).
    pub ram_mb: u64,
    /// Siempre Unknown en la detección sync de 5I.
    pub accel: Accel,
    pub quant_support: Vec<QuantLevel>,
}
