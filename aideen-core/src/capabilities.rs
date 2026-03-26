use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DeviceClass {
    Mobile,
    Desktop,
    Server,
    Unknown,
}

/// Always Unknown in sync detect().
/// Sprint 5K will add async probe to detect real GPU via wgpu.
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
    /// RAM in MB. 0 if unknown (WASM without DeviceMemory API).
    pub ram_mb: u64,
    /// Always Unknown in the 5I sync detection.
    pub accel: Accel,
    pub quant_support: Vec<QuantLevel>,
}
