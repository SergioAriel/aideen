// Aideen V8 - Javascript/WASM Interface
// Exposes the WebGPU Rust logic to the Chrome V8 Engine

use std::sync::Arc;
use wasm_bindgen::prelude::*;

use serde_json::Value;

/// Parsed tensor entry from the safetensors header
struct TensorMeta {
    name: String,
    offset: usize,
    length: usize,
}

/// Spec-compliant safetensors header parser.
///
/// Format (https://huggingface.co/docs/safetensors/index):
///   [8 bytes: u64 LE header_size] [header_size bytes: JSON] [data bytes]
///
/// Each JSON key (except `__metadata__`) maps to:
///   { "dtype": "F16", "shape": [...], "data_offsets": [start, end] }
///
/// `data_offsets` are byte offsets relative to the start of the data region
/// (i.e., byte 8 + header_size).
fn parse_safetensors_header(data: &[u8]) -> Result<Vec<TensorMeta>, JsValue> {
    if data.len() < 8 {
        return Err(JsValue::from_str(
            "safetensors: buffer too small (need ≥8 bytes for header length)",
        ));
    }

    let header_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let header_end = 8 + header_len;

    if data.len() < header_end {
        return Err(JsValue::from_str(&format!(
            "safetensors: truncated — need {} bytes, got {}",
            header_end,
            data.len()
        )));
    }

    let header_str = std::str::from_utf8(&data[8..header_end])
        .map_err(|e| JsValue::from_str(&format!("safetensors: invalid UTF-8 in header: {}", e)))?;

    let root: Value = serde_json::from_str(header_str)
        .map_err(|e| JsValue::from_str(&format!("safetensors: JSON parse error: {}", e)))?;

    let obj = root
        .as_object()
        .ok_or_else(|| JsValue::from_str("safetensors: header is not a JSON object"))?;

    let data_base = header_end;

    let metas = obj
        .iter()
        .filter(|(key, _)| *key != "__metadata__")
        .filter_map(|(name, entry)| {
            let offsets = entry.get("data_offsets")?.as_array()?;
            let start = offsets.first()?.as_u64()? as usize;
            let end = offsets.get(1)?.as_u64()? as usize;
            Some(TensorMeta {
                name: name.clone(),
                offset: data_base + start,
                length: end - start,
            })
        })
        .collect::<Vec<_>>();

    Ok(metas)
}

#[wasm_bindgen]
pub struct AideenEngine {
    // Arc allows the device and queue to be cloned into the async future_to_promise block.
    // In WASM everything is single-threaded, so Arc<T> is 'static even if T: !Send.
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    device_name: String,
    /// GPU buffers keyed by tensor name (e.g. "layers.0.fpm.A_log")
    weight_buffers: std::collections::HashMap<String, wgpu::Buffer>,
}

#[wasm_bindgen]
impl AideenEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            device: None,
            queue: None,
            device_name: "Aideen V8 WebGPU Engine (Initializing...)".to_string(),
            weight_buffers: std::collections::HashMap::new(),
        }
    }

    /// Asynchronously requests a WebGPU device from the Chrome browser
    #[wasm_bindgen]
    pub async fn initialize(&mut self) -> Result<JsValue, JsValue> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| JsValue::from_str("Failed to find a WebGPU adapter in this browser."))?;

        let subgroup_supported = adapter.features().contains(wgpu::Features::SUBGROUP);
        if subgroup_supported {
            web_sys::console::log_1(&JsValue::from_str("SUBGROUP supported."));
        } else {
            web_sys::console::log_1(&JsValue::from_str(
                "SUBGROUP not supported; using portable path.",
            ));
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Aideen Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to claim WebGPU device: {}", e)))?;

        // Wrap in Arc so they can be cloned into the run_inference async block
        self.device_name = adapter.get_info().name;
        self.device = Some(Arc::new(device));
        self.queue = Some(Arc::new(queue));

        Ok(JsValue::from_str(&self.device_name))
    }

    /// Parses raw `.safetensors` bytes and uploads every tensor directly into
    /// a dedicated `wgpu::Buffer` (VRAM) using `COPY_DST | STORAGE` usage flags.
    #[wasm_bindgen]
    pub fn load_safetensors(&mut self, buffer: &[u8]) -> Result<JsValue, JsValue> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| JsValue::from_str("Call initialize() before load_safetensors()"))?;
        let queue = self.queue.as_ref().unwrap();

        let metas = parse_safetensors_header(buffer)?;
        let tensor_count = metas.len();

        for meta in metas {
            let slice = &buffer[meta.offset..meta.offset + meta.length];
            let aligned_len = (slice.len() + 3) & !3;
            let mut padded = vec![0u8; aligned_len];
            padded[..slice.len()].copy_from_slice(slice);

            let gpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&meta.name),
                size: aligned_len as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            queue.write_buffer(&gpu_buffer, 0, &padded);
            self.weight_buffers.insert(meta.name, gpu_buffer);
        }

        Ok(JsValue::from_str(&format!(
            "✅ Loaded {} tensors into GPU VRAM ({:.2} MB)",
            tensor_count,
            buffer.len() as f64 / 1_048_576.0
        )))
    }

    #[wasm_bindgen]
    pub fn tensor_count(&self) -> usize {
        self.weight_buffers.len()
    }

    #[wasm_bindgen]
    pub fn device_name(&self) -> String {
        self.device_name.clone()
    }

    #[wasm_bindgen]
    pub fn list_weights(&self) -> String {
        let mut keys: Vec<&str> = self.weight_buffers.keys().map(|s| s.as_str()).collect();
        keys.sort();
        format!(
            "[{}]",
            keys.iter()
                .map(|k| format!("\"{}\"", k))
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    /// Run Loxi V8 forward pass. Returns a Promise<Float32Array>.
    ///
    /// The GPU pipeline (embedding → layers → lm_head → argmax) runs in the Web Worker.
    /// The map_async readback uses JsFuture::from(promise).await — correctly yields to the
    /// JavaScript event loop so the GPU callback can fire.
    #[wasm_bindgen]
    pub fn run_inference(&self, token_ids_js: &[u32]) -> js_sys::Promise {
        use crate::model::{forward::AideenForwardPass, AideenConfig};
        use wasm_bindgen_futures::future_to_promise;

        let device = match self.device.as_ref() {
            Some(d) => Arc::clone(d),
            None => {
                return js_sys::Promise::reject(&JsValue::from_str(
                    "run_inference: GPU not initialized — call initialize() first",
                ))
            }
        };
        let queue = match self.queue.as_ref() {
            Some(q) => Arc::clone(q),
            None => return js_sys::Promise::reject(&JsValue::from_str("queue not available")),
        };
        if self.weight_buffers.is_empty() {
            return js_sys::Promise::reject(&JsValue::from_str(
                "run_inference: no weights in VRAM — call load_safetensors() first",
            ));
        }
        if token_ids_js.is_empty() {
            return js_sys::Promise::reject(&JsValue::from_str(
                "run_inference: empty token sequence",
            ));
        }

        // Clone the weight buffer map so it can be moved into the async block
        // (wgpu::Buffer is 'static in WASM since everything is single-threaded)
        let weight_buffers: std::collections::HashMap<String, wgpu::Buffer> = {
            // We can't clone wgpu::Buffer directly, but we can share references.
            // Since this is single-threaded WASM, we use a reference-counted map.
            // For now, pass the buffers by reconstructing the forward pass each time.
            // The buffers live in self.weight_buffers — we pass them via a raw pointer approach.
            // SAFETY: In single-threaded WASM, self lives for the duration of the promise.
            // The weight_buffers HashMap will not be dropped while the Worker is running inference.
            //
            // TODO: migrate weight_buffers to Arc<HashMap<String, Arc<wgpu::Buffer>>>
            //       to make this fully safe. For now we use an empty map as a workaround
            //       and will wire the real map in the next step.
            std::collections::HashMap::new() // placeholder
        };
        let _ = weight_buffers; // suppress unused

        // To correctly share weight_buffers with the async block without unsafe,
        // we run the non-blocking parts synchronously and only await the map_async.
        let token_ids = token_ids_js.to_vec();

        // The forward pass is structured as:
        //   sync: embedding GPU dispatch
        //   sync: 40-layer stub dispatches
        //   sync: lm_head + argmax GPU dispatch + copy_to_staging
        //   ASYNC: staging buffer.map_async().await  ← only this needs the event loop
        //   sync: CPU reduce
        //
        // We get references (valid for 'static in WASM) to device/queue via Arc.
        // Weight buffers need special handling — see below.

        // APPROACH: pass weight_buffers as a raw pointer (safe in single-threaded WASM)
        let weight_ptr =
            &self.weight_buffers as *const std::collections::HashMap<String, wgpu::Buffer>;

        future_to_promise(async move {
            let config = AideenConfig::mini_v8();
            let pass = AideenForwardPass::new(config);

            // SAFETY: single-threaded WASM — self.weight_buffers lives as long as LoxiEngine
            // which outlives this Promise (the Worker holds the engine reference).
            let weight_buffers = unsafe { &*weight_ptr };

            let token = pass
                .run_async(&device, &queue, &token_ids, weight_buffers)
                .await?;

            let js_array = js_sys::Float32Array::new_with_length(1);
            js_array.copy_from(&[token]);
            Ok(js_array.into())
        })
    }
}
