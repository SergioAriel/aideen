//! Safetensors weight I/O for the AIDEEN DEQ model.
//!
//! Serializes and deserializes all [`MambaSlotReasoning`] and [`LmHead`]
//! weights into a single `.safetensors` file using the `safetensors` crate.
//!
//! Internally this delegates to the existing `export_weights()` /
//! `import_weights()` methods on each component, so the canonical tensor
//! names (`reasoning.*`, `head.*`) are preserved.

use crate::lm_head::LmHead;
use crate::mamba_slot_reasoning::MambaSlotReasoning;
use aideen_core::state::ArchitectureConfig;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;

// ── Shape metadata ──────────────────────────────────────────────────────────
//
// The existing `export_weights()` methods flatten every tensor to `Vec<f32>`.
// To round-trip through safetensors we need the original [rows, cols] (or [len])
// shape. We reconstruct that from the `ArchitectureConfig` and the tensor name.

fn tensor_shape(name: &str, config: &ArchitectureConfig) -> Vec<usize> {
    let d_r = config.d_r;
    let h_slots = config.h_slots;
    let vocab_size = config.vocab_size;

    match name {
        // ── MambaSlotReasoning: square d_r × d_r matrices ────────────────
        "reasoning.w_q"
        | "reasoning.w_k"
        | "reasoning.w_v"
        | "reasoning.w_o"
        | "reasoning.w_in"
        | "reasoning.w_hist_shared"
        | "reasoning.w_x"
        | "reasoning.w_out"
        | "reasoning.w_delta" => vec![d_r, d_r],

        // ── h_slots × d_r matrices ───────────────────────────────────────
        "reasoning.hist_slot_scale"
        | "reasoning.hist_slot_bias"
        | "reasoning.slot_anchor" => vec![h_slots, d_r],

        // ── h_slots vector ───────────────────────────────────────────────
        "reasoning.hist_gate_logit" => vec![h_slots],

        // ── d_r vectors ──────────────────────────────────────────────────
        "reasoning.a_log"
        | "reasoning.b_delta"
        | "reasoning.norm_scale" => vec![d_r],

        // ── scalars ──────────────────────────────────────────────────────
        "reasoning.damping" | "reasoning.residual_alpha" => vec![1],

        // ── LmHead ──────────────────────────────────────────────────────
        "head.w" => vec![vocab_size, d_r],
        "head.b" => vec![vocab_size],
        "head.g" => vec![d_r],

        // Fallback: 1-D with the data length (should not happen for known
        // weight names, but avoids panics on forward-compat additions).
        _ => vec![0], // caller will override with actual length
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Save all DEQ model weights to a safetensors file.
///
/// The file will contain every tensor produced by
/// [`MambaSlotReasoning::export_weights`] and [`LmHead::export_weights`],
/// stored as contiguous F32 data with their original shape metadata.
pub fn save_model(
    reasoning: &MambaSlotReasoning,
    lm_head: &LmHead,
    path: &str,
) -> Result<(), String> {
    let config = &reasoning.config;

    // Collect flat weight maps from both components.
    let mut all_weights: HashMap<String, Vec<f32>> = reasoning.export_weights();
    all_weights.extend(lm_head.export_weights());

    // Build TensorView list.  safetensors needs references that live until
    // serialization is done, so we pre-collect the byte slices.
    let byte_data: HashMap<String, Vec<u8>> = all_weights
        .iter()
        .map(|(name, data)| {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            (name.clone(), bytes)
        })
        .collect();

    let mut tensors: Vec<(String, TensorView<'_>)> = Vec::with_capacity(byte_data.len());
    for (name, bytes) in &byte_data {
        let mut shape = tensor_shape(name, config);
        // For the fallback case (unknown name), use actual element count.
        if shape == [0] {
            shape = vec![bytes.len() / 4];
        }
        let view = TensorView::new(Dtype::F32, shape, bytes)
            .map_err(|e| format!("Failed to create TensorView for {}: {}", name, e))?;
        tensors.push((name.clone(), view));
    }

    // Store the ArchitectureConfig as JSON in the __metadata__ section so
    // that `load_model` can reconstruct the config from the file alone when
    // needed in the future.  For now the caller supplies the config.
    let config_json = serde_json::to_string(config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;
    let mut meta_map = HashMap::new();
    meta_map.insert("architecture_config".to_string(), config_json);

    let serialized = safetensors::tensor::serialize(tensors, &Some(meta_map))
        .map_err(|e| format!("safetensors serialization failed: {}", e))?;

    std::fs::write(path, &serialized)
        .map_err(|e| format!("Failed to write {}: {}", path, e))?;

    Ok(())
}

/// Load all DEQ model weights from a safetensors file.
///
/// The caller provides the [`ArchitectureConfig`] that determines model
/// dimensions.  Fresh `MambaSlotReasoning` and `LmHead` instances are
/// created with random init, then their weights are overwritten via
/// `import_weights()`.
pub fn load_model(
    path: &str,
    config: &ArchitectureConfig,
) -> Result<(MambaSlotReasoning, LmHead), String> {
    let data =
        std::fs::read(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;

    let tensors = SafeTensors::deserialize(&data)
        .map_err(|e| format!("safetensors deserialization failed: {}", e))?;

    // Convert every tensor back into the HashMap<String, Vec<f32>> format
    // that import_weights() expects.
    let mut weight_map: HashMap<String, Vec<f32>> = HashMap::new();

    for (name, tensor) in tensors.tensors() {
        if tensor.dtype() != Dtype::F32 {
            return Err(format!(
                "Tensor {} has dtype {:?}, expected F32",
                name,
                tensor.dtype()
            ));
        }
        let raw = tensor.data();
        if raw.len() % 4 != 0 {
            return Err(format!(
                "Tensor {} data length {} is not a multiple of 4",
                name,
                raw.len()
            ));
        }
        let floats: Vec<f32> = raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        weight_map.insert(name.to_string(), floats);
    }

    // Build fresh instances and overwrite weights.
    let mut reasoning = MambaSlotReasoning::new(config.clone());
    reasoning.import_weights(&weight_map)?;

    let mut lm_head = LmHead::new(config.clone());
    lm_head.import_weights(&weight_map)?;

    Ok((reasoning, lm_head))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip test: save → load → compare all weights.
    #[test]
    fn roundtrip_save_load() {
        let mut config = ArchitectureConfig::default();
        // Use small dims so the test is fast.
        config.d_r = 16;
        config.h_slots = 2;
        config.vocab_size = 64;

        let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), 42);
        let lm_head = LmHead::new(config.clone());

        let dir = std::env::temp_dir().join("aideen_weights_io_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_model.safetensors");
        let path_str = path.to_str().unwrap();

        // Save
        save_model(&reasoning, &lm_head, path_str).unwrap();

        // Load
        let (loaded_reasoning, loaded_lm_head) = load_model(path_str, &config).unwrap();

        // Compare reasoning weights
        let orig = reasoning.export_weights();
        let loaded = loaded_reasoning.export_weights();
        assert_eq!(orig.len(), loaded.len(), "reasoning weight count mismatch");
        for (name, orig_data) in &orig {
            let loaded_data = loaded.get(name).unwrap_or_else(|| {
                panic!("Missing reasoning weight after load: {}", name)
            });
            assert_eq!(
                orig_data.len(),
                loaded_data.len(),
                "Size mismatch for {}",
                name
            );
            for (i, (a, b)) in orig_data.iter().zip(loaded_data.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-7,
                    "Value mismatch for {} at index {}: {} vs {}",
                    name,
                    i,
                    a,
                    b
                );
            }
        }

        // Compare lm_head weights
        let orig_head = lm_head.export_weights();
        let loaded_head = loaded_lm_head.export_weights();
        assert_eq!(orig_head.len(), loaded_head.len(), "head weight count mismatch");
        for (name, orig_data) in &orig_head {
            let loaded_data = loaded_head.get(name).unwrap_or_else(|| {
                panic!("Missing head weight after load: {}", name)
            });
            assert_eq!(
                orig_data.len(),
                loaded_data.len(),
                "Size mismatch for {}",
                name
            );
            for (i, (a, b)) in orig_data.iter().zip(loaded_data.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-7,
                    "Value mismatch for {} at index {}: {} vs {}",
                    name,
                    i,
                    a,
                    b
                );
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
