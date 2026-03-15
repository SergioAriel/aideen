/// Checkpoint save/load utilities for the Trainer.
///
/// Saves model weights (safetensors), optimizer state (bincode), and
/// metadata (JSON) into a checkpoint directory.

use crate::trainer::Trainer;
use aideen_backbone::weights_io;
use serde::{Deserialize, Serialize};

/// Metadata saved alongside the checkpoint.
#[derive(Serialize, Deserialize)]
struct CheckpointMeta {
    step_count: usize,
    lr: f32,
    config: aideen_core::state::ArchitectureConfig,
}

/// Save a full training checkpoint to the given directory.
///
/// Creates:
///   - `{dir}/model.safetensors` — model weights
///   - `{dir}/optimizer.bin`     — optimizer state (bincode)
///   - `{dir}/meta.json`        — step_count, lr, config
pub fn save_checkpoint(trainer: &Trainer, dir: &str) -> Result<(), String> {
    // 1. Create directory if needed
    std::fs::create_dir_all(dir)
        .map_err(|e| format!("Failed to create checkpoint dir {}: {}", dir, e))?;

    // 2. Save model weights via safetensors
    let model_path = format!("{}/model.safetensors", dir);
    weights_io::save_model(&trainer.reasoning, &trainer.lm_head, &model_path)?;

    // 3. Save optimizer state via bincode
    let opt_path = format!("{}/optimizer.bin", dir);
    let opt_bytes = bincode::serialize(&trainer.optimizer)
        .map_err(|e| format!("Failed to serialize optimizer: {}", e))?;
    std::fs::write(&opt_path, &opt_bytes)
        .map_err(|e| format!("Failed to write {}: {}", opt_path, e))?;

    // 4. Save metadata as JSON
    let meta = CheckpointMeta {
        step_count: trainer.step_count,
        lr: trainer.training_config.lr,
        config: trainer.config.clone(),
    };
    let meta_json = serde_json::to_string_pretty(&meta)
        .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
    let meta_path = format!("{}/meta.json", dir);
    std::fs::write(&meta_path, &meta_json)
        .map_err(|e| format!("Failed to write {}: {}", meta_path, e))?;

    Ok(())
}

/// Load a training checkpoint from the given directory.
///
/// Restores model weights, optimizer state (if present), and metadata (if present).
pub fn load_checkpoint(trainer: &mut Trainer, dir: &str) -> Result<(), String> {
    // 1. Load model weights
    let model_path = format!("{}/model.safetensors", dir);
    let (reasoning, lm_head) = weights_io::load_model(&model_path, &trainer.config)?;
    trainer.reasoning = reasoning;
    trainer.lm_head = lm_head;

    // 2. Load optimizer state if file exists
    let opt_path = format!("{}/optimizer.bin", dir);
    if std::path::Path::new(&opt_path).exists() {
        let opt_bytes = std::fs::read(&opt_path)
            .map_err(|e| format!("Failed to read {}: {}", opt_path, e))?;
        let optimizer = bincode::deserialize(&opt_bytes)
            .map_err(|e| format!("Failed to deserialize optimizer: {}", e))?;
        trainer.optimizer = optimizer;
    }

    // 3. Load metadata if file exists
    let meta_path = format!("{}/meta.json", dir);
    if std::path::Path::new(&meta_path).exists() {
        let meta_json = std::fs::read_to_string(&meta_path)
            .map_err(|e| format!("Failed to read {}: {}", meta_path, e))?;
        let meta: CheckpointMeta = serde_json::from_str(&meta_json)
            .map_err(|e| format!("Failed to parse metadata: {}", e))?;
        trainer.step_count = meta.step_count;
        trainer.training_config.lr = meta.lr;
        trainer.config = meta.config;
    }

    Ok(())
}
