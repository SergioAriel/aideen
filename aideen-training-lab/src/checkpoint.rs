use aideen_backbone::mamba_slot_reasoning::MambaSlotReasoning;
use aideen_core::model::AidenModel;
use std::path::Path;

/// Loads a `.aidn` checkpoint and returns a `MambaSlotReasoning`.
pub fn load<P: AsRef<Path>>(path: P) -> Result<MambaSlotReasoning, String> {
    let model = AidenModel::load(path.as_ref().to_str().ok_or("Invalid path")?)?;

    let mut reasoning = MambaSlotReasoning::new(model.config.clone());
    reasoning.import_weights(&model.weights)?;

    Ok(reasoning)
}
