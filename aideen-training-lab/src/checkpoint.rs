use aideen_backbone::fixed_point_memory_reasoning::FixedPointMemoryReasoning;
use aideen_core::model::AidenModel;
use std::path::Path;

/// Carga un checkpoint `.aidn` y retorna un `FixedPointMemoryReasoning`.
pub fn load<P: AsRef<Path>>(path: P) -> Result<FixedPointMemoryReasoning, String> {
    let model = AidenModel::load(path.as_ref().to_str().ok_or("Invalid path")?)?;

    let mut reasoning = FixedPointMemoryReasoning::new(model.config.clone());
    reasoning.import_weights(&model.weights)?;

    Ok(reasoning)
}
