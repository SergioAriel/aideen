use serde::{Deserialize, Serialize};

use crate::capabilities::QuantLevel;

/// Dominio del experto. String flexible ("math", "code", "reasoning", …).
/// Normalizar a lowercase en punto de ingesta si se requiere compat cross-lang.
pub type ExpertDomain = String;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtifactId {
    pub target_id: String,
    pub domain: ExpertDomain,
    pub version: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ArtifactKind {
    Expert,
    Backbone,
    Readout,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArtifactMeta {
    pub id: ArtifactId,
    pub kind: ArtifactKind,
    pub quant: QuantLevel,
    pub size_bytes: u64,
}
