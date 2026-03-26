use serde::{Deserialize, Serialize};

use crate::capabilities::QuantLevel;

/// Expert domain. Flexible string ("math", "code", "reasoning", ...).
/// Normalize to lowercase at ingestion if cross-lang compat is required.
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
