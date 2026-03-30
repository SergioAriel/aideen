use aideen_core::artifacts::ArtifactMeta;
use aideen_core::capabilities::NodeCapabilities;

pub trait SelectionPolicy: Send {
    fn select<'a>(
        &self,
        caps: &NodeCapabilities,
        available: &'a [ArtifactMeta],
    ) -> Vec<&'a ArtifactMeta>;
}

/// Default policy: filters artifacts whose QuantLevel is supported by the node.
pub struct CompatPolicy;

impl SelectionPolicy for CompatPolicy {
    fn select<'a>(
        &self,
        caps: &NodeCapabilities,
        available: &'a [ArtifactMeta],
    ) -> Vec<&'a ArtifactMeta> {
        available
            .iter()
            .filter(|a| caps.quant_support.contains(&a.quant))
            .collect()
    }
}
