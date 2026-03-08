use aideen_core::artifacts::ArtifactMeta;
use aideen_core::capabilities::NodeCapabilities;

use crate::artifacts::policy::SelectionPolicy;

pub struct ArtifactSelector {
    pub policy: Box<dyn SelectionPolicy>,
}

impl ArtifactSelector {
    pub fn new(policy: Box<dyn SelectionPolicy>) -> Self {
        Self { policy }
    }

    pub fn select<'a>(
        &self,
        caps: &NodeCapabilities,
        available: &'a [ArtifactMeta],
    ) -> Vec<&'a ArtifactMeta> {
        self.policy.select(caps, available)
    }
}
