pub mod signer;

use aideen_core::protocol::SignedUpdate;

/// Discovery evidence extracted from a NetMsg::Discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryEvidence {
    pub node_id: [u8; 32],
    pub target_id: String,
    pub q_total: f32,
    pub iters: u32,
    pub h_star_hash: [u8; 32],
    pub context_hash: [u8; 32],
    pub bundle_version: u64,
}

/// Result of a node replay verification.
#[derive(Debug, Clone)]
pub struct ReplayEvidence {
    pub sample_id: u64,
    pub node_id: [u8; 32],
    pub reproduced: bool,
    pub q_recomputed: f32,
    pub trace_digest: [u8; 32],
}

/// Routing and stability telemetry from a node.
#[derive(Debug, Clone)]
pub struct RouterStatsEvidence {
    pub node_id: [u8; 32],
    pub q_mean: f32,
    pub delta_norm_mean: f32,
    pub drops_count: u32,
}

/// Aggregated evidence bundle submitted to the Critic for update generation.
pub struct EvidenceBundle {
    pub discoveries: Vec<DiscoveryEvidence>,
    pub replay_results: Vec<ReplayEvidence>,
    pub router_stats: Vec<RouterStatsEvidence>,
}

/// Critic: proposes signed model updates from aggregated evidence.
pub trait Critic {
    fn propose_update(
        &mut self,
        target_id: &str,
        evidence: EvidenceBundle,
    ) -> Result<SignedUpdate, String>;
}
