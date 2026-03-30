use serde::{Deserialize, Serialize};

/// Canonical alias. Same type as node_id in NetMsg (no newtype to avoid conversions).
pub type NodeId = [u8; 32];

/// Expert peer metadata. Does not contain channels — description only.
/// Channels (`NetChannel`) are managed separately in `ExpertClient`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PeerEntry {
    pub node_id: NodeId,
    /// Future QUIC endpoint ("quic://host:port"). Empty in 5J (no real dial).
    pub endpoint: String,
    /// Domains served by this peer. Normalised to lowercase in PeerRegistry::insert().
    pub domains: Vec<String>,
    pub bundle_version: u64,
    /// Slot for TLS self-signed pinning (Sprint 5K). None in 5J.
    pub tls_fingerprint: Option<[u8; 32]>,
}

/// Update incremental del directorio de peers.
/// El camino normal (no bootstrap): un nodo entra/sale → 1 registro, no full-push.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PeerDelta {
    /// Monotonic. `apply_delta` rejects if epoch <= registry.epoch.
    pub epoch: u64,
    pub upserts: Vec<PeerEntry>,
    pub removes: Vec<NodeId>,
}
