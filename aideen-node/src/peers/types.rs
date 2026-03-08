use serde::{Deserialize, Serialize};

/// Alias canónico. Mismo tipo que node_id en NetMsg (no newtype para evitar conversiones).
pub type NodeId = [u8; 32];

/// Metadata de un peer experto. No contiene canales — solo descripción.
/// Los canales (`NetChannel`) se gestionan por separado en `ExpertClient`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct PeerEntry {
    pub node_id: NodeId,
    /// Endpoint QUIC futuro ("quic://host:port"). Vacío en 5J (sin dial real).
    pub endpoint: String,
    /// Dominios que sirve este peer. Normalizados a lowercase en PeerRegistry::insert().
    pub domains: Vec<String>,
    pub bundle_version: u64,
    /// Slot para TLS self-signed pinning (Sprint 5K). None en 5J.
    pub tls_fingerprint: Option<[u8; 32]>,
}

/// Update incremental del directorio de peers.
/// El camino normal (no bootstrap): un nodo entra/sale → 1 registro, no full-push.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PeerDelta {
    /// Monotónico. `apply_delta` rechaza si epoch <= registry.epoch.
    pub epoch: u64,
    pub upserts: Vec<PeerEntry>,
    pub removes: Vec<NodeId>,
}
