use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Cursor;
use zstd::stream::{decode_all, encode_all};

use crate::artifacts::ArtifactMeta;

/// Base model hash (to ensure the delta applies to the same global snapshot).
pub type ModelHash = [u8; 32];

/// Identifies which part of the model is updated (e.g., a specific expert).
pub type TargetId = String;

/// Which parameter is touched within the expert.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamId {
    W1,
    W2,
    B1,
    B2,
}

/// Delta cuantizado: real_delta ≈ q * scale
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QuantizedDelta {
    pub param: ParamId,
    pub scale: f32,    // de-quantization
    pub idx: Vec<u32>, // absolute indices, sorted
    pub q: Vec<i16>,   // same len as idx
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelBundleManifest {
    pub bundle_version: u64,
    pub experts: Vec<ExpertEntry>, // sorted by target_id
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExpertEntry {
    pub target_id: String,
    pub expert_hash: [u8; 32],
}

/// Update signed by the Critic.
/// IMPORTANT: the client verifies signature + model_hash + anti-replay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignedUpdate {
    /// Monotonic version per (target_id). Prevents simple replay.
    pub version: u64,

    /// Identifies the affected expert / module.
    pub target_id: TargetId,

    /// Current global bundle version
    pub bundle_version: u64,

    /// Global bundle hash for network alignment
    pub bundle_hash: [u8; 32],

    /// Hash of the base model on which the delta was computed.
    pub base_model_hash: ModelHash,

    /// Hash of the previous update (chain). Prevents rollbacks/forks.
    pub prev_update_hash: [u8; 32],

    /// Binary payload compressed with ZSTD containing Vec<QuantizedDelta>.
    pub payload: Vec<u8>,

    /// ed25519 signature of `signing_bytes()`.
    pub signature: Vec<u8>,
}

impl SignedUpdate {
    /// Canonical bytes that are signed/verified.
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256 + self.target_id.len() + self.payload.len());

        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(self.target_id.as_bytes());
        out.extend_from_slice(&self.bundle_version.to_le_bytes());
        out.extend_from_slice(&self.bundle_hash);
        out.extend_from_slice(&self.base_model_hash);
        out.extend_from_slice(&self.prev_update_hash);

        let payload_hash = Sha256::digest(&self.payload);
        out.extend_from_slice(&payload_hash);

        out
    }

    /// Hash of the update (for cryptographic sequential chaining).
    pub fn update_hash(&self) -> [u8; 32] {
        let h = Sha256::digest(self.signing_bytes());
        let mut out = [0u8; 32];
        out.copy_from_slice(&h);
        out
    }

    /// Verifies ed25519 signature with the Laboratory's public key.
    pub fn verify_signature(&self, public_key: &[u8; 32]) -> Result<(), String> {
        let vk = VerifyingKey::from_bytes(public_key).map_err(|e| e.to_string())?;
        let sig = Signature::from_slice(&self.signature).map_err(|e| e.to_string())?;
        vk.verify(&self.signing_bytes(), &sig)
            .map_err(|e| e.to_string())
    }
}

/// Helper: serialize a Vec<QuantizedDelta> to zstd-compressed payload bytes.
pub fn encode_payload_zstd<T: serde::Serialize>(obj: &T) -> Result<Vec<u8>, String> {
    let raw = bincode::serialize(obj).map_err(|e| e.to_string())?;
    encode_all(Cursor::new(raw), 3).map_err(|e| e.to_string())
}

/// Helper: deserialize compressed payload bytes to Vec<QuantizedDelta>.
pub fn decode_payload_zstd<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    let raw = decode_all(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    bincode::deserialize(&raw).map_err(|e| e.to_string())
}

/// Key delegation for rotation.
/// The node will validate this using its root key (ROOT_PK) embedded or injected at startup.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KeyDelegation {
    pub epoch: u64,
    pub critic_pk: [u8; 32],
    pub valid_from_unix: u64,
    pub valid_to_unix: u64,
    pub signature_by_root: Vec<u8>,
}

impl KeyDelegation {
    pub fn signing_bytes(&self) -> Vec<u8> {
        bincode::serialize(&(
            self.epoch,
            self.critic_pk,
            self.valid_from_unix,
            self.valid_to_unix,
        ))
        .expect("canonical serialize")
    }

    pub fn verify_signature(&self, root_pk: &[u8; 32]) -> Result<(), String> {
        let vk = VerifyingKey::from_bytes(root_pk).map_err(|e| e.to_string())?;
        let sig = Signature::from_slice(&self.signature_by_root).map_err(|e| e.to_string())?;
        vk.verify(&self.signing_bytes(), &sig)
            .map_err(|e| e.to_string())
    }
}

/// Ack discriminant: indicates which message is being acknowledged.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckKind {
    Delegation,
    Update,
    Discovery,
}

/// Canonical messages of the LOXI exchange network.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NetMsg {
    /// Initial client greeting identifying its protocol and topological state.
    Hello {
        node_id: [u8; 32],
        /// Wire protocol version (increment on breaking changes).
        protocol: u32,
        /// Version of the model bundle installed on the node.
        bundle_version: u64,
        /// SHA-256 hash of the bundle (topology alignment).
        bundle_hash: [u8; 32],
    },
    /// Operational key delegation from the coordinator
    Delegation(KeyDelegation),
    /// Topology update delivery
    Update(SignedUpdate),
    /// Typed receipt acknowledgement.
    Ack {
        kind: AckKind,
        version: u64,
        ok: bool,
    },
    /// Inference task sent by the origin node to an expert node.
    /// The expert receives H* (flattened as s_r), runs its DEQ, and returns ExpertResult.
    ExpertTask {
        /// Task identifier (correlates Request ↔ Response)
        task_id: [u8; 16],
        /// Expert that must process it
        target_id: TargetId,
        /// H* flattened (K×D_R floats in LE) — the starting point for refinement
        s_r: Vec<f32>,
        bundle_version: u64,
        /// Round number: 0=H_rough, 1=refine, 2=verify.
        /// Allows the expert to calibrate its response based on pipeline progress.
        round: u8,
        /// Maximum time in ms the expert can use to respond.
        time_budget_ms: u32,
    },
    /// Expert node result: Δ and quality metrics.
    ExpertResult {
        task_id: [u8; 16],
        target_id: TargetId,
        /// Delta Δ = h_next - h, serialized as LE floats (~8KB)
        delta: Vec<f32>,
        q_total: f32,
        iters: u32,
        stop: u8,
    },
    /// Discovery signal to the AiArchitect/Critic when Q >= Q_MIN_LEARN.
    ///
    /// Does not contain the full h* — only hashes for correlation and anti-spam.
    /// The Architect decides whether to request more info or trigger the Critic.
    Discovery {
        node_id: [u8; 32],
        target_id: TargetId,
        q_total: f32,
        iters: u32,
        /// 0 = Q_MIN_WRITE gate, 1 = Epsilon gate (ver StopReason)
        stop: u8,
        /// SHA-256 of h* serialized in LE bytes (correlation and dedup)
        h_star_hash: [u8; 32],
        /// SHA-256 of serialized s_context (for reproducibility in Critic)
        context_hash: [u8; 32],
        bundle_version: u64,
    },
    /// Lightweight stability metrics (telemetry, not discovery).
    /// For discovery signals use NetMsg::Discovery.
    Metrics {
        q_total: f32,
        iters: u32,
        stop: u8,
    },
    Ping,
    Pong,
    /// Protocol error (e.g., message received out of sequence).
    Error {
        code: u32,
        msg: String,
    },
    /// Reproducibility request: the Coordinator asks the node to re-execute a sample.
    ReplayRequest {
        sample_id: u64,
        context_hash: [u8; 32],
        h_star_hash: [u8; 32],
        seed: u64,
        iters: u32,
    },
    /// Node response after re-executing the requested sample.
    ReplayResponse {
        sample_id: u64,
        reproduced: bool,
        q_recomputed: f32,
        trace_digest: [u8; 32],
    },
    /// Routing telemetry emitted by NodeRunner every N ticks.
    /// Vec<(target_id, hit_count)> sorted by target_id — deterministic.
    RouterStats {
        node_id: [u8; 32],
        window_ticks: u32,
        q_mean: f32,
        q_min: f32,
        q_max: f32,
        expert_hits: Vec<(String, u32)>,
        unix_ts: u64,
        // Stability Pack:
        delta_norm_mean: f32,
        delta_norm_min: f32,
        delta_norm_max: f32,
        drops_count: u32,
        beta_mean: f32,
    },
    /// Node asks the Coordinator which artifacts it should have.
    /// caps_encoded: NodeCapabilities serialized with bincode (decode at the receiver).
    GetArtifactManifest {
        node_id: [u8; 32],
        caps_encoded: Vec<u8>,
    },
    /// Coordinator responds with the list of available artifacts for this node.
    ArtifactManifest {
        artifacts: Vec<ArtifactMeta>,
    },
}

impl NetMsg {
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| e.to_string())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| e.to_string())
    }
}
