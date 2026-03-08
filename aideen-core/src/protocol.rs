use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::Cursor;
use zstd::stream::{decode_all, encode_all};

use crate::artifacts::ArtifactMeta;

/// Hash de modelo base (para asegurar que el delta aplica sobre el mismo snapshot global).
pub type ModelHash = [u8; 32];

/// Identifica qué parte del modelo se actualiza (ej: un expert específico).
pub type TargetId = String;

/// Qué parámetro se toca dentro del expert.
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
    pub scale: f32,    // des-quantización
    pub idx: Vec<u32>, // indices absolutos, ordenados
    pub q: Vec<i16>,   // mismos len que idx
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelBundleManifest {
    pub bundle_version: u64,
    pub experts: Vec<ExpertEntry>, // ordenado por target_id
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExpertEntry {
    pub target_id: String,
    pub expert_hash: [u8; 32],
}

/// Update firmado por el Critic.
/// IMPORTANTE: el cliente verifica firma + model_hash + anti-replay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignedUpdate {
    /// Versión monotónica por (target_id). Previene replay simple.
    pub version: u64,

    /// Identifica el expert / módulo afectado.
    pub target_id: TargetId,

    /// Versión del bundle global actual
    pub bundle_version: u64,

    /// Hash del bundle global para alineación de red
    pub bundle_hash: [u8; 32],

    /// Hash del modelo base sobre el cual se calculó el delta.
    pub base_model_hash: ModelHash,

    /// Hash del update anterior (cadena). Previene rollbacks/forks.
    pub prev_update_hash: [u8; 32],

    /// Payload binario comprimido con ZSTD conteniendo Vec<QuantizedDelta>.
    pub payload: Vec<u8>,

    /// Firma ed25519 de `signing_bytes()`.
    pub signature: Vec<u8>,
}

impl SignedUpdate {
    /// Bytes canónicos que se firman/verifican.
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

    /// Hash del update (para encadenado secuencial criptográfico).
    pub fn update_hash(&self) -> [u8; 32] {
        let h = Sha256::digest(self.signing_bytes());
        let mut out = [0u8; 32];
        out.copy_from_slice(&h);
        out
    }

    /// Verifica firma ed25519 con la llave pública del Laboratorio.
    pub fn verify_signature(&self, public_key: &[u8; 32]) -> Result<(), String> {
        let vk = VerifyingKey::from_bytes(public_key).map_err(|e| e.to_string())?;
        let sig = Signature::from_slice(&self.signature).map_err(|e| e.to_string())?;
        vk.verify(&self.signing_bytes(), &sig)
            .map_err(|e| e.to_string())
    }
}

/// Helper: serializar un Vec<QuantizedDelta> a payload bytes comprimido con zstd.
pub fn encode_payload_zstd<T: serde::Serialize>(obj: &T) -> Result<Vec<u8>, String> {
    let raw = bincode::serialize(obj).map_err(|e| e.to_string())?;
    encode_all(Cursor::new(raw), 3).map_err(|e| e.to_string())
}

/// Helper: deserializar payload bytes comprimido a Vec<QuantizedDelta>.
pub fn decode_payload_zstd<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    let raw = decode_all(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    bincode::deserialize(&raw).map_err(|e| e.to_string())
}

/// Delegación de llaves para rotación.
/// El nodo validará esto usando su llave raíz (ROOT_PK) embebida o inyectada al inicio.
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

/// Discriminante de Ack: indica qué mensaje se está confirmando.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckKind {
    Delegation,
    Update,
    Discovery,
}

/// Mensajes canónicos de la red de intercambio de LOXI.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NetMsg {
    /// Saludo inicial del cliente identificando su protocolo y estado topológico.
    Hello {
        node_id: [u8; 32],
        /// Versión del protocolo wire (incrementar en breaking changes).
        protocol: u32,
        /// Versión del bundle de modelos instalado en el nodo.
        bundle_version: u64,
        /// Hash SHA-256 del bundle (alineación de topología).
        bundle_hash: [u8; 32],
    },
    /// Delegación de llave operativa del coordinador
    Delegation(KeyDelegation),
    /// Envío de actualización de topología
    Update(SignedUpdate),
    /// Confirmación tipificada de recepción.
    Ack {
        kind: AckKind,
        version: u64,
        ok: bool,
    },
    /// Tarea de inferencia enviada por el nodo origen a un nodo experto.
    /// El experto recibe H* (aplanado como s_r), ejecuta su DEQ, y devuelve ExpertResult.
    ExpertTask {
        /// Identificador de la tarea (correlaciona Request ↔ Response)
        task_id: [u8; 16],
        /// Expert que debe procesarlo
        target_id: TargetId,
        /// H* aplanado (K×D_R floats en LE) — el punto de partida para el refinement
        s_r: Vec<f32>,
        bundle_version: u64,
        /// Número de ronda: 0=H_rough, 1=refine, 2=verify.
        /// Permite que el expert calibre su respuesta según el avance del pipeline.
        round: u8,
        /// Máximo tiempo en ms que el expert puede usar para responder.
        time_budget_ms: u32,
    },
    /// Resultado del nodo experto: Δ y métricas de calidad.
    ExpertResult {
        task_id: [u8; 16],
        target_id: TargetId,
        /// Delta Δ = h_next - h, serializado como floats LE (~8KB)
        delta: Vec<f32>,
        q_total: f32,
        iters: u32,
        stop: u8,
    },
    /// Señal de discovery al AiArchitect/Critic cuando Q >= Q_MIN_LEARN.
    ///
    /// No contiene h* completo — solo hashes para correlación y anti-spam.
    /// El Architect decide si pide más info o dispara el Critic.
    Discovery {
        node_id: [u8; 32],
        target_id: TargetId,
        q_total: f32,
        iters: u32,
        /// 0 = Q_MIN_WRITE gate, 1 = Epsilon gate (ver StopReason)
        stop: u8,
        /// SHA-256 de h* serializado en LE bytes (correlación y dedup)
        h_star_hash: [u8; 32],
        /// SHA-256 del s_context serializado (para reproducibilidad en Critic)
        context_hash: [u8; 32],
        bundle_version: u64,
    },
    /// Métricas ligeras de estabilidad (telemetría, no discovery).
    /// Para señales de discovery usar NetMsg::Discovery.
    Metrics {
        q_total: f32,
        iters: u32,
        stop: u8,
    },
    Ping,
    Pong,
    /// Error de protocolo (ej: mensaje recibido fuera de secuencia).
    Error {
        code: u32,
        msg: String,
    },
    /// Solicitud de reproducibilidad: el Coordinator pide al nodo re-ejecutar un sample.
    ReplayRequest {
        sample_id: u64,
        context_hash: [u8; 32],
        h_star_hash: [u8; 32],
        seed: u64,
        iters: u32,
    },
    /// Respuesta del nodo tras re-ejecutar el sample solicitado.
    ReplayResponse {
        sample_id: u64,
        reproduced: bool,
        q_recomputed: f32,
        trace_digest: [u8; 32],
    },
    /// Telemetría de routing emitida por NodeRunner cada N ticks.
    /// Vec<(target_id, hit_count)> ordenado por target_id — determinista.
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
    /// Nodo pide al Coordinator qué artefactos le corresponden.
    /// caps_encoded: NodeCapabilities serializado con bincode (decodificar en el receptor).
    GetArtifactManifest {
        node_id: [u8; 32],
        caps_encoded: Vec<u8>,
    },
    /// Coordinator responde con la lista de artefactos disponibles para este nodo.
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
