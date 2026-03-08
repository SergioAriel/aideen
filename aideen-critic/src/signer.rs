use ed25519_dalek::{Signer, SigningKey};
use sha2::{Digest, Sha256};

use aideen_core::protocol::{encode_payload_zstd, ParamId, QuantizedDelta, SignedUpdate};

use crate::{Critic, EvidenceBundle};

/// Stub Critic implementation that generates deterministic SignedUpdates from evidence.
///
/// In production, real gradient computation happens out-of-band and is injected here.
/// For 6A pipeline testing, this produces a minimal zero-valued delta signed with
/// an ed25519 key derived from the evidence fingerprint.
pub struct CriticSigner {
    signing_key: SigningKey,
    pub verifying_key: [u8; 32],
    next_version: u64,
}

impl CriticSigner {
    pub fn new(signing_key: SigningKey) -> Self {
        let vk: [u8; 32] = signing_key.verifying_key().to_bytes();
        CriticSigner {
            signing_key,
            verifying_key: vk,
            next_version: 1,
        }
    }

    /// Generate a fresh CriticSigner with a random key.
    pub fn generate() -> Self {
        let mut rng = rand::rngs::OsRng;
        Self::new(SigningKey::generate(&mut rng))
    }

    /// Deterministic fingerprint of all evidence in the bundle.
    fn evidence_hash(evidence: &EvidenceBundle) -> [u8; 32] {
        let mut h = Sha256::new();
        for d in &evidence.discoveries {
            h.update(&d.h_star_hash);
            h.update(&d.context_hash);
        }
        for r in &evidence.replay_results {
            h.update(&r.trace_digest);
        }
        h.finalize().into()
    }
}

impl Critic for CriticSigner {
    fn propose_update(
        &mut self,
        target_id: &str,
        evidence: EvidenceBundle,
    ) -> Result<SignedUpdate, String> {
        let version = self.next_version;
        self.next_version += 1;

        // Stub delta: single zero-valued entry (placeholder for real gradient)
        let stub_delta = QuantizedDelta {
            param: ParamId::W1,
            scale: 0.0,
            idx: vec![0],
            q: vec![0],
        };
        let payload = encode_payload_zstd(&vec![stub_delta])?;

        let evidence_fp = Self::evidence_hash(&evidence);
        let bundle_version = evidence
            .discoveries
            .first()
            .map(|d| d.bundle_version)
            .unwrap_or(0);

        let mut update = SignedUpdate {
            version,
            target_id: target_id.to_string(),
            bundle_version,
            bundle_hash: evidence_fp,
            base_model_hash: [0u8; 32],
            prev_update_hash: [0u8; 32],
            payload,
            signature: vec![],
        };

        let sig = self.signing_key.sign(&update.signing_bytes());
        update.signature = sig.to_bytes().to_vec();
        Ok(update)
    }
}
