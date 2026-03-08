use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::governance::PubKey;

/// A single tamper-evident entry in the governance audit log.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GovernanceEvent {
    pub ts: u64,
    pub actor: PubKey,
    pub action: String,      // human-readable description
    pub prev_hash: [u8; 32], // hash of the previous event (zero for genesis)
    pub hash: [u8; 32],      // sha256(prev_hash ‖ ts ‖ actor ‖ action)
}

/// Append-only, tamper-evident governance audit log.
///
/// Each event's `hash` covers the previous event's `hash`, the timestamp,
/// the actor's key, and the action string — so tampering any event invalidates
/// all subsequent hashes.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct AuditLog {
    events: Vec<GovernanceEvent>,
}

impl AuditLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a new event to the log.
    pub fn append(&mut self, ts: u64, actor: PubKey, action: String) {
        let prev_hash = self.events.last().map(|e| e.hash).unwrap_or([0u8; 32]);
        let hash = compute_hash(prev_hash, ts, actor, &action);
        self.events.push(GovernanceEvent {
            ts,
            actor,
            action,
            prev_hash,
            hash,
        });
    }

    /// Verify the integrity of the entire chain.
    /// Returns Ok(()) if all hashes are consistent, Err with the position of the first break.
    pub fn verify_chain(&self) -> Result<(), String> {
        let mut expected_prev = [0u8; 32];
        for (i, event) in self.events.iter().enumerate() {
            if event.prev_hash != expected_prev {
                return Err(format!("chain break at index {}: prev_hash mismatch", i));
            }
            let expected_hash = compute_hash(event.prev_hash, event.ts, event.actor, &event.action);
            if event.hash != expected_hash {
                return Err(format!("chain break at index {}: hash mismatch", i));
            }
            expected_prev = event.hash;
        }
        Ok(())
    }

    pub fn events(&self) -> &[GovernanceEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Persist to disk with atomic write (tmp + rename).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    /// Load from disk. Returns empty log if file does not exist.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bincode::deserialize(&bytes).map_err(|e| e.to_string())
    }
}

fn compute_hash(prev_hash: [u8; 32], ts: u64, actor: PubKey, action: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(prev_hash);
    h.update(ts.to_le_bytes());
    h.update(actor);
    h.update(action.as_bytes());
    h.finalize().into()
}
