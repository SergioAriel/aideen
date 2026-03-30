use std::collections::HashMap;

use crate::peers::types::NodeId;

/// Trust decision returned by `verify_or_tofu`.
pub enum TrustDecision {
    /// Fingerprint matches the stored one — known peer.
    Trusted,
    /// First time seeing this peer — fingerprint stored (TOFU).
    TofuStored,
}

/// TLS fingerprint store. Implements TOFU and pinning in a single primitive.
///
/// Scope 5M: persistence on FS/OPFS.
pub struct TrustStore {
    known: HashMap<NodeId, [u8; 32]>,
}

impl TrustStore {
    pub fn new() -> Self {
        Self {
            known: HashMap::new(),
        }
    }

    /// TOFU + pinning in a single call.
    ///
    /// - `pinned_fp = Some(x)` → requires `observed == x` before any other check.
    /// - If the peer is already registered → requires `observed` to match the stored one.
    /// - First time → stores and returns `TofuStored`.
    pub fn verify_or_tofu(
        &mut self,
        node_id: NodeId,
        observed_fp: [u8; 32],
        pinned_fp: Option<[u8; 32]>,
    ) -> Result<TrustDecision, String> {
        // 1. Explicit pinning (most restrictive: fails even if TOFU is first time)
        if let Some(pinned) = pinned_fp {
            if observed_fp != pinned {
                return Err(format!(
                    "pinning mismatch for node {:02x}{:02x}{:02x}{:02x}",
                    node_id[0], node_id[1], node_id[2], node_id[3]
                ));
            }
        }

        // 2. Registro existente
        match self.known.get(&node_id) {
            Some(&stored) if stored != observed_fp => Err(format!(
                "TOFU mismatch for node {:02x}{:02x}{:02x}{:02x}: fingerprint changed",
                node_id[0], node_id[1], node_id[2], node_id[3]
            )),
            Some(_) => Ok(TrustDecision::Trusted),
            None => {
                self.known.insert(node_id, observed_fp);
                Ok(TrustDecision::TofuStored)
            }
        }
    }

    /// Queries the stored fingerprint for a peer.
    pub fn get(&self, node_id: &NodeId) -> Option<&[u8; 32]> {
        self.known.get(node_id)
    }

    /// Loads TrustStore from disk. If the file does not exist, returns `TrustStore::new()`.
    /// Scope 5N: migration to OPFS for browser.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let known: HashMap<NodeId, [u8; 32]> =
            bincode::deserialize(&bytes).map_err(|e| e.to_string())?;
        Ok(Self { known })
    }

    /// Persists to disk with atomic write (write tmp + rename).
    /// No crea directorios padre — el caller es responsable de que existan.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(&self.known).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}
