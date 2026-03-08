use std::collections::HashMap;

use crate::peers::types::NodeId;

/// Decisión de confianza devuelta por `verify_or_tofu`.
pub enum TrustDecision {
    /// Fingerprint coincide con el registrado — peer conocido.
    Trusted,
    /// Primera vez que se ve este peer — fingerprint guardado (TOFU).
    TofuStored,
}

/// Almacén de fingerprints TLS. Implementa TOFU y pinning en una sola primitiva.
///
/// Scope 5M: persistencia en FS/OPFS.
pub struct TrustStore {
    known: HashMap<NodeId, [u8; 32]>,
}

impl TrustStore {
    pub fn new() -> Self {
        Self {
            known: HashMap::new(),
        }
    }

    /// TOFU + pinning en una sola llamada.
    ///
    /// - `pinned_fp = Some(x)` → exige `observed == x` antes de cualquier otra comprobación.
    /// - Si el peer ya está registrado → exige que `observed` coincida con el almacenado.
    /// - Primera vez → almacena y retorna `TofuStored`.
    pub fn verify_or_tofu(
        &mut self,
        node_id: NodeId,
        observed_fp: [u8; 32],
        pinned_fp: Option<[u8; 32]>,
    ) -> Result<TrustDecision, String> {
        // 1. Pinning explícito (más restrictivo: falla aunque TOFU sea primera vez)
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

    /// Consulta el fingerprint almacenado para un peer.
    pub fn get(&self, node_id: &NodeId) -> Option<&[u8; 32]> {
        self.known.get(node_id)
    }

    /// Carga TrustStore desde disco. Si el fichero no existe, devuelve `TrustStore::new()`.
    /// Scope 5N: migración a OPFS para browser.
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

    /// Persiste a disco con escritura atómica (write tmp + rename).
    /// No crea directorios padre — el caller es responsable de que existan.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(&self.known).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }
}
