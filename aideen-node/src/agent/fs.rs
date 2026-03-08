#![cfg(not(target_arch = "wasm32"))]

use aideen_core::agent::{AgentEvent, AgentStore};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

/// Almacenamiento persistente del agente en disco (native only).
///
/// Layout:
///   <dir>/prefs.bin   — bincode(HashMap<String, String>)
///   <dir>/events.log  — secuencia de [u32 LE len][bincode(AgentEvent)]
pub struct FsAgentStore {
    dir: PathBuf,
    prefs: HashMap<String, String>,
}

impl FsAgentStore {
    /// Abre o crea el store en `<base_dir>/<agent_id>/`.
    pub fn open(base_dir: &str, agent_id: &str) -> Result<Self, String> {
        let dir = PathBuf::from(base_dir).join(agent_id);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

        let prefs_path = dir.join("prefs.bin");
        let prefs = if prefs_path.exists() {
            let bytes = std::fs::read(&prefs_path).map_err(|e| e.to_string())?;
            bincode::deserialize(&bytes).unwrap_or_default()
        } else {
            HashMap::new()
        };

        Ok(Self { dir, prefs })
    }

    fn prefs_path(&self) -> PathBuf {
        self.dir.join("prefs.bin")
    }
    fn events_path(&self) -> PathBuf {
        self.dir.join("events.log")
    }

    fn flush_prefs(&self) -> Result<(), String> {
        let bytes = bincode::serialize(&self.prefs).map_err(|e| e.to_string())?;
        std::fs::write(self.prefs_path(), &bytes).map_err(|e| e.to_string())
    }
}

impl AgentStore for FsAgentStore {
    fn get_pref(&self, key: &str) -> Option<String> {
        self.prefs.get(key).cloned()
    }

    fn set_pref(&mut self, key: &str, value: String) -> Result<(), String> {
        self.prefs.insert(key.to_string(), value);
        self.flush_prefs()
    }

    fn append_event(&mut self, event: AgentEvent) -> Result<(), String> {
        use std::fs::OpenOptions;
        let payload = bincode::serialize(&event).map_err(|e| e.to_string())?;
        let len = (payload.len() as u32).to_le_bytes();
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.events_path())
            .map_err(|e| e.to_string())?;
        f.write_all(&len).map_err(|e| e.to_string())?;
        f.write_all(&payload).map_err(|e| e.to_string())
    }

    fn recent_events(&self, limit: usize) -> Vec<AgentEvent> {
        // MVP: lee todos los frames, devuelve los últimos `limit` en orden inverso.
        let path = self.events_path();
        if !path.exists() {
            return vec![];
        }
        let Ok(bytes) = std::fs::read(&path) else {
            return vec![];
        };

        let mut events = Vec::new();
        let mut cursor = 0usize;
        while cursor + 4 <= bytes.len() {
            let len = u32::from_le_bytes([
                bytes[cursor],
                bytes[cursor + 1],
                bytes[cursor + 2],
                bytes[cursor + 3],
            ]) as usize;
            cursor += 4;
            if cursor + len > bytes.len() {
                break;
            }
            if let Ok(ev) = bincode::deserialize::<AgentEvent>(&bytes[cursor..cursor + len]) {
                events.push(ev);
            }
            cursor += len;
        }
        events.into_iter().rev().take(limit).collect()
    }
}
