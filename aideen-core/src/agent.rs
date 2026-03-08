use serde::{Deserialize, Serialize};

/// Eventos canónicos del agente LOXI.
///
/// Append-only: los backends los almacenan en orden temporal.
/// Versión 1: cubre ciclo cognitivo (Tick, Update, Delegation, Discovery)
/// y configuración del agente (PreferenceSet).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AgentEvent {
    /// El solver alcanzó un atractor de calidad aceptable.
    TickAttractor {
        q_total: f32,
        iters: u32,
        stop: u8,
        h_star_hash: [u8; 32],
        unix_ts: u64,
    },
    /// Se aplicó un SignedUpdate firmado por el Critic.
    UpdateApplied {
        version: u64,
        target_id: String,
        update_hash: [u8; 32],
        unix_ts: u64,
    },
    /// Se instaló una KeyDelegation del coordinador.
    DelegationInstalled {
        epoch: u64,
        critic_pk_hash: [u8; 32],
        unix_ts: u64,
    },
    /// El nodo emitió un Discovery.
    DiscoveryEmitted {
        q_total: f32,
        iters: u32,
        target_id: String,
        bundle_version: u64,
        unix_ts: u64,
    },
    /// El usuario (o sistema) modificó una preferencia del agente.
    PreferenceSet {
        key: String,
        value: String,
        unix_ts: u64,
    },
}

/// Contrato mínimo para la memoria del agente.
///
/// Backends: `OpfsAgentStore` (WASM), `FsAgentStore` (native).
/// No-op de producción: `NullAgentStore`.
pub trait AgentStore {
    /// Leer preferencia (KV exacto).
    fn get_pref(&self, key: &str) -> Option<String>;

    /// Escribir preferencia.
    fn set_pref(&mut self, key: &str, value: String) -> Result<(), String>;

    /// Agregar evento al log (append-only).
    fn append_event(&mut self, event: AgentEvent) -> Result<(), String>;

    /// Recuperar los últimos `limit` eventos (orden cronológico inverso).
    fn recent_events(&self, limit: usize) -> Vec<AgentEvent>;
}

/// Backend de producción "sin memoria de agente".
/// Todas las operaciones son no-op. Permite arrancar el nodo sin backend
/// de agente configurado.
pub struct NullAgentStore;

impl AgentStore for NullAgentStore {
    fn get_pref(&self, _key: &str) -> Option<String> {
        None
    }
    fn set_pref(&mut self, _key: &str, _value: String) -> Result<(), String> {
        Ok(())
    }
    fn append_event(&mut self, _event: AgentEvent) -> Result<(), String> {
        Ok(())
    }
    fn recent_events(&self, _limit: usize) -> Vec<AgentEvent> {
        vec![]
    }
}

/// Backend en memoria (in-process). Útil para nativo sin persistencia
/// o como base para implementaciones con flush a disco/OPFS.
pub struct InMemoryAgentStore {
    prefs: std::collections::HashMap<String, String>,
    events: Vec<AgentEvent>,
}

impl InMemoryAgentStore {
    pub fn new() -> Self {
        Self {
            prefs: std::collections::HashMap::new(),
            events: Vec::new(),
        }
    }
}

impl Default for InMemoryAgentStore {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentStore for InMemoryAgentStore {
    fn get_pref(&self, key: &str) -> Option<String> {
        self.prefs.get(key).cloned()
    }
    fn set_pref(&mut self, key: &str, value: String) -> Result<(), String> {
        self.prefs.insert(key.to_string(), value);
        Ok(())
    }
    fn append_event(&mut self, event: AgentEvent) -> Result<(), String> {
        self.events.push(event);
        Ok(())
    }
    fn recent_events(&self, limit: usize) -> Vec<AgentEvent> {
        self.events.iter().rev().take(limit).cloned().collect()
    }
}
