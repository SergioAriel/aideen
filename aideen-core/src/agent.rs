use serde::{Deserialize, Serialize};

/// Canonical LOXI agent events.
///
/// Append-only: backends store them in temporal order.
/// Version 1: covers cognitive cycle (Tick, Update, Delegation, Discovery)
/// and agent configuration (PreferenceSet).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AgentEvent {
    /// The solver reached an attractor of acceptable quality.
    TickAttractor {
        q_total: f32,
        iters: u32,
        stop: u8,
        h_star_hash: [u8; 32],
        unix_ts: u64,
    },
    /// A SignedUpdate signed by the Critic was applied.
    UpdateApplied {
        version: u64,
        target_id: String,
        update_hash: [u8; 32],
        unix_ts: u64,
    },
    /// A KeyDelegation from the coordinator was installed.
    DelegationInstalled {
        epoch: u64,
        critic_pk_hash: [u8; 32],
        unix_ts: u64,
    },
    /// The node emitted a Discovery.
    DiscoveryEmitted {
        q_total: f32,
        iters: u32,
        target_id: String,
        bundle_version: u64,
        unix_ts: u64,
    },
    /// The user (or system) modified an agent preference.
    PreferenceSet {
        key: String,
        value: String,
        unix_ts: u64,
    },
}

/// Minimal contract for the agent store.
///
/// Backends: `OpfsAgentStore` (WASM), `FsAgentStore` (native).
/// Production no-op: `NullAgentStore`.
pub trait AgentStore {
    /// Read preference (exact KV).
    fn get_pref(&self, key: &str) -> Option<String>;

    /// Write preference.
    fn set_pref(&mut self, key: &str, value: String) -> Result<(), String>;

    /// Append event to the log (append-only).
    fn append_event(&mut self, event: AgentEvent) -> Result<(), String>;

    /// Retrieve the last `limit` events (reverse chronological order).
    fn recent_events(&self, limit: usize) -> Vec<AgentEvent>;
}

/// Production backend with no agent memory.
/// All operations are no-op. Allows starting the node without a configured
/// agent backend.
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

/// In-memory backend (in-process). Useful for native without persistence
/// or as a base for implementations with disk/OPFS flush.
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
