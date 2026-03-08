use std::collections::HashMap;

use serde::{Deserialize, Serialize};

pub type NodeId = [u8; 32];

/// Reputation state for a single node in the learning plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeReputation {
    pub score: f32,
    pub reports: u64,
    pub replay_pass: u64,
    pub replay_fail: u64,
    pub last_q: f32,
    pub q_ema: f32,
    pub anomaly_count: u64,
    pub throttle_until: Option<u64>, // unix timestamp (seconds)
}

impl Default for NodeReputation {
    fn default() -> Self {
        NodeReputation {
            score: 1.0,
            reports: 0,
            replay_pass: 0,
            replay_fail: 0,
            last_q: 0.0,
            q_ema: 0.0,
            anomaly_count: 0,
            throttle_until: None,
        }
    }
}

impl NodeReputation {
    /// True if the node is currently throttled (contributions ignored by Critic).
    pub fn is_throttled(&self) -> bool {
        match self.throttle_until {
            None => false,
            Some(ts) => unix_now() < ts,
        }
    }

    /// Record a successful replay — small positive score adjustment.
    pub fn record_replay_pass(&mut self) {
        self.replay_pass += 1;
        self.reports += 1;
        self.score = (self.score * 0.95 + 0.05).min(1.0);
    }

    /// Record a failed replay — heavy penalty + exponential throttle.
    pub fn record_replay_fail(&mut self) {
        self.replay_fail += 1;
        self.reports += 1;
        self.score = (self.score * 0.7).max(0.0);
        let throttle_secs = (self.replay_fail * 30).min(3600);
        self.throttle_until = Some(unix_now() + throttle_secs);
    }

    /// Record a consistency anomaly — moderate score penalty.
    pub fn record_anomaly(&mut self) {
        self.anomaly_count += 1;
        self.score = (self.score * 0.85).max(0.0);
    }

    /// Update the node's Q EMA tracker.
    pub fn update_q(&mut self, q: f32) {
        self.last_q = q;
        self.q_ema = if self.q_ema == 0.0 {
            q
        } else {
            0.9 * self.q_ema + 0.1 * q
        };
    }
}

/// Per-node reputation registry for the Coordinator.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ReputationStore {
    pub by_node: HashMap<NodeId, NodeReputation>,
}

impl ReputationStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_insert(&mut self, node_id: NodeId) -> &mut NodeReputation {
        self.by_node.entry(node_id).or_default()
    }

    pub fn get(&self, node_id: &NodeId) -> Option<&NodeReputation> {
        self.by_node.get(node_id)
    }

    /// Fraction of replay attempts that failed for a given node (0.0 if no data).
    pub fn replay_fail_rate(&self, node_id: &NodeId) -> f32 {
        match self.by_node.get(node_id) {
            None => 0.0,
            Some(rep) => {
                let total = rep.replay_pass + rep.replay_fail;
                if total == 0 {
                    0.0
                } else {
                    rep.replay_fail as f32 / total as f32
                }
            }
        }
    }

    /// Persist to disk with atomic write (tmp + rename).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    /// Load from disk. Returns empty store if file does not exist.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bincode::deserialize(&bytes).map_err(|e| e.to_string())
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
