use std::collections::{HashMap, HashSet};

use crate::peers::types::{NodeId, PeerDelta, PeerEntry};

/// Indexed peer directory. O(1) lookup/update, O(k) query by domain.
///
/// Invariants:
/// - `by_domain[d]` never contains empty sets (cleanup on remove).
/// - All domains in `by_domain` are lowercase.
/// - `by_domain` and `by_id` are consistent: a NodeId in `by_id` ↔ appears in its domains.
pub struct PeerRegistry {
    epoch: u64,
    by_id: HashMap<NodeId, PeerEntry>,
    by_domain: HashMap<String, HashSet<NodeId>>,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self {
            epoch: 0,
            by_id: HashMap::new(),
            by_domain: HashMap::new(),
        }
    }

    /// Bootstrap: replaces everything. Sets epoch without validation (startup).
    pub fn set_snapshot(&mut self, epoch: u64, peers: Vec<PeerEntry>) {
        self.epoch = epoch;
        self.by_id.clear();
        self.by_domain.clear();
        for p in peers {
            self.insert(p);
        }
    }

    /// Update incremental. Rechaza si `delta.epoch <= self.epoch` (stale / replay).
    pub fn apply_delta(&mut self, delta: PeerDelta) -> Result<(), String> {
        if delta.epoch <= self.epoch {
            return Err(format!(
                "stale delta: epoch={} current={}",
                delta.epoch, self.epoch
            ));
        }
        self.epoch = delta.epoch;
        for p in delta.upserts {
            self.insert(p);
        }
        for id in delta.removes {
            self.remove(&id);
        }
        Ok(())
    }

    /// NodeIds of peers serving `domain` (automatic lowercase), deterministic order.
    pub fn node_ids_for_domain(&self, domain: &str) -> Vec<NodeId> {
        let domain = domain.to_lowercase();
        let mut ids: Vec<NodeId> = self
            .by_domain
            .get(&domain)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        ids.sort();
        ids
    }

    pub fn get(&self, id: &NodeId) -> Option<&PeerEntry> {
        self.by_id.get(id)
    }
    pub fn len(&self) -> usize {
        self.by_id.len()
    }
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    // ── Privados ──────────────────────────────────────────────────────────────

    fn insert(&mut self, mut p: PeerEntry) {
        // C: normalizar domains a lowercase al ingresar (evita fragmentación "Math" vs "math")
        p.domains = p.domains.iter().map(|d| d.to_lowercase()).collect();

        // Limpiar dominios anteriores si el entry ya existía (upsert)
        if let Some(old) = self.by_id.get(&p.node_id) {
            for d in &old.domains {
                if let Some(set) = self.by_domain.get_mut(d) {
                    set.remove(&p.node_id);
                    // B: limpiar keys vacíos para no ensuciar memoria
                    if set.is_empty() {
                        self.by_domain.remove(d);
                    }
                }
            }
        }
        for d in &p.domains {
            self.by_domain
                .entry(d.clone())
                .or_default()
                .insert(p.node_id);
        }
        self.by_id.insert(p.node_id, p);
    }

    fn remove(&mut self, id: &NodeId) {
        if let Some(old) = self.by_id.remove(id) {
            for d in &old.domains {
                if let Some(set) = self.by_domain.get_mut(d) {
                    set.remove(id);
                    // B: limpiar key vacío
                    if set.is_empty() {
                        self.by_domain.remove(d);
                    }
                }
            }
        }
    }
}
