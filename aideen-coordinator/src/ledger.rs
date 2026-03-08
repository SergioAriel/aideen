use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use aideen_core::protocol::SignedUpdate;

use crate::governance::{AdminAction, GovernanceGate, SignedAdminAction};

/// Lifecycle states of a model update in the rollout ledger.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UpdateStatus {
    Proposed,
    Canary,
    Promoted,
    Blocked { reason: String },
    RolledBack { reason: String },
    Archived,
}

/// Minimum time a canary must run before it can be promoted to full rollout.
/// Enforced by `UpdateLedger::transition()` when moving Canary → Promoted.
pub const CANARY_MIN_SECS: u64 = 300; // 5 minutes

/// Single entry in the update ledger.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LedgerEntry {
    pub update_id: [u8; 32], // SignedUpdate::update_hash()
    pub target_id: String,
    pub version: u64, // monotonic per target_id
    pub bundle_version: u64,
    pub delta_hash: [u8; 32], // sha256(payload) — for audit
    pub signature: Vec<u8>,   // ed25519 from Critic
    pub status: UpdateStatus,
    pub cohort_pct: u8,             // 1 | 5 | 20 | 100
    pub created_ts: u64,            // unix
    pub transitioned_ts: u64,       // last status change
    pub promote_not_before_ts: u64, // earliest ts at which Canary → Promoted is allowed (0 = not yet in Canary)
}

/// Tamper-evident ledger tracking all model update proposals and their lifecycle.
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct UpdateLedger {
    entries: HashMap<[u8; 32], LedgerEntry>, // keyed by update_id
    by_target: HashMap<String, Vec<[u8; 32]>>, // target_id → [update_id] sorted by version
}

impl UpdateLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Propose a new update. Returns the update_id on success.
    /// Fails if this update_id was already proposed.
    pub fn propose(
        &mut self,
        update: &SignedUpdate,
        cohort_pct: u8,
        now_ts: u64,
    ) -> Result<[u8; 32], String> {
        let update_id = update.update_hash();
        if self.entries.contains_key(&update_id) {
            return Err("update already in ledger".into());
        }
        let delta_hash: [u8; 32] = Sha256::digest(&update.payload).into();
        let entry = LedgerEntry {
            update_id,
            target_id: update.target_id.clone(),
            version: update.version,
            bundle_version: update.bundle_version,
            delta_hash,
            signature: update.signature.clone(),
            status: UpdateStatus::Proposed,
            cohort_pct,
            created_ts: now_ts,
            transitioned_ts: now_ts,
            promote_not_before_ts: 0,
        };
        self.by_target
            .entry(update.target_id.clone())
            .or_default()
            .push(update_id);
        self.entries.insert(update_id, entry);
        Ok(update_id)
    }

    /// Transition an entry to a new status. Validates the transition graph.
    ///
    /// Side effects:
    /// - Entering `Canary`: sets `promote_not_before_ts = now_ts + CANARY_MIN_SECS`.
    /// - Entering `Promoted`: enforces `now_ts >= promote_not_before_ts`.
    pub fn transition(
        &mut self,
        update_id: [u8; 32],
        new_status: UpdateStatus,
        now_ts: u64,
    ) -> Result<(), String> {
        let entry = self
            .entries
            .get_mut(&update_id)
            .ok_or("update_id not found")?;
        validate_transition(&entry.status, &new_status)?;

        // Time-delay enforcement: Canary must bake for at least CANARY_MIN_SECS before promote
        if matches!(new_status, UpdateStatus::Promoted) && now_ts < entry.promote_not_before_ts {
            return Err(format!(
                "promote too early: {} secs remaining (not_before={})",
                entry.promote_not_before_ts - now_ts,
                entry.promote_not_before_ts
            ));
        }

        // When entering Canary, record the earliest promote time
        if matches!(new_status, UpdateStatus::Canary) {
            entry.promote_not_before_ts = now_ts + CANARY_MIN_SECS;
        }

        entry.status = new_status;
        entry.transitioned_ts = now_ts;
        Ok(())
    }

    /// Governance-gated transition. Verifies the SignedAdminAction before mutating.
    pub fn transition_gated(
        &mut self,
        gate: &mut GovernanceGate,
        signed: &SignedAdminAction,
        now_ts: u64,
    ) -> Result<(), String> {
        gate.authorize(signed, now_ts)?;
        match &signed.action {
            AdminAction::Transition { update_id, to } => {
                self.transition(*update_id, to.clone(), now_ts)
            }
            _ => Err("expected AdminAction::Transition".into()),
        }
    }

    /// The active entry for a target (Canary or Promoted), if any.
    pub fn active_for_target(&self, target_id: &str) -> Option<&LedgerEntry> {
        let ids = self.by_target.get(target_id)?;
        ids.iter()
            .rev()
            .filter_map(|id| self.entries.get(id))
            .find(|e| matches!(e.status, UpdateStatus::Canary | UpdateStatus::Promoted))
    }

    pub fn get(&self, update_id: &[u8; 32]) -> Option<&LedgerEntry> {
        self.entries.get(update_id)
    }

    /// Persist to disk with atomic write (tmp + rename).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    /// Load from disk. Returns empty ledger if file does not exist.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bincode::deserialize(&bytes).map_err(|e| e.to_string())
    }
}

/// Valid state machine transitions for UpdateStatus.
fn validate_transition(from: &UpdateStatus, to: &UpdateStatus) -> Result<(), String> {
    let ok = matches!(
        (from, to),
        (UpdateStatus::Proposed, UpdateStatus::Canary)
            | (UpdateStatus::Proposed, UpdateStatus::Blocked { .. })
            | (UpdateStatus::Proposed, UpdateStatus::Archived)
            | (UpdateStatus::Canary, UpdateStatus::Promoted)
            | (UpdateStatus::Canary, UpdateStatus::RolledBack { .. })
            | (UpdateStatus::Canary, UpdateStatus::Blocked { .. })
            | (UpdateStatus::Promoted, UpdateStatus::Archived)
    );
    if ok {
        Ok(())
    } else {
        Err(format!("invalid transition {:?} → {:?}", from, to))
    }
}
