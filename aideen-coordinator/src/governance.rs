use std::collections::{HashMap, HashSet};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::ledger::UpdateStatus;

pub type PubKey = [u8; 32]; // ed25519 verifying key bytes
pub type Sig = Vec<u8>; // ed25519 signature bytes (64 bytes; Vec<u8> for serde compat)

/// Domain tag prepended to every SignedAdminAction payload before signing/verifying.
/// Prevents cross-context signature reuse (same payload, different meaning in another protocol).
const DOMAIN_TAG: &[u8] = b"AIDEEN_GOV_V1\x00";

/// Per-action timestamp skew windows.
///
/// Stricter windows for high-impact operations (promote, emergency stop) reduce the
/// replay window available to an attacker with a compromised or clock-skewed key.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActionSkew {
    /// Allowed clock skew for most actions (SetAutoRollout, Transition to Canary/Blocked/etc.).
    pub default_secs: u64,
    /// Tighter window for Transition{to: Promoted} — the most impactful non-emergency action.
    pub promote_secs: u64,
    /// Tightest window for EmergencyStop — should be applied near-real-time.
    pub emergency_secs: u64,
}

impl Default for ActionSkew {
    fn default() -> Self {
        ActionSkew {
            default_secs: 300,
            promote_secs: 60,
            emergency_secs: 30,
        }
    }
}

/// Governance action authorized by an admin key.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AdminAction {
    SetAutoRollout {
        enabled: bool,
    },
    Transition {
        update_id: [u8; 32],
        to: UpdateStatus,
    },
    /// Halt or unhalt the rollout system.
    ///
    /// When halted: all non-unhalt governance actions are rejected.
    /// Use `halted: false` to restore normal operation.
    EmergencyStop {
        halted: bool,
        reason: String,
    },
}

/// A governance action signed by an admin key with anti-replay nonce.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SignedAdminAction {
    pub key_id: PubKey,
    pub nonce: u64, // monotonic per key — prevents replay
    pub ts: u64,    // unix seconds — prevents stale reuse
    pub action: AdminAction,
    pub sig: Sig,
}

impl SignedAdminAction {
    /// Canonical bytes signed/verified: DOMAIN_TAG || bincode(action, nonce, ts).
    ///
    /// The domain tag prevents reusing a valid governance signature in any other
    /// context that shares the same ed25519 key material.
    pub fn signing_bytes(&self) -> Vec<u8> {
        let payload = bincode::serialize(&(&self.action, self.nonce, self.ts))
            .expect("canonical serialize must not fail");
        let mut out = Vec::with_capacity(DOMAIN_TAG.len() + payload.len());
        out.extend_from_slice(DOMAIN_TAG);
        out.extend_from_slice(&payload);
        out
    }

    /// Verify the ed25519 signature against key_id.
    pub fn verify(&self) -> Result<(), String> {
        let vk = VerifyingKey::from_bytes(&self.key_id).map_err(|e| e.to_string())?;
        let sig_arr: [u8; 64] = self
            .sig
            .as_slice()
            .try_into()
            .map_err(|_| format!("invalid signature length: {}", self.sig.len()))?;
        let sig = Signature::from_bytes(&sig_arr);
        vk.verify(&self.signing_bytes(), &sig)
            .map_err(|e| e.to_string())
    }
}

/// Governance gate: enforces allowlist, per-action timestamp skew, nonce anti-replay,
/// ed25519 signature, and system halt state.
///
/// Default: empty allowlist (no key authorized), `ActionSkew::default()`, not halted.
#[derive(Serialize, Deserialize, Debug)]
pub struct GovernanceGate {
    pub allow: HashSet<PubKey>,
    pub last_nonce: HashMap<PubKey, u64>,
    pub skew: ActionSkew,
    pub halted: bool,
}

impl GovernanceGate {
    pub fn new(allow: HashSet<PubKey>, skew: ActionSkew) -> Self {
        GovernanceGate {
            allow,
            last_nonce: HashMap::new(),
            skew,
            halted: false,
        }
    }

    /// True if the system is currently halted by a signed EmergencyStop action.
    pub fn is_halted(&self) -> bool {
        self.halted
    }

    /// Authorize a signed admin action.
    ///
    /// Check order:
    /// 1. Allowlist
    /// 2. Halt guard (reject all but EmergencyStop{halted:false} when system is halted)
    /// 3. Per-action timestamp skew
    /// 4. Nonce anti-replay
    /// 5. ed25519 signature
    ///
    /// On success: nonce is recorded and, for EmergencyStop, `self.halted` is updated.
    pub fn authorize(&mut self, signed: &SignedAdminAction, now_ts: u64) -> Result<(), String> {
        // 1. Allowlist
        if !self.allow.contains(&signed.key_id) {
            return Err("key not in allowlist".into());
        }

        // 2. Halt guard — only EmergencyStop{halted: false} is accepted when system is halted
        if self.halted {
            match &signed.action {
                AdminAction::EmergencyStop { halted: false, .. } => {}
                _ => {
                    return Err(
                        "system is halted — only EmergencyStop(halted: false) accepted".into(),
                    )
                }
            }
        }

        // 3. Per-action timestamp skew
        let max_skew = match &signed.action {
            AdminAction::Transition {
                to: UpdateStatus::Promoted,
                ..
            } => self.skew.promote_secs,
            AdminAction::EmergencyStop { .. } => self.skew.emergency_secs,
            _ => self.skew.default_secs,
        };
        if signed.ts.abs_diff(now_ts) > max_skew {
            return Err(format!(
                "timestamp skew {} > max {} (action: {:?})",
                signed.ts.abs_diff(now_ts),
                max_skew,
                std::mem::discriminant(&signed.action)
            ));
        }

        // 4. Nonce anti-replay
        let prev = self.last_nonce.get(&signed.key_id).copied().unwrap_or(0);
        if signed.nonce <= prev {
            return Err(format!("replayed nonce {} <= prev {}", signed.nonce, prev));
        }

        // 5. Signature
        signed.verify()?;

        // Commit: record nonce and apply EmergencyStop side effect
        self.last_nonce.insert(signed.key_id, signed.nonce);
        if let AdminAction::EmergencyStop { halted, .. } = &signed.action {
            self.halted = *halted;
        }

        Ok(())
    }

    /// Persist to disk with atomic write (tmp + rename).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    /// Load from disk. Returns default (empty allowlist, not halted) if file does not exist.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bincode::deserialize(&bytes).map_err(|e| e.to_string())
    }
}

impl Default for GovernanceGate {
    fn default() -> Self {
        Self::new(HashSet::new(), ActionSkew::default())
    }
}
