use std::collections::HashSet;

use aideen_coordinator::governance::{
    ActionSkew, AdminAction, GovernanceGate, PubKey, SignedAdminAction,
};
use aideen_coordinator::ledger::{UpdateLedger, UpdateStatus, CANARY_MIN_SECS};
use ed25519_dalek::{Signer, SigningKey};

// ── Helper ────────────────────────────────────────────────────────────────────

/// Build a properly signed `SignedAdminAction` using the given 32-byte seed.
fn signed_action(
    sk_seed: &[u8; 32],
    nonce: u64,
    ts: u64,
    action: AdminAction,
) -> SignedAdminAction {
    let sk = SigningKey::from_bytes(sk_seed);
    let key_id: PubKey = sk.verifying_key().to_bytes();

    let mut sa = SignedAdminAction {
        key_id,
        nonce,
        ts,
        action,
        sig: vec![0u8; 64],
    };
    let sig_bytes: [u8; 64] = sk.sign(&sa.signing_bytes()).to_bytes();
    sa.sig = sig_bytes.to_vec();
    sa
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gate_with_key(sk_seed: &[u8; 32]) -> (GovernanceGate, PubKey) {
    let sk = SigningKey::from_bytes(sk_seed);
    let key_id: PubKey = sk.verifying_key().to_bytes();
    let mut allow = HashSet::new();
    allow.insert(key_id);
    (GovernanceGate::new(allow, ActionSkew::default()), key_id)
}

// ── Original tests (API updated to ActionSkew) ────────────────────────────────

#[test]
fn test_governance_unknown_key_rejected() {
    let mut gate = GovernanceGate::new(HashSet::new(), ActionSkew::default());

    let sa = signed_action(
        &[0xAAu8; 32],
        1,
        now(),
        AdminAction::SetAutoRollout { enabled: true },
    );
    let err = gate.authorize(&sa, now()).unwrap_err();
    assert!(
        err.contains("allowlist"),
        "expected allowlist error, got: {}",
        err
    );
}

#[test]
fn test_governance_replay_nonce_rejected() {
    let sk_seed = [0xBBu8; 32];
    let (mut gate, _) = gate_with_key(&sk_seed);

    let ts = now();
    let sa1 = signed_action(
        &sk_seed,
        1,
        ts,
        AdminAction::SetAutoRollout { enabled: true },
    );
    gate.authorize(&sa1, ts).expect("first call must succeed");

    let sa2 = signed_action(
        &sk_seed,
        1,
        ts,
        AdminAction::SetAutoRollout { enabled: false },
    );
    let err = gate.authorize(&sa2, ts).unwrap_err();
    assert!(
        err.contains("replayed") || err.contains("nonce"),
        "expected nonce error, got: {}",
        err
    );
}

#[test]
fn test_governance_timestamp_skew_rejected() {
    let sk_seed = [0xCCu8; 32];
    let (mut gate, _) = gate_with_key(&sk_seed);

    // 600s stale → exceeds default_secs=300
    let stale_ts = now().saturating_sub(600);
    let sa = signed_action(
        &sk_seed,
        1,
        stale_ts,
        AdminAction::SetAutoRollout { enabled: true },
    );
    let err = gate.authorize(&sa, now()).unwrap_err();
    assert!(
        err.contains("skew") || err.contains("timestamp"),
        "expected skew error, got: {}",
        err
    );
}

#[test]
fn test_governance_transition_gated_ok() {
    use aideen_core::protocol::SignedUpdate;

    let sk_seed = [0xDDu8; 32];
    let (mut gate, _) = gate_with_key(&sk_seed);
    let mut ledger = UpdateLedger::new();

    let su = SignedUpdate {
        version: 1,
        target_id: "ai".into(),
        bundle_version: 1,
        bundle_hash: [0u8; 32],
        base_model_hash: [0u8; 32],
        prev_update_hash: [0u8; 32],
        payload: vec![],
        signature: vec![],
    };
    let update_id = ledger.propose(&su, 5, 1000).expect("propose must succeed");

    let ts = now();
    let sa = signed_action(
        &sk_seed,
        1,
        ts,
        AdminAction::Transition {
            update_id,
            to: UpdateStatus::Canary,
        },
    );
    ledger
        .transition_gated(&mut gate, &sa, ts)
        .expect("gated transition must succeed");

    assert_eq!(ledger.get(&update_id).unwrap().status, UpdateStatus::Canary);
}

#[test]
fn test_governance_set_auto_rollout_authorized() {
    let sk_seed = [0xEEu8; 32];
    let (mut gate, _) = gate_with_key(&sk_seed);

    let ts = now();
    let sa = signed_action(
        &sk_seed,
        1,
        ts,
        AdminAction::SetAutoRollout { enabled: true },
    );
    gate.authorize(&sa, ts)
        .expect("SetAutoRollout from authorized key must pass gate");
}

// ── New tests: per-action skew ────────────────────────────────────────────────

#[test]
fn test_governance_promote_skew_strict() {
    let sk_seed = [0x11u8; 32];
    // promote_secs = 60; a ts 200s old should be rejected for Canary→Promoted
    let skew = ActionSkew {
        default_secs: 300,
        promote_secs: 60,
        emergency_secs: 30,
    };
    let sk = SigningKey::from_bytes(&sk_seed);
    let key_id: PubKey = sk.verifying_key().to_bytes();
    let mut allow = HashSet::new();
    allow.insert(key_id);
    let mut gate = GovernanceGate::new(allow, skew);

    let ts_stale = now().saturating_sub(200); // 200s old
    let update_id = [0u8; 32];
    let sa = signed_action(
        &sk_seed,
        1,
        ts_stale,
        AdminAction::Transition {
            update_id,
            to: UpdateStatus::Promoted,
        },
    );

    let err = gate.authorize(&sa, now()).unwrap_err();
    assert!(
        err.contains("skew") || err.contains("timestamp"),
        "promote action with 200s-old ts must fail strict promote_secs=60, got: {}",
        err
    );
}

// ── New tests: emergency stop ─────────────────────────────────────────────────

#[test]
fn test_emergency_stop_halts_system() {
    let sk_seed = [0x22u8; 32];
    let (mut gate, _) = gate_with_key(&sk_seed);

    let ts = now();
    // Issue EmergencyStop{halted: true}
    let halt = signed_action(
        &sk_seed,
        1,
        ts,
        AdminAction::EmergencyStop {
            halted: true,
            reason: "test halt".into(),
        },
    );
    gate.authorize(&halt, ts)
        .expect("emergency stop must be accepted");
    assert!(gate.is_halted(), "gate must be halted after EmergencyStop");

    // Any other action is now rejected
    let sa = signed_action(
        &sk_seed,
        2,
        ts,
        AdminAction::SetAutoRollout { enabled: false },
    );
    let err = gate.authorize(&sa, ts).unwrap_err();
    assert!(
        err.contains("halted"),
        "non-unhalt action must fail when halted, got: {}",
        err
    );
}

#[test]
fn test_emergency_unhalt_restores_operations() {
    let sk_seed = [0x33u8; 32];
    let (mut gate, _) = gate_with_key(&sk_seed);

    let ts = now();
    // Halt
    gate.authorize(
        &signed_action(
            &sk_seed,
            1,
            ts,
            AdminAction::EmergencyStop {
                halted: true,
                reason: "halt".into(),
            },
        ),
        ts,
    )
    .unwrap();
    assert!(gate.is_halted());

    // Unhalt
    gate.authorize(
        &signed_action(
            &sk_seed,
            2,
            ts,
            AdminAction::EmergencyStop {
                halted: false,
                reason: "unhalt".into(),
            },
        ),
        ts,
    )
    .expect("unhalt must succeed even when system is halted");
    assert!(
        !gate.is_halted(),
        "gate must be unhalted after EmergencyStop(false)"
    );

    // Normal operations restored
    gate.authorize(
        &signed_action(
            &sk_seed,
            3,
            ts,
            AdminAction::SetAutoRollout { enabled: true },
        ),
        ts,
    )
    .expect("normal action must succeed after unhalt");
}

// ── New tests: time-delay for Canary → Promoted ───────────────────────────────

#[test]
fn test_time_delay_blocks_early_promote() {
    use aideen_core::protocol::SignedUpdate;

    let mut ledger = UpdateLedger::new();
    let su = SignedUpdate {
        version: 1,
        target_id: "ai".into(),
        bundle_version: 1,
        bundle_hash: [0u8; 32],
        base_model_hash: [0u8; 32],
        prev_update_hash: [0u8; 32],
        payload: vec![],
        signature: vec![],
    };

    let t0 = 1_000_000u64;
    let update_id = ledger.propose(&su, 5, t0).unwrap();
    ledger
        .transition(update_id, UpdateStatus::Canary, t0)
        .unwrap();

    // Try to promote 100s later — must fail (CANARY_MIN_SECS = 300)
    let err = ledger
        .transition(update_id, UpdateStatus::Promoted, t0 + 100)
        .unwrap_err();
    assert!(
        err.contains("early") || err.contains("remaining"),
        "promote before delay must fail, got: {}",
        err
    );
}

#[test]
fn test_time_delay_allows_promote_after_delay() {
    use aideen_core::protocol::SignedUpdate;

    let mut ledger = UpdateLedger::new();
    let su = SignedUpdate {
        version: 1,
        target_id: "ai".into(),
        bundle_version: 1,
        bundle_hash: [0u8; 32],
        base_model_hash: [0u8; 32],
        prev_update_hash: [0u8; 32],
        payload: vec![],
        signature: vec![],
    };

    let t0 = 2_000_000u64;
    let update_id = ledger.propose(&su, 5, t0).unwrap();
    ledger
        .transition(update_id, UpdateStatus::Canary, t0)
        .unwrap();

    // Promote exactly at t0 + CANARY_MIN_SECS — must succeed
    ledger
        .transition(update_id, UpdateStatus::Promoted, t0 + CANARY_MIN_SECS)
        .expect("promote after delay must succeed");

    assert_eq!(
        ledger.get(&update_id).unwrap().status,
        UpdateStatus::Promoted
    );
}
