#![cfg(not(target_arch = "wasm32"))]

use rand::Rng;

use aideen_coordinator::ledger::{UpdateLedger, UpdateStatus};
use aideen_coordinator::reputation::ReputationStore;

fn tmp_path(prefix: &str) -> std::path::PathBuf {
    let suffix: u64 = rand::thread_rng().gen();
    std::env::temp_dir().join(format!("aideen_{prefix}_{suffix}.bin"))
}

// ── Test 1: ledger persist roundtrip ─────────────────────────────────────────

#[test]
fn test_ledger_persist_roundtrip() {
    use aideen_core::protocol::SignedUpdate;

    let path = tmp_path("ledger");

    let su = SignedUpdate {
        version: 1,
        target_id: "ai".into(),
        bundle_version: 1,
        bundle_hash: [0u8; 32],
        base_model_hash: [0u8; 32],
        prev_update_hash: [0u8; 32],
        payload: vec![1, 2, 3],
        signature: vec![],
    };

    let mut ledger = UpdateLedger::new();
    let update_id = ledger.propose(&su, 5, 1000).expect("propose must succeed");
    ledger
        .transition(update_id, UpdateStatus::Canary, 1001)
        .expect("transition to Canary");

    ledger.flush(&path).expect("flush must succeed");

    let loaded = UpdateLedger::load(&path).expect("load must succeed");
    let entry = loaded
        .get(&update_id)
        .expect("entry must be present after load");

    assert_eq!(entry.status, UpdateStatus::Canary);
    assert_eq!(entry.target_id, "ai");
    assert_eq!(entry.cohort_pct, 5);

    let _ = std::fs::remove_file(&path);
}

// ── Test 2: reputation persist roundtrip ──────────────────────────────────────

#[test]
fn test_reputation_persist_roundtrip() {
    let path = tmp_path("reputation");
    let node_id = [0xFFu8; 32];

    let mut store = ReputationStore::new();
    store.get_or_insert(node_id).record_replay_fail();

    // Verify throttled before flush
    assert!(store.get(&node_id).unwrap().is_throttled());

    store.flush(&path).expect("flush must succeed");

    let loaded = ReputationStore::load(&path).expect("load must succeed");
    let rep = loaded
        .get(&node_id)
        .expect("node must be present after load");

    assert!(rep.is_throttled(), "throttle state must survive flush+load");
    assert!(rep.score < 1.0, "score penalty must survive flush+load");

    let _ = std::fs::remove_file(&path);
}
