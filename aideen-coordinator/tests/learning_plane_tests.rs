use aideen_coordinator::consistency::{ConsistencyChecker, ConsistencyVerdict};
use aideen_coordinator::reputation::ReputationStore;
use aideen_coordinator::shadow_eval::{CohortMetrics, ShadowEvalManager, ShadowVerdict};

// ── Reputation tests ──────────────────────────────────────────────────────────

#[test]
fn test_replay_pass_increases_reputation() {
    let mut store = ReputationStore::new();
    let node_id = [1u8; 32];

    let initial_score = store.get_or_insert(node_id).score;
    store.get_or_insert(node_id).record_replay_pass();

    let rep = store.get(&node_id).unwrap();
    assert!(
        rep.score >= initial_score,
        "score must not decrease after replay pass"
    );
    assert_eq!(rep.replay_pass, 1);
    assert!(!rep.is_throttled(), "replay pass must not throttle node");
}

#[test]
fn test_replay_fail_triggers_throttle() {
    let mut store = ReputationStore::new();
    let node_id = [2u8; 32];

    store.get_or_insert(node_id).record_replay_fail();

    let rep = store.get(&node_id).unwrap();
    assert!(
        rep.is_throttled(),
        "node must be throttled after replay fail"
    );
    assert!(rep.score < 1.0, "score must decrease after replay fail");
    assert_eq!(rep.replay_fail, 1);
}

// ── Consistency test ──────────────────────────────────────────────────────────

#[test]
fn test_consistency_spike_marks_anomaly() {
    let mut checker = ConsistencyChecker::new();

    // Low baseline: q_ema ← 0.3
    let v1 = checker.check_discovery(0.3, 10, [0u8; 32], [0u8; 32]);
    assert_eq!(v1, ConsistencyVerdict::Ok, "baseline must be Ok");

    // Spike: 0.95 > 0.3 * 3.0 = 0.9 — within [0,1] range but above EMA spike threshold
    let v2 = checker.check_discovery(0.95, 10, [0u8; 32], [0u8; 32]);
    assert!(
        matches!(v2, ConsistencyVerdict::Anomaly { .. }),
        "q spike must be detected as anomaly, got: {:?}",
        v2
    );
}

// ── Shadow eval tests ─────────────────────────────────────────────────────────

#[test]
fn test_shadow_eval_blocks_bad_update() {
    let mut mgr = ShadowEvalManager::new(0.05);

    mgr.set_control(CohortMetrics {
        q_mean: 0.70,
        drops_count: 5,
        delta_norm_mean: 0.1,
        sample_count: 100,
    });
    mgr.set_canary(CohortMetrics {
        q_mean: 0.50,
        drops_count: 15,
        delta_norm_mean: 0.3,
        sample_count: 100,
    });

    let verdict = mgr.evaluate();
    assert!(
        matches!(verdict, ShadowVerdict::Block { .. }),
        "bad canary (drops regression + q regression) must be blocked, got: {:?}",
        verdict
    );
}

#[test]
fn test_shadow_eval_promotes_good_update() {
    let mut mgr = ShadowEvalManager::new(0.05);

    mgr.set_control(CohortMetrics {
        q_mean: 0.60,
        drops_count: 5,
        delta_norm_mean: 0.1,
        sample_count: 100,
    });
    mgr.set_canary(CohortMetrics {
        q_mean: 0.70,
        drops_count: 4,
        delta_norm_mean: 0.1,
        sample_count: 100,
    });

    let verdict = mgr.evaluate();
    assert_eq!(
        verdict,
        ShadowVerdict::Promote,
        "good canary must be promoted"
    );
}
