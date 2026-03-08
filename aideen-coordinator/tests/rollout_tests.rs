use aideen_coordinator::cohort::{assign_pct, is_canary};
use aideen_coordinator::rollout::{
    CohortSnapshot, PromotionCriteria, RollbackCriteria, RolloutDecision, RolloutPolicy,
};

// ── Helper ────────────────────────────────────────────────────────────────────

fn policy() -> RolloutPolicy {
    RolloutPolicy::new(
        5,
        PromotionCriteria {
            min_q_improvement: 0.05,
            max_drops_ratio: 1.5,
            max_delta_norm_ratio: 2.0,
            max_replay_fail_rate: 0.05,
            min_canary_samples: 5,
        },
        RollbackCriteria {
            max_q_regression: 0.05,
            max_drops_ratio: 2.0,
            max_replay_fail_rate: 0.20,
        },
    )
}

// ── Test 1: good canary → Promote ─────────────────────────────────────────────

#[test]
fn test_rollout_promotes_good_canary() {
    let p = policy();
    let ctrl = CohortSnapshot {
        q_mean: 0.60,
        drops_count: 10,
        delta_norm_mean: 0.2,
        sample_count: 20,
    };
    let canary = CohortSnapshot {
        q_mean: 0.70,
        drops_count: 8,
        delta_norm_mean: 0.2,
        sample_count: 20,
    };

    assert_eq!(
        p.evaluate(&ctrl, &canary, 0.01),
        RolloutDecision::Promote,
        "canary with q+0.10 improvement and low drops must be promoted"
    );
}

// ── Test 2: high replay_fail_rate → Rollback ──────────────────────────────────

#[test]
fn test_rollout_rollback_high_replay_fail() {
    let p = policy();
    let ctrl = CohortSnapshot {
        q_mean: 0.60,
        drops_count: 5,
        delta_norm_mean: 0.2,
        sample_count: 10,
    };
    let canary = CohortSnapshot {
        q_mean: 0.65,
        drops_count: 5,
        delta_norm_mean: 0.2,
        sample_count: 10,
    };

    let decision = p.evaluate(&ctrl, &canary, 0.30); // 30% > 20% rollback threshold
    assert!(
        matches!(decision, RolloutDecision::Rollback { .. }),
        "replay_fail_rate=0.30 above max must trigger rollback, got: {:?}",
        decision
    );
}

// ── Test 3: drops regression 2× → Rollback ────────────────────────────────────

#[test]
fn test_rollout_rollback_drops_regression() {
    let p = policy();
    let ctrl = CohortSnapshot {
        q_mean: 0.60,
        drops_count: 10,
        delta_norm_mean: 0.2,
        sample_count: 10,
    };
    // canary drops 25 >= 2.0 * 10 = 20 → Rollback
    let canary = CohortSnapshot {
        q_mean: 0.60,
        drops_count: 25,
        delta_norm_mean: 0.2,
        sample_count: 10,
    };

    let decision = p.evaluate(&ctrl, &canary, 0.01);
    assert!(
        matches!(decision, RolloutDecision::Rollback { .. }),
        "drops ratio 2.5× above max 2.0 must rollback, got: {:?}",
        decision
    );
}

// ── Test 4: cohort assignment is deterministic ────────────────────────────────

#[test]
fn test_cohort_assignment_deterministic() {
    let target = "ai";
    let update_id = [0xABu8; 32];

    let node_a = [1u8; 32];
    let node_b = [2u8; 32];

    // Same inputs always give same result
    assert_eq!(
        assign_pct(&node_a, target, &update_id),
        assign_pct(&node_a, target, &update_id)
    );
    assert_eq!(
        assign_pct(&node_b, target, &update_id),
        assign_pct(&node_b, target, &update_id)
    );

    // Different node_ids likely give different bucket (sanity check)
    let pct_a = assign_pct(&node_a, target, &update_id);
    let pct_b = assign_pct(&node_b, target, &update_id);
    // Not guaranteed to differ, but [1;32] and [2;32] sha256 are very unlikely to collide
    assert_ne!(
        pct_a, pct_b,
        "different node_ids should map to different buckets"
    );

    // is_canary is consistent with assign_pct
    let pct = assign_pct(&node_a, target, &update_id);
    assert_eq!(is_canary(&node_a, target, &update_id, pct + 1), true);
    assert_eq!(is_canary(&node_a, target, &update_id, pct), false);
}
