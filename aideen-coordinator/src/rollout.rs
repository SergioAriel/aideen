/// Criteria for promoting a canary update to full rollout.
#[derive(Debug, Clone)]
pub struct PromotionCriteria {
    pub min_q_improvement: f32,    // canary.q_mean >= ctrl.q_mean + this
    pub max_drops_ratio: f32,      // canary.drops_count / ctrl.drops_count <= this
    pub max_delta_norm_ratio: f32, // canary.delta_norm_mean / ctrl.delta_norm_mean <= this
    pub max_replay_fail_rate: f32, // from ReputationStore
    pub min_canary_samples: u32,   // minimum RouterStats samples in canary cohort
}

impl Default for PromotionCriteria {
    fn default() -> Self {
        PromotionCriteria {
            min_q_improvement: 0.02,
            max_drops_ratio: 1.5,
            max_delta_norm_ratio: 2.0,
            max_replay_fail_rate: 0.05,
            min_canary_samples: 10,
        }
    }
}

/// Criteria for rolling back a canary update.
#[derive(Debug, Clone)]
pub struct RollbackCriteria {
    pub max_q_regression: f32,     // canary.q_mean < ctrl.q_mean - this
    pub max_drops_ratio: f32,      // canary.drops_count / ctrl.drops_count >= this
    pub max_replay_fail_rate: f32, // from ReputationStore
}

impl Default for RollbackCriteria {
    fn default() -> Self {
        RollbackCriteria {
            max_q_regression: 0.05,
            max_drops_ratio: 2.0,
            max_replay_fail_rate: 0.20,
        }
    }
}

/// Outcome of evaluating a canary rollout.
#[derive(Debug, PartialEq)]
pub enum RolloutDecision {
    Promote,
    Rollback { reason: String },
    Pending,
}

/// Metrics snapshot for one cohort (control or canary).
#[derive(Debug, Clone, Default)]
pub struct CohortSnapshot {
    pub q_mean: f32,
    pub drops_count: u32,
    pub delta_norm_mean: f32,
    pub sample_count: u32,
}

/// Policy that decides whether a canary update should be promoted, rolled back, or left pending.
#[derive(Debug, Clone)]
pub struct RolloutPolicy {
    pub canary_pct: u8, // e.g. 5 = 5% of nodes see the canary
    pub promote: PromotionCriteria,
    pub rollback: RollbackCriteria,
}

impl Default for RolloutPolicy {
    fn default() -> Self {
        RolloutPolicy {
            canary_pct: 5,
            promote: PromotionCriteria::default(),
            rollback: RollbackCriteria::default(),
        }
    }
}

impl RolloutPolicy {
    pub fn new(canary_pct: u8, promote: PromotionCriteria, rollback: RollbackCriteria) -> Self {
        RolloutPolicy {
            canary_pct,
            promote,
            rollback,
        }
    }

    /// Evaluate whether the canary cohort is ready to be promoted, rolled back, or left pending.
    ///
    /// `ctrl` and `canary` are aggregated RouterStats for each cohort.
    /// `replay_fail_rate` comes from `ReputationStore::replay_fail_rate()` averaged over canary nodes.
    pub fn evaluate(
        &self,
        ctrl: &CohortSnapshot,
        canary: &CohortSnapshot,
        replay_fail_rate: f32,
    ) -> RolloutDecision {
        // Not enough data yet
        if canary.sample_count < self.promote.min_canary_samples {
            return RolloutDecision::Pending;
        }

        // ── Rollback checks (fail-fast) ──────────────────────────────────────
        if replay_fail_rate > self.rollback.max_replay_fail_rate {
            return RolloutDecision::Rollback {
                reason: format!(
                    "replay_fail_rate {:.3} > max {:.3}",
                    replay_fail_rate, self.rollback.max_replay_fail_rate
                ),
            };
        }

        if ctrl.drops_count > 0 {
            let drops_ratio = canary.drops_count as f32 / ctrl.drops_count as f32;
            if drops_ratio >= self.rollback.max_drops_ratio {
                return RolloutDecision::Rollback {
                    reason: format!(
                        "drops ratio {:.2} >= max {:.2}",
                        drops_ratio, self.rollback.max_drops_ratio
                    ),
                };
            }
        }

        if canary.q_mean < ctrl.q_mean - self.rollback.max_q_regression {
            return RolloutDecision::Rollback {
                reason: format!(
                    "q regression: canary={:.3} < ctrl={:.3} - threshold={:.3}",
                    canary.q_mean, ctrl.q_mean, self.rollback.max_q_regression
                ),
            };
        }

        // ── Promote checks ───────────────────────────────────────────────────
        if replay_fail_rate > self.promote.max_replay_fail_rate {
            return RolloutDecision::Pending;
        }

        if ctrl.drops_count > 0 {
            let drops_ratio = canary.drops_count as f32 / ctrl.drops_count as f32;
            if drops_ratio > self.promote.max_drops_ratio {
                return RolloutDecision::Pending;
            }
        }

        if ctrl.delta_norm_mean > 0.0 {
            let norm_ratio = canary.delta_norm_mean / ctrl.delta_norm_mean;
            if norm_ratio > self.promote.max_delta_norm_ratio {
                return RolloutDecision::Pending;
            }
        }

        if canary.q_mean >= ctrl.q_mean + self.promote.min_q_improvement {
            return RolloutDecision::Promote;
        }

        RolloutDecision::Pending
    }
}
