use serde::{Deserialize, Serialize};

/// Aggregate metrics for a cohort of nodes (control or canary).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CohortMetrics {
    pub q_mean: f32,
    pub drops_count: u32,
    pub delta_norm_mean: f32,
    pub sample_count: u32,
}

/// Decision from shadow eval.
#[derive(Debug, PartialEq)]
pub enum ShadowVerdict {
    /// Canary is sufficiently better than control — promote the update.
    Promote,
    /// Canary is worse than control — block the update.
    Block { reason: String },
    /// Not enough data yet to decide.
    Pending,
}

/// Manages shadow evaluation cohorts (control vs canary) and gates model updates.
///
/// Rules:
/// - Require `min_canary_samples` before evaluating.
/// - Block if canary drops ≥ 2× control drops.
/// - Block if canary q_mean < control q_mean − threshold (quality regression).
/// - Promote if canary q_mean ≥ control q_mean + threshold.
/// - Otherwise Pending.
#[derive(Serialize, Deserialize)]
pub struct ShadowEvalManager {
    pub control: Option<CohortMetrics>,
    pub canary: Option<CohortMetrics>,
    q_improvement_threshold: f32,
    min_canary_samples: u32,
}

impl ShadowEvalManager {
    pub fn new(q_improvement_threshold: f32) -> Self {
        ShadowEvalManager {
            control: None,
            canary: None,
            q_improvement_threshold,
            min_canary_samples: 10,
        }
    }

    pub fn set_control(&mut self, m: CohortMetrics) {
        self.control = Some(m);
    }
    pub fn set_canary(&mut self, m: CohortMetrics) {
        self.canary = Some(m);
    }

    pub fn evaluate(&self) -> ShadowVerdict {
        let (ctrl, canary) = match (&self.control, &self.canary) {
            (Some(c), Some(k)) => (c, k),
            _ => return ShadowVerdict::Pending,
        };

        if canary.sample_count < self.min_canary_samples {
            return ShadowVerdict::Pending;
        }

        // Drops regression: canary ≥ 2× control
        if ctrl.drops_count > 0 && canary.drops_count >= ctrl.drops_count * 2 {
            return ShadowVerdict::Block {
                reason: format!(
                    "drops regression: canary={} >= 2x control={}",
                    canary.drops_count, ctrl.drops_count
                ),
            };
        }

        // Quality regression
        if canary.q_mean < ctrl.q_mean - self.q_improvement_threshold {
            return ShadowVerdict::Block {
                reason: format!(
                    "q regression: canary={:.3} < control={:.3} - threshold={:.3}",
                    canary.q_mean, ctrl.q_mean, self.q_improvement_threshold
                ),
            };
        }

        // Quality improvement → promote
        if canary.q_mean >= ctrl.q_mean + self.q_improvement_threshold {
            return ShadowVerdict::Promote;
        }

        ShadowVerdict::Pending
    }

    /// Persist to disk with atomic write.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush(&self, path: &std::path::Path) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &bytes).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())
    }

    /// Load from disk. Returns default if file does not exist.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        bincode::deserialize(&bytes).map_err(|e| e.to_string())
    }
}

impl Default for ShadowEvalManager {
    fn default() -> Self {
        Self::new(0.05)
    }
}
