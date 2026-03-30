/// Critic v0: Softmax Bandit with per-expert reputation.
///
/// # Design
///
/// Deliberately simple MVP: *online* reputation tracker per node.
/// Uses a UCB-1 Q(arm) estimator smoothed by softmax to select the
/// next expert, updated after each network response.
///
/// ## Why not gradient-based now
/// * Requires full H* + sample buffer → Phase 7+ complexity.
/// * Bandit/UCB is stochastically correct for the MVP (demo-ready).
/// * Compatible with the full Critic: same interface, richer logic.
///
/// ## Invariants
/// * `Critic::select()` is pure and deterministic given the current state.
/// * `Critic::update()` is the only mutation point.
/// * No I/O, no allocation except in `select()` for the probs vec.
///
/// # Usage
/// ```no_run
/// use aideen_node::critic::{Critic, CriticConfig};
/// let mut critic = Critic::new(CriticConfig::default());
/// let arms = &[[1u8; 32], [2u8; 32]];
///
/// // Probabilistic selection
/// let chosen = critic.select(arms).unwrap();
///
/// // Update after receiving expert response
/// critic.update(&arms[chosen], 0.85); // q_total = 0.85
/// ```
use std::collections::HashMap;

use crate::peers::NodeId;

// ── Config ────────────────────────────────────────────────────────────────────

/// Critic bandit parameters.
#[derive(Debug, Clone)]
pub struct CriticConfig {
    /// Softmax temperature. Higher → exploration; lower → exploitation.
    /// Typical range [0.1, 2.0].
    pub temperature: f32,
    /// UCB confidence bonus (exploration). C = 0 disables UCB.
    pub ucb_c: f32,
    /// Initial Q value for never-seen experts (initial optimism).
    pub q_init: f32,
    /// Incremental learning rate (exponential moving average).
    pub lr: f32,
}

impl Default for CriticConfig {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            ucb_c: 0.5,
            q_init: 0.6, // Start with moderate optimism
            lr: 0.1,
        }
    }
}

// ── Arm state ─────────────────────────────────────────────────────────────────

/// Internal state of an arm (one expert/node).
#[derive(Debug, Clone)]
struct ArmState {
    /// Q(arm): exponential moving average of received q_total values.
    q_est: f32,
    /// Number of times this arm was selected.
    n: u64,
}

// ── Critic ────────────────────────────────────────────────────────────────────

/// Critic v0: Softmax bandit with UCB bonus.
///
/// Thread-unsafe (exclusive use by NodeRunner in its thread).
/// For production: wrap in `Arc<Mutex<Critic>>` in the runner.
pub struct Critic {
    cfg: CriticConfig,
    arms: HashMap<NodeId, ArmState>,
    /// Global step (total updates). Required for UCB.
    t: u64,
}

impl Critic {
    pub fn new(cfg: CriticConfig) -> Self {
        Self {
            cfg,
            arms: HashMap::new(),
            t: 0,
        }
    }

    /// Returns the index of the selected expert from the `candidates` list.
    ///
    /// If the list is empty, returns None.
    /// If there is only one candidate, returns it without sampling (fast path).
    pub fn select(&self, candidates: &[NodeId]) -> Option<usize> {
        match candidates.len() {
            0 => None,
            1 => Some(0),
            n => {
                // Compute UCB scores
                let logT = ((self.t.max(1)) as f32).ln();
                let scores: Vec<f32> = candidates
                    .iter()
                    .map(|id| match self.arms.get(id) {
                        None => self.cfg.q_init + self.cfg.ucb_c * logT.sqrt(),
                        Some(a) => {
                            let ucb = if a.n == 0 {
                                f32::INFINITY
                            } else {
                                self.cfg.ucb_c * (logT / a.n as f32).sqrt()
                            };
                            a.q_est + ucb
                        }
                    })
                    .collect();

                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores
                    .iter()
                    .map(|s| ((s - max_s) / self.cfg.temperature).exp())
                    .collect();
                let sum: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

                // Deterministic time-based sampling for MVP (no external PRNG)
                // For production: use rand::thread_rng().
                let uniform = pseudo_uniform(self.t, candidates.len());
                let mut cumulative = 0.0f32;
                for (i, &p) in probs.iter().enumerate() {
                    cumulative += p;
                    if uniform <= cumulative || i == n - 1 {
                        return Some(i);
                    }
                }
                Some(n - 1)
            }
        }
    }

    /// Updates an expert's reputation after receiving its response.
    ///
    /// `node_id`: the expert that responded.
    /// `q_received`: observed quality (q_total from ExpertResult).
    pub fn update(&mut self, node_id: &NodeId, q_received: f32) {
        self.t += 1;
        let lr = self.cfg.lr;
        let q_init = self.cfg.q_init;
        let arm = self.arms.entry(*node_id).or_insert(ArmState {
            q_est: q_init,
            n: 0,
        });
        arm.n += 1;
        // Exponential moving average
        arm.q_est = arm.q_est + lr * (q_received - arm.q_est);
    }

    /// Current reputation of an expert (None if never seen).
    pub fn reputation(&self, node_id: &NodeId) -> Option<f32> {
        self.arms.get(node_id).map(|a| a.q_est)
    }

    /// Top-k experts sorted by descending reputation.
    /// Useful for diagnostics and logs.
    pub fn top_k(&self, k: usize) -> Vec<(NodeId, f32)> {
        let mut ranked: Vec<(NodeId, f32)> =
            self.arms.iter().map(|(id, a)| (*id, a.q_est)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(k);
        ranked
    }
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Pseudo-random generator in [0, 1) based on t and n.
/// Not cryptographically secure, only for MVP without rand dependency.
fn pseudo_uniform(t: u64, n: usize) -> f32 {
    // Xorshift64 single step
    let mut x = t.wrapping_add(6364136223846793005u64);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let bits = x.wrapping_mul(2685821657736338717u64);
    // Convert to [0, 1)
    let frac = (bits >> 11) as f32 / (1u64 << 53) as f32;
    // Adjust to [0, n) and back to [0, 1) relative to n (for softmax cumulative)
    // In practice for select we use frac directly as uniform [0,1)
    let _ = n; // n se usa en el caller para bounds
    frac
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_critic_select_empty_returns_none() {
        let critic = Critic::new(CriticConfig::default());
        assert!(critic.select(&[]).is_none());
    }

    #[test]
    fn test_critic_select_single_returns_zero() {
        let critic = Critic::new(CriticConfig::default());
        let ids = [[1u8; 32]];
        assert_eq!(critic.select(&ids), Some(0));
    }

    #[test]
    fn test_critic_update_increases_reputation_on_good_feedback() {
        let mut critic = Critic::new(CriticConfig {
            q_init: 0.5,
            lr: 0.5,
            ..Default::default()
        });
        let id = [1u8; 32];
        critic.update(&id, 0.9);
        // EMA: 0.5 + 0.5*(0.9 - 0.5) = 0.5 + 0.2 = 0.7
        let rep = critic.reputation(&id).unwrap();
        assert!((rep - 0.7).abs() < 1e-5, "Expected 0.7, got {rep}");
    }

    #[test]
    fn test_critic_top_k_ordered() {
        let mut critic = Critic::new(CriticConfig::default());
        critic.update(&[1u8; 32], 0.9);
        critic.update(&[2u8; 32], 0.5);
        critic.update(&[3u8; 32], 0.7);

        let top = critic.top_k(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, [1u8; 32], "id=1 must be first (q=0.9)");
        assert!(top[0].1 > top[1].1, "top_k must be sorted desc");
    }
}
