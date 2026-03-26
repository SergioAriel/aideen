use nalgebra::DVector;

use crate::state::HSlots;

pub const Q_MIN_LEARN: f32 = 0.6;
pub const Q_MIN_WRITE: f32 = 0.5;

/// Threshold below which the system decides to consult a live expert (1 hop).
/// If Q_semantic < Q_EXPERT_HOP, the system declares "I need help".
pub const Q_EXPERT_HOP: f32 = 0.45;

/// Physical metrics that determine attractor quality.
/// Q(h*) = 0.4*Stability + 0.3*Energy + 0.2*Oscillation + 0.1*Coherence
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMetrics {
    /// Residual stability: S = exp(-||h_{t+1} - h_t||)
    pub stability: f32,
    /// Integrated energy: E = exp(-||delta_s_r||)
    pub energy: f32,
    /// Oscillation: O = exp(-Var(||h_t - h_{t-1}||))
    pub oscillation: f32,
    /// Internal coherence: C = cos(delta_h, h*)
    pub coherence: f32,
    /// Total quality Q(h*) in the domain (0, 1]
    pub q_total: f32,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            stability: 0.0,
            energy: 0.0,
            oscillation: 0.0,
            coherence: 0.0,
            q_total: 0.0,
        }
    }
}

/// Computes the physical quality of an attractor h*.
/// Evaluated only when the system has reached equilibrium (or Control has stopped the loop).
pub fn compute_q(
    h_star: &DVector<f32>,
    h_prev: &DVector<f32>,    // h at t-1 for final stability
    delta_s_r: &DVector<f32>, // h* - s0
    oscillation_var: f32,     // Variance of delta_h norms during the loop
) -> QualityMetrics {
    // 1. Residual stability (Maximum if it stopped moving)
    let stability = (-(h_star - h_prev).norm()).exp();

    // 2. Integrated energy (Favors smooth convergence)
    let energy = (-delta_s_r.norm()).exp();

    // 3. Oscillation (Favors direct descent without vibration)
    let oscillation = (-oscillation_var).exp();

    // 4. Internal coherence (Alignment between change and final state)
    let coherence = {
        let dot = h_star.dot(delta_s_r);
        let norm = h_star.norm() * delta_s_r.norm() + 1e-8;
        (dot / norm).clamp(-1.0, 1.0)
    };

    // Final Formula: Domain (0, 1]
    let q_total = 0.4 * stability + 0.3 * energy + 0.2 * oscillation + 0.1 * coherence;

    QualityMetrics {
        stability,
        energy,
        oscillation,
        coherence,
        q_total,
    }
}

// ── Semantic Q ──────────────────────────────────────────────────────────────

/// Semantic quality signal: measures whether the system needs to consult the network.
///
/// Key difference from `QualityMetrics`:
/// - `QualityMetrics` measures whether the DEQ converged mathematically.
/// - `SemanticSignal` measures whether the result will be useful to the user.
///
/// Q_semantic = 0.5 * q_convergence
///            + 0.3 * slot_diversity   (distinct slots = richer H*)
///            + 0.2 * feedback_score   (real feedback signal, bootstrapped to 0.5)
#[derive(Debug, Clone, Copy)]
pub struct SemanticSignal {
    /// Attractor Q (from QualityMetrics.q_total)
    pub q_convergence: f32,
    /// How different the K slots are from each other.
    /// High = diversity = H* is rich. Low = all converged to the same = H* collapsed.
    pub slot_diversity: f32,
    /// User feedback signal: 1.0 = accepted / used, 0.0 = rejected / corrected.
    /// Bootstrapped to 0.5 until real signal is available.
    pub feedback_score: f32,
    /// Final combined score. If < Q_EXPERT_HOP → consult a live expert.
    pub q_semantic: f32,
}

impl SemanticSignal {
    /// Creates a signal with neutral feedback (no user data yet).
    pub fn bootstrapped(q_convergence: f32, slot_diversity: f32) -> Self {
        Self::new(q_convergence, slot_diversity, 0.5)
    }

    pub fn new(q_convergence: f32, slot_diversity: f32, feedback_score: f32) -> Self {
        let q_semantic = 0.5 * q_convergence + 0.3 * slot_diversity + 0.2 * feedback_score;
        Self {
            q_convergence,
            slot_diversity,
            feedback_score,
            q_semantic,
        }
    }

    /// Should a live expert be consulted?
    pub fn needs_expert(&self) -> bool {
        self.q_semantic < Q_EXPERT_HOP
    }

    /// Is it good enough to trigger Discovery to the Critic?
    pub fn qualifies_for_learning(&self) -> bool {
        self.q_semantic >= Q_MIN_LEARN
    }
}

/// Computes the diversity among the K slots of H*.
///
/// Idea: if all slots are equal, diversity is 0 (H* collapsed).
/// If slots are very different from each other, diversity is high (H* rich).
///
/// Implementation: average of pairwise slot distances, normalized.
pub fn compute_slot_diversity(h: &HSlots) -> f32 {
    let slots = h.slots;
    if slots < 2 {
        return 0.0;
    }

    let mut total_dist = 0.0;
    let mut pairs = 0u32;

    for i in 0..slots {
        for j in (i + 1)..slots {
            let diff = h.slot(i) - h.slot(j);
            total_dist += diff.norm();
            pairs += 1;
        }
    }

    let mean_dist = total_dist / pairs as f32;
    // Normalizamos con sigmoid para mantener el rango (0, 1)
    1.0 / (1.0 + (-mean_dist).exp())
}

/// Computes the mean energy of the K slots (average L2 norm).
///
/// A slot with energy close to 0 is a "dead slot" that did not differentiate
/// from the initial broadcast state. Useful for detecting representation collapse.
///
/// Output range: [0, ∞) — unbounded, the magnitude depends on the DEQ weights.
pub fn compute_slot_energy(h: &HSlots) -> f32 {
    let slots = h.slots;
    if slots == 0 {
        return 0.0;
    }
    (0..slots).map(|k| h.slot(k).norm()).sum::<f32>() / slots as f32
}

/// Routing decision: which path to take given the semantic state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Local H* is good enough. Go directly to the decoder.
    LocalOnly,
    /// H* is uncertain. Request contribution from a live expert (1 hop).
    ExpertHop,
    /// H* is excellent. Send Discovery to the Coordinator so the Critic can learn.
    Discovery,
}

/// Makes the routing decision based on Q_semantic.
pub fn decide_routing(signal: &SemanticSignal) -> RoutingDecision {
    if signal.needs_expert() {
        RoutingDecision::ExpertHop
    } else if signal.qualifies_for_learning() {
        RoutingDecision::Discovery
    } else {
        RoutingDecision::LocalOnly
    }
}
