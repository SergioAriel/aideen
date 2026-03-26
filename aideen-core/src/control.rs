/// Control modes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlMode {
    /// V0: Observe and measure without intervening (Transparent Mode)
    Observe,
    /// V1: Regulate, penalize, and stabilize
    Regulate,
}

/// Per-iteration control decision
#[derive(Debug, Clone)]
pub struct ControlDecision {
    /// Should the DEQ loop be stopped?
    pub stop: bool,
    /// Integration intensity coefficient (Attenuation)
    pub beta: f32,
    /// Is local memory persistence authorized?
    pub write_memory: bool,
    /// Is local learning (weight adjustment) authorized?
    pub allow_learning: bool,
}

/// Convergence and stopping control (The orchestrator)
pub trait Control {
    /// Maximum allowed iterations limit
    fn max_iters(&self) -> usize;

    /// Current operating mode
    fn mode(&self) -> ControlMode;

    /// Make a decision based on the current trajectory and entropy
    fn decide(&self, iter: usize, delta_norm: f32, entropy: f32) -> ControlDecision;

    /// Optional function to record trajectory metrics (V0)
    fn observe(&self, _iter: usize, _delta_norm: f32, _entropy: f32) {}
}
