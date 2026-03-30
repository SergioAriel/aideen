use crate::state::HSlots;
use nalgebra::DVector;

/// Iterative reasoning (DEQ-like) - The Δ proposal
pub trait Reasoning {
    /// Returns the architecture configuration.
    fn config(&self) -> &crate::state::ArchitectureConfig;

    /// Initializes the internal state H₀ from the global context S.
    /// By default, broadcasts the S_R subspace to the K slots.
    fn init(&self, s: &DVector<f32>) -> HSlots;

    /// Paso iterativo Hₜ₊₁ = f(Hₜ, S)
    /// Each slot is updated independently; an optional mixing layer
    /// can propagate information across slots.
    fn step(
        &self,
        h: &HSlots,
        s: &DVector<f32>,
        exec: Option<&mut dyn crate::compute::ComputeBackend>,
    ) -> HSlots;

    /// Temporal memory step M_t = g(M_{t-1}, H*).
    /// Executed once per token, after `step` has converged to a fixed-point
    /// solution H*. Updates the temporal state that will be passed to the
    /// next sequential token.
    fn temporal_step(&self, _m_prev: &HSlots, h_star: &HSlots) -> HSlots {
        // Default implementation: identity (no explicit short-term memory)
        // or overwrite with the last converged state.
        h_star.clone()
    }
}
