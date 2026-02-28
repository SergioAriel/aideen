use nalgebra::DVector;

/// Razonamiento iterativo (DEQ-like) - La propuesta de Δ
pub trait Reasoning {
    /// Inicializa el estado interno h₀
    fn init(&self, s: &DVector<f32>) -> DVector<f32>;

    /// Paso iterativo hₜ₊₁ = f(hₜ, S)
    fn step(&self, h: &DVector<f32>, s: &DVector<f32>) -> DVector<f32>;
}
