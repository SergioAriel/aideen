use nalgebra::DVector;

/// Razonamiento iterativo (DEQ-like) - La propuesta de Δ
pub trait Reasoning {
    /// Inicializa el estado interno h₀
    fn init(&self, s: &DVector<f32>) -> DVector<f32>;

    /// Paso iterativo hₜ₊₁ = f(hₜ, S)
    fn step(
        &self,
        h: &DVector<f32>,
        s: &DVector<f32>,
        exec: Option<&mut dyn crate::compute::ComputeBackend>,
    ) -> DVector<f32>;
}

/// Estimación del Jacobiano local por perturbación finita.
#[derive(Debug, Clone)]
pub struct JacobianEstimate {
    /// Respuesta del sistema (f(h, θ+ε) - f(h, θ))
    pub delta_h: DVector<f32>,
    /// Magnitud de la perturbación ε
    pub eps: f32,
    /// Índice del parámetro perturbado (opaco para el nodo)
    pub weight_index: usize,
}

/// Extensión para razonamiento capaz de aprendizaje local (Nivel 3).
pub trait MutableReasoning: Reasoning {
    /// Perturba un peso aleatorio y retorna su índice.
    fn perturb_weight(&mut self, eps: f32) -> usize;
    /// Revierte la perturbación en el índice dado.
    fn revert_weight(&mut self, index: usize, eps: f32);
    /// Aplica una actualización basada en el Jacobiano y un factor sign(ΔQ) * η.
    fn apply_update(&mut self, jacobian: &JacobianEstimate, step: f32);
}
