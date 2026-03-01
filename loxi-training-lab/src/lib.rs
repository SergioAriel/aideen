use loxi_core::reasoning::Reasoning;
use nalgebra::DVector;

/// Estimación del Jacobiano local.
#[derive(Debug, Clone)]
pub struct JacobianEstimate {
    pub delta_h: DVector<f32>,
    pub eps: f32,
    pub weight_index: usize,
}

/// Trait para razonamientos que soportan aprendizaje (MUDADO DE CORE).
pub trait MutableReasoning: Reasoning {
    fn perturb_weight(&mut self, eps: f32) -> usize;
    fn revert_weight(&mut self, index: usize, eps: f32);
    fn apply_update(&mut self, jacobian: &JacobianEstimate, step: f32);
}

/// Implementación de estimación de Jacobiano para uso en el laboratorio.
pub fn estimate_jacobian<R>(
    reasoning: &mut R,
    h_star: &DVector<f32>,
    s: &DVector<f32>,
    eps: f32,
) -> JacobianEstimate
where
    R: MutableReasoning,
{
    let h_base = reasoning.step(h_star, s, None);
    let idx = reasoning.perturb_weight(eps);
    let h_perturbed = reasoning.step(h_star, s, None);
    reasoning.revert_weight(idx, eps);

    JacobianEstimate {
        delta_h: h_perturbed - h_base,
        eps,
        weight_index: idx,
    }
}
