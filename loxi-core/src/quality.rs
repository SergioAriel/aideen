use nalgebra::DVector;

pub const Q_MIN_LEARN: f32 = 0.6;
pub const Q_MIN_WRITE: f32 = 0.5;

/// Métricas físicas que determinan la calidad de un atractor.
/// Q(h*) = 0.4*Stability + 0.3*Energy + 0.2*Oscillation + 0.1*Coherence
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMetrics {
    /// Estabilidad residual: S = exp(-||h_{t+1} - h_t||)
    pub stability: f32,
    /// Energía integrada: E = exp(-||delta_s_r||)
    pub energy: f32,
    /// Oscilación: O = exp(-Var(||h_t - h_{t-1}||))
    pub oscillation: f32,
    /// Coherencia interna: C = cos(delta_h, h*)
    pub coherence: f32,
    /// Calidad total Q(h*) en el dominio (0, 1]
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

/// Calcula la calidad física de un atractor h*.
/// Se evalúa solo cuando el sistema ha alcanzado el equilibrio (o Control ha detenido el loop).
pub fn compute_q(
    h_star: &DVector<f32>,
    h_prev: &DVector<f32>,    // h en t-1 para estabilidad final
    delta_s_r: &DVector<f32>, // h* - s0
    oscillation_var: f32,     // Varianza de las normas de delta_h durante el loop
) -> QualityMetrics {
    // 1. Estabilidad residual (Máximo si dejó de moverse)
    let stability = (-(h_star - h_prev).norm()).exp();

    // 2. Energía integrada (Favorece convergencia suave)
    let energy = (-delta_s_r.norm()).exp();

    // 3. Oscilación (Favorece caída directa sin vibración)
    let oscillation = (-oscillation_var).exp();

    // 4. Coherencia interna (Alineación entre cambio y estado final)
    let coherence = {
        let dot = h_star.dot(delta_s_r);
        let norm = h_star.norm() * delta_s_r.norm() + 1e-8;
        (dot / norm).clamp(-1.0, 1.0)
    };

    // Fórmula Final: Dominio (0, 1]
    let q_total = 0.4 * stability + 0.3 * energy + 0.2 * oscillation + 0.1 * coherence;

    QualityMetrics {
        stability,
        energy,
        oscillation,
        coherence,
        q_total,
    }
}
