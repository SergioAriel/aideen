use nalgebra::DVector;

use crate::state::HSlots;

pub const Q_MIN_LEARN: f32 = 0.6;
pub const Q_MIN_WRITE: f32 = 0.5;

/// Umbral por debajo del cual se decide consultar a un experto en vivo (1 hop).
/// Si Q_semantic < Q_EXPERT_HOP, el sistema declara "necesito ayuda".
pub const Q_EXPERT_HOP: f32 = 0.45;

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

// ── Q Semántico ──────────────────────────────────────────────────────────────

/// Señal semántica de calidad: mide si el sistema necesita consultar la red.
///
/// Diferencia clave con `QualityMetrics`:
/// - `QualityMetrics` mide si el DEQ convergió matemáticamente.
/// - `SemanticSignal` mide si el resultado va a ser útil para el usuario.
///
/// Q_semantic = 0.5 * q_convergence
///            + 0.3 * slot_diversity   (slots distintos = más riqueza en H*)
///            + 0.2 * feedback_score   (señal de feedback real, bootstrapped a 0.5)
#[derive(Debug, Clone, Copy)]
pub struct SemanticSignal {
    /// Q del atractor (viene de QualityMetrics.q_total)
    pub q_convergence: f32,
    /// Qué tan distintos son los K slots entre sí.
    /// Alto = diversidad = H* es rico. Bajo = todos convergieron a lo mismo = H* aplastado.
    pub slot_diversity: f32,
    /// Señal de feedback del usuario: 1.0 = aceptó / usó, 0.0 = rechazó / corrigió.
    /// Bootstrapped a 0.5 hasta tener señal real.
    pub feedback_score: f32,
    /// Score combinado final. Si < Q_EXPERT_HOP → consultar experto en vivo.
    pub q_semantic: f32,
}

impl SemanticSignal {
    /// Crea una señal con feedback neutro (sin datos de usuario aún).
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

    /// ¿Debe consultarse a un experto en vivo?
    pub fn needs_expert(&self) -> bool {
        self.q_semantic < Q_EXPERT_HOP
    }

    /// ¿Es suficientemente bueno para disparar Discovery al Critic?
    pub fn qualifies_for_learning(&self) -> bool {
        self.q_semantic >= Q_MIN_LEARN
    }
}

/// Calcula la diversidad entre los K slots de H*.
///
/// Idea: si todos los slots son iguales, la diversidad es 0 (H* aplastado).
/// Si los slots son muy distintos entre sí, la diversidad es alta (H* rico).
///
/// Implementación: promedio de las distancias entre pares de slots, normalizado.
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

/// Calcula la energía media de los K slots (norma L2 promedio).
///
/// Un slot con energía cercana a 0 es un "slot muerto" que no se diferenció
/// del estado de broadcast inicial. Útil para detectar colapso de la representación.
///
/// Rango de salida: [0, ∞) — no acotado, la magnitud depende de los pesos del DEQ.
pub fn compute_slot_energy(h: &HSlots) -> f32 {
    let slots = h.slots;
    if slots == 0 {
        return 0.0;
    }
    (0..slots).map(|k| h.slot(k).norm()).sum::<f32>() / slots as f32
}

/// Decisión de routing: qué camino tomar dado el estado semántico.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingDecision {
    /// H* local es suficientemente bueno. Ir directo al decoder.
    LocalOnly,
    /// H* es incierto. Pedir contribución a un experto en vivo (1 hop).
    ExpertHop,
    /// H* es excelente. Enviar Discovery al Coordinator para que el Critic aprenda.
    Discovery,
}

/// Toma la decisión de routing basada en Q_semantic.
pub fn decide_routing(signal: &SemanticSignal) -> RoutingDecision {
    if signal.needs_expert() {
        RoutingDecision::ExpertHop
    } else if signal.qualifies_for_learning() {
        RoutingDecision::Discovery
    } else {
        RoutingDecision::LocalOnly
    }
}
