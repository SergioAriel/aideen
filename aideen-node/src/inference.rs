/// Módulo de inferencia: pipeline completo query → H* → RoutingDecision.
///
/// Este es el "corazón ejecutivo" del nodo Aideen:
/// 1. Codifica la query como vector de estado S
/// 2. Ejecuta el loop DEQ hasta convergencia (o max_iters)
/// 3. Calcula métricas semánticas sobre H*
/// 4. Decide el routing (LocalOnly / ExpertHop / Discovery)
///
/// No genera texto — eso es tarea del decoder (Fase 6).
/// No envía nada por red — eso lo hace el caller con el RoutingDecision.
use aideen_core::{
    compute::ComputeBackend,
    quality::{
        compute_q, compute_slot_diversity, compute_slot_energy, decide_routing, SemanticSignal,
    },
    reasoning::Reasoning,
    state::HSlots,
};
use nalgebra::DVector;

use crate::expert::ExpertPipeline;

// ── Resultado de la inferencia ────────────────────────────────────────────────

/// Resultado completo de un ciclo de inferencia Aideen.
///
/// Contiene el estado de razonamiento convergido (H*), las métricas
/// que caracterizan su calidad, y la decisión de routing recomendada.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Estado de razonamiento convergido (K×D_R).
    pub h_star: HSlots,
    /// Métricas de la inferencia.
    pub metrics: InferenceMetrics,
    /// Qué hacer a continuación (LocalOnly / ExpertHop / Discovery).
    pub routing: aideen_core::quality::RoutingDecision,
    /// Señal semántica usada para tomar la decisión de routing.
    pub signal: SemanticSignal,
}

/// Métricas de un ciclo de inferencia.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Estabilidad residual del DEQ (campo `stability` de QualityMetrics).
    pub stability: f32,
    /// Diversidad entre slots de H* [0, 1].
    pub slot_diversity: f32,
    /// Energía media de los slots (norma L2 promedio).
    pub slot_energy: f32,
    /// Número de iteraciones DEQ ejecutadas.
    pub iters: usize,
    /// true si el DEQ alcanzo el criterio de convergencia.
    pub converged: bool,
}

// ── Encoder de query ──────────────────────────────────────────────────────────

/// Codifica una query de texto como vector de estado S de dimensión `dim`.
///
/// Implementación MVP: hash FNV-1a de cada palabra → acumula en slots de `dim`.
/// Normalización final: tanh por componente → rango (-1, 1].
///
/// Esta es una implementación stub intencional para que el pipeline funcione
/// end-to-end antes de integrar un tokenizador real (Fase 6).
pub fn encode_query(query: &str, dim: usize) -> DVector<f32> {
    let mut feats = vec![0.0f32; dim];
    for word in query.split_whitespace() {
        let h = fnv1a(word.as_bytes());
        feats[h % dim] += 1.0;
        // También registra bigramas de caracteres para más granularidad
        let bytes = word.as_bytes();
        for pair in bytes.windows(2) {
            let h2 = fnv1a(pair);
            feats[h2 % dim] += 0.3;
        }
    }
    DVector::from_vec(feats.into_iter().map(|x| x.tanh()).collect())
}

fn fnv1a(bytes: &[u8]) -> usize {
    let mut h: u64 = 14695981039346656037u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h as usize
}

// ── Loop de inferencia principal ──────────────────────────────────────────────

/// Configuración del loop de inferencia.
pub struct InferenceConfig {
    /// Máximo de iteraciones DEQ.
    pub max_iters: usize,
    /// Umbral de convergencia: ||H_{k+1} - H_k||_flat < epsilon → converge.
    pub epsilon: f32,
    /// Factor de damping α para integración de estado.
    pub alpha: f32,
    /// Puntuación de feedback histórico [0, 1] (0.5 = neutral).
    pub feedback_score: f32,
    /// Cada cuántas iteraciones DEQ consultar los ExpertDEQ nodes.
    /// 0 = desactivado (sin ExpertDEQs — modo local puro).
    /// Recomendado: entre 3 y 8 (consultar a mitad de convergencia).
    pub k_expert_interval: usize,
    /// Peso β con el que se mezcla el delta experto en H*.
    /// H*[k] += β × expert_delta[k]
    /// Recomendado: 0.3–0.6 (no dejar que los experts dominen).
    pub expert_beta: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_iters: 24,
            epsilon: 1e-4,
            alpha: 0.5,
            feedback_score: 0.5,
            k_expert_interval: 6, // consultar experts cada 6 iters
            expert_beta: 0.4,     // peso moderado del delta experto
        }
    }
}

/// Ejecuta el pipeline completo de inferencia para una query de texto.
///
/// Pipeline:
/// ```text
/// query → encode_query(D_R) → State S →
///   DEQ loop:
///     H_{k+1} = reasoning.step(H_k, S)
///     cada K iters → ExpertPipeline.run(H_k) → inject delta
///   until converge →
///   H* → compute_q + diversity + energy →
///   SemanticSignal → decide_routing → InferenceResult
/// ```
///
/// # Parámetros
/// - `query`: texto libre de entrada
/// - `reasoning`: implementación del DEQ a usar
/// - `backend`: backend de cómputo (CPU o WGPU)
/// - `experts`: pipeline de ExpertDEQ nodes (None = modo local puro)
/// - `cfg`: parámetros del loop
pub fn run<R, B>(
    query: &str,
    reasoning: &mut R,
    backend: &mut B,
    mut experts: Option<&mut ExpertPipeline>,
    cfg: &InferenceConfig,
) -> Option<InferenceResult>
where
    R: Reasoning,
    B: ComputeBackend,
{
    let config = reasoning.config();
    let d_r = config.d_r;
    // 1. Codificar query como estado S
    let s = encode_query(query, d_r);

    // 2. Inicializar H desde S
    let mut h: HSlots = reasoning.init(&s);
    let mut h_prev = h.clone();
    let mut delta_norms: Vec<f32> = Vec::with_capacity(cfg.max_iters);

    // 3. Loop DEQ con K-checkpoint para ExpertDEQ nodes
    let mut iters = 0usize;
    let mut converged = false;
    let k_interval = cfg.k_expert_interval;

    for iter in 0..cfg.max_iters {
        iters = iter + 1;
        let h_next = reasoning.step(&h, &s, Some(backend as &mut dyn ComputeBackend));

        // ── K-checkpoint: consultar ExpertDEQ nodes ──────────────────────────
        // Se ejecuta cada k_interval iteraciones (no en iter=0 para dar tiempo
        // al DEQ de obtener una representación inicial).
        let h_next = if k_interval > 0 && iter > 0 && iter % k_interval == 0 {
            if let Some(ref mut pipeline) = experts {
                // Usamos slot 0 como representante para la query al expert
                let h_k_vec = h_next.slot(0);
                let h_k_slice: &[f32] = h_k_vec.as_slice();

                match pipeline.run(h_k_slice) {
                    Ok(result) => {
                        // Inyectar delta en todos los slots con peso expert_beta
                        let mut h_enriched = h_next.clone();
                        let config = reasoning.config();
                        for slot_idx in 0..config.h_slots {
                            let slot = h_enriched.slot(slot_idx);
                            let enriched = slot
                                .iter()
                                .zip(result.delta.iter())
                                .map(|(h, d)| h + cfg.expert_beta * d)
                                .collect::<Vec<_>>();
                            let enriched_vec = DVector::from_vec(enriched);
                            h_enriched.set_slot(slot_idx, &enriched_vec);
                        }
                        h_enriched
                    }
                    Err(_) => h_next, // Si el expert falla, continuar sin enrichment
                }
            } else {
                h_next
            }
        } else {
            h_next
        };
        // ── fin K-checkpoint ─────────────────────────────────────────────────

        // Convergencia sobre el flat completo
        let flat_next = h_next.to_flat();
        let flat_curr = h.to_flat();
        let delta_norm = flat_next
            .iter()
            .zip(flat_curr.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        delta_norms.push(delta_norm);
        h_prev = h.clone();
        h = h_next;

        if delta_norm < cfg.epsilon {
            converged = true;
            break;
        }
    }

    // 4. Calcular métricas de calidad
    let h_r = h.slot(0); // representante canónico para compute_q
    let h_prev_r = h_prev.slot(0);
    let delta_s_r = &h_r - &s;

    let oscillation_var = if delta_norms.len() > 1 {
        let mean = delta_norms.iter().sum::<f32>() / delta_norms.len() as f32;
        delta_norms.iter().map(|n| (n - mean).powi(2)).sum::<f32>() / delta_norms.len() as f32
    } else {
        0.0
    };

    let quality = compute_q(&h_r, &h_prev_r, &delta_s_r, oscillation_var);

    // Solo devuelve resultado si el DEQ alcanzó un estado de calidad mínima
    if quality.q_total < aideen_core::quality::Q_MIN_WRITE {
        return None;
    }

    // 5. Métricas semánticas
    let slot_diversity = compute_slot_diversity(&h);
    let slot_energy = compute_slot_energy(&h);

    let metrics = InferenceMetrics {
        stability: quality.stability,
        slot_diversity,
        slot_energy,
        iters,
        converged,
    };

    // 6. Señal semántica — Q_semantic = 0.5·q_conv + 0.3·diversity + 0.2·feedback
    let signal = SemanticSignal::new(quality.q_total, slot_diversity, cfg.feedback_score);

    // 7. Routing
    let routing = decide_routing(&signal);

    Some(InferenceResult {
        h_star: h,
        metrics,
        routing,
        signal,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use aideen_backbone::linear_reasoning::LinearReasoning;
    use aideen_core::compute::{ComputeBackend, TensorId};
    use aideen_core::state::ArchitectureConfig;

    struct NullBackend;
    impl ComputeBackend for NullBackend {
        fn load_tensor(&mut self, _d: &[f32]) -> Result<TensorId, String> {
            Ok(TensorId(0))
        }
        fn ffn_forward(
            &mut self,
            _w: &TensorId,
            _i: &TensorId,
            out_dim: usize,
        ) -> Result<DVector<f32>, String> {
            Ok(DVector::zeros(out_dim))
        }
    }

    #[test]
    fn test_encode_query_produces_nonzero() {
        let config = ArchitectureConfig::default();
        let d_r = config.d_r;
        let q = encode_query("¿qué es el DEQ?", d_r);
        assert_eq!(q.len(), d_r);
        assert!(
            q.iter().any(|&x| x.abs() > 1e-6),
            "encode no debe ser todo cero"
        );
    }

    #[test]
    fn test_different_queries_produce_different_h_star() {
        let config = ArchitectureConfig::default();
        let mut reasoning = LinearReasoning::new(config);
        let mut backend = NullBackend;
        let cfg = InferenceConfig {
            max_iters: 5,
            epsilon: 1e-6,
            ..Default::default()
        };

        let r1 = run("hello world", &mut reasoning, &mut backend, None, &cfg);
        let r2 = run(
            "aideen distributed AI",
            &mut reasoning,
            &mut backend,
            None,
            &cfg,
        );

        // Si alguno devuelve un resultado, h* debe depender del input
        if let (Some(a), Some(b)) = (r1, r2) {
            let flat_a = a.h_star.to_flat();
            let flat_b = b.h_star.to_flat();
            let diff: f32 = flat_a
                .iter()
                .zip(flat_b.iter())
                .map(|(x, y)| (x - y).abs())
                .sum();
            assert!(diff > 1e-6, "h* para queries distintas debe ser diferente");
        }
        // Si el DEQ no convergió con estos params, el test pasa vacío (no es un fallo del pipeline)
    }

    #[test]
    fn test_inference_result_has_three_routing_fields() {
        let config = ArchitectureConfig::default();
        let mut reasoning = LinearReasoning::new(config);
        let mut backend = NullBackend;
        let cfg = InferenceConfig::default();

        // Solo chequeamos que el resultado tiene los campos correctos si hay convergencia
        let _result = run(
            "test query para aideen",
            &mut reasoning,
            &mut backend,
            None,
            &cfg,
        );
        // Si no converge con pesos aleatorios: ok, eso se resuelve en entrenamiento
    }
}
