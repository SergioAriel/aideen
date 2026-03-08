/// Critic v0: Bandit Softmax con reputación por experto.
///
/// # Diseño
///
/// MVP deliberadamente simple: tracker de «reputación» *online* por nodo.
/// Usa un estimador Q(arm) tipo UCB-1 suavizado por softmax para seleccionar
/// el próximo experto, actualizado tras cada respuesta de la red.
///
/// ## Por qué no gradient-based ahora
/// * Requiere H* completo + sample buffer → complejidad Fase 7+.
/// * Bandit/UCB es estocásticamente correcto para el MVP (demo-ready).
/// * Compatible con el Critic completo: misma interfaz, lógica más rica.
///
/// ## Invariantes
/// * `Critic::select()` es puro y determinista dado el estado actual.
/// * `Critic::update()` es el único punto de mutación.
/// * No hay I/O, sin allocación excepto en `select()` para el vec de probs.
///
/// # Uso
/// ```no_run
/// use aideen_node::critic::{Critic, CriticConfig};
/// let mut critic = Critic::new(CriticConfig::default());
/// let arms = &[[1u8; 32], [2u8; 32]];
///
/// // Selección probabilística
/// let chosen = critic.select(arms).unwrap();
///
/// // Actualización tras recibir respuesta del experto
/// critic.update(&arms[chosen], 0.85); // q_total = 0.85
/// ```
use std::collections::HashMap;

use crate::peers::NodeId;

// ── Config ────────────────────────────────────────────────────────────────────

/// Parámetros del Critic bandit.
#[derive(Debug, Clone)]
pub struct CriticConfig {
    /// Temperatura softmax. Más alto → exploración; más bajo → explotación.
    /// Rango típico [0.1, 2.0].
    pub temperature: f32,
    /// Bonus de confianza UCB (exploración). C = 0 deshabilita UCB.
    pub ucb_c: f32,
    /// Valor inicial de Q para expertos nunca vistos (optimismo inicial).
    pub q_init: f32,
    /// Tasa de aprendizaje incremental (exponential moving average).
    pub lr: f32,
}

impl Default for CriticConfig {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            ucb_c: 0.5,
            q_init: 0.6, // Arrancamos con optimismo moderado
            lr: 0.1,
        }
    }
}

// ── Arm state ─────────────────────────────────────────────────────────────────

/// Estado interno de un brazo (un experto/nodo).
#[derive(Debug, Clone)]
struct ArmState {
    /// Q(arm): media exponencial de q_total recibidos.
    q_est: f32,
    /// Número de veces que este arm fue seleccionado.
    n: u64,
}

// ── Critic ────────────────────────────────────────────────────────────────────

/// Critic v0: Softmax bandit con bonus UCB.
///
/// Thead-unsafe (uso exclusivo por NodeRunner en su hilo).
/// Para producción: encapsular en `Arc<Mutex<Critic>>` en el runner.
pub struct Critic {
    cfg: CriticConfig,
    arms: HashMap<NodeId, ArmState>,
    /// Paso global (total de updates). Necesario para UCB.
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

    /// Devuelve el índice del experto seleccionado de la lista `candidates`.
    ///
    /// Si la lista está vacía, devuelve None.
    /// Si hay un solo candidato, lo devuelve sin muestreo (fast path).
    pub fn select(&self, candidates: &[NodeId]) -> Option<usize> {
        match candidates.len() {
            0 => None,
            1 => Some(0),
            n => {
                // Calcular UCB scores
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

                // Muestreo determinista basado en tiempo para MVP (sin PRNG externo)
                // En producción: usar rand::thread_rng().
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

    /// Actualiza la reputación de un experto tras recibir su respuesta.
    ///
    /// `node_id`: el experto que respondió.
    /// `q_received`: calidad observada (q_total del ExpertResult).
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

    /// Reputación actual de un experto (None si nunca se ha visto).
    pub fn reputation(&self, node_id: &NodeId) -> Option<f32> {
        self.arms.get(node_id).map(|a| a.q_est)
    }

    /// Top-k expertos ordenados por reputación descendente.
    /// Útil para diagnóstico y logs.
    pub fn top_k(&self, k: usize) -> Vec<(NodeId, f32)> {
        let mut ranked: Vec<(NodeId, f32)> =
            self.arms.iter().map(|(id, a)| (*id, a.q_est)).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked.truncate(k);
        ranked
    }
}

// ── Helpers privados ──────────────────────────────────────────────────────────

/// Generador pseudo-aleatorio en [0, 1) basado en t y n.
/// No es criptográficamente seguro, solo para MVP sin dependencia rand.
fn pseudo_uniform(t: u64, n: usize) -> f32 {
    // Xorshift64 single step
    let mut x = t.wrapping_add(6364136223846793005u64);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    let bits = x.wrapping_mul(2685821657736338717u64);
    // Convertir a [0, 1)
    let frac = (bits >> 11) as f32 / (1u64 << 53) as f32;
    // Ajustar a [0, n) y volver a [0, 1) relativo a n (para softmax cumulative)
    // En realidad para select usamos directamente frac como uniforme [0,1)
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
        assert_eq!(top[0].0, [1u8; 32], "id=1 debe ser primero (q=0.9)");
        assert!(top[0].1 > top[1].1, "top_k debe estar ordenado desc");
    }
}
