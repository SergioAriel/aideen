use nalgebra::DVector;

use loxi_core::{
    control::{Control, ControlDecision},
    ethics::Ethics,
    memory::Memory,
    quality::{compute_q, QualityMetrics},
    reasoning::Reasoning,
    state::D_SIM,
};

use crate::engine::ComputeBackend;
use crate::network::Transport;

/// Nodo LOXI: orquesta el ciclo completo sin conocer implementaciones concretas.
pub struct LoxiNode<R, C, E, M, B, N> {
    // Estado cognitivo global [S_M | S_R | S_C | S_E | S_sim]
    pub state: DVector<f32>,

    // Contratos cognitivos (core)
    pub reasoning: R,
    pub control: C,
    pub ethics: E,
    pub memory: M,

    // Infraestructura (node)
    pub backend: B,
    pub network: N,

    // Parámetros de integración
    pub alpha: f32,
    pub epsilon: f32,
}

/// Razón por la cual se detuvo el tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StopReason {
    Control,
    Epsilon,
    Ethics,
}

/// Métricas de un tick para visualización y control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickMetrics {
    pub energy_r: f32,
    pub energy_sim: f32,
    pub energy_total: f32,
    pub iters: usize,
    pub converged: bool,
    pub stop_reason: StopReason,
    pub quality: QualityMetrics,
}

impl Default for TickMetrics {
    fn default() -> Self {
        Self {
            energy_r: 0.0,
            energy_sim: 0.0,
            energy_total: 0.0,
            iters: 0,
            converged: false,
            stop_reason: StopReason::Epsilon,
            quality: QualityMetrics::default(),
        }
    }
}

impl<R, C, E, M, B, N> LoxiNode<R, C, E, M, B, N>
where
    R: Reasoning,
    C: Control,
    E: Ethics,
    M: Memory,
    B: ComputeBackend,
    N: Transport,
{
    /// Un tick completo del nodo LOXI. Devuelve métricas si hubo integración.
    pub fn tick(&mut self) -> Option<TickMetrics> {
        let s0 = self.state.clone();

        // ── 1. Inicializar razonamiento ─────────────────────
        let mut h = self.reasoning.init(&s0);
        let mut h_prev;
        let mut delta_norms: Vec<f32> = Vec::new();

        // ── 2. Loop DEQ gobernado por Control ────────────────
        for iter in 0..self.control.max_iters() {
            let h_next = self.reasoning.step(&h, &s0);
            let delta_h = &h_next - &h;
            let delta_norm = delta_h.norm();
            delta_norms.push(delta_norm);

            let entropy = 0.0;
            self.control.observe(iter, delta_norm, entropy);
            let decision: ControlDecision = self.control.decide(iter, delta_norm, entropy);

            h_prev = h.clone();
            h = h_next;

            if decision.stop {
                let integrable_dim = self.state.len() - D_SIM;
                let h_r = h.rows(0, integrable_dim);
                let s_r0 = s0.rows(0, integrable_dim);
                let delta_s_r = &h_r - &s_r0;

                // Cálculo de Varianza de Oscilación sobre el subespacio R
                let oscillation_var = if delta_norms.len() > 1 {
                    let mean = delta_norms.iter().sum::<f32>() / delta_norms.len() as f32;
                    delta_norms.iter().map(|n| (n - mean).powi(2)).sum::<f32>()
                        / delta_norms.len() as f32
                } else {
                    0.0
                };

                let h_r_owned = h.rows(0, integrable_dim).into_owned();
                let h_prev_r_owned = h_prev.rows(0, integrable_dim).into_owned();
                let delta_s_r_owned = delta_s_r.into_owned();

                let quality = compute_q(
                    &h_r_owned,
                    &h_prev_r_owned,
                    &delta_s_r_owned,
                    oscillation_var,
                );

                // Recalculamos delta_s_r para evitar el move previo de into_owned()
                let integrable_dim = self.state.len() - D_SIM;
                // delta_s para integración (usamos h completa: 2560)
                let delta_s = &h - &s0;
                let delta_s_r_ref = delta_s.rows(0, integrable_dim);

                let mut proposed = self.state.clone();
                let mut energy_sq_r: f32 = 0.0;
                let mut energy_sq_sim: f32 = 0.0;

                // Integración de S_R (y otros cores)
                for (si, di) in proposed.as_mut_slice()[0..integrable_dim]
                    .iter_mut()
                    .zip(delta_s_r_ref.as_slice().iter())
                {
                    let c = (self.alpha * decision.beta * di).tanh();
                    *si += c;
                    energy_sq_r += c * c;
                }

                // Medición de S_sim (sin integración)
                let integrable_dim = self.state.len() - D_SIM;
                if h.len() >= integrable_dim + D_SIM {
                    let delta_s_sim =
                        &h.rows(integrable_dim, D_SIM) - &s0.rows(integrable_dim, D_SIM);
                    for di in delta_s_sim.as_slice().iter() {
                        energy_sq_sim += di * di;
                    }
                }

                let energy_r = energy_sq_r.sqrt();
                let energy_sim = energy_sq_sim.sqrt();
                let energy_total = (energy_sq_r + energy_sq_sim).sqrt();

                // Caso A: Cambio insignificante (Epsilon)
                if energy_r < self.epsilon {
                    return Some(TickMetrics {
                        energy_r,
                        energy_sim,
                        energy_total,
                        iters: iter + 1,
                        converged: true,
                        stop_reason: StopReason::Epsilon,
                        quality,
                    });
                }

                // Caso B: Violación ética
                if self.ethics.violates(&proposed) {
                    return Some(TickMetrics {
                        energy_r,
                        energy_sim,
                        energy_total,
                        iters: iter + 1,
                        converged: false,
                        stop_reason: StopReason::Ethics,
                        quality,
                    });
                }

                // Caso C: Control detuvo el loop
                self.state = proposed;
                return Some(TickMetrics {
                    energy_r,
                    energy_sim,
                    energy_total,
                    iters: iter + 1,
                    converged: true,
                    stop_reason: StopReason::Control,
                    quality,
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use loxi_backbone::tensor::Tensor;
    use loxi_core::control::ControlMode;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct MockBackend;
    impl ComputeBackend for MockBackend {
        fn load_tensor(&mut self, _tensor: &Tensor) -> Result<(), String> {
            Ok(())
        }
        fn execute_shader(&mut self, _shader_id: &str) -> Result<Tensor, String> {
            Ok(Tensor::new(vec![1], vec![0.0]))
        }
    }

    struct MockTransport;
    impl Transport for MockTransport {
        fn connect(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn send_chunk(&mut self, _chunk: &Tensor) -> Result<(), String> {
            Ok(())
        }
        fn receive_chunk(&mut self) -> Result<Tensor, String> {
            Ok(Tensor::new(vec![1], vec![0.0]))
        }
    }

    struct MockReasoning {
        step_return_len: usize,
    }
    impl Reasoning for MockReasoning {
        fn init(&self, state: &DVector<f32>) -> DVector<f32> {
            state.rows(0, 2048).into_owned()
        }
        fn step(&self, _h: &DVector<f32>, _x: &DVector<f32>) -> DVector<f32> {
            DVector::zeros(self.step_return_len)
        }
    }

    struct MockControl {
        stop_at_iter: usize,
        beta: f32,
        write_memory: bool,
    }
    impl Control for MockControl {
        fn max_iters(&self) -> usize {
            10
        }
        fn mode(&self) -> ControlMode {
            ControlMode::Observe
        }
        fn decide(&self, iter: usize, _delta_norm: f32, _entropy: f32) -> ControlDecision {
            ControlDecision {
                stop: iter >= self.stop_at_iter,
                beta: self.beta,
                write_memory: self.write_memory,
            }
        }
    }

    struct MockEthics {
        will_violate: bool,
    }
    impl Ethics for MockEthics {
        fn fingerprint(&self) -> [u8; 32] {
            [0; 32]
        }
        fn violates(&self, _state: &DVector<f32>) -> bool {
            self.will_violate
        }
        fn project(&self, state: &DVector<f32>) -> DVector<f32> {
            state.clone()
        }
    }

    struct MockMemory {
        writes: Rc<RefCell<usize>>,
    }
    impl Memory for MockMemory {
        fn write(&mut self, _invariant: DVector<f32>) {
            *self.writes.borrow_mut() += 1;
        }
        fn query(&self, _q: &DVector<f32>, _k: usize) -> Vec<DVector<f32>> {
            vec![]
        }
    }

    fn build_test_node(
        stop_iter: usize,
        ethics_violate: bool,
    ) -> (
        LoxiNode<MockReasoning, MockControl, MockEthics, MockMemory, MockBackend, MockTransport>,
        Rc<RefCell<usize>>,
    ) {
        let writes = Rc::new(RefCell::new(0));
        let memory = MockMemory {
            writes: Rc::clone(&writes),
        };

        let node = LoxiNode {
            state: DVector::zeros(2560),
            reasoning: MockReasoning {
                step_return_len: 2048,
            },
            control: MockControl {
                stop_at_iter: stop_iter,
                beta: 1.0,
                write_memory: true,
            },
            ethics: MockEthics {
                will_violate: ethics_violate,
            },
            memory,
            backend: MockBackend,
            network: MockTransport,
            alpha: 0.5,
            epsilon: 0.001,
        };
        (node, writes)
    }

    #[test]
    fn test_node_compiles_and_runs() {
        let (mut node, _) = build_test_node(2, false);
        let metrics = node.tick();
        assert!(metrics.is_some());
        let m = metrics.unwrap();
        assert!(m.converged);
        assert_eq!(m.stop_reason, StopReason::Epsilon);
    }
}
