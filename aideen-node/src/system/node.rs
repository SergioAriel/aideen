use nalgebra::DVector;

use aideen_core::{
    control::{Control, ControlDecision},
    ethics::Ethics,
    memory::Memory,
    quality::{compute_q, QualityMetrics, Q_MIN_LEARN, Q_MIN_WRITE},
    reasoning::Reasoning,
    state::HSlots,
};

/// AIDEEN Node: orchestrates the full cycle without knowing concrete implementations.
/// The network is external to the DEQ loop — messages are emitted from outside tick().
pub struct AideenNode<R, C, E, M, B> {
    // Estado cognitivo global [S_M | S_R | S_C | S_E | S_sim]
    pub state: DVector<f32>,

    // Cognitive contracts (core)
    pub reasoning: R,
    pub control: C,
    pub ethics: E,
    pub memory: M,

    // Compute infrastructure
    pub backend: B,

    // Integration parameters
    pub alpha: f32,
    pub epsilon: f32,
}

/// Reason why the tick stopped.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StopReason {
    Control,
    Epsilon,
    Ethics,
    LowQuality,
    ReachedAttractor,
}

/// Tick metrics for visualisation and control.
#[derive(Debug, Clone)]
pub struct TickMetrics {
    pub energy_r: f32,
    pub energy_sim: f32,
    pub energy_total: f32,
    pub iters: usize,
    pub converged: bool,
    pub stop_reason: StopReason,
    pub is_attractor: bool,
    pub allow_learning: bool,
    pub quality: QualityMetrics,
    /// Cognitive fixed point (K×D_R slots). Only present when is_attractor == true.
    pub h_star: Option<HSlots>,
}

/// Tick output signal for the laboratory (PRIVATE/EXTERNAL).
/// Internal struct — never travels over the network directly.
/// To emit to the network use `to_discovery_msg()`.
#[derive(Debug, Clone)]
pub struct LearningSignal {
    pub allow_learning: bool,
    pub q_total: f32,
    pub h_star: HSlots,
    pub s_context: DVector<f32>,
}

impl LearningSignal {
    /// Converts the internal signal to the canonical wire-format for the AiArchitect.
    /// Only hashes travel — full h* remains local.
    pub fn to_discovery_msg(
        &self,
        node_id: [u8; 32],
        target_id: String,
        iters: u32,
        stop: u8,
        bundle_version: u64,
    ) -> aideen_core::protocol::NetMsg {
        use sha2::{Digest, Sha256};

        let h_flat = self.h_star.to_flat();
        let h_bytes: Vec<u8> = h_flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        let h_star_hash: [u8; 32] = Sha256::digest(&h_bytes).into();

        let ctx_bytes: Vec<u8> = self
            .s_context
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let context_hash: [u8; 32] = Sha256::digest(&ctx_bytes).into();

        aideen_core::protocol::NetMsg::Discovery {
            node_id,
            target_id,
            q_total: self.q_total,
            iters,
            stop,
            h_star_hash,
            context_hash,
            bundle_version,
        }
    }
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
            is_attractor: false,
            allow_learning: false,
            quality: QualityMetrics::default(),
            h_star: None,
        }
    }
}

impl<R, C, E, M, B> AideenNode<R, C, E, M, B>
where
    R: Reasoning,
    C: Control,
    E: Ethics,
    M: Memory,
    B: aideen_core::compute::ComputeBackend,
{
    /// Constitutional Attractor Definition (AIDEEN):
    /// h* es atractor <=> ||h_{t+1} - h_t|| < epsilon AND Q(h*) >= Q_MIN_WRITE
    pub fn is_attractor_state(&self, delta_norm: f32, q: f32) -> bool {
        delta_norm < self.epsilon && q >= Q_MIN_WRITE
    }

    /// Writes the context vector into the S_sim subspace (not integrable).
    /// Call before tick() to inject external document context.
    /// `ctx` must have length D_SIM; if shorter, the rest stays at 0.
    pub fn set_context(&mut self, ctx: &DVector<f32>) {
        let config = self.reasoning.config();
        let d_reasoning = config.d_reasoning();
        let d_sim = config.d_sim;
        let len = ctx.len().min(d_sim);
        self.state.as_mut_slice()[d_reasoning..d_reasoning + len]
            .copy_from_slice(&ctx.as_slice()[..len]);
        for v in &mut self.state.as_mut_slice()[d_reasoning + len..d_reasoning + d_sim] {
            *v = 0.0;
        }
    }

    /// Injects a delta into the reasoning subspace S_R.
    /// state[OFF_R .. OFF_R+D_R] += delta
    /// Call before tick() for warm-start with expert contributions.
    pub fn inject_delta_r(&mut self, delta: &[f32]) {
        let config = self.reasoning.config();
        let off_r = config.off_r();
        let d_r = config.d_r;
        assert_eq!(delta.len(), d_r, "inject_delta_r: delta must be d_r={d_r}");
        let sr = &mut self.state.as_mut_slice()[off_r..off_r + d_r];
        for (s, d) in sr.iter_mut().zip(delta.iter()) {
            *s += d;
        }
    }

    /// Same as inject_delta_r but applies β in-place, without allocating an extra Vec.
    /// state[OFF_R .. OFF_R+D_R] += beta * delta
    pub fn inject_delta_r_scaled(&mut self, delta: &[f32], beta: f32) {
        let config = self.reasoning.config();
        let off_r = config.off_r();
        let d_r = config.d_r;
        assert_eq!(
            delta.len(),
            d_r,
            "inject_delta_r_scaled: delta must be d_r={d_r}"
        );
        let sr = &mut self.state.as_mut_slice()[off_r..off_r + d_r];
        for (s, d) in sr.iter_mut().zip(delta.iter()) {
            *s += beta * d;
        }
    }

    /// A full tick of the AIDEEN node. Returns metrics if integration occurred.
    pub fn tick(&mut self) -> Option<TickMetrics> {
        let s0 = self.state.clone();

        // ── 1. Initialise reasoning ───────────────────────
        // Warm-start: if there are similar attractors in memory, we use the closest
        // as h_0; otherwise we fall back to the Reasoning's normal init().
        // Initialise H as HSlots from the global state S0
        let h_init: HSlots = self.reasoning.init(&s0);
        // warm-start: if similar attractors exist in memory use slot(0) of the closest
        let mem_hint = self.memory.query(&h_init.slot(0), 1);
        let mut h: HSlots = if let Some(hint_vec) = mem_hint.into_iter().next() {
            // Reconstruct HSlots from the memory DVector (broadcast to slot 0, rest from init)
            let mut warm = h_init.clone();
            warm.set_slot(0, &hint_vec);
            warm
        } else {
            h_init
        };
        let mut h_prev: HSlots = h.clone();
        let mut delta_norms: Vec<f32> = Vec::new();

        // ── 2. DEQ loop governed by Control ──────────────────
        for iter in 0..self.control.max_iters() {
            let h_next = self.reasoning.step(&h, &s0, Some(&mut self.backend));

            // Convergence: norm of the difference between the flats of H_k+1 and H_k
            let flat_next = h_next.to_flat();
            let flat_curr = h.to_flat();
            let delta_norm = flat_next
                .iter()
                .zip(flat_curr.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            delta_norms.push(delta_norm);

            let entropy = 0.0;
            self.control.observe(iter, delta_norm, entropy);
            let decision: ControlDecision = self.control.decide(iter, delta_norm, entropy);

            h_prev = h.clone();
            h = h_next;

            if decision.stop {
                // Use slot(0) as canonical representative of H* for compute_q
                // (compute_q works with DVector; slot richness is measured by SemanticSignal)
                let h_r_owned = h.slot(0);
                let h_prev_r_owned = h_prev.slot(0);
                let config = self.reasoning.config();
                let s_r0 = s0.rows(config.off_r(), config.d_r).into_owned();
                let delta_s_r_owned = &h_r_owned - &s_r0;

                // Oscillation variance
                let oscillation_var = if delta_norms.len() > 1 {
                    let mean = delta_norms.iter().sum::<f32>() / delta_norms.len() as f32;
                    delta_norms.iter().map(|n| (n - mean).powi(2)).sum::<f32>()
                        / delta_norms.len() as f32
                } else {
                    0.0
                };

                let quality = compute_q(
                    &h_r_owned,
                    &h_prev_r_owned,
                    &delta_s_r_owned,
                    oscillation_var,
                );

                // --- GATING N2 (Física Cognitiva) ---
                let is_attractor = self.is_attractor_state(delta_norm, quality.q_total);

                if !is_attractor {
                    return Some(TickMetrics {
                        energy_r: 0.0,
                        energy_sim: 0.0,
                        energy_total: 0.0,
                        iters: iter + 1,
                        converged: false,
                        stop_reason: StopReason::LowQuality,
                        is_attractor: false,
                        allow_learning: false,
                        quality,
                        h_star: None,
                    });
                }

                // Stable integration: apply delta H*[slot0] over the S_R subspace (D_R dims).
                // delta_i = h_flat[i] - s0[i]  →  c = tanh(α·β·delta)  →  proposed[i] += c
                let h_flat = h.to_flat();
                let mut proposed = self.state.clone();
                let mut energy_sq_r = 0.0;

                let d_r = self.reasoning.config().d_r;
                for i in 0..d_r {
                    let di = h_flat[i] - s0[i];
                    let c = (self.alpha * decision.beta * di).tanh();
                    proposed[i] += c;
                    energy_sq_r += c * c;
                }

                let energy_r = energy_sq_r.sqrt();
                let energy_total = energy_r; // S_sim no cambia con H*

                if self.ethics.violates(&proposed) {
                    return Some(TickMetrics {
                        energy_r,
                        energy_sim: 0.0,
                        energy_total,
                        iters: iter + 1,
                        converged: false,
                        stop_reason: StopReason::Ethics,
                        is_attractor: false,
                        allow_learning: false,
                        quality,
                        h_star: None,
                    });
                }

                let allow_learning = quality.q_total >= Q_MIN_LEARN;
                let write_memory = decision.write_memory && (quality.q_total >= Q_MIN_WRITE);

                if write_memory {
                    // Store slot 0 of H* as representative in memory
                    self.memory.write(h.slot(0));
                }

                self.state = proposed;
                let stop_reason = if delta_norm < self.epsilon {
                    StopReason::ReachedAttractor
                } else {
                    StopReason::Control
                };

                return Some(TickMetrics {
                    energy_r,
                    energy_sim: 0.0,
                    energy_total,
                    iters: iter + 1,
                    converged: true,
                    stop_reason,
                    is_attractor: true,
                    allow_learning,
                    quality,
                    h_star: Some(h.clone()),
                });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aideen_core::control::ControlMode;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct MockBackend;
    impl aideen_core::compute::ComputeBackend for MockBackend {
        fn load_tensor(&mut self, _data: &[f32]) -> Result<aideen_core::compute::TensorId, String> {
            Ok(aideen_core::compute::TensorId(0))
        }
        fn ffn_forward(
            &mut self,
            _w: &aideen_core::compute::TensorId,
            _i: &aideen_core::compute::TensorId,
            out_dim: usize,
        ) -> Result<nalgebra::DVector<f32>, String> {
            Ok(nalgebra::DVector::zeros(out_dim))
        }
    }

    struct MockReasoning {
        config: aideen_core::state::ArchitectureConfig,
    }
    impl Reasoning for MockReasoning {
        fn config(&self) -> &aideen_core::state::ArchitectureConfig {
            &self.config
        }
        fn init(&self, s: &DVector<f32>) -> aideen_core::state::HSlots {
            let config = self.config();
            aideen_core::state::HSlots::from_broadcast(
                &s.rows(config.off_r(), config.d_r).into_owned(),
                config,
            )
        }
        fn step(
            &self,
            h: &aideen_core::state::HSlots,
            _x: &DVector<f32>,
            _exec: Option<&mut dyn aideen_core::compute::ComputeBackend>,
        ) -> aideen_core::state::HSlots {
            h.clone()
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
                allow_learning: true,
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
        AideenNode<MockReasoning, MockControl, MockEthics, MockMemory, MockBackend>,
        Rc<RefCell<usize>>,
    ) {
        let writes = Rc::new(RefCell::new(0));
        let memory = MockMemory {
            writes: Rc::clone(&writes),
        };

        let config = aideen_core::state::ArchitectureConfig::default();
        let node = AideenNode {
            state: DVector::zeros(config.total_size()),
            reasoning: MockReasoning {
                config: config.clone(),
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
        assert_eq!(m.stop_reason, StopReason::ReachedAttractor);
    }
}
