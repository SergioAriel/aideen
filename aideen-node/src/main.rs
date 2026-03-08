use aideen_core::control::{Control, ControlDecision};
use aideen_core::ethics::Ethics;
use aideen_core::memory::Memory;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots, State};

use aideen_core::compute::ComputeBackend;
use aideen_engine::gpu::WgpuBackend;
use aideen_node::system::node::{AideenNode, TickMetrics};

use nalgebra::DVector;
use rand::Rng;

// ─── Implementaciones Temporales (Dummies) ───────────────────────

struct DummyEthics;
impl Ethics for DummyEthics {
    fn fingerprint(&self) -> [u8; 32] {
        [0; 32]
    }
    fn violates(&self, _state: &DVector<f32>) -> bool {
        false
    }
    fn project(&self, state: &DVector<f32>) -> DVector<f32> {
        state.clone()
    }
}

struct DummyMemory;
impl Memory for DummyMemory {
    fn write(&mut self, _invariant: DVector<f32>) {}
    fn query(&self, _q: &DVector<f32>, _k: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}

struct StressControl {
    beta: f32,
    max_iters: usize,
}
impl Control for StressControl {
    fn max_iters(&self) -> usize {
        self.max_iters
    }
    fn mode(&self) -> aideen_core::control::ControlMode {
        aideen_core::control::ControlMode::Observe
    }
    fn decide(&self, iter: usize, _delta_norm: f32, _entropy: f32) -> ControlDecision {
        ControlDecision {
            stop: iter >= self.max_iters - 1,
            beta: self.beta,
            write_memory: false,
            allow_learning: true,
        }
    }
}

// Reasoning inestable para Test D
struct UnstableReasoning {
    config: ArchitectureConfig,
    mode: u8, // 0: oscillation, 1: divergence
}
impl aideen_core::reasoning::Reasoning for UnstableReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.config
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        let d_r = self.config.d_r;
        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };
        HSlots::from_broadcast(&s_r, &self.config)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn aideen_core::compute::ComputeBackend>,
    ) -> HSlots {
        let h_slots = self.config.h_slots;
        let mut next = HSlots::zeros(&self.config);
        for k in 0..h_slots {
            let mut slot = h.slot(k);
            if self.mode == 0 {
                // Oscilación: voltea y amplía lentamente
                slot.iter_mut().for_each(|v| *v *= -1.1);
            } else {
                // Divergencia lineal
                slot.iter_mut().for_each(|v| *v += 10.0);
            }
            next.set_slot(k, &slot);
        }
        next
    }
}

#[tokio::main]
async fn main() {
    println!("🟥 NIVEL 2 — STRESS DINÁMICO AIDEEN");
    println!("──────────────────────────────────────────────────");

    // A. Sensibilidad a Alpha
    test_alpha_sweep().await;

    // B. Sensibilidad a Beta
    test_beta_sweep().await;

    // C. Perturbación Adversaria en S_sim
    test_adversarial_sim().await;

    // D. No convergencia forzada
    test_unstable_reasoning().await;
}

async fn test_alpha_sweep() {
    println!("\n[TEST A: BARRIDO DE ALPHA]");
    let alphas = [0.01, 0.05, 0.1, 0.3, 0.7, 1.0];

    for alpha in alphas {
        let mut node = build_base_node(alpha, 1.0, 5).await;
        println!(">>> Ejecutando alpha = {}", alpha);
        run_sim_quiet(&mut node);
    }
}

async fn test_beta_sweep() {
    println!("\n[TEST B: BARRIDO DE BETA (CONTROL)]");
    let betas = [0.1, 0.5, 1.0, 2.0, 5.0];

    for beta in betas {
        let mut node = build_base_node(0.1, beta, 5).await;
        println!(">>> Ejecutando beta = {}", beta);
        run_sim_quiet(&mut node);
    }
}

async fn test_adversarial_sim() {
    println!("\n[TEST C: PERTURBACIÓN ADVERSARIA EN S_SIM]");

    // 1. Ruido aleatorio grande
    println!(">>> 1. Ruido grande en S_sim");
    let mut node_large = build_base_node(0.1, 1.0, 5).await;
    let config = node_large.reasoning.config().clone();
    let d_sim = config.d_sim;
    let mut rng = rand::thread_rng();
    let noise: Vec<f32> = (0..d_sim).map(|_| rng.gen_range(-10.0..10.0)).collect();
    let off_sim = config.off_sim();
    node_large.state.as_mut_slice()[off_sim..off_sim + d_sim].copy_from_slice(&noise);
    run_sim_quiet(&mut node_large);

    // 2. Ruido pequeño persistente (inyectado cada tick)
    println!(">>> 2. Ruido persistente");
    let mut node_persist = build_base_node(0.1, 1.0, 5).await;
    for tick in 1..=5 {
        let noise: Vec<f32> = (0..d_sim).map(|_| rng.gen_range(-0.1..0.1)).collect();
        node_persist.state.as_mut_slice()[off_sim..off_sim + d_sim].copy_from_slice(&noise);
        if let Some(m) = node_persist.tick() {
            print_metrics("PERSIST", tick, &m);
        }
    }
}

async fn test_unstable_reasoning() {
    println!("\n[TEST D: NO CONVERGENCIA FORZADA]");

    let config = ArchitectureConfig::default();
    let d_r = config.d_r;
    let total_size = config.total_size();

    println!(">>> 1. Oscilación divergente");
    let mut state_osc = DVector::zeros(total_size);
    state_osc.rows_mut(0, d_r).add_scalar_mut(1.0);

    let mut node_osc = AideenNode {
        state: state_osc,
        reasoning: UnstableReasoning {
            config: config.clone(),
            mode: 0,
        },
        control: StressControl {
            beta: 1.0,
            max_iters: 10,
        },
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().await.unwrap(),
        alpha: 0.1,
        epsilon: 1e-4,
    };
    run_sim_quiet(&mut node_osc);

    println!(">>> 2. Divergencia lineal");
    let mut state_div = DVector::zeros(total_size);
    state_div.rows_mut(0, d_r).add_scalar_mut(1.0);

    let mut node_div = AideenNode {
        state: state_div,
        reasoning: UnstableReasoning {
            config: config.clone(),
            mode: 1,
        },
        control: StressControl {
            beta: 1.0,
            max_iters: 10,
        },
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().await.unwrap(),
        alpha: 0.1,
        epsilon: 1e-4,
    };
    run_sim_quiet(&mut node_div);
}

// --- Helpers ---

// Reasoning simulado para pruebas de stress
struct StableMockReasoning {
    config: ArchitectureConfig,
}
impl aideen_core::reasoning::Reasoning for StableMockReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.config
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        let d_r = self.config.d_r;
        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };
        HSlots::from_broadcast(&s_r, &self.config)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn aideen_core::compute::ComputeBackend>,
    ) -> HSlots {
        let h_slots = self.config.h_slots;
        let d_r = self.config.d_r;
        let mut next = HSlots::zeros(&self.config);
        for k in 0..h_slots {
            let slot = h.slot(k) + DVector::from_element(d_r, 0.0000001);
            next.set_slot(k, &slot);
        }
        next
    }
}

async fn build_base_node(
    alpha: f32,
    beta: f32,
    iters: usize,
) -> AideenNode<StableMockReasoning, StressControl, DummyEthics, DummyMemory, WgpuBackend> {
    let config = ArchitectureConfig::default();
    let mut state = State::new(config.clone());
    state.inject_delta_r(&vec![1.0; config.d_r]);

    AideenNode {
        state: state.s,
        reasoning: StableMockReasoning { config },
        control: StressControl {
            beta,
            max_iters: iters,
        },
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().await.unwrap(),
        alpha,
        epsilon: 1e-4,
    }
}

fn run_sim_quiet<R, C, E, M, B>(node: &mut AideenNode<R, C, E, M, B>)
where
    R: aideen_core::reasoning::Reasoning,
    C: aideen_core::control::Control,
    E: aideen_core::ethics::Ethics,
    M: aideen_core::memory::Memory,
    B: ComputeBackend,
{
    // Solo mostramos el primer y último tick relevante para no saturar
    let mut last_m: Option<TickMetrics> = None;
    for tick in 1..=5 {
        if let Some(m) = node.tick() {
            if tick == 1 {
                print_metrics("START", tick, &m);
            }
            last_m = Some(m);
        } else {
            break;
        }
    }
    if let Some(m) = last_m {
        print_metrics("END  ", 5, &m);
    }
}
fn print_metrics(tag: &str, tick: usize, m: &TickMetrics) {
    println!(
        "   {:7} | T{:02} | iters={} | ER={:.2} | ET={:.2} | stop={:?} | Q={:.3} | attractor={} | learn={}",
        tag,
        tick,
        m.iters,
        m.energy_r,
        m.energy_total,
        m.stop_reason,
        m.quality.q_total,
        m.is_attractor,
        m.allow_learning
    );
}
