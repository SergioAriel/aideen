use loxi_core::control::{Control, ControlDecision};
use loxi_core::ethics::Ethics;
use loxi_core::memory::Memory;
use loxi_core::state::{State, D_R, D_SIM};

use loxi_backbone::ffn_reasoning::FfnReasoning;
use loxi_node::engine::wgpu_backend::WgpuBackend;
use loxi_node::engine::ComputeBackend;
use loxi_node::network::Transport;
use loxi_node::system::node::{LoxiNode, TickMetrics};

use loxi_backbone::tensor::Tensor;
use nalgebra::DVector;
use rand::Rng;

// ─── Implementaciones Temporales (Dummies) ───────────────────────

struct DummyTransport;
impl Transport for DummyTransport {
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
    fn mode(&self) -> loxi_core::control::ControlMode {
        loxi_core::control::ControlMode::Observe
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
    mode: u8, // 0: oscillation, 1: divergence
}
impl loxi_core::reasoning::Reasoning for UnstableReasoning {
    fn init(&self, s: &DVector<f32>) -> DVector<f32> {
        s.clone()
    }
    fn step(&self, h: &DVector<f32>, _s: &DVector<f32>) -> DVector<f32> {
        let mut next = h.clone();
        if self.mode == 0 {
            // Oscilación: voltea el signo del subespacio de razonamiento (0 a 2048)
            for val in next.rows_mut(0, 2048).iter_mut() {
                *val *= -1.1; // Crece y oscila
            }
        } else {
            // Divergencia lineal
            for val in next.rows_mut(0, 2048).iter_mut() {
                *val += 10.0;
            }
        }
        next
    }
}

fn main() {
    println!("🟥 NIVEL 2 — STRESS DINÁMICO LOXI");
    println!("──────────────────────────────────────────────────");

    // A. Sensibilidad a Alpha
    test_alpha_sweep();

    // B. Sensibilidad a Beta
    test_beta_sweep();

    // C. Perturbación Adversaria en S_sim
    test_adversarial_sim();

    // D. No convergencia forzada
    test_unstable_reasoning();
}

fn test_alpha_sweep() {
    println!("\n[TEST A: BARRIDO DE ALPHA]");
    let alphas = [0.01, 0.05, 0.1, 0.3, 0.7, 1.0];

    for alpha in alphas {
        let mut node = build_base_node(alpha, 1.0, 5);
        println!(">>> Ejecutando alpha = {}", alpha);
        run_sim_quiet(&mut node);
    }
}

fn test_beta_sweep() {
    println!("\n[TEST B: BARRIDO DE BETA (CONTROL)]");
    let betas = [0.1, 0.5, 1.0, 2.0, 5.0];

    for beta in betas {
        let mut node = build_base_node(0.1, beta, 5);
        println!(">>> Ejecutando beta = {}", beta);
        run_sim_quiet(&mut node);
    }
}

fn test_adversarial_sim() {
    println!("\n[TEST C: PERTURBACIÓN ADVERSARIA EN S_SIM]");

    // 1. Ruido aleatorio grande
    println!(">>> 1. Ruido grande en S_sim");
    let mut node_large = build_base_node(0.1, 1.0, 5);
    let mut rng = rand::rng();
    let noise: Vec<f32> = (0..D_SIM).map(|_| rng.random_range(-10.0..10.0)).collect();
    node_large.state.as_mut_slice()[2048..].copy_from_slice(&noise);
    run_sim_quiet(&mut node_large);

    // 2. Ruido pequeño persistente (inyectado cada tick)
    println!(">>> 2. Ruido persistente");
    let mut node_persist = build_base_node(0.1, 1.0, 5);
    for tick in 1..=5 {
        let noise: Vec<f32> = (0..D_SIM).map(|_| rng.random_range(-0.1..0.1)).collect();
        node_persist.state.as_mut_slice()[2048..].copy_from_slice(&noise);
        if let Some(m) = node_persist.tick() {
            print_metrics("PERSIST", tick, &m);
        }
    }
}

fn test_unstable_reasoning() {
    println!("\n[TEST D: NO CONVERGENCIA FORZADA]");

    println!(">>> 1. Oscilación divergente");
    let mut state_osc = DVector::zeros(2560);
    state_osc.rows_mut(0, 2048).add_scalar_mut(1.0); // Inyectamos 1.0 en todo el subespacio de razonamiento

    let mut node_osc = LoxiNode {
        state: state_osc,
        reasoning: UnstableReasoning { mode: 0 },
        control: StressControl {
            beta: 1.0,
            max_iters: 10,
        },
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().unwrap(),
        network: DummyTransport,
        alpha: 0.1,
        epsilon: 1e-4,
    };
    run_sim_quiet(&mut node_osc);

    println!(">>> 2. Divergencia lineal");
    let mut state_div = DVector::zeros(2560);
    state_div.rows_mut(0, 2048).add_scalar_mut(1.0);

    let mut node_div = LoxiNode {
        state: state_div,
        reasoning: UnstableReasoning { mode: 1 },
        control: StressControl {
            beta: 1.0,
            max_iters: 10,
        },
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().unwrap(),
        network: DummyTransport,
        alpha: 0.1,
        epsilon: 1e-4,
    };
    run_sim_quiet(&mut node_div);
}

// --- Helpers ---

// Reasoning simulado para pruebas de stress
struct StableMockReasoning;
impl loxi_core::reasoning::Reasoning for StableMockReasoning {
    fn init(&self, s: &DVector<f32>) -> DVector<f32> {
        let d_reasoning = s.len() - D_SIM;
        s.rows(0, d_reasoning).into_owned()
    }
    fn step(&self, h: &DVector<f32>, _s: &DVector<f32>) -> DVector<f32> {
        // Retorna h con un ruido mínimo para simular convergencia rápida y alta calidad
        h.clone() + DVector::from_element(h.len(), 0.00001)
    }
}

fn build_base_node(
    alpha: f32,
    beta: f32,
    iters: usize,
) -> LoxiNode<
    StableMockReasoning,
    StressControl,
    DummyEthics,
    DummyMemory,
    WgpuBackend,
    DummyTransport,
> {
    let mut state = State::new();
    state.inject_delta_r(&vec![1.0; D_R]);

    LoxiNode {
        state: state.s,
        reasoning: StableMockReasoning,
        control: StressControl {
            beta,
            max_iters: iters,
        },
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().unwrap(),
        network: DummyTransport,
        alpha,
        epsilon: 1e-4,
    }
}

fn run_sim_quiet<R, C, E, M, B, N>(node: &mut LoxiNode<R, C, E, M, B, N>)
where
    R: loxi_core::reasoning::Reasoning,
    C: loxi_core::control::Control,
    E: loxi_core::ethics::Ethics,
    M: loxi_core::memory::Memory,
    B: ComputeBackend,
    N: Transport,
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
