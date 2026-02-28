use loxi_core::control::{Control, ControlDecision};
use loxi_core::ethics::Ethics;
use loxi_core::memory::Memory;
use loxi_core::state::{State, D_R, D_SIM};

use loxi_backbone::ffn_reasoning::FfnReasoning;
use loxi_node::engine::wgpu_backend::WgpuBackend;
use loxi_node::engine::ComputeBackend;
use loxi_node::network::Transport;
use loxi_node::system::node::{LoxiNode, StopReason};

use loxi_backbone::tensor::Tensor;
use nalgebra::DVector;

// ─── Implementaciones Temporales (Dummies) para Visualización ───

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

struct SimpleControl;
impl Control for SimpleControl {
    fn max_iters(&self) -> usize {
        5
    }
    fn mode(&self) -> loxi_core::control::ControlMode {
        loxi_core::control::ControlMode::Observe
    }
    fn decide(&self, iter: usize, _delta_norm: f32, _entropy: f32) -> ControlDecision {
        ControlDecision {
            stop: iter >= 4,
            beta: 1.0,
            write_memory: false,
        }
    }
}

fn main() {
    println!("🚀 Visualización de Trayectorias LOXI (Nivel 1: Dinámica Observable)");
    println!("──────────────────────────────────────────────────");
    println!("V0: Control en modo OBSERVE. S_sim es Volátil.");

    // 1. ESCENARIO A: Razonamiento Matemático (Afecta S_R)
    println!("\n[Escenario A: Inyección en S_R - Integración Persistente]");
    let mut state_a = State::new();
    state_a.inject_delta_r(&vec![1.0; D_R]);

    let mut node_a = LoxiNode {
        state: state_a.s,
        reasoning: FfnReasoning::new(4096),
        control: SimpleControl,
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().unwrap(),
        network: DummyTransport,
        alpha: 0.1,
        epsilon: 1e-4,
    };

    run_sim("RAZONAMIENTO", &mut node_a);

    // 2. ESCENARIO B: Ficción / Simulación (Afecta S_sim)
    println!("\n[Escenario B: Inyección en S_sim - Pura Volatilidad]");
    let mut state_b = State::new();
    state_b.write_sim(&vec![2.0; D_SIM]);

    let mut node_b = LoxiNode {
        state: state_b.s,
        reasoning: FfnReasoning::new(4096),
        control: SimpleControl,
        ethics: DummyEthics,
        memory: DummyMemory,
        backend: WgpuBackend::new().unwrap(),
        network: DummyTransport,
        alpha: 0.1,
        epsilon: 1e-4,
    };

    run_sim("FICCIÓN", &mut node_b);

    println!("\n──────────────────────────────────────────────────");
    println!("✅ Conclusión: S_R converge y persiste. S_sim se disipa (Energy R -> 0).");
}

fn run_sim<R, C, E, M, B, N>(name: &str, node: &mut LoxiNode<R, C, E, M, B, N>)
where
    R: loxi_core::reasoning::Reasoning,
    C: loxi_core::control::Control,
    E: loxi_core::ethics::Ethics,
    M: loxi_core::memory::Memory,
    B: ComputeBackend,
    N: Transport,
{
    for tick in 1..=10 {
        if let Some(m) = node.tick() {
            println!(
                "{} | Tick {:02} | iters={} | ER={:.2} | ESIM={:.2} | ET={:.2} | stop={:?} | conv={}",
                name, tick, m.iters, m.energy_r, m.energy_sim, m.energy_total, m.stop_reason, m.converged
            );

            if m.stop_reason == StopReason::Epsilon {
                println!("{} | Tick {:02} | [Atractor Alcanzado]", name, tick + 1);
                break;
            }
        } else {
            println!(
                "{} | Tick {:02} | [Inhibido / Sin Convergencia]",
                name, tick
            );
            break;
        }
    }
}
