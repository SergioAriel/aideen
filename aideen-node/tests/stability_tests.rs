#![cfg(not(target_arch = "wasm32"))]

use aideen_core::agent::NullAgentStore;
use aideen_core::compute::{ComputeBackend, TensorId};
use aideen_core::control::{Control, ControlDecision, ControlMode};
use aideen_core::doc_memory::NullDocMemory;
use aideen_core::ethics::Ethics;
use aideen_core::memory::Memory;
use aideen_core::protocol::NetMsg;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use aideen_node::expert::{ExpertClient, ExpertPipeline, RunResult, UniformRouter};
use aideen_node::network::in_process::InProcessChannel;
use aideen_node::network::NetChannel;
use aideen_node::runner::{NodeRunner, RouterStatsAccumulator};
use aideen_node::system::node::AideenNode;
use nalgebra::DVector;

// ── Mocks comunes ─────────────────────────────────────────────────────────────

struct MockB;
impl ComputeBackend for MockB {
    fn load_tensor(&mut self, _: &[f32]) -> Result<TensorId, String> {
        Ok(TensorId(0))
    }
    fn ffn_forward(
        &mut self,
        _: &TensorId,
        _: &TensorId,
        out_dim: usize,
    ) -> Result<DVector<f32>, String> {
        Ok(DVector::zeros(out_dim))
    }
}
struct MockR(ArchitectureConfig);
impl Reasoning for MockR {
    fn config(&self) -> &ArchitectureConfig {
        &self.0
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&s.rows(0, self.0.d_r).into_owned(), &self.0)
    }
    fn step(&self, h: &HSlots, _: &DVector<f32>, _: Option<&mut dyn ComputeBackend>) -> HSlots {
        h.clone()
    }
}
struct MockC;
impl Control for MockC {
    fn max_iters(&self) -> usize {
        1
    }
    fn mode(&self) -> ControlMode {
        ControlMode::Observe
    }
    fn decide(&self, _: usize, _: f32, _: f32) -> ControlDecision {
        ControlDecision {
            stop: true,
            beta: 1.0,
            write_memory: false,
            allow_learning: false,
        }
    }
}
struct MockE;
impl Ethics for MockE {
    fn fingerprint(&self) -> [u8; 32] {
        [0; 32]
    }
    fn violates(&self, _: &DVector<f32>) -> bool {
        false
    }
    fn project(&self, s: &DVector<f32>) -> DVector<f32> {
        s.clone()
    }
}
struct MockM;
impl Memory for MockM {
    fn write(&mut self, _: DVector<f32>) {}
    fn query(&self, _: &DVector<f32>, _: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}

fn build_runner() -> NodeRunner<MockR, MockC, MockE, MockM, MockB> {
    let config = ArchitectureConfig::default();
    let node = AideenNode {
        state: DVector::zeros(config.d_global()),
        reasoning: MockR(config),
        control: MockC,
        ethics: MockE,
        memory: MockM,
        backend: MockB,
        alpha: 0.1,
        epsilon: 1e-4,
    };
    NodeRunner::new(
        node,
        Box::new(NullAgentStore),
        Box::new(NullDocMemory),
        [0u8; 32],
        1,
        "test".into(),
    )
}

/// Servidor que responde directamente con ExpertResult de delta fijo (bypasa ExpertService).
/// Esto garantiza deltas controlados independientemente del Reasoning.
fn spawn_fixed_delta_server(delta_fill: f32, config: ArchitectureConfig) -> Box<dyn NetChannel> {
    let (client_ch, mut server_ch) = InProcessChannel::pair();
    std::thread::spawn(move || loop {
        let task = match server_ch.recv() {
            Ok(t) => t,
            Err(_) => break,
        };
        let (task_id, target_id) = match task {
            NetMsg::ExpertTask {
                task_id, target_id, ..
            } => (task_id, target_id),
            _ => break,
        };
        let result = NetMsg::ExpertResult {
            task_id,
            target_id,
            delta: vec![delta_fill; config.d_r],
            q_total: 0.9,
            iters: 1,
            stop: 1,
        };
        if server_ch.send(result).is_err() {
            break;
        }
    });
    Box::new(client_ch)
}

// ── Test 1: outlier hard drop — 3 peers, peer_c outlier ──────────────────────
//
// Con k=3: sorted=[32, 32, 3200], mediana-low = sorted[1] = 32
// factor * mediana = 2.0 * 32 = 64. peer_c (norm=3200) > 64 → dropeado.

#[test]
fn test_outlier_hard_drop_two_peers() {
    let config = ArchitectureConfig::default();
    let peer_a: [u8; 32] = [1u8; 32];
    let peer_b: [u8; 32] = [2u8; 32];
    let peer_c: [u8; 32] = [3u8; 32];

    let client = ExpertClient::new(vec![
        (peer_a, spawn_fixed_delta_server(1.0, config.clone())), // norma ≈ 32 (si d_r=1024?) No, config.d_r default=512.
        (peer_b, spawn_fixed_delta_server(1.0, config.clone())), // norma ≈ sqrt(512*1^2) ≈ 22.6
        (peer_c, spawn_fixed_delta_server(100.0, config.clone())), // norma ≈ sqrt(512*100^2) ≈ 2262
    ]);

    let mut pipeline = ExpertPipeline {
        router: Box::new(UniformRouter { k: 3 }),
        client,
        bundle_version: 1,
        delta_cap_global: None,
        outlier_factor: Some(2.0),
    };

    let h_k = vec![0.0f32; config.d_r];
    let result = pipeline.run(&h_k).unwrap();

    assert_eq!(
        result.drops_count, 1,
        "peer_c debe ser dropeado como outlier"
    );
    // Con peer_c excluido, delta ≈ [1.0; 512], norma ≈ 22.6
    let norm: f32 = result.delta.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        norm < 200.0,
        "norma del delta combinado debe ser << 2262; got {norm}"
    );
}

// ── Test 2: outlier desactivado cuando factor=None ────────────────────────────

#[test]
fn test_outlier_disabled_when_factor_none() {
    let config = ArchitectureConfig::default();
    let peer_a: [u8; 32] = [1u8; 32];
    let peer_b: [u8; 32] = [2u8; 32];
    let peer_c: [u8; 32] = [3u8; 32];

    let mut pipeline = ExpertPipeline {
        router: Box::new(UniformRouter { k: 3 }),
        client: ExpertClient::new(vec![
            (peer_a, spawn_fixed_delta_server(1.0, config.clone())),
            (peer_b, spawn_fixed_delta_server(1.0, config.clone())),
            (peer_c, spawn_fixed_delta_server(100.0, config.clone())),
        ]),
        bundle_version: 1,
        delta_cap_global: None,
        outlier_factor: None,
    };

    let h_k = vec![0.0f32; config.d_r];
    let result = pipeline.run(&h_k).unwrap();

    assert_eq!(
        result.drops_count, 0,
        "sin outlier_factor, drops_count debe ser 0"
    );
}

// ── Test 3: k=1 → outlier check imposible → drops_count == 0 ─────────────────

#[test]
fn test_single_peer_no_outlier_check() {
    let config = ArchitectureConfig::default();
    let peer_a: [u8; 32] = [1u8; 32];

    let mut pipeline = ExpertPipeline {
        router: Box::new(UniformRouter { k: 1 }),
        client: ExpertClient::new(vec![(
            peer_a,
            spawn_fixed_delta_server(100.0, config.clone()),
        )]),
        bundle_version: 1,
        delta_cap_global: None,
        outlier_factor: Some(2.0),
    };

    let h_k = vec![0.0f32; config.d_r];
    let result = pipeline.run(&h_k).unwrap();

    assert_eq!(
        result.drops_count, 0,
        "k=1: sin comparación posible → drops_count == 0"
    );
}

// ── Test 4: β se reduce con delta_norm grande ─────────────────────────────────

#[test]
fn test_beta_reduces_with_large_delta() {
    let mut runner = build_runner();
    let config = ArchitectureConfig::default();

    // delta_norm=50, q_mean=1.0 → β_raw = 1.0 / 51 ≈ 0.0196 → clampeado a beta_min
    let big_norm = RunResult {
        delta: vec![0.0f32; config.d_r],
        q_mean: 1.0,
        delta_norm: 50.0,
        drops_count: 0,
    };
    let beta_big = runner.apply_expert_result(&big_norm);

    // delta_norm=0, q_mean=1.0 → β_raw = 1.0 / 1.0 = 1.0 → beta_max
    let small_norm = RunResult {
        delta: vec![0.0f32; config.d_r],
        q_mean: 1.0,
        delta_norm: 0.0,
        drops_count: 0,
    };
    let beta_small = runner.apply_expert_result(&small_norm);

    assert!(
        beta_big < beta_small,
        "β(norm=50) debe ser < β(norm=0); got big={beta_big}, small={beta_small}"
    );
    assert!(
        (beta_small - 1.0).abs() < 1e-5,
        "β(norm=0, q=1) debe == beta_max=1.0; got {beta_small}"
    );
}

// ── Test 5: β se reduce con q_mean baja ──────────────────────────────────────

#[test]
fn test_beta_reduces_with_low_quality() {
    let mut runner = build_runner();
    let config = ArchitectureConfig::default();

    // q_mean=0 → β_raw=0 → clampeado a beta_min=0.05
    let zero_quality = RunResult {
        delta: vec![0.0f32; config.d_r],
        q_mean: 0.0,
        delta_norm: 0.0,
        drops_count: 0,
    };
    let beta_zero_q = runner.apply_expert_result(&zero_quality);
    assert!(
        (beta_zero_q - runner.beta_min).abs() < 1e-5,
        "β(q=0) debe == beta_min={}; got {beta_zero_q}",
        runner.beta_min
    );

    // q_mean=1.0, delta_norm=0 → β_raw=1.0 → beta_max=1.0
    let full_quality = RunResult {
        delta: vec![0.0f32; config.d_r],
        q_mean: 1.0,
        delta_norm: 0.0,
        drops_count: 0,
    };
    let beta_full_q = runner.apply_expert_result(&full_quality);
    assert!(
        (beta_full_q - runner.beta0).abs() < 1e-5,
        "β(q=1, norm=0) debe == beta0={}; got {beta_full_q}",
        runner.beta0
    );
}

// ── Test 6: RouterStatsAccumulator flush con campos Stability Pack ────────────

#[test]
fn test_stats_acc_expert_flush() {
    let mut acc = RouterStatsAccumulator::new(2);
    // Necesitamos 2 ticks para flush_every=2
    acc.record(0.8, Some("expert_a"));
    acc.record(0.9, Some("expert_a"));
    // Registrar 2 consultas expert con (delta_norm, drops, beta)
    acc.record_expert(5.0, 1, 0.8);
    acc.record_expert(3.0, 0, 0.9);

    let msg = acc
        .flush([0u8; 32])
        .expect("debe producir RouterStats tras 2 ticks");

    match msg {
        NetMsg::RouterStats {
            delta_norm_mean,
            delta_norm_min,
            delta_norm_max,
            drops_count,
            beta_mean,
            ..
        } => {
            assert!(
                (delta_norm_mean - 4.0).abs() < 1e-4,
                "delta_norm_mean debe ≈ 4.0; got {delta_norm_mean}"
            );
            assert!(
                (delta_norm_min - 3.0).abs() < 1e-4,
                "delta_norm_min debe ≈ 3.0; got {delta_norm_min}"
            );
            assert!(
                (delta_norm_max - 5.0).abs() < 1e-4,
                "delta_norm_max debe ≈ 5.0; got {delta_norm_max}"
            );
            assert_eq!(drops_count, 1, "drops_count debe == 1");
            assert!(
                (beta_mean - 0.85).abs() < 1e-4,
                "beta_mean debe ≈ 0.85; got {beta_mean}"
            );
        }
        _ => panic!("expected RouterStats"),
    }
}
