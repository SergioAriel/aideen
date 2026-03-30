#![cfg(not(target_arch = "wasm32"))]

use aideen_core::compute::{ComputeBackend, TensorId};
use aideen_core::control::{Control, ControlDecision, ControlMode};
use aideen_core::ethics::Ethics;
use aideen_core::memory::Memory;
use aideen_core::protocol::NetMsg;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use aideen_node::expert::{Aggregator, ExpertClient, ExpertPipeline, ExpertService, UniformRouter};
use aideen_node::network::in_process::InProcessChannel;
use aideen_node::network::NetChannel;
use aideen_node::system::node::AideenNode;
use nalgebra::DVector;

// ── Mock Reasoning ───────────────────────────────────────────────────────────

/// Converges fast: δ_step ≈ 1e-7 * sqrt(dim) — suitable for early-stop tests.
struct StableMockReasoning(ArchitectureConfig);
impl Reasoning for StableMockReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.0
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&s.rows(0, self.0.d_r).into_owned(), &self.0)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn ComputeBackend>,
    ) -> HSlots {
        let mut next = HSlots::zeros(&self.0);
        for k in 0..self.0.h_slots {
            let slot = h.slot(k) + DVector::from_element(self.0.d_r, 1e-7f32);
            next.set_slot(k, &slot);
        }
        next
    }
}

/// Diverges linearly: h += 10.0 per step — suitable for clamp tests.
struct UnstableMockReasoning(ArchitectureConfig);
impl Reasoning for UnstableMockReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.0
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&s.rows(0, self.0.d_r).into_owned(), &self.0)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn ComputeBackend>,
    ) -> HSlots {
        let mut next = HSlots::zeros(&self.0);
        for k in 0..self.0.h_slots {
            let slot = h.slot(k) + DVector::from_element(self.0.d_r, 10.0f32);
            next.set_slot(k, &slot);
        }
        next
    }
}

// ── Mocks para construir AideenNode en Test 3 ─────────────────────────────────

struct TestBackend;
impl ComputeBackend for TestBackend {
    fn load_tensor(&mut self, _data: &[f32]) -> Result<TensorId, String> {
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

/// Reasoning for the querying node. Converges in 1 step.
struct NodeReasoning(ArchitectureConfig);
impl Reasoning for NodeReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.0
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&s.rows(0, self.0.d_r).into_owned(), &self.0)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn ComputeBackend>,
    ) -> HSlots {
        h.clone()
    }
}

struct TestControl;
impl Control for TestControl {
    fn max_iters(&self) -> usize {
        1
    }
    fn mode(&self) -> ControlMode {
        ControlMode::Observe
    }
    fn decide(&self, _iter: usize, _dn: f32, _e: f32) -> ControlDecision {
        ControlDecision {
            stop: true,
            beta: 1.0,
            write_memory: false,
            allow_learning: true,
        }
    }
}

struct TestEthics;
impl Ethics for TestEthics {
    fn fingerprint(&self) -> [u8; 32] {
        [0; 32]
    }
    fn violates(&self, _state: &DVector<f32>) -> bool {
        false
    }
    fn project(&self, s: &DVector<f32>) -> DVector<f32> {
        s.clone()
    }
}

struct TestMemory;
impl Memory for TestMemory {
    fn write(&mut self, _: DVector<f32>) {}
    fn query(&self, _: &DVector<f32>, _: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}

fn build_test_node(
    config: ArchitectureConfig,
) -> AideenNode<NodeReasoning, TestControl, TestEthics, TestMemory, TestBackend> {
    AideenNode {
        state: DVector::zeros(config.d_global()),
        reasoning: NodeReasoning(config),
        control: TestControl,
        ethics: TestEthics,
        memory: TestMemory,
        backend: TestBackend,
        alpha: 0.1,
        epsilon: 1e-4,
    }
}

// ── Test 1: ExpertService delta básico (k_max gate) ──────────────────────────

#[test]
fn test_expert_service_delta() {
    let config = ArchitectureConfig::default();
    let svc = ExpertService {
        reasoning: StableMockReasoning(config.clone()),
        k_max: 3,
        eps_step: 1e-10, // no corta antes
        delta_cap: 100.0,
    };
    let task = NetMsg::ExpertTask {
        task_id: [0u8; 16],
        target_id: "test".into(),
        s_r: vec![1.0f32; config.d_r],
        bundle_version: 1,
        round: 0,
        time_budget_ms: 5000,
    };
    let result = svc.process(&task).unwrap();
    match result {
        NetMsg::ExpertResult {
            delta,
            q_total,
            iters,
            stop,
            ..
        } => {
            assert_eq!(delta.len(), config.h_slots * config.d_r);
            assert!(q_total > 0.0 && q_total <= 1.0);
            assert_eq!(iters, 3); // agotó k_max
            assert_eq!(stop, 1); // k_max gate
        }
        _ => panic!("expected ExpertResult"),
    }
}

// ── Test 2: Aggregator combina con clamp global ──────────────────────────────

#[test]
fn test_aggregator_weighted_combine() {
    let config = ArchitectureConfig::default();
    let r1 = NetMsg::ExpertResult {
        task_id: [0; 16],
        target_id: "t".into(),
        delta: vec![1.0f32; config.d_r],
        q_total: 0.9,
        iters: 1,
        stop: 1,
    };
    let r2 = NetMsg::ExpertResult {
        task_id: [0; 16],
        target_id: "t".into(),
        delta: vec![3.0f32; config.d_r],
        q_total: 0.8,
        iters: 1,
        stop: 1,
    };
    // Sin clamp: promedio ponderado uniforme = 2.0
    let combined = Aggregator::combine(&[0.5, 0.5], &[r1, r2], None).unwrap();
    assert_eq!(combined.len(), config.d_r);
    assert!((combined[0] - 2.0).abs() < 1e-5);

    // Con delta_cap_global: delta grande debe clamparse
    let r3 = NetMsg::ExpertResult {
        task_id: [0; 16],
        target_id: "t".into(),
        delta: vec![100.0f32; config.d_r],
        q_total: 0.5,
        iters: 1,
        stop: 1,
    };
    let clamped = Aggregator::combine(&[1.0], &[r3], Some(1.0)).unwrap();
    let norm: f32 = clamped.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(norm <= 1.0 + 1e-5, "norm={norm} debe ser <= 1.0");
}

// ── Test 3: Pipeline E2E con InProcessChannel ────────────────────────────────

#[test]
fn test_expert_pipeline_inprocess() {
    let config = ArchitectureConfig::default();
    let (client_ch, mut server_ch) = InProcessChannel::pair();

    let config_clone = config.clone();
    std::thread::spawn(move || {
        let svc = ExpertService {
            reasoning: StableMockReasoning(config_clone),
            k_max: 1,
            eps_step: 1e-10,
            delta_cap: 100.0,
        };
        loop {
            let Ok(task) = server_ch.recv() else { break };
            let result = svc.process(&task).unwrap();
            if server_ch.send(result).is_err() {
                break;
            }
        }
    });

    let peer_id: [u8; 32] = [1u8; 32];
    let mut pipeline = ExpertPipeline {
        router: Box::new(UniformRouter { k: 1 }),
        client: ExpertClient::new(vec![(peer_id, Box::new(client_ch) as Box<dyn NetChannel>)]),
        bundle_version: 1,
        delta_cap_global: None,
        outlier_factor: None,
    };

    let h_k = vec![1.0f32; config.d_r];
    let result = pipeline.run(&h_k).unwrap();
    assert_eq!(result.delta.len(), config.h_slots * config.d_r);

    let mut node = build_test_node(config.clone());
    node.inject_delta_r(&result.delta[..config.d_r]);
    assert!(node.tick().is_some());
}

// ── Test 4: early-stop y clamp por delta_cap ────────────────────────────────

#[test]
fn test_expert_service_early_stop_and_cap() {
    let config = ArchitectureConfig::default();
    let svc_stable = ExpertService {
        reasoning: StableMockReasoning(config.clone()),
        k_max: 50,
        eps_step: 1e-4,
        delta_cap: 100.0,
    };
    let task = NetMsg::ExpertTask {
        task_id: [0; 16],
        target_id: "t".into(),
        s_r: vec![0.0f32; config.d_r],
        bundle_version: 1,
        round: 0,
        time_budget_ms: 5000,
    };
    match svc_stable.process(&task).unwrap() {
        NetMsg::ExpertResult { iters, stop, .. } => {
            assert!(iters < 50, "debería cortar antes de k_max; iters={iters}");
            assert_eq!(stop, 0, "stop debe ser eps_step gate (0)");
        }
        _ => panic!("expected ExpertResult"),
    }

    let svc_unstable = ExpertService {
        reasoning: UnstableMockReasoning(config.clone()),
        k_max: 3,
        eps_step: 1e-10,
        delta_cap: 1.0,
    };
    match svc_unstable.process(&task).unwrap() {
        NetMsg::ExpertResult { delta, stop, .. } => {
            let norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                norm <= 1.0 + 1e-5,
                "delta debe estar clampeado; norm={norm}"
            );
            assert_eq!(stop, 2, "stop debe ser delta_cap gate (2)");
        }
        _ => panic!("expected ExpertResult"),
    }
}
