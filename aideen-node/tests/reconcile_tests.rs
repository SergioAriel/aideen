#![cfg(not(target_arch = "wasm32"))]

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::DVector;

use aideen_core::agent::NullAgentStore;
use aideen_core::compute::{ComputeBackend, TensorId};
use aideen_core::control::{Control, ControlDecision, ControlMode};
use aideen_core::doc_memory::NullDocMemory;
use aideen_core::ethics::Ethics;
use aideen_core::memory::Memory;
use aideen_core::reasoning::Reasoning;

use aideen_node::expert::ExpertClient;
use aideen_node::network::channel_factory::{ChannelFactory, DialResult};
use aideen_node::network::in_process::InProcessChannel;
use aideen_node::peers::types::{NodeId, PeerEntry};
use aideen_node::runner::NodeRunner;
use aideen_node::system::node::AideenNode;

// ── Mocks ─────────────────────────────────────────────────────────────────────

struct MockB;
impl ComputeBackend for MockB {
    fn load_tensor(&mut self, _: &[f32]) -> Result<TensorId, String> {
        Ok(TensorId(0))
    }
    fn ffn_forward(
        &mut self,
        _: &TensorId,
        _: &TensorId,
        d: usize,
    ) -> Result<DVector<f32>, String> {
        Ok(DVector::zeros(d))
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

/// MockChannelFactory que cuenta dials y retorna InProcessChannel cliente.
struct MockChannelFactory {
    dial_calls: Arc<AtomicUsize>,
}
impl ChannelFactory for MockChannelFactory {
    fn dial(&self, _peer: &PeerEntry) -> Result<DialResult, String> {
        self.dial_calls.fetch_add(1, Ordering::SeqCst);
        let (client_ch, _server_ch) = InProcessChannel::pair();
        Ok(DialResult {
            channel: Box::new(client_ch),
            fingerprint: [0u8; 32],
        })
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

fn make_peer(id: NodeId) -> PeerEntry {
    PeerEntry {
        node_id: id,
        endpoint: "127.0.0.1:9999".into(),
        domains: vec!["ai".into()],
        bundle_version: 1,
        tls_fingerprint: None,
    }
}

// ── Test 1: no redial if peer is already in ExpertClient ──────────────────────

#[test]
fn test_reconcile_does_not_redial_existing_peer() {
    let dial_calls = Arc::new(AtomicUsize::new(0));
    let mut runner = build_runner();
    runner.channel_factory = Arc::new(MockChannelFactory {
        dial_calls: dial_calls.clone(),
    });
    runner.set_peer_snapshot(1, vec![make_peer([1u8; 32])]);

    let mut client = ExpertClient::new(vec![]);

    // First reconcile → must dial
    runner.reconcile_expert_client("ai", &mut client);
    assert_eq!(
        dial_calls.load(Ordering::SeqCst),
        1,
        "first reconcile must dial"
    );
    assert_eq!(runner.last_reconcile_stats.dial_attempts, 1);
    assert_eq!(runner.last_reconcile_stats.dial_success, 1);

    // Second reconcile → peer already in client → no extra dial
    runner.reconcile_expert_client("ai", &mut client);
    assert_eq!(
        dial_calls.load(Ordering::SeqCst),
        1,
        "reconcile with active peer must not redial"
    );
    assert_eq!(
        runner.last_reconcile_stats.dial_attempts, 0,
        "second reconcile: 0 attempts"
    );
}

// ── Test 2: reconcile respeta circuit breaker open ────────────────────────────

#[test]
fn test_reconcile_skips_when_breaker_open() {
    let dial_calls = Arc::new(AtomicUsize::new(0));
    let peer_id: NodeId = [2u8; 32];

    let mut runner = build_runner();
    runner.channel_factory = Arc::new(MockChannelFactory {
        dial_calls: dial_calls.clone(),
    });
    runner.set_peer_snapshot(1, vec![make_peer(peer_id)]);

    // Pre-open breaker with long TTL
    runner.peer_failures.entry(peer_id).or_default().open_until =
        Some(Instant::now() + Duration::from_secs(60));

    let mut client = ExpertClient::new(vec![]);
    runner.reconcile_expert_client("ai", &mut client);

    assert_eq!(
        dial_calls.load(Ordering::SeqCst),
        0,
        "breaker open → 0 dials"
    );
    assert_eq!(
        runner.last_reconcile_stats.breaker_skips, 1,
        "must count 1 breaker skip"
    );
    assert_eq!(runner.last_reconcile_stats.dial_attempts, 0);
}
