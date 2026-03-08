#![cfg(not(target_arch = "wasm32"))]

use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::DVector;

use aideen_core::agent::NullAgentStore;
use aideen_core::compute::{ComputeBackend, TensorId};
use aideen_core::control::{Control, ControlDecision, ControlMode};
use aideen_core::doc_memory::NullDocMemory;
use aideen_core::ethics::Ethics;
use aideen_core::memory::Memory;
use aideen_core::protocol::NetMsg;
use aideen_core::reasoning::Reasoning;

use aideen_node::runner::NodeRunner;
use aideen_node::system::node::AideenNode;

// ── Mocks (same pattern as reconcile_tests) ───────────────────────────────────

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
        [0xABu8; 32], // node_id — non-zero to verify it influences trace_digest
        1,
        "test".into(),
    )
}

// ── test_node_replay_response_deterministic_digest ────────────────────────────

#[test]
fn test_node_replay_response_deterministic_digest() {
    let runner = build_runner();

    let req = NetMsg::ReplayRequest {
        sample_id: 42,
        context_hash: [0xCDu8; 32],
        h_star_hash: [0xEFu8; 32],
        seed: 99999,
        iters: 10,
    };

    // Two identical calls must produce byte-identical responses
    let resp1 = runner.handle_replay_request(&req);
    let resp2 = runner.handle_replay_request(&req);

    match (&resp1, &resp2) {
        (
            NetMsg::ReplayResponse {
                sample_id: s1,
                trace_digest: d1,
                reproduced: r1,
                q_recomputed: q1,
            },
            NetMsg::ReplayResponse {
                sample_id: s2,
                trace_digest: d2,
                reproduced: r2,
                q_recomputed: q2,
            },
        ) => {
            assert_eq!(s1, s2, "sample_id must match");
            assert_eq!(d1, d2, "trace_digest must be deterministic");
            assert_eq!(q1, q2, "q_recomputed must be deterministic");
            assert!(*r1 && *r2, "reproduced must be true");
            assert_eq!(*s1, 42u64, "sample_id must equal request");
        }
        _ => panic!("expected ReplayResponse, got {:?} / {:?}", resp1, resp2),
    }

    // A different node_id must produce a different digest
    let runner2 = {
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
            [0x00u8; 32], // different node_id
            1,
            "test".into(),
        )
    };

    let resp3 = runner2.handle_replay_request(&req);
    if let (
        NetMsg::ReplayResponse {
            trace_digest: d1, ..
        },
        NetMsg::ReplayResponse {
            trace_digest: d3, ..
        },
    ) = (&resp1, &resp3)
    {
        assert_ne!(
            d1, d3,
            "different node_ids must produce different trace_digests"
        );
    } else {
        panic!("expected ReplayResponse from runner2, got {:?}", resp3);
    }
}
