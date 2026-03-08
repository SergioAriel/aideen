#![cfg(not(target_arch = "wasm32"))]

use aideen_node::peers::{NodeId, PeerDelta, PeerEntry, PeerRegistry};

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_peer(id: u8, domains: &[&str]) -> PeerEntry {
    PeerEntry {
        node_id: [id; 32],
        endpoint: format!("127.0.0.1:{}", 9000 + id as u16),
        domains: domains.iter().map(|d| d.to_string()).collect(),
        bundle_version: 1,
        tls_fingerprint: None,
    }
}

// ── Test 1: set_snapshot indexa correctamente ─────────────────────────────────

#[test]
fn test_registry_snapshot_indexed() {
    let mut reg = PeerRegistry::new();
    let peer_a: NodeId = [1u8; 32];
    let peer_b: NodeId = [2u8; 32];

    reg.set_snapshot(
        1,
        vec![make_peer(1, &["math", "physics"]), make_peer(2, &["math"])],
    );

    assert_eq!(reg.len(), 2);
    assert!(reg.get(&peer_a).is_some());
    assert!(reg.get(&peer_b).is_some());

    let ids = reg.node_ids_for_domain("math");
    assert_eq!(ids.len(), 2);
    assert!(ids.contains(&peer_a));
    assert!(ids.contains(&peer_b));

    let physics_ids = reg.node_ids_for_domain("physics");
    assert_eq!(physics_ids.len(), 1);
    assert!(physics_ids.contains(&peer_a));
}

// ── Test 2: apply_delta upsert + remove ──────────────────────────────────────

#[test]
fn test_registry_delta_upsert_remove() {
    let mut reg = PeerRegistry::new();
    reg.set_snapshot(1, vec![make_peer(1, &["math"]), make_peer(2, &["math"])]);

    // Upsert peer 3, remove peer 2
    let delta = PeerDelta {
        epoch: 2,
        upserts: vec![make_peer(3, &["chemistry"])],
        removes: vec![[2u8; 32]],
    };
    reg.apply_delta(delta).expect("delta debe aplicarse");

    assert_eq!(reg.len(), 2, "peer_2 eliminado, peer_3 agregado");
    assert!(reg.get(&[3u8; 32]).is_some());
    assert!(reg.get(&[2u8; 32]).is_none());

    // math solo debe tener peer 1 ahora
    let math_ids = reg.node_ids_for_domain("math");
    assert_eq!(math_ids, vec![[1u8; 32]]);

    // chemistry debe tener peer 3
    let chem_ids = reg.node_ids_for_domain("chemistry");
    assert_eq!(chem_ids, vec![[3u8; 32]]);
}

// ── Test 3: apply_delta stale rechazado ──────────────────────────────────────

#[test]
fn test_registry_delta_stale_rejected() {
    let mut reg = PeerRegistry::new();
    reg.set_snapshot(5, vec![make_peer(1, &["math"])]);

    // epoch igual → rechazado
    let err = reg.apply_delta(PeerDelta {
        epoch: 5,
        upserts: vec![],
        removes: vec![],
    });
    assert!(err.is_err(), "epoch igual debe rechazarse");

    // epoch menor → rechazado
    let err = reg.apply_delta(PeerDelta {
        epoch: 3,
        upserts: vec![],
        removes: vec![],
    });
    assert!(err.is_err(), "epoch anterior debe rechazarse");

    // estado no alterado
    assert_eq!(reg.epoch(), 5);
    assert_eq!(reg.len(), 1);
}

// ── Test 4: domain index consistente en upsert/remove ────────────────────────

#[test]
fn test_registry_domain_index_consistent() {
    let mut reg = PeerRegistry::new();
    reg.set_snapshot(1, vec![make_peer(1, &["math"])]);

    // Upsert peer 1 con dominios cambiados: de "math" a "physics"
    let delta = PeerDelta {
        epoch: 2,
        upserts: vec![PeerEntry {
            node_id: [1u8; 32],
            endpoint: "127.0.0.1:9001".into(),
            domains: vec!["physics".into()],
            bundle_version: 2,
            tls_fingerprint: None,
        }],
        removes: vec![],
    };
    reg.apply_delta(delta).unwrap();

    // "math" ya no debe tener ningún peer
    assert!(
        reg.node_ids_for_domain("math").is_empty(),
        "math debe estar vacío tras upsert sin ese dominio"
    );

    // "physics" debe tener peer 1
    assert_eq!(reg.node_ids_for_domain("physics"), vec![[1u8; 32]]);
}

// ── Test 5: NodeRunner API de peers ──────────────────────────────────────────

mod runner_peer_api {
    use aideen_node::peers::{PeerDelta, PeerEntry};

    use aideen_core::state::{ArchitectureConfig, HSlots};
    use aideen_core::{
        agent::NullAgentStore,
        compute::{ComputeBackend, TensorId},
        control::{Control, ControlDecision, ControlMode},
        doc_memory::NullDocMemory,
        ethics::Ethics,
        memory::Memory,
        reasoning::Reasoning,
    };
    use aideen_node::runner::NodeRunner;
    use aideen_node::system::node::AideenNode;
    use nalgebra::DVector;

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

    #[test]
    fn test_runner_peer_api() {
        let mut runner = build_runner();

        runner.set_peer_snapshot(
            1,
            vec![PeerEntry {
                node_id: [10u8; 32],
                endpoint: "127.0.0.1:9010".into(),
                domains: vec!["math".into()],
                bundle_version: 1,
                tls_fingerprint: None,
            }],
        );
        assert_eq!(runner.peer_registry.len(), 1);

        let ids = runner.peer_ids_for_domain("math");
        assert_eq!(ids, vec![[10u8; 32]]);

        // apply_delta: añadir peer 11 en "science"
        runner
            .apply_peer_delta(PeerDelta {
                epoch: 2,
                upserts: vec![PeerEntry {
                    node_id: [11u8; 32],
                    endpoint: "127.0.0.1:9011".into(),
                    domains: vec!["science".into()],
                    bundle_version: 1,
                    tls_fingerprint: None,
                }],
                removes: vec![],
            })
            .unwrap();

        assert_eq!(runner.peer_registry.len(), 2);
        assert_eq!(runner.peer_ids_for_domain("science"), vec![[11u8; 32]]);
    }
}

// ── Test 6: registry → sorted ids → ExpertClient (wiring sin dial real) ──────

#[test]
fn test_pipeline_with_registry_wiring() {
    use aideen_core::compute::{ComputeBackend, TensorId};
    use aideen_core::reasoning::Reasoning;
    use aideen_core::state::{ArchitectureConfig, HSlots};
    use aideen_node::expert::{ExpertClient, ExpertPipeline, ExpertService, UniformRouter};
    use aideen_node::network::in_process::InProcessChannel;
    use aideen_node::network::NetChannel;
    use nalgebra::DVector;

    let config = ArchitectureConfig::default();

    // Mock reasoning: identity (delta = 0)
    struct IdentityReasoning(ArchitectureConfig);
    impl Reasoning for IdentityReasoning {
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

    // Construcción del registry con 1 peer en "math"
    let mut reg = PeerRegistry::new();
    let peer_id: NodeId = [42u8; 32];
    reg.set_snapshot(
        1,
        vec![PeerEntry {
            node_id: peer_id,
            endpoint: "127.0.0.1:9042".into(),
            domains: vec!["math".into()],
            bundle_version: 1,
            tls_fingerprint: None,
        }],
    );

    // Obtener los ids del dominio "math"
    let domain_ids = reg.node_ids_for_domain("math");
    assert_eq!(domain_ids, vec![peer_id]);

    let config_clone = config.clone();
    // Servidor expert en thread
    let (client_ch, mut server_ch) = InProcessChannel::pair();
    std::thread::spawn(move || {
        let svc = ExpertService {
            reasoning: IdentityReasoning(config_clone),
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

    // Construir ExpertClient desde los domain_ids del registry
    let peers: Vec<(NodeId, Box<dyn NetChannel>)> = domain_ids
        .into_iter()
        .zip(std::iter::once(Box::new(client_ch) as Box<dyn NetChannel>))
        .collect();

    let mut pipeline = ExpertPipeline {
        router: Box::new(UniformRouter { k: 1 }),
        client: ExpertClient::new(peers),
        bundle_version: 1,
        delta_cap_global: None,
        outlier_factor: None,
    };

    let h_k = vec![0.5f32; config.d_r];
    let result = pipeline.run(&h_k).unwrap();
    assert_eq!(
        result.delta.len(),
        config.h_slots * config.d_r,
        "delta debe tener dim H_SLOTS*D_R"
    );
}
