use aideen_backbone::ffn_reasoning::FfnReasoning;
use aideen_core::protocol::{
    encode_payload_zstd, AckKind, KeyDelegation, NetMsg, QuantizedDelta, SignedUpdate,
};
use aideen_core::state::{ArchitectureConfig, HSlots};
use aideen_node::network::quic_channel::QuicChannel;
use aideen_node::network::NetChannel;
use aideen_node::system::node::{AideenNode, LearningSignal};
use aideen_node::update::UpdateManager;
/// E2E: AIDEEN canonical network pipeline over real QUIC (quinn).
///
/// Same three invariants as net_pipeline_e2e, but the channel is
/// QuicChannel::pair_local() con framing u32 LE + bincode.
///
///   1. HANDSHAKE  — Hello → Delegation → Ack
///   2. DISCOVERY  — tick() allow_learning=true → Discovery enviado
///   3. UPDATE LOOP — SignedUpdate → on_message → Ack
use ed25519_dalek::{Signer, SigningKey};
use nalgebra::DVector;
use rand::rngs::OsRng;

// ── Helpers de llaves ─────────────────────────────────────────────────────

fn generate_keys() -> (SigningKey, [u8; 32]) {
    let sk = SigningKey::generate(&mut OsRng);
    let pk = sk.verifying_key().to_bytes();
    (sk, pk)
}

fn create_delegation(root_sk: &SigningKey, critic_pk: [u8; 32], epoch: u64) -> KeyDelegation {
    let mut d = KeyDelegation {
        epoch,
        critic_pk,
        valid_from_unix: 0,
        valid_to_unix: u64::MAX,
        signature_by_root: vec![],
    };
    d.signature_by_root = root_sk.sign(&d.signing_bytes()).to_bytes().to_vec();
    d
}

fn create_update(
    critic_sk: &SigningKey,
    version: u64,
    target_id: &str,
    base_model_hash: [u8; 32],
) -> SignedUpdate {
    let payload = encode_payload_zstd(&Vec::<QuantizedDelta>::new()).unwrap();
    let mut u = SignedUpdate {
        version,
        target_id: target_id.to_string(),
        bundle_version: 0,
        bundle_hash: [0u8; 32],
        base_model_hash,
        prev_update_hash: [0u8; 32],
        payload,
        signature: vec![],
    };
    u.signature = critic_sk.sign(&u.signing_bytes()).to_bytes().to_vec();
    u
}

// ── Mocks mínimos ─────────────────────────────────────────────────────────

struct MockBackend;
impl aideen_core::compute::ComputeBackend for MockBackend {
    fn load_tensor(&mut self, _d: &[f32]) -> Result<aideen_core::compute::TensorId, String> {
        Ok(aideen_core::compute::TensorId(0))
    }
    fn ffn_forward(
        &mut self,
        _w: &aideen_core::compute::TensorId,
        _i: &aideen_core::compute::TensorId,
        out_dim: usize,
    ) -> Result<DVector<f32>, String> {
        Ok(DVector::zeros(out_dim))
    }
}

struct FixedReasoning(ArchitectureConfig);
impl aideen_core::reasoning::Reasoning for FixedReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.0
    }
    fn init(&self, _s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&DVector::from_element(self.0.d_r, 0.3f32), &self.0)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn aideen_core::compute::ComputeBackend>,
    ) -> HSlots {
        h.clone()
    }
}

struct ImmediateStop;
impl aideen_core::control::Control for ImmediateStop {
    fn max_iters(&self) -> usize {
        10
    }
    fn mode(&self) -> aideen_core::control::ControlMode {
        aideen_core::control::ControlMode::Observe
    }
    fn decide(&self, _i: usize, _dn: f32, _e: f32) -> aideen_core::control::ControlDecision {
        aideen_core::control::ControlDecision {
            stop: true,
            beta: 1.0,
            write_memory: false,
            allow_learning: true,
        }
    }
}

struct PassEthics;
impl aideen_core::ethics::Ethics for PassEthics {
    fn fingerprint(&self) -> [u8; 32] {
        [0; 32]
    }
    fn violates(&self, _s: &DVector<f32>) -> bool {
        false
    }
    fn project(&self, s: &DVector<f32>) -> DVector<f32> {
        s.clone()
    }
}

struct NullMemory;
impl aideen_core::memory::Memory for NullMemory {
    fn write(&mut self, _v: DVector<f32>) {}
    fn query(&self, _q: &DVector<f32>, _k: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}

// ── Test 1: Handshake sobre QUIC ─────────────────────────────────────────

#[test]
fn test_quic_handshake_hello_delegation_ack() {
    let (mut client, mut server) = QuicChannel::pair_local();

    let (root_sk, root_pk) = generate_keys();
    let (_, critic_pk) = generate_keys();
    let delegation = create_delegation(&root_sk, critic_pk, 1);

    // cliente → Hello
    client
        .send(NetMsg::Hello {
            node_id: [1u8; 32],
            protocol: 1,
            bundle_version: 0,
            bundle_hash: [0u8; 32],
        })
        .unwrap();

    // servidor recibe Hello, envía Delegation
    let msg = server.recv().unwrap();
    assert!(matches!(msg, NetMsg::Hello { .. }), "esperaba Hello");
    server.send(NetMsg::Delegation(delegation)).unwrap();

    // cliente recibe Delegation → on_message → Ack
    let msg = client.recv().unwrap();
    let mut manager = UpdateManager::new(root_pk);
    let config = ArchitectureConfig::default();
    let mut expert = FfnReasoning::new(64, config);
    let ack = manager.on_message(&mut expert, msg).unwrap();

    assert!(
        matches!(
            ack,
            NetMsg::Ack {
                kind: AckKind::Delegation,
                version: 1,
                ok: true
            }
        ),
        "Ack esperado kind=Delegation version=1 ok=true, recibido: {:?}",
        ack
    );
    client.send(ack).unwrap();

    // servidor confirma Ack
    let ack_rcv = server.recv().unwrap();
    assert!(matches!(ack_rcv, NetMsg::Ack { ok: true, .. }));
}

// ── Test 2: Discovery sobre QUIC ─────────────────────────────────────────

#[test]
fn test_quic_discovery_emitted_when_allow_learning() {
    let (mut client, mut server) = QuicChannel::pair_local();
    let config = ArchitectureConfig::default();

    let mut node = AideenNode {
        state: DVector::zeros(config.d_global()),
        reasoning: FixedReasoning(config.clone()),
        control: ImmediateStop,
        ethics: PassEthics,
        memory: NullMemory,
        backend: MockBackend,
        alpha: 0.5,
        epsilon: 1e-3,
    };

    let m = node.tick().expect("tick debe producir métricas");
    assert!(m.allow_learning, "Q debe habilitar discovery");

    let signal = LearningSignal {
        allow_learning: m.allow_learning,
        q_total: m.quality.q_total,
        h_star: m.h_star.expect("h_star debe existir en atractor"),
        s_context: DVector::zeros(config.d_r),
    };

    let discovery =
        signal.to_discovery_msg([42u8; 32], "expert_0".to_string(), m.iters as u32, 0, 1);
    client.send(discovery).unwrap();

    let rcv = server.recv().unwrap();
    match rcv {
        NetMsg::Discovery {
            node_id,
            target_id,
            q_total,
            h_star_hash,
            bundle_version,
            ..
        } => {
            assert_eq!(node_id, [42u8; 32]);
            assert_eq!(target_id, "expert_0");
            assert!(q_total >= 0.6, "q_total debe ser >= Q_MIN_LEARN");
            assert_ne!(h_star_hash, [0u8; 32], "h_star_hash no debe ser cero");
            assert_eq!(bundle_version, 1);
        }
        other => panic!("esperaba NetMsg::Discovery, recibido: {:?}", other),
    }
}

// ── Test 3: Update loop sobre QUIC ────────────────────────────────────────

#[test]
fn test_quic_update_loop_server_sends_client_applies_ack() {
    let (mut client, mut server) = QuicChannel::pair_local();

    let (root_sk, root_pk) = generate_keys();
    let (critic_sk, critic_pk) = generate_keys();
    let delegation = create_delegation(&root_sk, critic_pk, 1);

    let mut manager = UpdateManager::new(root_pk);
    let config = ArchitectureConfig::default();
    let mut expert = FfnReasoning::new(64, config);
    manager.apply_delegation(&delegation).unwrap();

    let base_hash = expert.current_model_hash();
    let update = create_update(&critic_sk, 1, "expert_0", base_hash);
    server.send(NetMsg::Update(update.clone())).unwrap();

    let msg = client.recv().unwrap();
    let response = manager.on_message(&mut expert, msg).unwrap();

    assert!(
        matches!(
            response,
            NetMsg::Ack {
                kind: AckKind::Update,
                version: 1,
                ok: true
            }
        ),
        "Ack esperado kind=Update version=1, recibido: {:?}",
        response
    );
    client.send(response).unwrap();

    let ack = server.recv().unwrap();
    assert!(
        matches!(
            ack,
            NetMsg::Ack {
                kind: AckKind::Update,
                ok: true,
                version: 1
            }
        ),
        "servidor esperaba Ack kind=Update ok=true version=1"
    );
}
