#![cfg(not(target_arch = "wasm32"))]

use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::DVector;

use aideen_core::{
    agent::{AgentEvent, AgentStore, InMemoryAgentStore},
    compute::ComputeBackend,
    control::{Control, ControlDecision, ControlMode},
    doc_memory::{DocMemory, DocMeta, NullDocMemory},
    ethics::Ethics,
    memory::Memory,
    protocol::NetMsg,
    reasoning::Reasoning,
};
use aideen_node::{
    runner::{build_context_features, NodeRunner, RuntimeContext, TickOutcome},
    system::node::AideenNode,
};

// ── Mocks mínimos ──────────────────────────────────────────────────────────────

struct MockBackend;
impl ComputeBackend for MockBackend {
    fn load_tensor(&mut self, _data: &[f32]) -> Result<aideen_core::compute::TensorId, String> {
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

struct MockReasoning {
    config: ArchitectureConfig,
}
impl Reasoning for MockReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.config
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&s.rows(0, self.config.d_r).into_owned(), &self.config)
    }
    fn step(
        &self,
        _h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn ComputeBackend>,
    ) -> HSlots {
        HSlots::zeros(&self.config) // zeros → delta_norm ≈ 0 → converge inmediato
    }
}

/// State-sensitive Reasoning: blends slots with the S_sim region (where
/// `set_context()` escribe los features documentales). Converge en 1 paso.
struct ContextSensitiveReasoning {
    config: ArchitectureConfig,
}
impl Reasoning for ContextSensitiveReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.config
    }
    fn init(&self, s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&s.rows(0, self.config.d_r).into_owned(), &self.config)
    }
    fn step(&self, h: &HSlots, s: &DVector<f32>, _exec: Option<&mut dyn ComputeBackend>) -> HSlots {
        // S_sim es donde set_context() escribe los features de contexto
        let s_sim = s
            .rows(self.config.d_reasoning(), self.config.d_sim)
            .into_owned();
        // Pad a D_R con ceros para hacer el blend
        let mut sim_padded = DVector::<f32>::zeros(self.config.d_r);
        for (i, v) in s_sim.iter().enumerate() {
            sim_padded[i] = *v;
        }
        let mut next = HSlots::zeros(&self.config);
        for k in 0..self.config.h_slots {
            // Converges immediately in 1 iteration to avoid depending on H_SLOTS or epsilon
            next.set_slot(k, &sim_padded);
        }
        next
    }
}

struct MockControl;
impl Control for MockControl {
    fn max_iters(&self) -> usize {
        3
    }
    fn mode(&self) -> ControlMode {
        ControlMode::Observe
    }
    fn decide(&self, iter: usize, _delta_norm: f32, _entropy: f32) -> ControlDecision {
        ControlDecision {
            stop: iter >= 2,
            beta: 1.0,
            write_memory: false,
            allow_learning: true,
        }
    }
}

struct MockEthics;
impl Ethics for MockEthics {
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

struct MockMemory;
impl Memory for MockMemory {
    fn write(&mut self, _invariant: DVector<f32>) {}
    fn query(&self, _q: &DVector<f32>, _k: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}

// ── Constructores de runner ────────────────────────────────────────────────────

fn build_runner(
    agent_store: Box<dyn AgentStore + Send>,
    doc_memory: Box<dyn DocMemory + Send>,
) -> NodeRunner<MockReasoning, MockControl, MockEthics, MockMemory, MockBackend> {
    let node = AideenNode {
        state: DVector::zeros(2560),
        reasoning: MockReasoning {
            config: ArchitectureConfig::default(),
        },
        control: MockControl,
        ethics: MockEthics,
        memory: MockMemory,
        backend: MockBackend,
        alpha: 0.1,
        epsilon: 1e-4,
    };
    NodeRunner::new(
        node,
        agent_store,
        doc_memory,
        [0u8; 32],
        0,
        "test".to_string(),
    )
}

fn build_ctx_runner(
    agent_store: Box<dyn AgentStore + Send>,
    doc_memory: Box<dyn DocMemory + Send>,
) -> NodeRunner<ContextSensitiveReasoning, MockControl, MockEthics, MockMemory, MockBackend> {
    build_ctx_runner_with_id(agent_store, doc_memory, [0u8; 32], 0, "test".to_string())
}

fn build_ctx_runner_with_id(
    agent_store: Box<dyn AgentStore + Send>,
    doc_memory: Box<dyn DocMemory + Send>,
    node_id: [u8; 32],
    bundle_version: u64,
    target_id: String,
) -> NodeRunner<ContextSensitiveReasoning, MockControl, MockEthics, MockMemory, MockBackend> {
    let node = AideenNode {
        state: DVector::zeros(2560),
        reasoning: ContextSensitiveReasoning {
            config: ArchitectureConfig::default(),
        },
        control: MockControl,
        ethics: MockEthics,
        memory: MockMemory,
        backend: MockBackend,
        alpha: 0.1,
        epsilon: 1e-4,
    };
    NodeRunner::new(
        node,
        agent_store,
        doc_memory,
        node_id,
        bundle_version,
        target_id,
    )
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn tmp_dir() -> std::path::PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("aideen_runner_test_{}", n))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn meta(title: &str) -> DocMeta {
    DocMeta {
        title: title.into(),
        locator: format!("file:{}.txt", title),
        mime: "text/plain".into(),
        len_bytes: 0,
        added_unix: unix_now(),
    }
}

// ── Test 1 ─────────────────────────────────────────────────────────────────────

/// tick() emite TickAttractor si el nodo alcanza un atractor de calidad.
#[test]
fn tick_produces_attractor_event() {
    let store = Box::new(InMemoryAgentStore::new());
    let doc_mem = Box::new(NullDocMemory);
    let mut runner = build_runner(store, doc_mem);

    let metrics = runner.tick();
    assert!(metrics.is_some(), "tick debe devolver métricas");

    let m = metrics.unwrap();
    if m.is_attractor {
        let events = runner.recent_events(5);
        assert!(!events.is_empty(), "debe haber al menos un evento");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AgentEvent::TickAttractor { .. })),
            "el evento debe ser TickAttractor"
        );
    }
}

// ── Test 2 ─────────────────────────────────────────────────────────────────────

/// add_document/search/locate + restart: los documentos persisten.
#[test]
fn add_doc_search_locate_survive_restart() {
    use aideen_node::{agent::FsAgentStore, doc_memory::FsDocMemory};

    let base = tmp_dir();
    let agent = "runner_node";
    let content = b"aideen motor cognitivo distribuido aideen".to_vec();

    let doc_id = {
        let store = Box::new(FsAgentStore::open(base.to_str().unwrap(), agent).unwrap());
        let doc_mem = Box::new(FsDocMemory::open(base.to_str().unwrap(), agent).unwrap());
        let mut runner = build_runner(store, doc_mem);

        let doc_id = runner.add_document(meta("Motor"), content.clone()).unwrap();

        let hits = runner.search_docs("aideen", 5);
        assert!(!hits.is_empty(), "search debe encontrar el documento");
        assert_eq!(hits[0].doc_id, doc_id);

        let offs = runner.locate(doc_id, b"aideen", 5);
        assert!(
            offs.len() >= 2,
            "locate debe encontrar las 2 ocurrencias de 'aideen'"
        );
        assert_eq!(offs[0].0, 0, "primera ocurrencia en byte 0");

        doc_id
    };

    let doc_mem2 = FsDocMemory::open(base.to_str().unwrap(), agent).unwrap();
    let hits2 = doc_mem2.search("aideen", 5);
    assert!(!hits2.is_empty(), "document must survive restart");
    assert_eq!(hits2[0].doc_id, doc_id);
}

// ── Test 3 ─────────────────────────────────────────────────────────────────────

/// build_context_features is deterministic and produces a non-null vector with real hits.
#[test]
fn doc_context_changes_state() {
    use aideen_node::{agent::FsAgentStore, doc_memory::FsDocMemory};

    let base = tmp_dir();
    let agent = "ctx_state";

    let store = Box::new(FsAgentStore::open(base.to_str().unwrap(), agent).unwrap());
    let doc_mem = Box::new(FsDocMemory::open(base.to_str().unwrap(), agent).unwrap());
    let mut runner = build_ctx_runner(store, doc_mem);

    runner
        .add_document(meta("Motor"), b"aideen motor cognitivo aideen".to_vec())
        .unwrap();

    let hits = runner.search_docs("aideen", 5);
    assert!(!hits.is_empty(), "the doc must appear in the index");

    let feats1 = build_context_features(
        &RuntimeContext {
            docs: hits.clone(),
            prefs: std::collections::HashMap::new(),
            recent_events: vec![],
        },
        512,
    );
    let feats2 = build_context_features(
        &RuntimeContext {
            docs: hits,
            prefs: std::collections::HashMap::new(),
            recent_events: vec![],
        },
        512,
    );

    assert_eq!(
        feats1, feats2,
        "build_context_features must be deterministic"
    );
    assert!(
        feats1.norm() > 0.0,
        "features with real hits must be non-null"
    );
}

// ── Test 4 ─────────────────────────────────────────────────────────────────────

/// Two different queries produce different h* when Reasoning is sensitive to S_sim.
#[test]
fn context_affects_attractor() {
    use aideen_node::{agent::FsAgentStore, doc_memory::FsDocMemory};

    let base = tmp_dir();
    let agent = "ctx_attractor";

    let store = Box::new(FsAgentStore::open(base.to_str().unwrap(), agent).unwrap());
    let doc_mem = Box::new(FsDocMemory::open(base.to_str().unwrap(), agent).unwrap());
    let mut runner = build_ctx_runner(store, doc_mem);

    runner
        .add_document(meta("Motor"), b"aideen motor cognitivo".to_vec())
        .unwrap();

    let out_a = runner
        .tick_with_query("aideen", 5)
        .expect("debe producir outcome (query A)");
    println!("DEBUG TEST A: converged={}, is_attractor={}, delta_norm(via stop_reason)={:?}, q_total={:?}, epsilon={:?}", out_a.metrics.converged, out_a.metrics.is_attractor, out_a.metrics.stop_reason, out_a.metrics.quality.q_total, runner.node.epsilon);
    assert!(
        out_a.metrics.is_attractor,
        "debe alcanzar atractor con query A"
    );
    let h_star_a = out_a.metrics.h_star.unwrap();

    runner.node.state = DVector::zeros(2560);
    let out_b = runner
        .tick_with_query("zzznotfound", 5)
        .expect("debe producir outcome (query B)");
    assert!(
        out_b.metrics.is_attractor,
        "debe alcanzar atractor con query B"
    );
    let h_star_b = out_b.metrics.h_star.unwrap();

    let flat_a = h_star_a.to_flat();
    let flat_b = h_star_b.to_flat();
    let diff: f32 = flat_a
        .iter()
        .zip(flat_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "diferente contexto debe producir diferente h* (diff={diff})"
    );
}

// ── Test 5 ─────────────────────────────────────────────────────────────────────

/// Con delegated=true, tick_with_query produce NetMsg::Discovery y registra
/// AgentEvent::DiscoveryEmitted con los campos correctos.
#[test]
fn discovery_emitted_when_delegated() {
    use aideen_node::{agent::FsAgentStore, doc_memory::FsDocMemory};

    let base = tmp_dir();
    let agent = "disc_delegated";

    let node_id: [u8; 32] = [1u8; 32];
    let bundle_version: u64 = 7;
    let target_id = "coordinator".to_string();

    let store = Box::new(FsAgentStore::open(base.to_str().unwrap(), agent).unwrap());
    let doc_mem = Box::new(FsDocMemory::open(base.to_str().unwrap(), agent).unwrap());
    let mut runner =
        build_ctx_runner_with_id(store, doc_mem, node_id, bundle_version, target_id.clone());
    runner.set_delegated(true);

    runner
        .add_document(meta("Doc"), b"aideen aprendizaje cognitivo".to_vec())
        .unwrap();

    let TickOutcome { metrics, discovery } = runner
        .tick_with_query("aideen", 5)
        .expect("debe producir outcome");

    assert!(metrics.is_attractor, "debe alcanzar atractor");
    assert!(
        metrics.allow_learning,
        "allow_learning debe ser true (q >= Q_MIN_LEARN)"
    );
    assert!(
        discovery.is_some(),
        "Discovery debe existir cuando allow_learning && delegated"
    );

    if let Some(NetMsg::Discovery {
        node_id: nid,
        target_id: tid,
        bundle_version: bv,
        q_total,
        ..
    }) = discovery
    {
        assert_eq!(nid, node_id, "node_id debe coincidir");
        assert_eq!(tid, target_id, "target_id debe coincidir");
        assert_eq!(bv, bundle_version, "bundle_version debe coincidir");
        assert!(q_total > 0.0, "q_total debe ser positivo");
    }

    let events = runner.recent_events(10);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AgentEvent::DiscoveryEmitted { .. })),
        "DiscoveryEmitted debe registrarse en agent_store"
    );
}

// ── Test 6 ─────────────────────────────────────────────────────────────────────

/// Without delegation, tick_with_query does not emit Discovery even if allow_learning=true.
#[test]
fn discovery_not_emitted_without_delegation() {
    use aideen_node::{agent::FsAgentStore, doc_memory::FsDocMemory};

    let base = tmp_dir();
    let agent = "disc_nodelegation";

    let store = Box::new(FsAgentStore::open(base.to_str().unwrap(), agent).unwrap());
    let doc_mem = Box::new(FsDocMemory::open(base.to_str().unwrap(), agent).unwrap());
    // delegated = false (default)
    let mut runner =
        build_ctx_runner_with_id(store, doc_mem, [0u8; 32], 0, "coordinator".to_string());

    runner
        .add_document(meta("Doc2"), b"aideen sistema cognitivo".to_vec())
        .unwrap();

    let TickOutcome { metrics, discovery } = runner
        .tick_with_query("aideen", 5)
        .expect("debe producir outcome");

    assert!(metrics.allow_learning, "allow_learning debe ser true");
    assert!(
        discovery.is_none(),
        "sin delegación no debe emitir Discovery"
    );

    let events = runner.recent_events(10);
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, AgentEvent::DiscoveryEmitted { .. })),
        "sin delegación no debe registrarse DiscoveryEmitted"
    );
}
