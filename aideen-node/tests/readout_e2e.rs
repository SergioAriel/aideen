/// E2E: Readout canónico sobre h* del ciclo DEQ.
///
/// Invariantes verificados:
///   1. tick() con convergencia instantánea → h_star = Some(h*)
///   2. LinearReadout::identity devuelve exactamente h*
///   3. LinearReadout::new(W, b) devuelve W·h* + b exacto
///   4. Cuando Ethics viola → h_star = None (gating)
///   5. allow_learning refleja Q >= Q_MIN_LEARN
use aideen_backbone::readout::LinearReadout;
use aideen_core::readout::Readout;
use aideen_core::state::{ArchitectureConfig, HSlots};
use aideen_node::system::node::AideenNode;
use nalgebra::{DMatrix, DVector};

// ── Constantes del test ────────────────────────────────────────────────────

/// Valor fijo al que converge el MockReasoning
const H_VAL: f32 = 0.3;

// ── Mocks mínimos ──────────────────────────────────────────────────────────

struct MockBackend;
impl aideen_core::compute::ComputeBackend for MockBackend {
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

/// Reasoning que converge instantáneamente a `[H_VAL; D_R]` en todos los slots.
struct FixedPointReasoning(ArchitectureConfig);
impl aideen_core::reasoning::Reasoning for FixedPointReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.0
    }
    fn init(&self, _s: &DVector<f32>) -> HSlots {
        HSlots::from_broadcast(&DVector::from_element(self.0.d_r, H_VAL), &self.0)
    }
    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        _exec: Option<&mut dyn aideen_core::compute::ComputeBackend>,
    ) -> HSlots {
        h.clone() // h_next == h → delta_norm = 0
    }
}

/// Control que para en la primera iteración (iter=0).
struct ImmediateStop;
impl aideen_core::control::Control for ImmediateStop {
    fn max_iters(&self) -> usize {
        10
    }
    fn mode(&self) -> aideen_core::control::ControlMode {
        aideen_core::control::ControlMode::Observe
    }
    fn decide(&self, _iter: usize, _dn: f32, _e: f32) -> aideen_core::control::ControlDecision {
        aideen_core::control::ControlDecision {
            stop: true, // siempre para en iter=0
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
    fn violates(&self, _state: &DVector<f32>) -> bool {
        false
    }
    fn project(&self, state: &DVector<f32>) -> DVector<f32> {
        state.clone()
    }
}

struct FailEthics;
impl aideen_core::ethics::Ethics for FailEthics {
    fn fingerprint(&self) -> [u8; 32] {
        [0; 32]
    }
    fn violates(&self, _state: &DVector<f32>) -> bool {
        true
    }
    fn project(&self, state: &DVector<f32>) -> DVector<f32> {
        state.clone()
    }
}

struct NullMemory;
impl aideen_core::memory::Memory for NullMemory {
    fn write(&mut self, _v: DVector<f32>) {}
    fn query(&self, _q: &DVector<f32>, _k: usize) -> Vec<DVector<f32>> {
        vec![]
    }
}

// ── Helper ─────────────────────────────────────────────────────────────────

fn assert_dvec_approx(a: &DVector<f32>, b: &DVector<f32>, tol: f32) {
    assert_eq!(a.len(), b.len(), "dimensión distinta");
    for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (ai - bi).abs() < tol,
            "índice {}: {} vs {} (tol={})",
            i,
            ai,
            bi,
            tol
        );
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

/// Readout identidad: output debe ser exactamente h*.
#[test]
fn test_readout_identity_on_attractor() {
    let config = ArchitectureConfig::default();
    let mut node = AideenNode {
        state: DVector::zeros(config.d_global()),
        reasoning: FixedPointReasoning(config.clone()),
        control: ImmediateStop,
        ethics: PassEthics,
        memory: NullMemory,
        backend: MockBackend,
        alpha: 0.5,
        epsilon: 1e-3,
    };

    let m = node.tick().expect("tick debe producir métricas");
    assert!(m.is_attractor, "debe ser atractor");
    assert!(m.allow_learning, "Q=0.7 >= Q_MIN_LEARN=0.6");

    let h_star_slots = m.h_star.as_ref().expect("h_star debe existir en atractor");
    let h_star = h_star_slots.slot(0);
    assert_eq!(h_star.len(), config.d_r);

    let readout = LinearReadout::identity(config.d_r);
    let output = readout.readout(&h_star);

    assert_dvec_approx(&output, &h_star, 1e-6);
}

/// Readout proyección 2D: W·h* + b exacto.
#[test]
fn test_readout_projection_on_attractor() {
    let config = ArchitectureConfig::default();
    let mut node = AideenNode {
        state: DVector::zeros(config.d_global()),
        reasoning: FixedPointReasoning(config.clone()),
        control: ImmediateStop,
        ethics: PassEthics,
        memory: NullMemory,
        backend: MockBackend,
        alpha: 0.5,
        epsilon: 1e-3,
    };

    let m = node.tick().expect("tick debe producir métricas");
    let h_star_slots = m.h_star.as_ref().expect("h_star debe existir");
    let h_star = h_star_slots.slot(0);

    // W: [2 × D_R] selecciona h[0] y h[1]; b = [100, 200]
    let out_dim = 2;
    let mut w = DMatrix::<f32>::zeros(out_dim, config.d_r);
    w[(0, 0)] = 1.0;
    w[(1, 1)] = 1.0;
    let b = DVector::from_vec(vec![100.0_f32, 200.0_f32]);

    let readout = LinearReadout::new(w, b);
    let output = readout.readout(&h_star);

    // h_star slot0 = [H_VAL; D_R], output[i] = H_VAL + bias[i]
    assert!(
        (output[0] - (H_VAL + 100.0)).abs() < 1e-5,
        "output[0] = {}",
        output[0]
    );
    assert!(
        (output[1] - (H_VAL + 200.0)).abs() < 1e-5,
        "output[1] = {}",
        output[1]
    );
}

/// Gating ético: si Ethics viola, h_star debe ser None.
#[test]
fn test_h_star_absent_on_ethics_violation() {
    let config = ArchitectureConfig::default();
    let mut node = AideenNode {
        state: DVector::zeros(config.d_global()),
        reasoning: FixedPointReasoning(config.clone()),
        control: ImmediateStop,
        ethics: FailEthics,
        memory: NullMemory,
        backend: MockBackend,
        alpha: 0.5,
        epsilon: 1e-3,
    };

    let m = node.tick().expect("tick debe producir métricas");
    assert!(!m.is_attractor, "ética violada → no es atractor");
    assert!(
        m.h_star.is_none(),
        "h_star debe ser None cuando ética falla"
    );
}
