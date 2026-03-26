use crate::spectral_norm;
use aideen_core::{
    block_backend::BlockBackend,
    compute::ComputeBackend,
    reasoning::Reasoning,
    state::{ArchitectureConfig, HSlots},
};
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

/// MambaSlotReasoning — el bloque `f` real del DEQ.
pub struct MambaSlotReasoning {
    pub config: ArchitectureConfig,

    // ── Cross-slot attention ─────────────────────────────────────────────────
    #[cfg(feature = "lab")]
    pub w_q: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_q: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub w_k: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_k: DMatrix<f32>,
    // Per-slot bias for Q and K: [h_slots × d_r], appended to W_q/W_k GPU buffers.
    // Breaks permutation symmetry so slots can develop distinct attention roles.
    #[cfg(feature = "lab")]
    pub q_bias: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) q_bias: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub k_bias: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) k_bias: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub w_v: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_v: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub w_o: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_o: DMatrix<f32>,

    // ── Input injection ──────────────────────────────────────────────────────
    #[cfg(feature = "lab")]
    pub w_in: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_in: DMatrix<f32>,

    // ── History interface into the DEQ ─────────────────────────────────────
    #[cfg(feature = "lab")]
    pub w_hist_shared: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_hist_shared: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub hist_slot_scale: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) hist_slot_scale: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub hist_slot_bias: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) hist_slot_bias: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub hist_gate_logit: DVector<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) hist_gate_logit: DVector<f32>,
    #[cfg(feature = "lab")]
    pub slot_anchor: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) slot_anchor: DMatrix<f32>,

    // ── Mamba SSM por slot ───────────────────────────────────────────────────
    #[cfg(feature = "lab")]
    pub a_log: DMatrix<f32>, // [h_slots × d_r] row-major, per-slot decay
    #[cfg(not(feature = "lab"))]
    pub(crate) a_log: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub w_x: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_x: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub w_out: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_out: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub w_delta: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_delta: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub b_delta: DVector<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) b_delta: DVector<f32>,

    // ── Dynamic history gate (h^k-dependent slot query) ───────────────────────
    #[cfg(feature = "lab")]
    pub w_gate_hist: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_gate_hist: DMatrix<f32>,

    // ── Forget gate for M state ───────────────────────────────────────────────
    // f[s] = σ(b_f[s] + dot(W_f[s,:], H_curr) / √d)
    // M_t = a * f[s] * m_prev + (1-a) * x_proj
    // W_f zeros + b_f=3.0 → f≈0.95 (near-identity at init, ∂/∂h=0)
    #[cfg(feature = "lab")]
    pub w_forget: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_forget: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub b_forget: DVector<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) b_forget: DVector<f32>,

    // ── LayerNorm por slot ───────────────────────────────────────────────────
    #[cfg(feature = "lab")]
    pub norm_scale: DVector<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) norm_scale: DVector<f32>,

    // ── Backend GPU opcional ─────────────────────────────────────────────────
    backend: RefCell<Option<Box<dyn BlockBackend>>>,

    // ── Picard β-relaxation ───────────────────────────────────────────────
    pub damping: f32,
    /// Residual mixing inside f(h): combined += residual_alpha * h.
    /// 1.0 = legacy behavior, <1.0 = contractivity-friendly experimental mode.
    pub residual_alpha: f32,
}

impl MambaSlotReasoning {
    fn matrix_to_row_major(m: &DMatrix<f32>) -> Vec<f32> {
        let rows = m.nrows();
        let cols = m.ncols();
        let mut data = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[i * cols + j] = m[(i, j)];
            }
        }
        data
    }

    fn xavier_mat_with_rng<R: Rng + ?Sized>(
        rng: &mut R,
        d_r: usize,
        xavier_range: f32,
    ) -> DMatrix<f32> {
        DMatrix::from_fn(d_r, d_r, |_, _| rng.gen_range(-xavier_range..xavier_range))
    }

    #[allow(dead_code)]
    fn identity_like_mat_with_rng<R: Rng + ?Sized>(
        rng: &mut R,
        d_r: usize,
        noise: f32,
    ) -> DMatrix<f32> {
        let jitter = DMatrix::from_fn(d_r, d_r, |_, _| rng.gen_range(-noise..noise));
        DMatrix::identity(d_r, d_r) + jitter
    }

    fn scaled_identity_like_mat_with_rng<R: Rng + ?Sized>(
        rng: &mut R,
        d_r: usize,
        scale: f32,
        noise: f32,
    ) -> DMatrix<f32> {
        let jitter = DMatrix::from_fn(d_r, d_r, |_, _| rng.gen_range(-noise..noise));
        DMatrix::identity(d_r, d_r) * scale + jitter
    }

    fn default_slot_anchor(config: &ArchitectureConfig) -> DMatrix<f32> {
        let h_slots = config.h_slots;
        let d_r = config.d_r;
        DMatrix::from_fn(h_slots, d_r, |slot, dim| {
            // Deterministic slot identity basis. This breaks permutation symmetry in H0
            // without injecting large energy that would destabilize the DEQ operator.
            let centered = slot as f32 - (h_slots.saturating_sub(1) as f32 * 0.5);
            let phase = (((slot + 1) * (dim + 3) * 17) % 2048) as f32 / 1024.0 - 1.0;
            centered * 2.0e-4f32 + phase * 2.5e-5f32
        })
    }

    pub fn new(config: ArchitectureConfig) -> Self {
        let mut rng = thread_rng();
        Self::new_with_rng(config, &mut rng)
    }

    pub fn new_with_seed(config: ArchitectureConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        Self::new_with_rng(config, &mut rng)
    }

    fn new_with_rng<R: Rng + ?Sized>(config: ArchitectureConfig, rng: &mut R) -> Self {
        let d_r = config.d_r;
        let h_slots = config.h_slots;
        // Xavier Uniform initialization: range = sqrt(3 / fan_in)
        let xavier_range = (3.0 / d_r as f32).sqrt();

        let mut w_q = Self::xavier_mat_with_rng(rng, d_r, xavier_range);
        let mut w_k = Self::xavier_mat_with_rng(rng, d_r, xavier_range);
        let mut w_v = Self::xavier_mat_with_rng(rng, d_r, xavier_range);
        let mut w_o = Self::xavier_mat_with_rng(rng, d_r, xavier_range);
        let mut w_in = Self::xavier_mat_with_rng(rng, d_r, xavier_range);
        // Historical interface: 0.15*I — increased from 0.05 after DEQ stabilization (contr≈0.21).
        // hist_ctx has ∂/∂h=0 so W_hist does not enter the Jacobian; increasing it is safe.
        // Target: hist/inj ≈ 0.25 (up from 0.08) for meaningful SSM gradient path.
        let w_hist_shared = Self::scaled_identity_like_mat_with_rng(rng, d_r, 0.15_f32, 0.01_f32);
        // The temporal carrier is applied as a residual around identity:
        //   x_proj = h_unit + W_x h_unit
        //   M_t    = m_inner + W_out m_inner
        // Therefore W_x/W_out should initialize near zero, not near identity. This keeps
        // the carrier alive by construction while still allowing learned deviations later.
        let w_x = Self::scaled_identity_like_mat_with_rng(rng, d_r, 0.0f32, 0.01f32);
        let w_out = Self::scaled_identity_like_mat_with_rng(rng, d_r, 0.0f32, 0.01f32);
        // Per-slot W_delta: random small init to break slot symmetry.
        // zeros would give delta_input[s]=W_delta[s]*M=0 for all slots → identical M trajectories
        // → identical A_log gradients → slots never specialize regardless of training length.
        // Small random noise gives distinct delta_input per slot from token 2 onward.
        let w_delta = DMatrix::from_fn(h_slots * d_r, d_r, |_, _| {
            rng.gen_range(-0.01_f32..0.01_f32)
        });

        // DEQ requiere σ(J_f) < 1 desde la inicialización.
        // El Jacobiano compuesto de atención (W_o · softmax · W_v · W_q) puede superar 1
        // con Xavier estándar. Renormalizamos las matrices de atención a σ ≤ 0.10
        // para garantizar σ(J_attn) << 1 antes del primer token de entrenamiento.
        // residual_alpha=0.0 es necesario — incluso alpha=0.2 lleva contr→1.
        let deq_threshold = 0.10_f32;
        let win_threshold = 0.30_f32;
        let n_iter = 20;
        spectral_norm::normalize_if_needed(&mut w_q, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_k, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_v, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_o, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_in, win_threshold, n_iter);

        let hist_slot_scale = DMatrix::from_fn(h_slots, d_r, |slot, _| {
            // Keep multiplicative history adaptation off at initialization. A non-zero
            // diagonal scale would create an implicit bypass from M_{t-1} into the DEQ.
            let _ = slot;
            0.0
        });
        let hist_slot_bias = DMatrix::from_fn(h_slots, d_r, |slot, _| {
            // Break slot permutation symmetry structurally. Without a slot-specific additive
            // anchor, all slots remain exchangeable because H0 is broadcast and the early
            // historical carrier M_{t-1} is nearly identical across slots.
            let centered = slot as f32 - (h_slots.saturating_sub(1) as f32 * 0.5);
            centered * 2.5e-3f32 + rng.gen_range(-5.0e-4f32..5.0e-4f32)
        });
        let hist_gate_logit = DVector::from_fn(h_slots, |slot, _| {
            // Gate inicial: alpha_target ≈ 0.10 con piso 0.07 y techo 0.20.
            // sigma(g) = (alpha - alpha_min)/(alpha_max - alpha_min) = 0.23077 -> g ≈ -1.204.
            let base = -1.204_f32;
            let centered = slot as f32 - (h_slots.saturating_sub(1) as f32 * 0.5);
            base + centered * 0.02
        });
        let slot_anchor = Self::default_slot_anchor(&config);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_hist_shared,
            hist_slot_scale,
            hist_slot_bias,
            hist_gate_logit,
            slot_anchor,
            // Mamba-style timescale prior: decay spread a ∈ [0.80, 0.999] across slots.
            // Biased toward long-term memory so the forget gate has useful history to gate.
            // Gradient + forget gate can push individual slots faster if needed.
            a_log: DMatrix::from_fn(h_slots, d_r, |slot, _| {
                let a_min = 0.80_f32;
                let a_max = 0.999_f32;
                let alpha = if h_slots <= 1 {
                    0.5
                } else {
                    slot as f32 / (h_slots - 1) as f32
                };
                let a = a_min + alpha * (a_max - a_min);
                (a / (1.0 - a)).ln() // logit(a): slot 0 = 1.39 (moderate), slot 7 = 6.91 (very slow)
            }),
            w_x,
            w_out,
            w_delta,
            b_delta: DVector::zeros(d_r),
            // W_gate_hist: zeros init → hist_mod = 1+tanh(0) = 1 at step 0 (identity behavior).
            w_gate_hist: DMatrix::zeros(h_slots, d_r),
            // Forget gate: W_f=zeros, b_f=3.0 → f≈0.95 (near-identity at init).
            // Allows model to learn to forget selectively once training establishes gradients.
            w_forget: DMatrix::zeros(h_slots, d_r),
            b_forget: DVector::from_element(h_slots, 3.0_f32),
            norm_scale: DVector::from_element(d_r, 1.0_f32),
            // Q/K per-slot bias: random init to immediately break slot symmetry.
            // Zero init would give gscore=0 (uniform attn → zero gradient → saddle point forever).
            // randn×0.2 gives distinct Q/K per slot from step 1 → attn_ent can fall below log(K).
            q_bias: DMatrix::from_fn(h_slots, d_r, |_, _| rng.gen_range(-0.2_f32..0.2_f32)),
            k_bias: DMatrix::from_fn(h_slots, d_r, |_, _| rng.gen_range(-0.2_f32..0.2_f32)),
            backend: RefCell::new(None),
            damping: 0.70_f32,
            residual_alpha: 0.0_f32,
            config,
        }
    }

    pub fn with_damping(mut self, beta: f32) -> Self {
        self.damping = beta.clamp(0.01, 1.0);
        self
    }

    pub fn with_residual_alpha(mut self, alpha: f32) -> Self {
        self.residual_alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Returns W_in as a flat GPU buffer: per-slot W_in with deterministic jitter.
    /// This breaks slot symmetry at step 1 while keeping σ(W_in) bounded.
    pub fn w_in_gpu_flat(&self) -> Vec<f32> {
        let h_slots = self.config.h_slots;
        let d_r = self.config.d_r;
        let base = Self::matrix_to_row_major(&self.w_in);
        let mut hasher = DefaultHasher::new();
        for v in &base {
            hasher.write_u32(v.to_bits());
        }
        let seed = hasher.finish();
        let mut out = Vec::with_capacity(h_slots * base.len());
        let _attn_t = 0.10_f32;
        let win_t = 0.30_f32;
        let n_iter = 20;
        // Disable per-slot jitter to remove seed-dependent W_in variance during diagnosis.
        let jitter = 0.0_f32;
        for s in 0..h_slots {
            let slot_seed = seed ^ (s as u64).wrapping_mul(0x9E3779B97F4A7C15);
            let mut mat = self.w_in.clone();
            if jitter > 0.0 {
                let mut rng = StdRng::seed_from_u64(slot_seed);
                for r in 0..d_r {
                    for c in 0..d_r {
                        mat[(r, c)] += rng.gen_range(-jitter..jitter);
                    }
                }
            }
            spectral_norm::normalize_if_needed(&mut mat, win_t, n_iter);
            out.extend_from_slice(&Self::matrix_to_row_major(&mat));
        }
        out
    }

    /// Row-major flat slice of a_log for GPU upload: [h_slots × d_r].
    pub fn a_log_gpu_flat(&self) -> Vec<f32> {
        Self::matrix_to_row_major(&self.a_log)
    }

    /// GPU flat layout for W_v: [h_slots × d_r×d_r matrices] (no bias — W_o absorbs output bias).
    /// Each slot matrix is the shared w_v plus a deterministic per-slot jitter (smaller than Q/K).
    /// Shader accesses slot s: W_v[s*d*d + j*d + d_out].
    pub fn w_v_gpu_flat(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let base = self.w_v.as_slice(); // column-major, d*d floats
        let mut v = Vec::with_capacity(h * d * d);
        for s in 0..h {
            for (i, &x) in base.iter().enumerate() {
                // Smaller jitter than Q/K (0.03 vs 0.05) to keep V more stable
                let jitter =
                    ((s * d * d + i) as f32 * 0.0001_f32 + s as f32 * 0.3_f32).sin() * 0.03_f32;
                v.push(x + jitter);
            }
        }
        v
    }

    /// GPU flat layout for W_q: [h_slots × d_r×d_r matrices | h_slots×d_r q_bias (row-major)].
    /// Each slot matrix is the shared w_q plus a deterministic per-slot jitter for symmetry breaking.
    /// Shader accesses slot s matrix: W_q[s*d*d + j*d + d_out].
    /// Bias: W_q[h_slots*d*d + s*d + d_out].
    pub fn w_q_gpu_flat(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let base = self.w_q.as_slice(); // column-major, d*d floats
        let mut v = Vec::with_capacity(h * d * d + h * d);
        for s in 0..h {
            for (i, &x) in base.iter().enumerate() {
                // Deterministic jitter: sin(index) * scale — same for same init, different per slot
                let jitter = ((s * d * d + i) as f32 * 0.0001_f32).sin() * 0.05_f32;
                v.push(x + jitter);
            }
        }
        v.extend_from_slice(&Self::matrix_to_row_major(&self.q_bias));
        v
    }

    /// GPU flat layout for W_k: [h_slots × d_r×d_r matrices | h_slots×d_r k_bias (row-major)].
    pub fn w_k_gpu_flat(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let base = self.w_k.as_slice();
        let mut v = Vec::with_capacity(h * d * d + h * d);
        for s in 0..h {
            for (i, &x) in base.iter().enumerate() {
                let jitter = ((s * d * d + i) as f32 * 0.0001_f32).sin() * 0.05_f32;
                v.push(x + jitter);
            }
        }
        v.extend_from_slice(&Self::matrix_to_row_major(&self.k_bias));
        v
    }

    pub fn renormalize_weights(&mut self) {
        // Umbral 0.10 para matrices de atención — necesario para mantener σ(J_attn) < 1
        // durante el entrenamiento (no solo en la inicialización).
        // w_x y w_out (Mamba externo) usan umbral 0.70 — no afectan la contractividad del DEQ.
        let attn_t = 0.10_f32;
        let win_t = 0.30_f32;
        let mamba_t = 0.70_f32;
        let n = 20;
        spectral_norm::normalize_if_needed(&mut self.w_q, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_k, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_v, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_o, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_in, win_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_hist_shared, 1.5, n);
        spectral_norm::normalize_if_needed(&mut self.w_x, mamba_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_out, mamba_t, n);
    }

    pub fn spectral_norms(&self) -> [f32; 8] {
        let n = 10;
        [
            spectral_norm::spectral_norm(&self.w_q, n),
            spectral_norm::spectral_norm(&self.w_k, n),
            spectral_norm::spectral_norm(&self.w_v, n),
            spectral_norm::spectral_norm(&self.w_o, n),
            spectral_norm::spectral_norm(&self.w_in, n),
            spectral_norm::spectral_norm(&self.w_hist_shared, n),
            spectral_norm::spectral_norm(&self.w_x, n),
            spectral_norm::spectral_norm(&self.w_out, n),
        ]
    }

    pub fn set_backend(&mut self, b: impl BlockBackend + 'static) {
        *self.backend.borrow_mut() = Some(Box::new(b));
    }

    pub fn clear_backend(&mut self) {
        *self.backend.borrow_mut() = None;
    }

    pub fn has_backend(&self) -> bool {
        self.backend.borrow().is_some()
    }

    pub fn reset(&mut self) {
        // CPU side is currently stateless inside reasoning,
        // but this is called by the Trainer during EOS Reset.
    }

    pub fn export_weights(&self) -> std::collections::HashMap<String, Vec<f32>> {
        let mut weights = std::collections::HashMap::new();
        weights.insert(
            "reasoning.w_q".to_string(),
            Self::matrix_to_row_major(&self.w_q),
        );
        weights.insert(
            "reasoning.w_k".to_string(),
            Self::matrix_to_row_major(&self.w_k),
        );
        weights.insert(
            "reasoning.q_bias".to_string(),
            Self::matrix_to_row_major(&self.q_bias),
        );
        weights.insert(
            "reasoning.k_bias".to_string(),
            Self::matrix_to_row_major(&self.k_bias),
        );
        weights.insert(
            "reasoning.w_v".to_string(),
            Self::matrix_to_row_major(&self.w_v),
        );
        weights.insert(
            "reasoning.w_o".to_string(),
            Self::matrix_to_row_major(&self.w_o),
        );
        weights.insert(
            "reasoning.w_in".to_string(),
            Self::matrix_to_row_major(&self.w_in),
        );
        weights.insert(
            "reasoning.w_hist_shared".to_string(),
            Self::matrix_to_row_major(&self.w_hist_shared),
        );
        weights.insert(
            "reasoning.hist_slot_scale".to_string(),
            Self::matrix_to_row_major(&self.hist_slot_scale),
        );
        weights.insert(
            "reasoning.hist_slot_bias".to_string(),
            Self::matrix_to_row_major(&self.hist_slot_bias),
        );
        weights.insert(
            "reasoning.hist_gate_logit".to_string(),
            self.hist_gate_logit.as_slice().to_vec(),
        );
        weights.insert(
            "reasoning.slot_anchor".to_string(),
            Self::matrix_to_row_major(&self.slot_anchor),
        );
        weights.insert(
            "reasoning.w_x".to_string(),
            Self::matrix_to_row_major(&self.w_x),
        );
        weights.insert(
            "reasoning.w_out".to_string(),
            Self::matrix_to_row_major(&self.w_out),
        );
        weights.insert(
            "reasoning.w_delta".to_string(),
            Self::matrix_to_row_major(&self.w_delta),
        );
        weights.insert(
            "reasoning.b_delta".to_string(),
            self.b_delta.as_slice().to_vec(),
        );
        weights.insert(
            "reasoning.w_gate_hist".to_string(),
            Self::matrix_to_row_major(&self.w_gate_hist),
        );
        weights.insert(
            "reasoning.w_forget".to_string(),
            Self::matrix_to_row_major(&self.w_forget),
        );
        weights.insert(
            "reasoning.b_forget".to_string(),
            self.b_forget.as_slice().to_vec(),
        );
        weights.insert(
            "reasoning.a_log".to_string(),
            Self::matrix_to_row_major(&self.a_log),
        );
        weights.insert(
            "reasoning.norm_scale".to_string(),
            self.norm_scale.as_slice().to_vec(),
        );
        weights.insert("reasoning.damping".to_string(), vec![self.damping]);
        weights.insert(
            "reasoning.residual_alpha".to_string(),
            vec![self.residual_alpha],
        );

        weights
    }

    pub fn history_params_gpu_layout(
        &self,
    ) -> (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        (
            Self::matrix_to_row_major(&self.w_hist_shared),
            Self::matrix_to_row_major(&self.hist_slot_scale),
            Self::matrix_to_row_major(&self.hist_slot_bias),
            self.hist_gate_logit.as_slice().to_vec(),
            Self::matrix_to_row_major(&self.slot_anchor),
            Self::matrix_to_row_major(&self.w_delta),
            self.b_delta.as_slice().to_vec(),
            Self::matrix_to_row_major(&self.w_gate_hist),
            Self::matrix_to_row_major(&self.w_forget),
            self.b_forget.as_slice().to_vec(),
        )
    }

    pub fn import_weights(
        &mut self,
        weights: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<(), String> {
        let d_r = self.config.d_r;
        let get_mat = |name: &str| -> Result<nalgebra::DMatrix<f32>, String> {
            let data = weights
                .get(name)
                .ok_or_else(|| format!("Weight {} not found", name))?;
            if data.len() != d_r * d_r {
                return Err(format!(
                    "Weight {} size mismatch: expected {}, got {}",
                    name,
                    d_r * d_r,
                    data.len()
                ));
            }
            Ok(nalgebra::DMatrix::from_row_slice(d_r, d_r, data))
        };

        self.w_q = get_mat("reasoning.w_q")?;
        self.w_k = get_mat("reasoning.w_k")?;
        // q_bias/k_bias: graceful fallback to zeros for checkpoints that predate this feature.
        let h_slots = self.config.h_slots;
        if let Some(data) = weights.get("reasoning.q_bias") {
            if data.len() == h_slots * d_r {
                self.q_bias = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
            }
        }
        if let Some(data) = weights.get("reasoning.k_bias") {
            if data.len() == h_slots * d_r {
                self.k_bias = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
            }
        }
        self.w_v = get_mat("reasoning.w_v")?;
        self.w_o = get_mat("reasoning.w_o")?;
        self.w_in = get_mat("reasoning.w_in")?;
        self.w_hist_shared = get_mat("reasoning.w_hist_shared")?;
        {
            let data = weights
                .get("reasoning.hist_slot_scale")
                .ok_or("reasoning.hist_slot_scale not found")?;
            let h_slots = self.config.h_slots;
            if data.len() != h_slots * d_r {
                return Err(format!(
                    "Weight reasoning.hist_slot_scale size mismatch: expected {}, got {}",
                    h_slots * d_r,
                    data.len()
                ));
            }
            self.hist_slot_scale = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
        }
        {
            let data = weights
                .get("reasoning.hist_slot_bias")
                .ok_or("reasoning.hist_slot_bias not found")?;
            let h_slots = self.config.h_slots;
            if data.len() != h_slots * d_r {
                return Err(format!(
                    "Weight reasoning.hist_slot_bias size mismatch: expected {}, got {}",
                    h_slots * d_r,
                    data.len()
                ));
            }
            self.hist_slot_bias = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
        }
        self.hist_gate_logit = nalgebra::DVector::from_vec(
            weights
                .get("reasoning.hist_gate_logit")
                .ok_or("reasoning.hist_gate_logit not found")?
                .clone(),
        );
        {
            let h_slots = self.config.h_slots;
            if let Some(data) = weights.get("reasoning.slot_anchor") {
                if data.len() != h_slots * d_r {
                    return Err(format!(
                        "Weight reasoning.slot_anchor size mismatch: expected {}, got {}",
                        h_slots * d_r,
                        data.len()
                    ));
                }
                self.slot_anchor = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
            } else {
                self.slot_anchor = Self::default_slot_anchor(&self.config);
            }
        }
        self.w_x = get_mat("reasoning.w_x")?;
        self.w_out = get_mat("reasoning.w_out")?;
        {
            let h_slots = self.config.h_slots;
            let data = weights
                .get("reasoning.w_delta")
                .ok_or("reasoning.w_delta not found")?;
            self.w_delta = if data.len() == d_r * d_r {
                // Legacy checkpoint: broadcast single matrix to all slots.
                let base = nalgebra::DMatrix::from_row_slice(d_r, d_r, data);
                let flat: Vec<f32> = (0..h_slots)
                    .flat_map(|_| Self::matrix_to_row_major(&base))
                    .collect();
                nalgebra::DMatrix::from_row_slice(h_slots * d_r, d_r, &flat)
            } else if data.len() == h_slots * d_r * d_r {
                nalgebra::DMatrix::from_row_slice(h_slots * d_r, d_r, data)
            } else {
                return Err(format!(
                    "reasoning.w_delta size mismatch: expected {} or {}, got {}",
                    d_r * d_r,
                    h_slots * d_r * d_r,
                    data.len()
                ));
            };
        }
        self.b_delta = nalgebra::DVector::from_vec(
            weights
                .get("reasoning.b_delta")
                .ok_or("reasoning.b_delta not found")?
                .clone(),
        );
        // w_gate_hist: optional — zeros for old checkpoints that predate this feature.
        {
            let h_slots = self.config.h_slots;
            if let Some(data) = weights.get("reasoning.w_gate_hist") {
                if data.len() == h_slots * d_r {
                    self.w_gate_hist = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
                }
            }
        }
        // w_forget / b_forget: optional — zeros + bias 3.0 for old checkpoints.
        {
            let h_slots = self.config.h_slots;
            if let Some(data) = weights.get("reasoning.w_forget") {
                if data.len() == h_slots * d_r {
                    self.w_forget = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
                }
            }
            if let Some(data) = weights.get("reasoning.b_forget") {
                if data.len() == h_slots {
                    self.b_forget = nalgebra::DVector::from_vec(data.clone());
                }
            }
        }

        {
            let h_slots = self.config.h_slots;
            let d_r = self.config.d_r;
            let data = weights
                .get("reasoning.a_log")
                .ok_or("reasoning.a_log not found")?;
            self.a_log = if data.len() == d_r {
                // Legacy checkpoint: broadcast single row to all slots.
                let flat: Vec<f32> = (0..h_slots).flat_map(|_| data.iter().copied()).collect();
                nalgebra::DMatrix::from_row_slice(h_slots, d_r, &flat)
            } else if data.len() == h_slots * d_r {
                nalgebra::DMatrix::from_row_slice(h_slots, d_r, data)
            } else {
                return Err(format!(
                    "reasoning.a_log size mismatch: expected {} or {}, got {}",
                    d_r,
                    h_slots * d_r,
                    data.len()
                ));
            };
        }
        self.norm_scale = nalgebra::DVector::from_vec(
            weights
                .get("reasoning.norm_scale")
                .ok_or("reasoning.norm_scale not found")?
                .clone(),
        );
        self.damping = weights
            .get("reasoning.damping")
            .ok_or("reasoning.damping not found")?[0];
        self.residual_alpha = weights
            .get("reasoning.residual_alpha")
            .map(|v| v[0])
            .unwrap_or(1.0);

        Ok(())
    }

    pub fn save_checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), String> {
        use aideen_core::model::AidenModel;

        let mut model = AidenModel::new(self.config.clone());
        model.weights = self.export_weights();
        model
            .metadata
            .insert("type".to_string(), "MambaSlotReasoning".to_string());

        model.save(path.as_ref().to_str().ok_or("Invalid path")?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        config: ArchitectureConfig,
        w_q: nalgebra::DMatrix<f32>,
        w_k: nalgebra::DMatrix<f32>,
        w_v: nalgebra::DMatrix<f32>,
        w_o: nalgebra::DMatrix<f32>,
        w_in: nalgebra::DMatrix<f32>,
        w_hist_shared: nalgebra::DMatrix<f32>,
        hist_slot_scale: nalgebra::DMatrix<f32>,
        hist_slot_bias: nalgebra::DMatrix<f32>,
        hist_gate_logit: nalgebra::DVector<f32>,
        slot_anchor: nalgebra::DMatrix<f32>,
        w_x: nalgebra::DMatrix<f32>,
        w_out: nalgebra::DMatrix<f32>,
        w_delta: nalgebra::DMatrix<f32>,
        b_delta: nalgebra::DVector<f32>,
        a_log: nalgebra::DMatrix<f32>,
        norm_scale: nalgebra::DVector<f32>,
        damping: f32,
        residual_alpha: f32,
    ) -> Self {
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_hist_shared,
            hist_slot_scale,
            hist_slot_bias,
            hist_gate_logit,
            slot_anchor,
            w_x,
            w_out,
            w_delta,
            b_delta,
            w_gate_hist: nalgebra::DMatrix::zeros(config.h_slots, config.d_r),
            a_log,
            norm_scale,
            q_bias: nalgebra::DMatrix::zeros(config.h_slots, config.d_r),
            k_bias: nalgebra::DMatrix::zeros(config.h_slots, config.d_r),
            w_forget: nalgebra::DMatrix::zeros(config.h_slots, config.d_r),
            b_forget: nalgebra::DVector::from_element(config.h_slots, 3.0_f32),
            backend: RefCell::new(None),
            damping,
            residual_alpha,
            config,
        }
    }

    fn rms_norm(&self, v: &DVector<f32>) -> DVector<f32> {
        let rms = (v.map(|x| x * x).mean() + 1e-6).sqrt();
        v.zip_map(&self.norm_scale, |x, s| s * x / rms)
    }

    fn cross_slot_attn(&self, h: &HSlots) -> HSlots {
        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;
        let scale = (d_r as f32).sqrt().recip();

        let qs: Vec<DVector<f32>> = (0..h_slots).map(|k| &self.w_q * h.slot(k)).collect();
        let ks: Vec<DVector<f32>> = (0..h_slots).map(|k| &self.w_k * h.slot(k)).collect();
        let vs: Vec<DVector<f32>> = (0..h_slots).map(|k| &self.w_v * h.slot(k)).collect();

        let mut next = HSlots::zeros(&self.config);
        for q_idx in 0..h_slots {
            let raw_scores: Vec<f32> = ks.iter().map(|k| qs[q_idx].dot(k) * scale).collect();

            let max_s = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = raw_scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let attn: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

            let mixed: DVector<f32> = attn
                .iter()
                .zip(vs.iter())
                .map(|(a, v)| v * *a)
                .fold(DVector::zeros(d_r), |acc, v| acc + v);

            next.set_slot(q_idx, &(&self.w_o * mixed));
        }
        next
    }
}

impl Reasoning for MambaSlotReasoning {
    fn config(&self) -> &ArchitectureConfig {
        &self.config
    }

    fn init(&self, s: &DVector<f32>) -> HSlots {
        let d_r = self.config.d_r;
        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };
        let mut h0 = HSlots::from_broadcast(&s_r, &self.config);
        for k in 0..self.config.h_slots {
            for d in 0..d_r {
                h0.data[(k, d)] += self.slot_anchor[(k, d)];
            }
        }
        h0
    }

    fn step(&self, h: &HSlots, s: &DVector<f32>, _exec: Option<&mut dyn ComputeBackend>) -> HSlots {
        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;

        let h_attn = self.cross_slot_attn(h);

        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };
        let input_signal = &self.w_in * &s_r;

        let mut next = HSlots::zeros(&self.config);
        for k in 0..h_slots {
            let h_slot = h.slot(k);
            let attn_k = h_attn.slot(k);

            // v14: En el loop DEQ, solo combinamos atención e inyección de contexto.
            // La conexión residual interna ha sido purgada para forzar p(J)<1.
            let combined = attn_k + &input_signal;
            let slot_bias = self.slot_anchor.row(k).transpose().into_owned();
            let f_h = self.rms_norm(&(combined + slot_bias));
            let damped = spectral_norm::damped_update(&h_slot, &f_h, self.damping);
            next.set_slot(k, &damped);
        }
        next
    }

    fn temporal_step(&self, m_prev: &HSlots, h_star: &HSlots) -> HSlots {
        let h_slots = self.config.h_slots;
        let mut next_m = HSlots::zeros(&self.config);

        for k in 0..h_slots {
            let m_k = m_prev.slot(k);
            let h_k = h_star.slot(k);
            let h_rms = ((h_k.dot(&h_k) / h_k.len() as f32) + 1e-6).sqrt();
            let h_unit = h_k / h_rms;

            // The temporal carrier should depend on the content of H*, not on the raw amplitude
            // of the fixed point. As the DEQ learns, ||H*|| can contract substantially; using the
            // unit-RMS slot state keeps the external memory alive without changing Picard itself.
            let a_bar = DVector::from_fn(self.config.d_r, |d, _| {
                1.0 / (1.0 + self.a_log[(k, d)].exp())
            });
            let b_bar = a_bar.map(|a| 1.0 - a);
            let mut x_proj = h_unit.clone_owned();
            let wx_max = 0.5_f32;
            for i in 0..self.config.d_r {
                let wx = wx_max * self.w_x[(i, i)].tanh();
                x_proj[i] += wx * h_unit[i];
            }
            let m_k_next = a_bar.zip_map(&m_k, |a, m| a * m) + b_bar.zip_map(&x_proj, |b, x| b * x);

            // Keep an identity carrier path so the temporal memory cannot self-annihilate
            // when W_x/W_out are trainable.
            let out_k = m_k_next.clone_owned() + &self.w_out * m_k_next;
            next_m.set_slot(k, &out_k);
        }
        next_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_query(seed: f32, d_r: usize) -> DVector<f32> {
        let mut v = DVector::zeros(d_r);
        for i in 0..d_r {
            v[i] = (seed * (i + 1) as f32).sin() * 0.5;
        }
        v
    }

    #[test]
    fn step_is_sensitive_to_input() {
        let config = ArchitectureConfig::default();
        let r = MambaSlotReasoning::new(config.clone());
        let s_a = make_query(1.0, config.d_r);
        let s_b = make_query(2.0, config.d_r);
        let h0 = r.init(&s_a);

        let h_a = r.step(&h0, &s_a, None);
        let h_b = r.step(&h0, &s_b, None);

        assert_ne!(
            h_a.to_flat(),
            h_b.to_flat(),
            "step debe ser sensible al input s"
        );
    }

    #[test]
    fn step_is_stable_no_explosion() {
        let config = ArchitectureConfig::default();
        let r = MambaSlotReasoning::new(config.clone());
        let s = make_query(1.0, config.d_r);
        let mut h = r.init(&s);
        for _ in 0..30 {
            h = r.step(&h, &s, None);
        }
        let energy: f32 = h.to_flat().iter().map(|x| x * x).sum();
        assert!(energy.is_finite(), "Energy no debe explotar: {energy}");
        assert!(energy < 1e6, "Energy demasiado alta: {energy}");
    }
}
