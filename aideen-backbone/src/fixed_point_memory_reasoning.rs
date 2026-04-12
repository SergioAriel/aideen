use crate::{deq_mode::DeqRuntimeConfig, spectral_norm};
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

/// FixedPointMemoryReasoning — el bloque `f` real del DEQ.
pub struct FixedPointMemoryReasoning {
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

    // ── Fixed-Point Memory SSM por slot ───────────────────────────────────────────────────
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

    // ── Memory write gate prior ───────────────────────────────────────────────
    // z[s] = σ(b_write[s] + dot(W_write[s,:], c) / √d)
    // Controls how much new slot memory may be written from the current token state.
    #[cfg(feature = "lab")]
    pub w_write_gate: DMatrix<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) w_write_gate: DMatrix<f32>,
    #[cfg(feature = "lab")]
    pub b_write_mem: DVector<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) b_write_mem: DVector<f32>,

    // ── Retain gate (low-rank, r=32) ─────────────────────────────────────────
    // retain[s,d] = σ(W_down[s] · (W_up[s] · c) + b_retain[s,d])
    // m_next = (I + W_out) * (a * m_prev + (1 - a) * (h_unit + W_x h_unit + write))
    // replaces uniform fatigue decay with input-dependent selective forgetting
    // W_up: [h_slots × d_r × r], W_down: [h_slots × r × d_r], b_retain: [h_slots × d_r]
    #[cfg(feature = "lab")]
    pub w_retain_up: Vec<DMatrix<f32>>, // h_slots matrices of shape (d_r, r)
    #[cfg(not(feature = "lab"))]
    pub(crate) w_retain_up: Vec<DMatrix<f32>>,
    #[cfg(feature = "lab")]
    pub w_retain_down: Vec<DMatrix<f32>>, // h_slots matrices of shape (r, d_r)
    #[cfg(not(feature = "lab"))]
    pub(crate) w_retain_down: Vec<DMatrix<f32>>,
    #[cfg(feature = "lab")]
    pub b_retain: DMatrix<f32>, // [h_slots × d_r]
    #[cfg(not(feature = "lab"))]
    pub(crate) b_retain: DMatrix<f32>,
    // ── Memory read projections (low-rank, r=32) ───────────────────────────
    // q_mem[s] = W_q_mem[s] · h_s, k_mem[s] = W_k_mem[s] · m_s
    // Used for cross-slot relational memory read over M_{t-1}.
    #[cfg(feature = "lab")]
    pub w_q_mem: Vec<DMatrix<f32>>, // h_slots matrices of shape (d_r, r)
    #[cfg(not(feature = "lab"))]
    pub(crate) w_q_mem: Vec<DMatrix<f32>>,
    #[cfg(feature = "lab")]
    pub w_k_mem: Vec<DMatrix<f32>>, // h_slots matrices of shape (d_r, r)
    #[cfg(not(feature = "lab"))]
    pub(crate) w_k_mem: Vec<DMatrix<f32>>,
    #[cfg(feature = "lab")]
    pub b_read_mem: DVector<f32>, // [h_slots], slot-wise read bias for FPM memory attention
    #[cfg(not(feature = "lab"))]
    pub(crate) b_read_mem: DVector<f32>,

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

impl FixedPointMemoryReasoning {
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
            // Equal-norm deterministic slot code: break permutation symmetry without giving
            // edge slots larger amplitude than middle slots.
            let theta = std::f32::consts::TAU * (slot as f32) / (h_slots.max(1) as f32);
            let phi0 = 0.061 * dim as f32;
            let phi1 = 0.113 * dim as f32;
            1.0e-3f32 * (theta + phi0).sin() + 1.0e-3f32 * (theta + phi1).cos()
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
        let slot_coord_mode = DeqRuntimeConfig::from_env().has_explicit_slot_comparison();
        // Xavier Uniform initialization: range = sqrt(3 / fan_in)
        let xavier_range = (3.0 / d_r as f32).sqrt();
        // Attention query/key scale: much smaller than Xavier to guarantee near-zero QK^T logits
        // at initialization. With xavier_range≈0.077 and d=512, random QK^T scores have
        // variance that can place α[s,s] as low as ~3e-5 (measured), killing gradient flow.
        // Scale 0.001 * xavier_range gives |QK^T/√d| < 1e-6 → softmax within 1% of uniform.
        // Wv/Wo are NOT scaled down — they don't gate gradients through the softmax Jacobian.
        let attn_qk_range = xavier_range * 0.001_f32;

        let mut w_q = Self::xavier_mat_with_rng(rng, d_r, attn_qk_range);
        let mut w_k = if slot_coord_mode {
            // In slot coordination we want Q and K to start from the same global geometry so
            // their per-slot specialization can become mutually legible in the logits.
            let mut aligned = w_q.clone();
            let noise = 0.05 * attn_qk_range;
            for r in 0..d_r {
                for c in 0..d_r {
                    aligned[(r, c)] += rng.gen_range(-noise..noise);
                }
            }
            aligned
        } else {
            Self::xavier_mat_with_rng(rng, d_r, attn_qk_range)
        };
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
        let deq_threshold = if slot_coord_mode { 0.30_f32 } else { 0.10_f32 };
        let win_threshold = 0.30_f32;
        let n_iter = 20;
        spectral_norm::normalize_if_needed(&mut w_q, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_k, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_v, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_o, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_in, win_threshold, n_iter);

        let hist_slot_scale = DMatrix::from_fn(h_slots, d_r, |slot, _| {
            // Exact zero left the multiplicative history branch effectively dormant in
            // practice: checkpoints kept hist_slot_scale≈0 and history contributed almost
            // nothing even when the rest of the temporal path was alive. A tiny slot-specific
            // seed preserves the "near-off" invariant while restoring a learnable signal path.
            let centered = slot as f32 - (h_slots.saturating_sub(1) as f32 * 0.5);
            centered * 5.0e-5f32 + rng.gen_range(-1.0e-4f32..1.0e-4f32)
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
        let mem_read_xavier = (3.0f32 / d_r.max(1) as f32).sqrt();
        let mem_read_align_noise = 0.10 * mem_read_xavier;
        let w_q_mem: Vec<DMatrix<f32>> = (0..h_slots)
            .map(|_| {
                DMatrix::from_fn(d_r, 32, |_, _| {
                    rng.gen_range(-mem_read_xavier..mem_read_xavier)
                })
            })
            .collect();
        let w_k_mem: Vec<DMatrix<f32>> = w_q_mem
            .iter()
            .map(|base| {
                DMatrix::from_fn(d_r, 32, |i, j| {
                    base[(i, j)] + rng.gen_range(-mem_read_align_noise..mem_read_align_noise)
                })
            })
            .collect();
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
            // Fixed-Point Memory-style timescale prior: decay spread a ∈ [0.80, 0.999] across slots.
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
            // Write gate prior: start conservative so cold-start memory does not immediately
            // saturate writes before the slot has learned what is worth storing.
            w_write_gate: DMatrix::zeros(h_slots, d_r),
            b_write_mem: DVector::from_fn(h_slots, |slot, _| {
                let centered = slot as f32 - (h_slots.saturating_sub(1) as f32 * 0.5);
                centered * 0.02
            }),
            // Retain gate (low-rank r=32):
            // W_up: small random init (d_r × r), W_down: zeros (r × d_r)
            // → retain = σ(0 + b_retain) = σ(2.0) ≈ 0.88 at init.
            // This keeps retention as the prior, but avoids nearly freezing write budget
            // before the low-rank gate has learned any slot-dependent structure.
            w_retain_up: (0..h_slots)
                .map(|_| DMatrix::from_fn(d_r, 32, |_, _| rng.gen_range(-0.01_f32..0.01_f32)))
                .collect(),
            w_retain_down: (0..h_slots).map(|_| DMatrix::zeros(32, d_r)).collect(),
            b_retain: DMatrix::from_element(h_slots, d_r, 2.0_f32),
            // Memory read projections: q/k start from the same base geometry so the content
            // read has an immediately legible dot-product space, instead of two unrelated
            // random subspaces that make the first reads effectively arbitrary.
            w_q_mem,
            w_k_mem,
            b_read_mem: DVector::from_fn(h_slots, |slot, _| {
                let centered = slot as f32 - (h_slots.saturating_sub(1) as f32 * 0.5);
                centered * 0.02 + rng.gen_range(-5.0e-4f32..5.0e-4f32)
            }),
            norm_scale: DVector::from_element(d_r, 1.0_f32),
            // Q/K per-slot bias: break permutation symmetry with a *shared slot code*,
            // not two independent random tables. If q_bias and k_bias are unrelated,
            // q(slot_i) has no initial geometric reason to prefer k(slot_i) over k(slot_j),
            // so slot attention can stay near-uniform and starve Q/K specialization.
            q_bias: {
                let mut slot_code = DMatrix::zeros(h_slots, d_r);
                for slot in 0..h_slots {
                    let theta = std::f32::consts::TAU * (slot as f32) / (h_slots.max(1) as f32);
                    for dim in 0..d_r {
                        let phi0 = 0.047 * dim as f32;
                        let phi1 = 0.089 * dim as f32;
                        slot_code[(slot, dim)] =
                            0.08_f32 * (theta + phi0).sin()
                            + 0.08_f32 * (theta + phi1).cos()
                            + rng.gen_range(-0.005_f32..0.005_f32);
                    }
                }
                slot_code
            },
            k_bias: {
                let mut slot_code = DMatrix::zeros(h_slots, d_r);
                for slot in 0..h_slots {
                    let theta = std::f32::consts::TAU * (slot as f32) / (h_slots.max(1) as f32);
                    for dim in 0..d_r {
                        let phi0 = 0.047 * dim as f32;
                        let phi1 = 0.089 * dim as f32;
                        slot_code[(slot, dim)] =
                            0.08_f32 * (theta + phi0).sin()
                            + 0.08_f32 * (theta + phi1).cos()
                            + rng.gen_range(-0.005_f32..0.005_f32);
                    }
                }
                slot_code
            },
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
    /// We replicate the same learned matrix for every slot; slot-specific structure must come
    /// from explicit slot parameters and learned dynamics, not from upload-time perturbations.
    /// Shader accesses slot s: W_v[s*d*d + j*d + d_out].
    pub fn w_v_gpu_flat(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let base = self.w_v.as_slice(); // column-major, d*d floats
        let mut v = Vec::with_capacity(h * d * d);
        for _ in 0..h {
            v.extend_from_slice(base);
        }
        v
    }

    /// GPU flat layout for W_o: [h_slots × d_r×d_r matrices].
    /// We keep W_o identical across slots at upload time; slot-specific behavior should come
    /// from learned per-slot parameters and states, not from a hidden upload-time perturbation.
    /// Shader accesses slot s: W_o[s*d*d + j*d + d_out].
    pub fn w_o_gpu_flat(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let slot_coord_mode = DeqRuntimeConfig::from_env().has_explicit_slot_comparison();
        let attn_t = if slot_coord_mode { 0.30_f32 } else { 0.10_f32 };
        let n_iter = 20;
        let mut v = Vec::with_capacity(h * d * d);
        for _ in 0..h {
            let mut mat = self.w_o.clone();
            spectral_norm::normalize_if_needed(&mut mat, attn_t, n_iter);
            v.extend_from_slice(mat.as_slice());
        }
        v
    }

    /// GPU flat layout for W_q: [h_slots × d_r×d_r matrices | h_slots×d_r q_bias (row-major)].
    /// Each slot matrix starts from the same learned base; slot identity comes from q_bias,
    /// slot_anchor, and learned per-slot GPU updates, not from hidden upload-time noise.
    /// Shader accesses slot s matrix: W_q[s*d*d + j*d + d_out].
    /// Bias: W_q[h_slots*d*d + s*d + d_out].
    pub fn w_q_gpu_flat(&self) -> Vec<f32> {
        let d = self.config.d_r;
        let h = self.config.h_slots;
        let base = self.w_q.as_slice(); // column-major, d*d floats
        let mut v = Vec::with_capacity(h * d * d + h * d);
        for _ in 0..h {
            v.extend_from_slice(base);
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
        for _ in 0..h {
            v.extend_from_slice(base);
        }
        v.extend_from_slice(&Self::matrix_to_row_major(&self.k_bias));
        v
    }

    pub fn renormalize_weights(&mut self) {
        // Umbral 0.10 para matrices de atención — necesario para mantener σ(J_attn) < 1
        // durante el entrenamiento (no solo en la inicialización).
        // w_x y w_out (Fixed-Point Memory externo) usan umbral 0.70 — no afectan la contractividad del DEQ.
        let slot_coord_mode = DeqRuntimeConfig::from_env().has_explicit_slot_comparison();
        let attn_t = if slot_coord_mode { 0.30_f32 } else { 0.10_f32 };
        let win_t = 0.30_f32;
        let fpm_t = 0.70_f32;
        let n = 20;
        spectral_norm::normalize_if_needed(&mut self.w_q, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_k, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_v, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_o, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_in, win_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_hist_shared, 1.5, n);
        spectral_norm::normalize_if_needed(&mut self.w_x, fpm_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_out, fpm_t, n);
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
            "reasoning.w_write_gate".to_string(),
            Self::matrix_to_row_major(&self.w_write_gate),
        );
        weights.insert(
            "reasoning.b_write_mem".to_string(),
            self.b_write_mem.as_slice().to_vec(),
        );
        weights.insert(
            "reasoning.w_retain_up".to_string(),
            self.w_retain_up
                .iter()
                .flat_map(Self::matrix_to_row_major)
                .collect(),
        );
        weights.insert(
            "reasoning.w_retain_down".to_string(),
            self.w_retain_down
                .iter()
                .flat_map(Self::matrix_to_row_major)
                .collect(),
        );
        weights.insert(
            "reasoning.b_retain".to_string(),
            Self::matrix_to_row_major(&self.b_retain),
        );
        weights.insert(
            "reasoning.w_q_mem".to_string(),
            self.w_q_mem
                .iter()
                .flat_map(Self::matrix_to_row_major)
                .collect(),
        );
        weights.insert(
            "reasoning.w_k_mem".to_string(),
            self.w_k_mem
                .iter()
                .flat_map(Self::matrix_to_row_major)
                .collect(),
        );
        weights.insert(
            "reasoning.b_read_mem".to_string(),
            self.b_read_mem.as_slice().to_vec(),
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
        Vec<f32>, // w_hist_shared
        Vec<f32>, // hist_slot_scale
        Vec<f32>, // hist_slot_bias
        Vec<f32>, // hist_gate_logit
        Vec<f32>, // slot_anchor
        Vec<f32>, // w_delta
        Vec<f32>, // b_delta
        Vec<f32>, // w_gate_hist
        Vec<f32>, // w_write_gate
        Vec<f32>, // b_write_mem
        Vec<f32>, // w_retain_up  (h_slots × d_r × r, row-major per slot)
        Vec<f32>, // w_retain_down (h_slots × r × d_r, row-major per slot)
        Vec<f32>, // b_retain (h_slots × d_r, row-major)
        Vec<f32>, // w_q_mem (h_slots × d_r × r, row-major per slot)
        Vec<f32>, // w_k_mem (h_slots × d_r × r, row-major per slot)
        Vec<f32>, // b_read_mem (h_slots)
    ) {
        let w_retain_up_flat: Vec<f32> = self
            .w_retain_up
            .iter()
            .flat_map(|m| Self::matrix_to_row_major(m))
            .collect();
        let w_retain_down_flat: Vec<f32> = self
            .w_retain_down
            .iter()
            .flat_map(|m| Self::matrix_to_row_major(m))
            .collect();
        let w_q_mem_flat: Vec<f32> = self
            .w_q_mem
            .iter()
            .flat_map(|m| Self::matrix_to_row_major(m))
            .collect();
        let w_k_mem_flat: Vec<f32> = self
            .w_k_mem
            .iter()
            .flat_map(|m| Self::matrix_to_row_major(m))
            .collect();
        (
            Self::matrix_to_row_major(&self.w_hist_shared),
            Self::matrix_to_row_major(&self.hist_slot_scale),
            Self::matrix_to_row_major(&self.hist_slot_bias),
            self.hist_gate_logit.as_slice().to_vec(),
            Self::matrix_to_row_major(&self.slot_anchor),
            Self::matrix_to_row_major(&self.w_delta),
            self.b_delta.as_slice().to_vec(),
            Self::matrix_to_row_major(&self.w_gate_hist),
            Self::matrix_to_row_major(&self.w_write_gate),
            self.b_write_mem.as_slice().to_vec(),
            w_retain_up_flat,
            w_retain_down_flat,
            Self::matrix_to_row_major(&self.b_retain),
            w_q_mem_flat,
            w_k_mem_flat,
            self.b_read_mem.as_slice().to_vec(),
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
        // Write gate params: prefer dedicated names; fall back to legacy forget-gate keys
        // from earlier memory experiments for checkpoint compatibility.
        {
            let h_slots = self.config.h_slots;
            if let Some(data) = weights.get("reasoning.w_write_gate") {
                if data.len() == h_slots * d_r {
                    self.w_write_gate = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
                }
            } else if let Some(data) = weights.get("reasoning.w_forget") {
                if data.len() == h_slots * d_r {
                    self.w_write_gate = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
                }
            }
            if let Some(data) = weights.get("reasoning.b_write_mem") {
                if data.len() == h_slots {
                    self.b_write_mem = nalgebra::DVector::from_vec(data.clone());
                }
            } else if let Some(data) = weights.get("reasoning.b_forget") {
                if data.len() == h_slots {
                    self.b_write_mem = nalgebra::DVector::from_vec(data.clone());
                }
            }
        }
        {
            let h_slots = self.config.h_slots;
            const RETAIN_RANK: usize = 32;
            if let Some(data) = weights.get("reasoning.w_retain_up") {
                let expected = h_slots * d_r * RETAIN_RANK;
                if data.len() != expected {
                    return Err(format!(
                        "reasoning.w_retain_up size mismatch: expected {}, got {}",
                        expected,
                        data.len()
                    ));
                }
                self.w_retain_up = data
                    .chunks_exact(d_r * RETAIN_RANK)
                    .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, RETAIN_RANK, chunk))
                    .collect();
            }
            if let Some(data) = weights.get("reasoning.w_retain_down") {
                let expected = h_slots * RETAIN_RANK * d_r;
                if data.len() != expected {
                    return Err(format!(
                        "reasoning.w_retain_down size mismatch: expected {}, got {}",
                        expected,
                        data.len()
                    ));
                }
                self.w_retain_down = data
                    .chunks_exact(RETAIN_RANK * d_r)
                    .map(|chunk| nalgebra::DMatrix::from_row_slice(RETAIN_RANK, d_r, chunk))
                    .collect();
            }
            if let Some(data) = weights.get("reasoning.b_retain") {
                if data.len() != h_slots * d_r {
                    return Err(format!(
                        "reasoning.b_retain size mismatch: expected {}, got {}",
                        h_slots * d_r,
                        data.len()
                    ));
                }
                self.b_retain = nalgebra::DMatrix::from_row_slice(h_slots, d_r, data);
            }
            if let Some(data) = weights.get("reasoning.w_q_mem") {
                let expected = h_slots * d_r * RETAIN_RANK;
                if data.len() != expected {
                    return Err(format!(
                        "reasoning.w_q_mem size mismatch: expected {}, got {}",
                        expected,
                        data.len()
                    ));
                }
                self.w_q_mem = data
                    .chunks_exact(d_r * RETAIN_RANK)
                    .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, RETAIN_RANK, chunk))
                    .collect();
            }
            if let Some(data) = weights.get("reasoning.w_k_mem") {
                let expected = h_slots * d_r * RETAIN_RANK;
                if data.len() != expected {
                    return Err(format!(
                        "reasoning.w_k_mem size mismatch: expected {}, got {}",
                        expected,
                        data.len()
                    ));
                }
                self.w_k_mem = data
                    .chunks_exact(d_r * RETAIN_RANK)
                    .map(|chunk| nalgebra::DMatrix::from_row_slice(d_r, RETAIN_RANK, chunk))
                    .collect();
            }
            if let Some(data) = weights.get("reasoning.b_read_mem") {
                if data.len() != h_slots {
                    return Err(format!(
                        "reasoning.b_read_mem size mismatch: expected {}, got {}",
                        h_slots,
                        data.len()
                    ));
                }
                self.b_read_mem = nalgebra::DVector::from_vec(data.clone());
            } else {
                // Backward compatibility: old checkpoints reused hist_gate_logit as the
                // read-bias prior. Keep that behavior only as import fallback.
                self.b_read_mem = self.hist_gate_logit.clone();
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
            .insert("type".to_string(), "FixedPointMemoryReasoning".to_string());

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
            w_write_gate: nalgebra::DMatrix::zeros(config.h_slots, config.d_r),
            b_write_mem: nalgebra::DVector::zeros(config.h_slots),
            w_retain_up: (0..config.h_slots)
                .map(|_| nalgebra::DMatrix::zeros(config.d_r, 32))
                .collect(),
            w_retain_down: (0..config.h_slots)
                .map(|_| nalgebra::DMatrix::zeros(32, config.d_r))
                .collect(),
            b_retain: nalgebra::DMatrix::from_element(config.h_slots, config.d_r, 2.0_f32),
            w_q_mem: (0..config.h_slots)
                .map(|_| nalgebra::DMatrix::zeros(config.d_r, 32))
                .collect(),
            w_k_mem: (0..config.h_slots)
                .map(|_| nalgebra::DMatrix::zeros(config.d_r, 32))
                .collect(),
            b_read_mem: nalgebra::DVector::zeros(config.h_slots),
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

    fn project_input_signal(&self, s: &DVector<f32>) -> DVector<f32> {
        let d_r = self.config.d_r;
        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };
        &self.w_in * &s_r
    }

    fn hist_alpha_min_target() -> f32 {
        0.070
    }

    fn hist_alpha_max() -> f32 {
        0.20
    }

    fn hist_gate_alpha(&self, slot: usize) -> f32 {
        let sigma = 1.0 / (1.0 + (-self.hist_gate_logit[slot]).exp());
        Self::hist_alpha_min_target()
            + (Self::hist_alpha_max() - Self::hist_alpha_min_target()) * sigma
    }

    /// Builds the historical context injected into the DEQ as a token-fixed signal.
    ///
    /// This intentionally depends on the previous temporal carrier `m_prev` and the current
    /// token injection scale, but not on the current Picard iterate `h`. That preserves the
    /// invariant we want for stability: history participates in the fixed point as a frozen
    /// context instead of re-entering the Jacobian at every iteration.
    pub fn fixed_hist_ctx(&self, m_prev: Option<&HSlots>, s: &DVector<f32>) -> HSlots {
        let Some(m_prev) = m_prev else {
            return HSlots::zeros(&self.config);
        };

        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;
        let input_signal = self.project_input_signal(s);
        let inj_rms = (input_signal.map(|x| x * x).mean() + 1e-6).sqrt();
        let mut out = HSlots::zeros(&self.config);

        for k in 0..h_slots {
            let prev_m = m_prev.slot(k);
            let prev_rms = (prev_m.map(|x| x * x).mean() + 1e-6).sqrt();
            let prev_unit = &prev_m / prev_rms;
            let slot_scale = self.hist_slot_scale.row(k).transpose().into_owned();

            let mut u = &self.w_hist_shared * &prev_m;
            for dim in 0..d_r {
                u[dim] += slot_scale[dim] * prev_unit[dim];
            }

            let u_rms = (u.map(|x| x * x).mean() + 1e-6).sqrt();
            let scale = if u_rms > 1e-6 {
                (inj_rms / u_rms).min(1.0)
            } else {
                1.0
            };
            let alpha = self.hist_gate_alpha(k);
            out.set_slot(k, &(u * (alpha * scale)));
        }

        out
    }

    pub fn step_with_fixed_hist_ctx(
        &self,
        h: &HSlots,
        s: &DVector<f32>,
        hist_ctx: &HSlots,
        _exec: Option<&mut dyn ComputeBackend>,
    ) -> HSlots {
        let h_slots = self.config.h_slots;
        let h_attn = self.cross_slot_attn(h);
        let input_signal = self.project_input_signal(s);

        let mut next = HSlots::zeros(&self.config);
        for k in 0..h_slots {
            let h_slot = h.slot(k);
            let attn_k = h_attn.slot(k);
            let hist_k = hist_ctx.slot(k);
            let slot_bias = self.slot_anchor.row(k).transpose().into_owned();
            let combined = attn_k + &input_signal + hist_k;
            let f_h = self.rms_norm(&(combined + slot_bias));
            let damped = spectral_norm::damped_update(&h_slot, &f_h, self.damping);
            next.set_slot(k, &damped);
        }
        next
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

impl Reasoning for FixedPointMemoryReasoning {
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
        let hist_ctx = HSlots::zeros(&self.config);
        self.step_with_fixed_hist_ctx(h, s, &hist_ctx, _exec)
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
        let r = FixedPointMemoryReasoning::new(config.clone());
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
        let r = FixedPointMemoryReasoning::new(config.clone());
        let s = make_query(1.0, config.d_r);
        let mut h = r.init(&s);
        for _ in 0..30 {
            h = r.step(&h, &s, None);
        }
        let energy: f32 = h.to_flat().iter().map(|x| x * x).sum();
        assert!(energy.is_finite(), "Energy no debe explotar: {energy}");
        assert!(energy < 1e6, "Energy demasiado alta: {energy}");
    }

    #[test]
    fn fixed_hist_ctx_is_zero_without_memory() {
        let config = ArchitectureConfig::default();
        let r = FixedPointMemoryReasoning::new(config.clone());
        let s = make_query(1.0, config.d_r);
        let hist = r.fixed_hist_ctx(None, &s);
        assert!(
            hist.to_flat().iter().all(|v| v.abs() < 1e-8),
            "Sin m_prev el hist_ctx debe ser exactamente nulo"
        );
    }

    #[test]
    fn zero_fixed_hist_ctx_matches_plain_step() {
        let config = ArchitectureConfig::default();
        let r = FixedPointMemoryReasoning::new(config.clone());
        let s = make_query(1.0, config.d_r);
        let h0 = r.init(&s);
        let zero_hist = HSlots::zeros(&config);

        let plain = r.step(&h0, &s, None);
        let with_hist = r.step_with_fixed_hist_ctx(&h0, &s, &zero_hist, None);

        for (a, b) in plain.to_flat().iter().zip(with_hist.to_flat().iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "step_with_fixed_hist_ctx(0) debe coincidir con step: {a} vs {b}"
            );
        }
    }
}
