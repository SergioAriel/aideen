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

/// Gradients from a single Picard step backward pass.
#[cfg(feature = "lab")]
pub struct StepGrads {
    /// dL/dh^(ℓ) — gradient to propagate to the previous Picard iteration
    pub grad_h_prev: HSlots,
    /// dL/dW_q: [d_r x d_r]
    pub grad_w_q: DMatrix<f32>,
    /// dL/dW_k: [d_r x d_r]
    pub grad_w_k: DMatrix<f32>,
    /// dL/dW_v: [d_r x d_r]
    pub grad_w_v: DMatrix<f32>,
    /// dL/dW_o: [d_r x d_r]
    pub grad_w_o: DMatrix<f32>,
    /// dL/dW_in: [d_r x d_r]
    pub grad_w_in: DMatrix<f32>,
    /// dL/d_norm_scale: [d_r]
    pub grad_norm_scale: DVector<f32>,
    /// dL/d_slot_anchor: [h_slots x d_r]
    pub grad_slot_anchor: DMatrix<f32>,
    /// dL/ds — gradient w.r.t. the input signal
    pub grad_s: DVector<f32>,
}

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
    pub a_log: DVector<f32>,
    #[cfg(not(feature = "lab"))]
    pub(crate) a_log: DVector<f32>,
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

    fn xavier_mat_with_rng<R: Rng + ?Sized>(rng: &mut R, d_r: usize, xavier_range: f32) -> DMatrix<f32> {
        DMatrix::from_fn(d_r, d_r, |_, _| rng.gen_range(-xavier_range..xavier_range))
    }

    fn identity_like_mat_with_rng<R: Rng + ?Sized>(rng: &mut R, d_r: usize, noise: f32) -> DMatrix<f32> {
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
            centered * 2.0e-3f32 + phase * 2.5e-4f32
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
        // The historical interface must start semantically neutral. If W_hist is non-zero at
        // initialization, the model injects a random history vector before the DEQ has learned
        // how to use it, which is exactly the failure mode we see under seeded stress: no-mamba
        // remains contractive while hist_gated enters BOOST/FAIL immediately. Zero-init keeps
        // the branch equivalent to no-mamba at step 0 while preserving non-zero gradients into
        // W_hist through g_u and the normalized carrier.
        let w_hist_shared = DMatrix::zeros(d_r, d_r);
        // The temporal carrier is applied as a residual around identity:
        //   x_proj = h_unit + W_x h_unit
        //   M_t    = m_inner + W_out m_inner
        // Therefore W_x/W_out should initialize near zero, not near identity. This keeps
        // the carrier alive by construction while still allowing learned deviations later.
        let w_x = Self::scaled_identity_like_mat_with_rng(rng, d_r, 0.0f32, 0.01f32);
        let w_out = Self::scaled_identity_like_mat_with_rng(rng, d_r, 0.0f32, 0.01f32);
        // Selective branch starts neutral: delta=0 => a_t = a_base.
        let w_delta = DMatrix::zeros(d_r, d_r);

        // DEQ requiere σ(J_f) < 1 desde la inicialización.
        // El Jacobiano compuesto de atención (W_o · softmax · W_v · W_q) puede superar 1
        // con Xavier estándar. Renormalizamos las matrices de atención a σ ≤ 0.10
        // para garantizar σ(J_attn) << 1 antes del primer token de entrenamiento.
        // residual_alpha=0.0 es necesario — incluso alpha=0.2 lleva contr→1.
        let deq_threshold = 0.10_f32;
        let n_iter = 20;
        spectral_norm::normalize_if_needed(&mut w_q, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_k, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_v, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_o, deq_threshold, n_iter);
        spectral_norm::normalize_if_needed(&mut w_in, deq_threshold, n_iter);

        let hist_slot_scale =
            DMatrix::from_fn(h_slots, d_r, |slot, _| {
                // Keep multiplicative history adaptation off at initialization. A non-zero
                // diagonal scale would create an implicit bypass from M_{t-1} into the DEQ.
                let _ = slot;
                0.0
            });
        let hist_slot_bias =
            DMatrix::from_fn(h_slots, d_r, |slot, _| {
                // Break slot permutation symmetry structurally. Without a slot-specific additive
                // anchor, all slots remain exchangeable because H0 is broadcast and the early
                // historical carrier M_{t-1} is nearly identical across slots.
                let centered = slot as f32 - (h_slots.saturating_sub(1) as f32 * 0.5);
                centered * 2.5e-3f32 + rng.gen_range(-5.0e-4f32..5.0e-4f32)
            });
        let hist_gate_logit = DVector::from_fn(h_slots, |slot, _| {
            // The historical branch is now stable enough that the next bottleneck is
            // under-learning, not contractivity. Start the gate at alpha≈0.14 with a
            // positive floor (alpha_min=0.08, alpha_max=0.28) so the interface has
            // enough energy to learn without relying on optimizer-only fixes.
            let base = -0.847_297_85_f32; // alpha = 0.14 for alpha_min=0.08, alpha_max=0.28
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
            a_log: DVector::from_element(d_r, -0.5_f32), // keep frozen in production path
            w_x,
            w_out,
            w_delta,
            b_delta: DVector::zeros(d_r),
            norm_scale: DVector::from_element(d_r, 1.0_f32),
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

    pub fn renormalize_weights(&mut self) {
        // Umbral 0.10 para matrices de atención — necesario para mantener σ(J_attn) < 1
        // durante el entrenamiento (no solo en la inicialización).
        // w_x y w_out (Mamba externo) usan umbral 0.70 — no afectan la contractividad del DEQ.
        let attn_t = 0.10_f32;
        let mamba_t = 0.70_f32;
        let n = 20;
        spectral_norm::normalize_if_needed(&mut self.w_q, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_k, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_v, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_o, attn_t, n);
        spectral_norm::normalize_if_needed(&mut self.w_in, attn_t, n);
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
        weights.insert("reasoning.w_q".to_string(), Self::matrix_to_row_major(&self.w_q));
        weights.insert("reasoning.w_k".to_string(), Self::matrix_to_row_major(&self.w_k));
        weights.insert("reasoning.w_v".to_string(), Self::matrix_to_row_major(&self.w_v));
        weights.insert("reasoning.w_o".to_string(), Self::matrix_to_row_major(&self.w_o));
        weights.insert("reasoning.w_in".to_string(), Self::matrix_to_row_major(&self.w_in));
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
        weights.insert("reasoning.w_x".to_string(), Self::matrix_to_row_major(&self.w_x));
        weights.insert("reasoning.w_out".to_string(), Self::matrix_to_row_major(&self.w_out));
        weights.insert("reasoning.w_delta".to_string(), Self::matrix_to_row_major(&self.w_delta));
        weights.insert(
            "reasoning.b_delta".to_string(),
            self.b_delta.as_slice().to_vec(),
        );
        weights.insert(
            "reasoning.a_log".to_string(),
            self.a_log.as_slice().to_vec(),
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
    ) {
        (
            Self::matrix_to_row_major(&self.w_hist_shared),
            Self::matrix_to_row_major(&self.hist_slot_scale),
            Self::matrix_to_row_major(&self.hist_slot_bias),
            self.hist_gate_logit.as_slice().to_vec(),
            Self::matrix_to_row_major(&self.slot_anchor),
            Self::matrix_to_row_major(&self.w_delta),
            self.b_delta.as_slice().to_vec(),
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
        self.w_delta = get_mat("reasoning.w_delta")?;
        self.b_delta = nalgebra::DVector::from_vec(
            weights
                .get("reasoning.b_delta")
                .ok_or("reasoning.b_delta not found")?
                .clone(),
        );

        self.a_log = nalgebra::DVector::from_vec(
            weights
                .get("reasoning.a_log")
                .ok_or("reasoning.a_log not found")?
                .clone(),
        );
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
        a_log: nalgebra::DVector<f32>,
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
            a_log,
            norm_scale,
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

    /// Backward pass through one Picard iteration (the `step` method).
    ///
    /// Given dL/dh_next (gradient of loss w.r.t. the output of this step),
    /// computes gradients w.r.t. all parameters and the input h_prev.
    ///
    /// The forward computation per slot k is:
    ///   1. attn_out = cross_slot_attn(h_prev)        — via Q, K, V, O projections
    ///   2. input_signal = W_in @ s_r
    ///   3. combined = attn_out[k] + input_signal + slot_anchor[k]
    ///   4. f_h = RMSNorm(combined, norm_scale)
    ///   5. h_next[k] = damping * f_h + (1 - damping) * h_prev[k]
    #[cfg(feature = "lab")]
    pub fn step_backward(
        &self,
        h_prev: &HSlots,
        s: &DVector<f32>,
        dl_dh_next: &HSlots,
    ) -> StepGrads {
        let d_r = self.config.d_r;
        let h_slots = self.config.h_slots;
        let scale = (d_r as f32).sqrt().recip();
        let eps = 1e-6_f32;
        let beta = self.damping;

        // Prepare s_r (same truncation / zero-pad as forward)
        let s_r = if s.len() >= d_r {
            s.rows(0, d_r).into_owned()
        } else {
            let mut v = DVector::zeros(d_r);
            v.rows_mut(0, s.len()).copy_from(&s.rows(0, s.len()));
            v
        };

        // ── Re-run forward intermediates we need for backward ────────────────
        let qs: Vec<DVector<f32>> = (0..h_slots).map(|k| &self.w_q * h_prev.slot(k)).collect();
        let ks: Vec<DVector<f32>> = (0..h_slots).map(|k| &self.w_k * h_prev.slot(k)).collect();
        let vs: Vec<DVector<f32>> = (0..h_slots).map(|k| &self.w_v * h_prev.slot(k)).collect();

        let input_signal = &self.w_in * &s_r;

        // Per-query attention intermediates
        let mut attn_weights_all: Vec<Vec<f32>> = Vec::with_capacity(h_slots);
        let mut mixed_all: Vec<DVector<f32>> = Vec::with_capacity(h_slots);
        let mut combined_all: Vec<DVector<f32>> = Vec::with_capacity(h_slots);

        for q_idx in 0..h_slots {
            let raw_scores: Vec<f32> = ks.iter().map(|k| qs[q_idx].dot(k) * scale).collect();
            let max_s = raw_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = raw_scores.iter().map(|s_| (s_ - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let attn: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

            let mixed: DVector<f32> = attn
                .iter()
                .zip(vs.iter())
                .map(|(a, v)| v * *a)
                .fold(DVector::zeros(d_r), |acc, v| acc + v);

            let attn_out = &self.w_o * &mixed;
            let slot_bias = self.slot_anchor.row(q_idx).transpose().into_owned();
            let combined = attn_out + &input_signal + slot_bias;

            attn_weights_all.push(attn);
            mixed_all.push(mixed);
            combined_all.push(combined);
        }

        // ── Initialize gradient accumulators ─────────────────────────────────
        let mut grad_h_prev = HSlots::zeros(&self.config);
        let mut grad_w_q = DMatrix::zeros(d_r, d_r);
        let mut grad_w_k = DMatrix::zeros(d_r, d_r);
        let mut grad_w_v = DMatrix::zeros(d_r, d_r);
        let mut grad_w_o = DMatrix::zeros(d_r, d_r);
        let mut grad_w_in = DMatrix::zeros(d_r, d_r);
        let mut grad_norm_scale = DVector::zeros(d_r);
        let mut grad_slot_anchor = DMatrix::zeros(h_slots, d_r);
        let mut grad_s = DVector::zeros(d_r);

        // ── Backprop per slot k ──────────────────────────────────────────────
        for k in 0..h_slots {
            let dl_dhk = dl_dh_next.slot(k);

            // Step 5 backward: h_next[k] = beta * f_h + (1 - beta) * h_prev[k]
            // dL/d_f_h = beta * dL/dh_next[k]
            let dl_df_h = &dl_dhk * beta;
            // dL/dh_prev[k] += (1 - beta) * dL/dh_next[k]
            let mut h_prev_grad_k = &dl_dhk * (1.0 - beta);

            // Step 4 backward: f_h = RMSNorm(combined, norm_scale)
            //   y_i = norm_scale_i * combined_i / rms
            //   rms = sqrt(mean(combined^2) + eps)
            let combined = &combined_all[k];
            let n = combined.len() as f32;
            let mean_sq: f32 = combined.iter().map(|v| v * v).sum::<f32>() / n;
            let rms = (mean_sq + eps).sqrt();
            let x_hat = combined / rms;

            // dL/d_norm_scale += dL/d_f_h * x_hat  (element-wise)
            grad_norm_scale += dl_df_h.component_mul(&x_hat);

            // dL/d_combined = (1/rms) * (dl_df_h . norm_scale - x_hat * mean(dl_df_h . norm_scale . x_hat))
            let dl_dy_g = dl_df_h.component_mul(&self.norm_scale);
            let mean_term: f32 = dl_dy_g.component_mul(&x_hat).sum() / n;
            let dl_dcombined = (&dl_dy_g - &x_hat * mean_term) / rms;

            // Step 3 backward: combined = attn_out + input_signal + slot_anchor[k]
            let dl_dattn_out = dl_dcombined.clone();
            let dl_dinput_signal = dl_dcombined.clone();
            // dL/d_slot_anchor[k] = dL/d_combined
            for d in 0..d_r {
                grad_slot_anchor[(k, d)] += dl_dcombined[d];
            }

            // Step 2 backward: input_signal = W_in @ s_r
            // dL/dW_in += dL/d_input_signal @ s_r^T
            grad_w_in += &dl_dinput_signal * s_r.transpose();
            // dL/ds_r += W_in^T @ dL/d_input_signal
            grad_s += self.w_in.transpose() * &dl_dinput_signal;

            // Step 1 backward: through W_o and attention
            // attn_out = W_o @ mixed
            // dL/d_mixed = W_o^T @ dL/d_attn_out
            let dl_dmixed = self.w_o.transpose() * &dl_dattn_out;
            // dL/dW_o += dL/d_attn_out @ mixed^T
            grad_w_o += &dl_dattn_out * mixed_all[k].transpose();

            // Backward through attention weighted sum: mixed = sum_j(attn[j] * v[j])
            // dL/d_attn[j] = dL/d_mixed . v[j]
            // dL/d_v[j] += attn[j] * dL/d_mixed
            let attn = &attn_weights_all[k];
            let mut dl_dattn: Vec<f32> = Vec::with_capacity(h_slots);
            for j in 0..h_slots {
                dl_dattn.push(dl_dmixed.dot(&vs[j]));
            }

            // Backward through softmax: p = softmax(scores)
            // dL/d_scores[j] = p[j] * (dL/dp[j] - sum_m(p[m] * dL/dp[m]))
            let dot_sum: f32 = attn.iter().zip(dl_dattn.iter()).map(|(a, d)| a * d).sum();
            let mut dl_dscores: Vec<f32> = Vec::with_capacity(h_slots);
            for j in 0..h_slots {
                dl_dscores.push(attn[j] * (dl_dattn[j] - dot_sum));
            }

            // Backward through scores: scores[j] = q[k] . k[j] * scale
            // dL/d_q[k] += sum_j(dL/d_scores[j] * scale * k[j])
            // dL/d_k[j] += dL/d_scores[j] * scale * q[k]
            let mut dl_dq_k = DVector::zeros(d_r);
            for j in 0..h_slots {
                let s_j = dl_dscores[j] * scale;
                dl_dq_k += &ks[j] * s_j;

                // dL/d_k[j] from this query slot k
                let dl_dk_j = &qs[k] * s_j;
                // dL/dW_k += dL/d_k[j] @ h_prev[j]^T
                grad_w_k += &dl_dk_j * h_prev.slot(j).transpose();
                // dL/dh_prev[j] += W_k^T @ dL/d_k[j]
                let contrib = self.w_k.transpose() * &dl_dk_j;
                let cur = grad_h_prev.slot(j);
                grad_h_prev.set_slot(j, &(cur + contrib));

                // dL/d_v[j] from attention weighting
                let dl_dv_j = &dl_dmixed * attn[j];
                // dL/dW_v += dL/d_v[j] @ h_prev[j]^T
                grad_w_v += &dl_dv_j * h_prev.slot(j).transpose();
                // dL/dh_prev[j] += W_v^T @ dL/d_v[j]
                let contrib_v = self.w_v.transpose() * &dl_dv_j;
                let cur = grad_h_prev.slot(j);
                grad_h_prev.set_slot(j, &(cur + contrib_v));
            }

            // dL/dW_q += dL/d_q[k] @ h_prev[k]^T
            grad_w_q += &dl_dq_k * h_prev.slot(k).transpose();
            // dL/dh_prev[k] += W_q^T @ dL/d_q[k]
            h_prev_grad_k += self.w_q.transpose() * &dl_dq_k;

            // Accumulate h_prev gradient for slot k
            let cur = grad_h_prev.slot(k);
            grad_h_prev.set_slot(k, &(cur + h_prev_grad_k));
        }

        StepGrads {
            grad_h_prev,
            grad_w_q,
            grad_w_k,
            grad_w_v,
            grad_w_o,
            grad_w_in,
            grad_norm_scale,
            grad_slot_anchor,
            grad_s,
        }
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
            let a_bar = self.a_log.map(|a| 1.0 / (1.0 + a.exp()));
            let b_bar = a_bar.map(|a| 1.0 - a);
            let x_proj = h_unit.clone_owned() + &self.w_x * h_unit;
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
