use crate::spectral_norm;
use aideen_core::{
    block_backend::BlockBackend,
    compute::ComputeBackend,
    reasoning::Reasoning,
    state::{ArchitectureConfig, HSlots},
};
use nalgebra::{DMatrix, DVector};
use std::cell::RefCell;

/// MambaSlotReasoning — el bloque `f` real del DEQ.
pub struct MambaSlotReasoning {
    pub config: ArchitectureConfig,

    // ── Cross-slot attention ─────────────────────────────────────────────────
    pub(crate) w_q: DMatrix<f32>,
    pub(crate) w_k: DMatrix<f32>,
    pub(crate) w_v: DMatrix<f32>,
    pub(crate) w_o: DMatrix<f32>,

    // ── Input injection ──────────────────────────────────────────────────────
    pub(crate) w_in: DMatrix<f32>,

    // ── Mamba SSM por slot ───────────────────────────────────────────────────
    pub(crate) a_log: DVector<f32>,
    pub(crate) w_x: DMatrix<f32>,
    pub(crate) w_out: DMatrix<f32>,

    // ── LayerNorm por slot ───────────────────────────────────────────────────
    pub(crate) norm_scale: DVector<f32>,

    // ── Backend GPU opcional ─────────────────────────────────────────────────
    backend: RefCell<Option<Box<dyn BlockBackend>>>,

    // ── Picard β-relaxation ───────────────────────────────────────────────
    pub damping: f32,
}

impl MambaSlotReasoning {
    pub fn new(config: ArchitectureConfig) -> Self {
        let d_r = config.d_r;
        let scale = (d_r as f32).sqrt().recip();
        let small = 0.01_f32;

        let eye_s = DMatrix::identity(d_r, d_r) * scale;
        let eye_sm = DMatrix::identity(d_r, d_r) * small;

        Self {
            w_q: eye_s.clone(),
            w_k: eye_s.clone(),
            w_v: eye_s.clone(),
            w_o: DMatrix::identity(d_r, d_r),
            w_in: eye_sm,
            a_log: DVector::from_element(d_r, -0.5_f32),
            w_x: eye_s.clone(),
            w_out: DMatrix::identity(d_r, d_r) * 0.9,
            norm_scale: DVector::from_element(d_r, 1.0_f32),
            backend: RefCell::new(None),
            damping: 0.9_f32,
            config,
        }
    }

    pub fn with_damping(mut self, beta: f32) -> Self {
        self.damping = beta.clamp(0.01, 1.0);
        self
    }

    pub fn renormalize_weights(&mut self) {
        let t = 0.99_f32;
        let n = 10;
        self.w_q = spectral_norm::normalize(&self.w_q, t, n);
        self.w_k = spectral_norm::normalize(&self.w_k, t, n);
        self.w_v = spectral_norm::normalize(&self.w_v, t, n);
        self.w_o = spectral_norm::normalize(&self.w_o, t, n);
        self.w_in = spectral_norm::normalize(&self.w_in, t, n);
        self.w_x = spectral_norm::normalize(&self.w_x, t, n);
        self.w_out = spectral_norm::normalize(&self.w_out, t, n);
    }

    pub fn spectral_norms(&self) -> [f32; 7] {
        let n = 10;
        [
            spectral_norm::spectral_norm(&self.w_q, n),
            spectral_norm::spectral_norm(&self.w_k, n),
            spectral_norm::spectral_norm(&self.w_v, n),
            spectral_norm::spectral_norm(&self.w_o, n),
            spectral_norm::spectral_norm(&self.w_in, n),
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

    pub fn export_weights(&self) -> std::collections::HashMap<String, Vec<f32>> {
        let mut weights = std::collections::HashMap::new();

        weights.insert("reasoning.w_q".to_string(), self.w_q.as_slice().to_vec());
        weights.insert("reasoning.w_k".to_string(), self.w_k.as_slice().to_vec());
        weights.insert("reasoning.w_v".to_string(), self.w_v.as_slice().to_vec());
        weights.insert("reasoning.w_o".to_string(), self.w_o.as_slice().to_vec());
        weights.insert("reasoning.w_in".to_string(), self.w_in.as_slice().to_vec());
        weights.insert("reasoning.w_x".to_string(), self.w_x.as_slice().to_vec());
        weights.insert(
            "reasoning.w_out".to_string(),
            self.w_out.as_slice().to_vec(),
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

        weights
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
            Ok(nalgebra::DMatrix::from_vec(d_r, d_r, data.clone()))
        };

        self.w_q = get_mat("reasoning.w_q")?;
        self.w_k = get_mat("reasoning.w_k")?;
        self.w_v = get_mat("reasoning.w_v")?;
        self.w_o = get_mat("reasoning.w_o")?;
        self.w_in = get_mat("reasoning.w_in")?;
        self.w_x = get_mat("reasoning.w_x")?;
        self.w_out = get_mat("reasoning.w_out")?;

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
        w_x: nalgebra::DMatrix<f32>,
        w_out: nalgebra::DMatrix<f32>,
        a_log: nalgebra::DVector<f32>,
        norm_scale: nalgebra::DVector<f32>,
        damping: f32,
    ) -> Self {
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            w_in,
            w_x,
            w_out,
            a_log,
            norm_scale,
            backend: RefCell::new(None),
            damping,
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

    fn mamba_step(&self, h_slot: &DVector<f32>) -> DVector<f32> {
        let d_r = self.config.d_r;
        let x: Vec<f32> = (&self.w_x * h_slot).data.as_vec().clone();
        let a: Vec<f32> = self.a_log.data.as_vec().clone();
        let dt: Vec<f32> = vec![1.0_f32; d_r];
        let b: Vec<f32> = vec![1.0_f32; d_r];
        let c: Vec<f32> = vec![1.0_f32; d_r];

        let y = {
            let mut backend_ref = self.backend.borrow_mut();
            if let Some(backend) = backend_ref.as_mut() {
                match backend.mamba_batch_step(&x, &dt, &a, &b, &c) {
                    Ok(y) => DVector::from_vec(y),
                    Err(e) => {
                        eprintln!("[MambaSlotReasoning] GPU error: {e} — CPU fallback");
                        self.mamba_step_cpu(h_slot)
                    }
                }
            } else {
                self.mamba_step_cpu(h_slot)
            }
        };

        &self.w_out * y
    }

    fn mamba_step_cpu(&self, h_slot: &DVector<f32>) -> DVector<f32> {
        let a_bar = self.a_log.map(|a| a.exp());
        let b_bar = a_bar.map(|a| 1.0 - a);
        let x_proj = &self.w_x * h_slot;
        a_bar.zip_map(h_slot, |a, h| a * h) + b_bar.zip_map(&x_proj, |b, x| b * x)
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
        HSlots::from_broadcast(&s_r, &self.config)
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
            let mamba_k = self.mamba_step(&h_slot);

            let combined = attn_k + mamba_k + &input_signal + &h_slot;
            let f_h = self.rms_norm(&combined);
            let damped = spectral_norm::damped_update(&h_slot, &f_h, self.damping);
            next.set_slot(k, &damped);
        }
        next
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
