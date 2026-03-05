use aideen_core::compute::{ComputeBackend, TensorId};
use aideen_core::protocol::{decode_payload_zstd, ParamId, SignedUpdate};
use aideen_core::reasoning::Reasoning;
use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::{DMatrix, DVector};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::HashMap;

/// Backbone: Red Feed-Forward de 2 capas con expansión.
pub struct FfnReasoning {
    pub config: ArchitectureConfig,
    w1: DMatrix<f32>,
    b1: DVector<f32>,
    w2: DMatrix<f32>,
    b2: DVector<f32>,
    // ⚠️ Constitutional Note:
    // Interior mutability is ONLY allowed for GPU tensor handles (TensorId).
    // No learning, no weight mutation, no semantic state is stored here.
    w1_id: RefCell<Option<TensorId>>,
    w2_id: RefCell<Option<TensorId>>,
}

impl FfnReasoning {
    pub fn new(hidden_dim: usize, config: ArchitectureConfig) -> Self {
        let d_r = config.d_r;
        let w1 = DMatrix::from_fn(hidden_dim, d_r, |r, c| {
            let val = ((r * 31 + c * 17) % 256) as f32 / 256.0;
            val * 0.1 - 0.05
        });
        let b1 = DVector::zeros(hidden_dim);

        let w2 = DMatrix::from_fn(d_r, hidden_dim, |r, c| {
            let val = ((r * 19 + c * 23) % 256) as f32 / 256.0;
            val * 0.1 - 0.05
        });
        let b2 = DVector::zeros(d_r);

        Self {
            config,
            w1,
            b1,
            w2,
            b2,
            w1_id: RefCell::new(None),
            w2_id: RefCell::new(None),
        }
    }
}

impl Reasoning for FfnReasoning {
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

    fn step(
        &self,
        h: &HSlots,
        _s: &DVector<f32>,
        mut exec: Option<&mut dyn ComputeBackend>,
    ) -> HSlots {
        let mut next = HSlots::zeros(&self.config);

        // ── 1. Paso FFN por slot (independiente) ────────────────────────────
        let h_slots = self.config.h_slots;
        let d_r = self.config.d_r;

        for k in 0..h_slots {
            let h_slot = h.slot(k);
            let h_r = h_slot.rows(0, d_r).into_owned();

            let updated_r = if let Some(be) = exec.as_mut() {
                // Rama GPU (misma lógica que antes, trasladada por slot)
                if self.w1_id.borrow().is_none() {
                    *self.w1_id.borrow_mut() = be.load_tensor(self.w1.as_slice()).ok();
                }
                if self.w2_id.borrow().is_none() {
                    *self.w2_id.borrow_mut() = be.load_tensor(self.w2.as_slice()).ok();
                }

                if let (Some(w1_id), Some(w2_id)) =
                    (self.w1_id.borrow().as_ref(), self.w2_id.borrow().as_ref())
                {
                    if let Ok(i_id) = be.load_tensor(h_r.as_slice()) {
                        let out_dim_1 = self.w1.nrows();
                        if let Ok(mut h_mid) = be.ffn_forward(w1_id, &i_id, out_dim_1) {
                            h_mid += &self.b1;
                            for v in h_mid.iter_mut() {
                                *v = v.tanh();
                            }
                            if let Ok(mid_id) = be.load_tensor(h_mid.as_slice()) {
                                let out_dim_2 = self.w2.nrows();
                                if let Ok(mut h_out_r) = be.ffn_forward(w2_id, &mid_id, out_dim_2) {
                                    h_out_r += &self.b2;
                                    for v in h_out_r.iter_mut() {
                                        *v = v.tanh();
                                    }
                                    h_out_r
                                } else {
                                    cpu_ffn(&self.w1, &self.b1, &self.w2, &self.b2, &h_r)
                                }
                            } else {
                                cpu_ffn(&self.w1, &self.b1, &self.w2, &self.b2, &h_r)
                            }
                        } else {
                            cpu_ffn(&self.w1, &self.b1, &self.w2, &self.b2, &h_r)
                        }
                    } else {
                        cpu_ffn(&self.w1, &self.b1, &self.w2, &self.b2, &h_r)
                    }
                } else {
                    cpu_ffn(&self.w1, &self.b1, &self.w2, &self.b2, &h_r)
                }
            } else {
                // Rama CPU
                cpu_ffn(&self.w1, &self.b1, &self.w2, &self.b2, &h_r)
            };

            next.set_slot(k, &updated_r);
        }

        // ── 2. Slot-mixing: residual de la media ────────────────────────────
        let mean_slot: DVector<f32> = {
            let sum = (0..h_slots).fold(DVector::zeros(d_r), |acc, k| acc + next.slot(k));
            sum / (h_slots as f32)
        };
        for k in 0..h_slots {
            let mut s = next.slot(k);
            s += mean_slot.scale(0.1);
            next.set_slot(k, &s);
        }

        next
    }
}

/// FFN CPU de 2 capas: tanh(W2 · tanh(W1 · h + b1) + b2)
fn cpu_ffn(
    w1: &DMatrix<f32>,
    b1: &DVector<f32>,
    w2: &DMatrix<f32>,
    b2: &DVector<f32>,
    h: &DVector<f32>,
) -> DVector<f32> {
    let mut mid = w1 * h + b1;
    for v in mid.iter_mut() {
        *v = v.tanh();
    }
    let mut out = w2 * &mid + b2;
    for v in out.iter_mut() {
        *v = v.tanh();
    }
    out
}

pub struct ReplayGuard {
    pub last_version: HashMap<String, u64>,
    pub last_update_hash: HashMap<String, [u8; 32]>,
}

impl Default for ReplayGuard {
    fn default() -> Self {
        Self {
            last_version: HashMap::new(),
            last_update_hash: HashMap::new(),
        }
    }
}

impl FfnReasoning {
    /// Hash del "modelo base" actual (w1, w2, b1, b2).
    pub fn current_model_hash(&self) -> [u8; 32] {
        let mut h = Sha256::new();

        h.update(bytemuck::cast_slice(self.w1.as_slice()));
        h.update(bytemuck::cast_slice(self.w2.as_slice()));
        h.update(bytemuck::cast_slice(self.b1.as_slice()));
        h.update(bytemuck::cast_slice(self.b2.as_slice()));

        let out = h.finalize();
        let mut mh = [0u8; 32];
        mh.copy_from_slice(&out);
        mh
    }

    /// ÚNICA entrada pública para mutación: update firmado criptográficamente.
    pub fn apply_signed_update(
        &mut self,
        update: &SignedUpdate,
        critic_public_key: &[u8; 32],
        guard: &mut ReplayGuard,
    ) -> Result<(), String> {
        // 1) Firma
        update.verify_signature(critic_public_key)?;

        // 2) Anti-replay por version monotónica (Mas rápido de chequear)
        let last_v = guard
            .last_version
            .get(&update.target_id)
            .copied()
            .unwrap_or(0);
        if update.version <= last_v {
            return Err("replay detected: version not increasing".into());
        }

        // 3) base_model_hash debe matchear el modelo actual
        let my_hash = self.current_model_hash();
        if update.base_model_hash != my_hash {
            return Err("base_model_hash mismatch (client model != update base)".into());
        }

        // 4) Encadenado opcional (previene forks/rollbacks ocultos)
        let expected_prev = guard
            .last_update_hash
            .get(&update.target_id)
            .copied()
            .unwrap_or([0u8; 32]);

        if expected_prev != [0u8; 32] && update.prev_update_hash != expected_prev {
            return Err("prev_update_hash mismatch (fork/replay)".into());
        }

        // 5) Decode payload zstd+bincode
        let deltas: Vec<aideen_core::protocol::QuantizedDelta> =
            decode_payload_zstd(&update.payload)?;

        // 6) Aplicar deltas
        for d in deltas {
            match d.param {
                ParamId::W1 => {
                    let w = self.w1.as_mut_slice();
                    for (&i, &qi) in d.idx.iter().zip(d.q.iter()) {
                        let k = i as usize;
                        if k >= w.len() {
                            return Err("W1 delta index out of bounds".into());
                        }
                        w[k] += (qi as f32) * d.scale;
                    }
                }
                ParamId::W2 => {
                    let w = self.w2.as_mut_slice();
                    for (&i, &qi) in d.idx.iter().zip(d.q.iter()) {
                        let k = i as usize;
                        if k >= w.len() {
                            return Err("W2 delta index out of bounds".into());
                        }
                        w[k] += (qi as f32) * d.scale;
                    }
                }
                ParamId::B1 => {
                    let b = self.b1.as_mut_slice();
                    for (&i, &qi) in d.idx.iter().zip(d.q.iter()) {
                        let k = i as usize;
                        if k >= b.len() {
                            return Err("B1 delta index out of bounds".into());
                        }
                        b[k] += (qi as f32) * d.scale;
                    }
                }
                ParamId::B2 => {
                    let b = self.b2.as_mut_slice();
                    for (&i, &qi) in d.idx.iter().zip(d.q.iter()) {
                        let k = i as usize;
                        if k >= b.len() {
                            return Err("B2 delta index out of bounds".into());
                        }
                        b[k] += (qi as f32) * d.scale;
                    }
                }
            }
        }

        // 7) Invalidar descriptores GPU para que se recarguen
        *self.w1_id.borrow_mut() = None;
        *self.w2_id.borrow_mut() = None;

        // 8) Commit estado al Replay Guard
        guard
            .last_version
            .insert(update.target_id.clone(), update.version);
        guard
            .last_update_hash
            .insert(update.target_id.clone(), update.update_hash());

        Ok(())
    }

    /// Acceso de alta velocidad a los pesos para el backprop. SOLO LABORATORIO PRIVADO.
    #[cfg(feature = "lab")]
    pub fn w1_mut(&mut self) -> &mut nalgebra::DMatrix<f32> {
        &mut self.w1
    }

    /// Serializa los pesos hacia un buffer (Snapshot)
    #[cfg(feature = "lab")]
    pub fn get_weights_snapshot(&self) -> Vec<f32> {
        self.w1.as_slice().to_vec()
    }

    /// Aplica un buffer de pesos crudo. SOLO LABORATORIO PRIVADO.
    #[cfg(feature = "lab")]
    pub fn apply_weights_snapshot(&mut self, snapshot: &[f32]) {
        assert_eq!(snapshot.len(), self.w1.len());
        self.w1.as_mut_slice().copy_from_slice(snapshot);
    }
}
