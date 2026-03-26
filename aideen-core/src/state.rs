use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Architecture and execution configuration for an AIDEEN instance.
/// Allows the engine to be dynamic and adapt to the loaded file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub d_m: usize,
    pub d_r: usize,
    pub d_c: usize,
    pub d_e: usize,
    pub d_sim: usize,
    pub h_slots: usize,
    pub vocab_size: usize,

    // -- Execution / Design Parameters --
    /// Maximum context window.
    pub ctx_len: usize,
    /// Maximum DEQ iterations.
    pub max_deq_iters: usize,
    /// DEQ convergence epsilon.
    pub deq_epsilon: f32,
    /// Adjoint Picard iterations (Implicit Backpropagation).
    pub adj_iters: usize,
    /// Train the DEQ core?
    pub train_deq: bool,
    /// DEQ gradient scale.
    pub deq_grad_scale: f32,
    /// Spectral renormalization every N steps.
    pub renorm_every_steps: usize,
    /// Number of samples for Sampled Softmax.
    pub num_samples: usize,
    /// Penalty factor to prevent weight growth in DEQ.
    pub weight_decay: f32,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            d_m: 1024, // Robust brain
            d_r: 512,  // Equilibrium space (stable on current GPU)
            d_c: 256,
            d_e: 256,
            d_sim: 1024,
            h_slots: 8,        // 8 Slots — multi-scale temporal specialization
            vocab_size: 50257, // MUST match your tokenizer.json
            ctx_len: 256,      // Memory window for chat
            max_deq_iters: 16, // v14 (Optimized after sweep: guarantees 100% conv with alpha=0)
            deq_epsilon: 1e-4,
            adj_iters: 6, // contr≈0.20 → residual error 0.20^6≈6e-5, practically exact
            train_deq: true,
            deq_grad_scale: 0.01,
            renorm_every_steps: 4, // Cada 4 steps: ~0.25x overhead vs inline, σ controlada en ventana corta
            num_samples: 512,
            weight_decay: 0.01,
        }
    }
}

impl ArchitectureConfig {
    pub fn d_global(&self) -> usize {
        self.d_m + self.d_r + self.d_c + self.d_e + self.d_sim
    }

    pub fn total_size(&self) -> usize {
        self.d_global()
    }

    pub fn d_reasoning(&self) -> usize {
        self.d_m + self.d_r + self.d_c + self.d_e
    }

    pub fn off_r(&self) -> usize {
        self.d_m
    }

    pub fn off_sim(&self) -> usize {
        self.d_reasoning()
    }
}

/// Iterative reasoning state: K slots × D_R floats.
/// H*[k] is the reasoning vector of the k-th slot.
#[derive(Clone, Debug)]
pub struct HSlots {
    pub data: DMatrix<f32>,
    pub slots: usize,
    pub d_r: usize,
}

impl HSlots {
    /// Initializes all slots to zero using the configuration.
    pub fn zeros(config: &ArchitectureConfig) -> Self {
        HSlots {
            data: DMatrix::zeros(config.h_slots, config.d_r),
            slots: config.h_slots,
            d_r: config.d_r,
        }
    }

    /// Builds HSlots by copying the same vector s into the K slots.
    pub fn from_broadcast(s: &DVector<f32>, config: &ArchitectureConfig) -> Self {
        assert_eq!(s.len(), config.d_r, "broadcast requires s.len() == d_r");
        let mut m = DMatrix::zeros(config.h_slots, config.d_r);
        for k in 0..config.h_slots {
            m.row_mut(k).copy_from(&s.transpose());
        }
        HSlots {
            data: m,
            slots: config.h_slots,
            d_r: config.d_r,
        }
    }

    /// Extracts a slot as a DVector.
    pub fn slot(&self, k: usize) -> DVector<f32> {
        self.data.row(k).transpose()
    }

    /// Writes a DVector into slot k.
    pub fn set_slot(&mut self, k: usize, v: &DVector<f32>) {
        self.data.row_mut(k).copy_from(&v.transpose());
    }

    /// Flattens for network transmission.
    pub fn to_flat(&self) -> Vec<f32> {
        // Export in row-major ([slot][d]) to match `from_flat` and GPU buffers.
        let mut out = Vec::with_capacity(self.slots * self.d_r);
        for k in 0..self.slots {
            for d in 0..self.d_r {
                out.push(self.data[(k, d)]);
            }
        }
        out
    }

    /// Reconstructs from flattened bytes.
    pub fn from_flat(flat: &[f32], config: &ArchitectureConfig) -> Self {
        assert_eq!(flat.len(), config.h_slots * config.d_r);
        HSlots {
            data: DMatrix::from_row_slice(config.h_slots, config.d_r, flat),
            slots: config.h_slots,
            d_r: config.d_r,
        }
    }
}

/// Global cognitive state
#[derive(Clone, Debug)]
pub struct State {
    pub s: DVector<f32>,
    pub config: ArchitectureConfig,
}

impl State {
    pub fn new(config: ArchitectureConfig) -> Self {
        let d_global = config.d_global();
        Self {
            s: DVector::zeros(d_global),
            config,
        }
    }

    // ── slices de solo lectura ───────────────────────────

    pub fn m(&self) -> &[f32] {
        &self.s.as_slice()[0..self.config.d_m]
    }

    pub fn r(&self) -> &[f32] {
        let start = self.config.d_m;
        let end = start + self.config.d_r;
        &self.s.as_slice()[start..end]
    }

    pub fn c(&self) -> &[f32] {
        let start = self.config.d_m + self.config.d_r;
        let end = start + self.config.d_c;
        &self.s.as_slice()[start..end]
    }

    pub fn e(&self) -> &[f32] {
        let start = self.config.d_m + self.config.d_r + self.config.d_c;
        let end = start + self.config.d_e;
        &self.s.as_slice()[start..end]
    }

    pub fn sim(&self) -> &[f32] {
        let start = self.config.d_m + self.config.d_r + self.config.d_c + self.config.d_e;
        let end = start + self.config.d_sim;
        &self.s.as_slice()[start..end]
    }

    // ── escritura controlada ────────────────────────────

    /// Injects delta ONLY into S_R
    pub fn inject_delta_r(&mut self, delta_r: &[f32]) {
        assert_eq!(delta_r.len(), self.config.d_r);
        let start = self.config.d_m;
        let r_slice = &mut self.s.as_mut_slice()[start..start + self.config.d_r];
        for (ri, di) in r_slice.iter_mut().zip(delta_r.iter()) {
            *ri += di;
        }
    }

    /// Writes simulated state
    pub fn write_sim(&mut self, sim: &[f32]) {
        assert_eq!(sim.len(), self.config.d_sim);
        let start = self.config.d_m + self.config.d_r + self.config.d_c + self.config.d_e;
        let sim_slice = &mut self.s.as_mut_slice()[start..start + self.config.d_sim];
        sim_slice.copy_from_slice(sim);
    }

    /// Clears the simulation
    pub fn clear_sim(&mut self) {
        let start = self.config.d_m + self.config.d_r + self.config.d_c + self.config.d_e;
        let sim_slice = &mut self.s.as_mut_slice()[start..start + self.config.d_sim];
        for v in sim_slice.iter_mut() {
            *v = 0.0;
        }
    }

    /// Extracts S_R as a DVector.
    pub fn r_vec(&self) -> DVector<f32> {
        DVector::from_row_slice(self.r())
    }
}
