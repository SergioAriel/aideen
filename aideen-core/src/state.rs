use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// Configuración de la arquitectura y ejecución de una instancia de AIDEEN.
/// Permite que el motor sea dinámico y se adapte al archivo cargado.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArchitectureConfig {
    pub d_m: usize,
    pub d_r: usize,
    pub d_c: usize,
    pub d_e: usize,
    pub d_sim: usize,
    pub h_slots: usize,
    pub vocab_size: usize,

    // -- Parámetros de Ejecución / Diseño --
    /// Ventana de contexto máxima.
    pub ctx_len: usize,
    /// Máximo de iteraciones DEQ.
    pub max_deq_iters: usize,
    /// Epsilon de convergencia DEQ.
    pub deq_epsilon: f32,
    /// Iteraciones del Gradiente Conjugado (Retropropagación Implícita).
    pub cg_iters: usize,
    /// ¿Entrenar el núcleo DEQ?
    pub train_deq: bool,
    /// Escala del gradiente DEQ.
    pub deq_grad_scale: f32,
    /// Renormalización espectral cada N steps.
    pub renorm_every_steps: usize,
    /// Número de muestras para Sampled Softmax.
    pub num_samples: usize,
    /// Factor de penalización para evitar crecimiento de pesos en DEQ.
    pub weight_decay: f32,
}

impl Default for ArchitectureConfig {
    fn default() -> Self {
        Self {
            d_m: 1024, // Cerebro robusto
            d_r: 1024, // Espacio de equilibrio
            d_c: 256,
            d_e: 256,
            d_sim: 1024,
            h_slots: 4,        // 4 Slots = Alta velocidad + Razonamiento paralelo
            vocab_size: 50257, // DEBE coincidir con tu tokenizer.json
            ctx_len: 256,      // Ventana de memoria para chat
            max_deq_iters: 1,
            deq_epsilon: 1e-4,
            cg_iters: 6, // Precisión de gradiente estable
            train_deq: true,
            deq_grad_scale: 0.1,
            renorm_every_steps: 50, // Mantiene el radio espectral a raya
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

/// Estado de razonamiento iterativo: K slots × D_R floats.
/// H*[k] es el vector de razonamiento del k-ésimo slot.
#[derive(Clone, Debug)]
pub struct HSlots {
    pub data: DMatrix<f32>,
    pub slots: usize,
    pub d_r: usize,
}

impl HSlots {
    /// Inicializa todos los slots a cero usando la configuración.
    pub fn zeros(config: &ArchitectureConfig) -> Self {
        HSlots {
            data: DMatrix::zeros(config.h_slots, config.d_r),
            slots: config.h_slots,
            d_r: config.d_r,
        }
    }

    /// Construye HSlots copiando el mismo vector s en los K slots.
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

    /// Extrae un slot como DVector.
    pub fn slot(&self, k: usize) -> DVector<f32> {
        self.data.row(k).transpose()
    }

    /// Escribe un DVector en el slot k.
    pub fn set_slot(&mut self, k: usize, v: &DVector<f32>) {
        self.data.row_mut(k).copy_from(&v.transpose());
    }

    /// Aplana para envío por red.
    pub fn to_flat(&self) -> Vec<f32> {
        self.data.as_slice().to_vec()
    }

    /// Reconstruye desde bytes aplanados.
    pub fn from_flat(flat: &[f32], config: &ArchitectureConfig) -> Self {
        assert_eq!(flat.len(), config.h_slots * config.d_r);
        HSlots {
            data: DMatrix::from_row_slice(config.h_slots, config.d_r, flat),
            slots: config.h_slots,
            d_r: config.d_r,
        }
    }
}

/// Estado cognitivo global
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

    /// Inyecta delta SOLO en S_R
    pub fn inject_delta_r(&mut self, delta_r: &[f32]) {
        assert_eq!(delta_r.len(), self.config.d_r);
        let start = self.config.d_m;
        let r_slice = &mut self.s.as_mut_slice()[start..start + self.config.d_r];
        for (ri, di) in r_slice.iter_mut().zip(delta_r.iter()) {
            *ri += di;
        }
    }

    /// Escribe estado simulado
    pub fn write_sim(&mut self, sim: &[f32]) {
        assert_eq!(sim.len(), self.config.d_sim);
        let start = self.config.d_m + self.config.d_r + self.config.d_c + self.config.d_e;
        let sim_slice = &mut self.s.as_mut_slice()[start..start + self.config.d_sim];
        sim_slice.copy_from_slice(sim);
    }

    /// Limpia la simulación
    pub fn clear_sim(&mut self) {
        let start = self.config.d_m + self.config.d_r + self.config.d_c + self.config.d_e;
        let sim_slice = &mut self.s.as_mut_slice()[start..start + self.config.d_sim];
        for v in sim_slice.iter_mut() {
            *v = 0.0;
        }
    }

    /// Extrae S_R como DVector.
    pub fn r_vec(&self) -> DVector<f32> {
        DVector::from_row_slice(self.r())
    }
}
