use nalgebra::DVector;

/// Dimensiones canónicas
pub const D_M: usize = 512;
pub const D_R: usize = 1024;
pub const D_C: usize = 256;
pub const D_E: usize = 256;
pub const D_SIM: usize = 512;

pub const D_GLOBAL: usize = D_M + D_R + D_C + D_E + D_SIM;

/// Estado cognitivo global
#[derive(Clone, Debug)]
pub struct State {
    pub s: DVector<f32>,
}

impl State {
    pub fn new() -> Self {
        Self {
            s: DVector::zeros(D_GLOBAL),
        }
    }

    // ── slices de solo lectura ───────────────────────────

    pub fn m(&self) -> &[f32] {
        &self.s.as_slice()[0..D_M]
    }

    pub fn r(&self) -> &[f32] {
        &self.s.as_slice()[D_M..D_M + D_R]
    }

    pub fn c(&self) -> &[f32] {
        &self.s.as_slice()[D_M + D_R..D_M + D_R + D_C]
    }

    pub fn e(&self) -> &[f32] {
        &self.s.as_slice()[D_M + D_R + D_C..D_M + D_R + D_C + D_E]
    }

    pub fn sim(&self) -> &[f32] {
        &self.s.as_slice()[D_M + D_R + D_C + D_E..D_GLOBAL]
    }

    // ── escritura controlada ────────────────────────────

    /// Inyecta delta SOLO en S_R (razonamiento integrable)
    pub fn inject_delta_r(&mut self, delta_r: &[f32]) {
        assert_eq!(delta_r.len(), D_R);
        let r_slice = &mut self.s.as_mut_slice()[D_M..D_M + D_R];
        for (ri, di) in r_slice.iter_mut().zip(delta_r.iter()) {
            *ri += di;
        }
    }

    /// Escribe estado simulado (NO integrable, descartable)
    pub fn write_sim(&mut self, sim: &[f32]) {
        assert_eq!(sim.len(), D_SIM);
        let sim_slice = &mut self.s.as_mut_slice()[D_M + D_R + D_C + D_E..D_GLOBAL];
        sim_slice.copy_from_slice(sim);
    }

    /// Limpia la simulación (opcional por tick)
    pub fn clear_sim(&mut self) {
        let sim_slice = &mut self.s.as_mut_slice()[D_M + D_R + D_C + D_E..D_GLOBAL];
        for v in sim_slice.iter_mut() {
            *v = 0.0;
        }
    }
}
