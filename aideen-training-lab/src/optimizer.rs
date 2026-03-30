//! Optimizador Adam para pesos nalgebra DMatrix/DVector.

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// Adam optimizer con first/second moment tracking.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    // Moments by parameter name
    m_mat: HashMap<String, DMatrix<f32>>,
    v_mat: HashMap<String, DMatrix<f32>>,
    m_vec: HashMap<String, DVector<f32>>,
    v_vec: HashMap<String, DVector<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m_mat: HashMap::new(),
            v_mat: HashMap::new(),
            m_vec: HashMap::new(),
            v_vec: HashMap::new(),
            t: 0,
        }
    }

    /// Increment the step counter (call once per train_step).
    pub fn tick(&mut self) {
        self.t += 1;
    }

    pub fn step_count(&self) -> usize {
        self.t
    }

    // ── Accessors for synchronising GPU ↔ CPU moments ─────────────────────
    pub fn set_mat(&mut self, key: &str, val: DMatrix<f32>) {
        self.m_mat.insert(key.to_string(), val);
    }
    pub fn get_mat(&self, key: &str) -> Option<&DMatrix<f32>> {
        self.m_mat.get(key)
    }
    pub fn set_vec(&mut self, key: &str, val: DVector<f32>) {
        self.m_vec.insert(key.to_string(), val);
    }
    pub fn get_vec(&self, key: &str) -> Option<&DVector<f32>> {
        self.m_vec.get(key)
    }

    /// Updates a DMatrix in-place with Adam.
    pub fn step_matrix(&mut self, name: &str, w: &mut DMatrix<f32>, grad: &DMatrix<f32>) {
        let key = name.to_string();
        let (nrows, ncols) = (w.nrows(), w.ncols());

        let m = self
            .m_mat
            .entry(key.clone())
            .or_insert_with(|| DMatrix::zeros(nrows, ncols));
        let v = self
            .v_mat
            .entry(key)
            .or_insert_with(|| DMatrix::zeros(nrows, ncols));

        // m = β₁·m + (1-β₁)·grad
        *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
        // v = β₂·v + (1-β₂)·grad²
        *v = &*v * self.beta2 + grad.map(|g| g * g) * (1.0 - self.beta2);

        // Bias correction
        let t = self.t.max(1) as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        // w -= lr · m_hat / (sqrt(v_hat) + eps)
        for (w_i, (m_i, v_i)) in w.iter_mut().zip(m.iter().zip(v.iter())) {
            let m_hat = m_i / bc1;
            let v_hat = v_i / bc2;
            *w_i -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }

    /// Updates a DVector in-place with Adam.
    pub fn step_vector(&mut self, name: &str, w: &mut DVector<f32>, grad: &DVector<f32>) {
        let key = name.to_string();
        let len = w.len();

        let m = self
            .m_vec
            .entry(key.clone())
            .or_insert_with(|| DVector::zeros(len));
        let v = self.v_vec.entry(key).or_insert_with(|| DVector::zeros(len));

        *m = &*m * self.beta1 + grad * (1.0 - self.beta1);
        *v = &*v * self.beta2 + grad.map(|g| g * g) * (1.0 - self.beta2);

        let t = self.t.max(1) as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (w_i, (m_i, v_i)) in w.iter_mut().zip(m.iter().zip(v.iter())) {
            let m_hat = m_i / bc1;
            let v_hat = v_i / bc2;
            *w_i -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

impl Adam {
    // ─────────────────────────────────────────────────────────────────────
    // Checkpointing — binary serialization without extra dependencies
    //
    // File format (.opt):
    //   [0..8]   magic "AIDENOPT"
    //   [8..16]  step count (u64 LE)
    //   then N entries:
    //     [u32 LE] key length in bytes
    //     [u8 * key_len] key UTF-8
    //     [u8]  type: 0 = vec (DVector/DMatrix flattened), 1 = mat
    //     [u32 LE] number of f32 elements
    //     [f32 * n LE] data
    // ─────────────────────────────────────────────────────────────────────

    pub fn save_state(&self, path: &str) -> Result<(), String> {
        use std::io::Write;

        let mut f = std::fs::File::create(path)
            .map_err(|e| format!("save_state create '{}': {}", path, e))?;

        f.write_all(b"AIDENOPT").map_err(|e| e.to_string())?;
        f.write_all(&(self.t as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Helper closure para escribir un tensor plano.
        let write_entry =
            |f: &mut std::fs::File, prefix: &str, key: &str, data: &[f32]| -> Result<(), String> {
                let full_key = format!("{prefix}.{key}");
                let key_bytes = full_key.as_bytes();
                f.write_all(&(key_bytes.len() as u32).to_le_bytes())
                    .map_err(|e| e.to_string())?;
                f.write_all(key_bytes).map_err(|e| e.to_string())?;
                f.write_all(&(data.len() as u32).to_le_bytes())
                    .map_err(|e| e.to_string())?;
                f.write_all(bytemuck::cast_slice(data))
                    .map_err(|e| e.to_string())
            };

        for (k, v) in &self.m_vec {
            write_entry(&mut f, "m_vec", k, v.as_slice())?;
        }
        for (k, v) in &self.v_vec {
            write_entry(&mut f, "v_vec", k, v.as_slice())?;
        }
        for (k, m) in &self.m_mat {
            write_entry(&mut f, "m_mat", k, m.as_slice())?;
        }
        for (k, m) in &self.v_mat {
            write_entry(&mut f, "v_mat", k, m.as_slice())?;
        }

        Ok(())
    }

    pub fn load_state(&mut self, path: &str) -> Result<(), String> {
        use std::io::Read;

        let mut data = Vec::new();
        std::fs::File::open(path)
            .map_err(|e| format!("load_state open '{}': {}", path, e))?
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;

        if data.len() < 16 || &data[0..8] != b"AIDENOPT" {
            return Err("Archivo .opt inválido o corrupto".to_string());
        }

        self.t = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;

        let mut pos = 16usize;
        while pos + 8 <= data.len() {
            let key_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + key_len + 4 > data.len() {
                break;
            }
            let full_key = std::str::from_utf8(&data[pos..pos + key_len])
                .map_err(|e| e.to_string())?
                .to_string();
            pos += key_len;
            let n_elems = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + n_elems * 4 > data.len() {
                break;
            }
            let floats: Vec<f32> = data[pos..pos + n_elems * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            pos += n_elems * 4;

            if let Some(rest) = full_key.strip_prefix("m_vec.") {
                self.m_vec
                    .insert(rest.to_string(), DVector::from_vec(floats));
            } else if let Some(rest) = full_key.strip_prefix("v_vec.") {
                self.v_vec
                    .insert(rest.to_string(), DVector::from_vec(floats));
            } else if let Some(rest) = full_key.strip_prefix("m_mat.") {
                let side = (floats.len() as f64).sqrt() as usize;
                let mat = if side * side == floats.len() {
                    DMatrix::from_column_slice(side, side, &floats)
                } else {
                    // Matriz no cuadrada (p.ej. d_r × vocab): almacenar como n×1
                    DMatrix::from_column_slice(floats.len(), 1, &floats)
                };
                self.m_mat.insert(rest.to_string(), mat);
            } else if let Some(rest) = full_key.strip_prefix("v_mat.") {
                let side = (floats.len() as f64).sqrt() as usize;
                let mat = if side * side == floats.len() {
                    DMatrix::from_column_slice(side, side, &floats)
                } else {
                    DMatrix::from_column_slice(floats.len(), 1, &floats)
                };
                self.v_mat.insert(rest.to_string(), mat);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_reduces_simple_quadratic() {
        // Minimizar f(w) = ||w - target||² con Adam
        let target = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut w = DVector::zeros(3);
        let mut opt = Adam::new(0.1);

        for _ in 0..100 {
            opt.tick();
            let grad = &w - &target; // df/dw = 2(w - target), ignoramos el 2
            opt.step_vector("w", &mut w, &grad);
        }

        let dist = (&w - &target).norm();
        assert!(dist < 0.1, "Adam debería converger, dist={dist}");
    }
}
