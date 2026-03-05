use aideen_core::readout::Readout;
use nalgebra::{DMatrix, DVector};

/// Readout lineal canónico: output = W * h* + b
///
/// Dimensiones:
///   w: [out_dim × h_dim]
///   b: [out_dim]
///
/// Para testear sin LLM: Output = DVector<f32> (embedding / scores abstractos).
/// El caller decide cómo interpretar el output (clasificación, logits, acción).
pub struct LinearReadout {
    pub w: DMatrix<f32>, // [out_dim × h_dim]
    pub b: DVector<f32>, // [out_dim]
}

impl LinearReadout {
    /// Readout identidad: out_dim == h_dim, W=I, b=0.
    /// Útil para tests y para exponer h* directamente.
    pub fn identity(dim: usize) -> Self {
        Self {
            w: DMatrix::identity(dim, dim),
            b: DVector::zeros(dim),
        }
    }

    /// Readout de proyección: out_dim != h_dim.
    pub fn new(w: DMatrix<f32>, b: DVector<f32>) -> Self {
        assert_eq!(
            w.nrows(),
            b.len(),
            "LinearReadout: w.nrows() debe coincidir con b.len()"
        );
        Self { w, b }
    }
}

impl Readout for LinearReadout {
    type Output = DVector<f32>;

    fn readout(&self, h_star: &DVector<f32>) -> Self::Output {
        &self.w * h_star + &self.b
    }
}
