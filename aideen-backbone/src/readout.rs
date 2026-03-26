use aideen_core::readout::Readout;
use nalgebra::{DMatrix, DVector};

/// Canonical linear readout: output = W * h* + b
///
/// Dimensions:
///   w: [out_dim × h_dim]
///   b: [out_dim]
///
/// For testing without an LLM: Output = DVector<f32> (embedding / abstract scores).
/// The caller decides how to interpret the output (classification, logits, action).
pub struct LinearReadout {
    pub w: DMatrix<f32>, // [out_dim × h_dim]
    pub b: DVector<f32>, // [out_dim]
}

impl LinearReadout {
    /// Identity readout: out_dim == h_dim, W=I, b=0.
    /// Useful for tests and for exposing h* directly.
    pub fn identity(dim: usize) -> Self {
        Self {
            w: DMatrix::identity(dim, dim),
            b: DVector::zeros(dim),
        }
    }

    /// Projection readout: out_dim != h_dim.
    pub fn new(w: DMatrix<f32>, b: DVector<f32>) -> Self {
        assert_eq!(
            w.nrows(),
            b.len(),
            "LinearReadout: w.nrows() must match b.len()"
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
