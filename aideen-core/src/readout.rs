use nalgebra::DVector;

/// Canonical readout contract: projects h* into an observable output.
///
/// The readout is the only legitimate output of the LOXI loop.
/// There is no next_token, there is no sampling.
/// The response emerges from h* — the cognitive fixed point.
pub trait Readout {
    type Output;
    fn readout(&self, h_star: &DVector<f32>) -> Self::Output;
}
