use nalgebra::DVector;

/// Ethics core: Non-negotiable geometric constraints
pub trait Ethics {
    /// Projects state S onto the safe manifold
    fn project(&self, s: &DVector<f32>) -> DVector<f32>;

    /// Detects if there is a direct violation (hard boolean)
    fn violates(&self, s: &DVector<f32>) -> bool;

    /// Ontological fingerprint of the system (Ethics Fingerprint)
    fn fingerprint(&self) -> [u8; 32];
}
