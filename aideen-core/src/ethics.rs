use nalgebra::DVector;

/// Núcleo ético: Restricciones geométricas no negociables
pub trait Ethics {
    /// Proyecta el estado S al manifold seguro
    fn project(&self, s: &DVector<f32>) -> DVector<f32>;

    /// Detecta si hay violación directa (booleano duro)
    fn violates(&self, s: &DVector<f32>) -> bool;

    /// Huella ontológica del sistema (Ethics Fingerprint)
    fn fingerprint(&self) -> [u8; 32];
}
