use nalgebra::DVector;

/// Contrato canónico de readout: proyecta h* en un output observable.
///
/// El readout es la única salida legítima del loop LOXI.
/// No existe next_token, no existe sampling.
/// La respuesta emerge de h* — el punto fijo cognitivo.
pub trait Readout {
    type Output;
    fn readout(&self, h_star: &DVector<f32>) -> Self::Output;
}
