use crate::state::HSlots;

/// `ModelHead` — contrato para cualquier head sobre H*.
///
/// Un head toma el atractor H* del DEQ y produce una salida observable.
/// Es la única puerta de salida legítima del loop Loxi.
///
/// ## Heads disponibles en aideen-backbone
/// - `LmHead`    → tokens de texto (generación autoregresiva)
/// - `EmbedHead` → vector de embedding (D_R dims)
/// - `ClassHead` → clase predicha (usize)
///
/// ## Diseño
/// El contrato es genérico sobre `Output` para que cada head pueda
/// retornar el tipo apropiado sin boxing ni dyn overhead.
pub trait ModelHead {
    type Output;

    /// Proyectar H* al espacio de salida.
    fn forward(&self, h_star: &HSlots) -> Self::Output;
}
