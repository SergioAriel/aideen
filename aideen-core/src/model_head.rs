use crate::state::HSlots;

/// `ModelHead` — contract for any head on H*.
///
/// A head takes the DEQ attractor H* and produces an observable output.
/// It is the only legitimate output gate of the Loxi loop.
///
/// ## Available heads in aideen-backbone
/// - `LmHead`    → text tokens (autoregressive generation)
/// - `EmbedHead` → embedding vector (D_R dims)
/// - `ClassHead` → predicted class (usize)
///
/// ## Design
/// The contract is generic over `Output` so that each head can
/// return the appropriate type without boxing or dyn overhead.
pub trait ModelHead {
    type Output;

    /// Project H* to the output space.
    fn forward(&self, h_star: &HSlots) -> Self::Output;
}
