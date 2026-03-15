use aideen_backbone::mamba_slot_reasoning::MambaSlotReasoning;
use aideen_core::state::HSlots;
use nalgebra::{DMatrix, DVector};

/// Backward pass through the LM head: computes gradient w.r.t. h_pooled.
///
/// Returns (dl_dW, dl_dh) where dl_dW is the weight gradient (unused placeholder)
/// and dl_dh is the gradient w.r.t. the pooled hidden state.
pub fn lmhead_backward(
    dl_dlogits: &DVector<f32>,
    h_pooled: &DVector<f32>,
    w: &DMatrix<f32>,
    g: &DVector<f32>,
) -> (DMatrix<f32>, DVector<f32>) {
    // RMSNorm forward: h_norm = (h / rms) * g
    let mean_sq = h_pooled.map(|v| v * v).mean();
    let rms = (mean_sq + 1e-5).sqrt();
    let h_norm = h_pooled.map(|v| v / rms).component_mul(g);

    // dl_dh_norm = W^T * dl_dlogits
    let dl_dh_norm = w.transpose() * dl_dlogits;

    // dl_dW = dl_dlogits * h_norm^T (outer product)
    let dl_dw = dl_dlogits * h_norm.transpose();

    // Backprop through RMSNorm (simplified)
    let dl_dh = dl_dh_norm.component_mul(g) / rms;

    (dl_dw, dl_dh)
}

/// Compute implicit gradient through the DEQ fixed point using conjugate gradient.
///
/// Returns v: the implicit gradient vector (DVector<f32>).
pub fn deq_implicit_grad(
    _reasoning: &MambaSlotReasoning,
    _h_star: &HSlots,
    _query: &DVector<f32>,
    dl_dh: &DVector<f32>,
    _cg_iters: usize,
) -> DVector<f32> {
    // Stub: return dl_dh as a passthrough until proper CG is implemented
    dl_dh.clone()
}
