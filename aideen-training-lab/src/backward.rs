/// Backward pass utilities for DEQ unrolled training.
///
/// This module implements "unrolled differentiation": we treat N Picard iterations
/// as N explicit layers and backpropagate through them, accumulating parameter
/// gradients at each layer.

use aideen_backbone::lm_head::{LmHead, LmHeadGrads};
use aideen_backbone::mamba_slot_reasoning::{MambaSlotReasoning, StepGrads};
use aideen_core::reasoning::Reasoning;
use aideen_core::state::HSlots;
use nalgebra::{DMatrix, DVector};

/// Accumulated gradients from the full unrolled backward pass.
pub struct BackwardResult {
    pub loss: f32,
    pub grad_w_q: DMatrix<f32>,
    pub grad_w_k: DMatrix<f32>,
    pub grad_w_v: DMatrix<f32>,
    pub grad_w_o: DMatrix<f32>,
    pub grad_w_in: DMatrix<f32>,
    pub grad_norm_scale: DVector<f32>,
    pub grad_slot_anchor: DMatrix<f32>,
    pub grad_s: DVector<f32>,
    pub grad_lm: LmHeadGrads,
}

/// Run forward through `max_iters` Picard iterations, then backpropagate
/// through all of them (unrolled differentiation) to compute parameter gradients.
///
/// # Arguments
/// * `reasoning` - The MambaSlotReasoning block (the DEQ operator f)
/// * `lm_head`   - The language model head (projects H* to logits)
/// * `s`         - The global state vector (input to the DEQ)
/// * `target`    - The target token id for cross-entropy loss
/// * `max_iters` - Number of Picard iterations to unroll
///
/// # Returns
/// A `BackwardResult` containing the loss, all parameter gradients, and the
/// LmHead gradients.
pub fn unrolled_backward(
    reasoning: &MambaSlotReasoning,
    lm_head: &LmHead,
    s: &DVector<f32>,
    target: u32,
    max_iters: usize,
) -> BackwardResult {
    let d_r = reasoning.config.d_r;
    let h_slots = reasoning.config.h_slots;

    // ── 1. Forward pass: save all intermediate H states ──────────────────
    // h_states[0] = H_0 (from init), h_states[i] = H after i-th Picard step
    let mut h_states: Vec<HSlots> = Vec::with_capacity(max_iters + 1);
    h_states.push(reasoning.init(s));

    for i in 0..max_iters {
        let h_next = reasoning.step(&h_states[i], s, None);
        h_states.push(h_next);
    }

    // ── 2. Compute loss and dL/dH* via lm_head.backward() ───────────────
    let h_star = &h_states[max_iters];
    let (loss, grad_h_star, grad_lm) = lm_head.backward(h_star, target);

    // ── 3. Backprop through Picard iterations in reverse order ───────────
    // dl_dh is the gradient flowing backward; starts as dL/dH*
    let mut dl_dh = grad_h_star;

    // Accumulated parameter gradients (summed over all iterations)
    let mut grad_w_q = DMatrix::zeros(d_r, d_r);
    let mut grad_w_k = DMatrix::zeros(d_r, d_r);
    let mut grad_w_v = DMatrix::zeros(d_r, d_r);
    let mut grad_w_o = DMatrix::zeros(d_r, d_r);
    let mut grad_w_in = DMatrix::zeros(d_r, d_r);
    let mut grad_norm_scale = DVector::zeros(d_r);
    let mut grad_slot_anchor = DMatrix::zeros(h_slots, d_r);
    let mut grad_s = DVector::zeros(d_r);

    for i in (0..max_iters).rev() {
        // Backprop through iteration i: H_{i+1} = step(H_i, s)
        let step_grads: StepGrads = reasoning.step_backward(&h_states[i], s, &dl_dh);

        // Accumulate parameter gradients
        grad_w_q += &step_grads.grad_w_q;
        grad_w_k += &step_grads.grad_w_k;
        grad_w_v += &step_grads.grad_w_v;
        grad_w_o += &step_grads.grad_w_o;
        grad_w_in += &step_grads.grad_w_in;
        grad_norm_scale += &step_grads.grad_norm_scale;
        grad_slot_anchor += &step_grads.grad_slot_anchor;
        grad_s += &step_grads.grad_s;

        // Propagate gradient to previous iteration
        dl_dh = step_grads.grad_h_prev;
    }

    BackwardResult {
        loss,
        grad_w_q,
        grad_w_k,
        grad_w_v,
        grad_w_o,
        grad_w_in,
        grad_norm_scale,
        grad_slot_anchor,
        grad_s,
        grad_lm,
    }
}
