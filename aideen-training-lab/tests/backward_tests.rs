/// Gradient check tests for unrolled DEQ backward pass.
///
/// Uses finite differences to verify that the analytical gradients computed by
/// `step_backward` and `unrolled_backward` match numerical gradients.

use aideen_backbone::lm_head::LmHead;
use aideen_backbone::mamba_slot_reasoning::MambaSlotReasoning;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::ArchitectureConfig;
use aideen_training::backward::unrolled_backward;
use nalgebra::DVector;

fn small_config() -> ArchitectureConfig {
    ArchitectureConfig {
        d_m: 16,
        d_r: 16,
        d_c: 8,
        d_e: 8,
        d_sim: 16,
        h_slots: 2,
        vocab_size: 50,
        ctx_len: 16,
        max_deq_iters: 3,
        deq_epsilon: 1e-4,
        cg_iters: 4,
        train_deq: true,
        deq_grad_scale: 0.01,
        renorm_every_steps: 16,
        num_samples: 32,
        weight_decay: 0.01,
    }
}

/// Helper to compute loss given a MambaSlotReasoning and LmHead.
fn compute_loss(
    reasoning: &MambaSlotReasoning,
    lm_head: &LmHead,
    s: &DVector<f32>,
    target: u32,
    max_iters: usize,
) -> f32 {
    let mut h = reasoning.init(s);
    for _ in 0..max_iters {
        h = reasoning.step(&h, s, None);
    }
    lm_head.forward_loss(&h, target)
}

/// Numerically check gradient of a single scalar element in a weight matrix.
///
/// Perturbs the (row, col) element of the matrix obtained via `get_mat`,
/// computes finite-difference gradient, and compares to the analytical value.
fn check_matrix_grad_element(
    config: &ArchitectureConfig,
    seed: u64,
    target: u32,
    max_iters: usize,
    param_name: &str,
    row: usize,
    col: usize,
    analytical_grad: f32,
    eps: f32,
) {
    let s = make_s(config);

    // Compute f(x + eps)
    let mut r_plus = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_plus = make_lm_head(config);
    {
        let mat = get_mat_mut(&mut r_plus, param_name);
        mat[(row, col)] += eps;
    }
    let loss_plus = compute_loss(&r_plus, &lm_plus, &s, target, max_iters);

    // Compute f(x - eps)
    let mut r_minus = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_minus = make_lm_head(config);
    {
        let mat = get_mat_mut(&mut r_minus, param_name);
        mat[(row, col)] -= eps;
    }
    let loss_minus = compute_loss(&r_minus, &lm_minus, &s, target, max_iters);

    let numerical_grad = (loss_plus - loss_minus) / (2.0 * eps);

    // Both should be finite and non-zero (or at least the analytical one should be)
    assert!(
        numerical_grad.is_finite(),
        "{param_name}[{row},{col}]: numerical gradient is not finite: {numerical_grad}"
    );
    assert!(
        analytical_grad.is_finite(),
        "{param_name}[{row},{col}]: analytical gradient is not finite: {analytical_grad}"
    );

    // Relative error check with tolerance for deep chain numerical issues
    let abs_diff = (analytical_grad - numerical_grad).abs();
    let denom = analytical_grad.abs().max(numerical_grad.abs()).max(1e-7);
    let rel_err = abs_diff / denom;

    // Allow 15% relative tolerance — the chain is deep and f32 precision degrades
    assert!(
        rel_err < 0.15 || abs_diff < 1e-5,
        "{param_name}[{row},{col}]: analytical={analytical_grad:.6e}, numerical={numerical_grad:.6e}, rel_err={rel_err:.4} (>15%)"
    );
}

fn make_s(config: &ArchitectureConfig) -> DVector<f32> {
    let d_global = config.d_m + config.d_r + config.d_c + config.d_e + config.d_sim;
    DVector::from_fn(d_global, |i, _| ((i as f32 + 1.0) * 0.7).sin() * 0.3)
}

fn make_lm_head(config: &ArchitectureConfig) -> LmHead {
    LmHead::new(config.clone())
}

/// Get a mutable reference to a named weight matrix in MambaSlotReasoning.
fn get_mat_mut<'a>(
    r: &'a mut MambaSlotReasoning,
    name: &str,
) -> &'a mut nalgebra::DMatrix<f32> {
    match name {
        "w_q" => &mut r.w_q,
        "w_k" => &mut r.w_k,
        "w_v" => &mut r.w_v,
        "w_o" => &mut r.w_o,
        "w_in" => &mut r.w_in,
        _ => panic!("Unknown parameter: {name}"),
    }
}

#[test]
fn gradients_are_finite_and_nonzero() {
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 3;

    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);

    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);

    // Check loss is finite
    assert!(result.loss.is_finite(), "loss is not finite: {}", result.loss);
    assert!(result.loss > 0.0, "loss should be positive: {}", result.loss);

    // Check all gradient matrices are finite
    for (name, grad) in [
        ("w_q", &result.grad_w_q),
        ("w_k", &result.grad_w_k),
        ("w_v", &result.grad_w_v),
        ("w_o", &result.grad_w_o),
        ("w_in", &result.grad_w_in),
    ] {
        let all_finite = grad.iter().all(|v| v.is_finite());
        assert!(all_finite, "grad_{name} contains non-finite values");

        let has_nonzero = grad.iter().any(|v| v.abs() > 1e-15);
        assert!(has_nonzero, "grad_{name} is all zeros");
    }

    // Check norm_scale gradient
    assert!(
        result.grad_norm_scale.iter().all(|v| v.is_finite()),
        "grad_norm_scale contains non-finite values"
    );

    // Check slot_anchor gradient
    assert!(
        result.grad_slot_anchor.iter().all(|v| v.is_finite()),
        "grad_slot_anchor contains non-finite values"
    );

    // Check grad_s
    assert!(
        result.grad_s.iter().all(|v| v.is_finite()),
        "grad_s contains non-finite values"
    );
}

#[test]
fn numerical_gradient_check_w_in() {
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 3;
    let eps = 1e-3;

    // Compute analytical gradients
    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);
    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);

    // Check a few elements of w_in
    let test_positions = [(0, 0), (1, 3), (5, 2), (0, 7), (3, 3)];
    for (row, col) in test_positions {
        let analytical = result.grad_w_in[(row, col)];
        check_matrix_grad_element(
            &config, seed, target, max_iters,
            "w_in", row, col, analytical, eps,
        );
    }
}

#[test]
fn numerical_gradient_check_w_q() {
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 3;
    let eps = 1e-3;

    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);
    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);

    let test_positions = [(0, 0), (2, 1), (4, 3), (7, 7)];
    for (row, col) in test_positions {
        let analytical = result.grad_w_q[(row, col)];
        check_matrix_grad_element(
            &config, seed, target, max_iters,
            "w_q", row, col, analytical, eps,
        );
    }
}

#[test]
fn numerical_gradient_check_w_k() {
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 3;
    let eps = 1e-3;

    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);
    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);

    let test_positions = [(0, 0), (1, 5), (3, 2)];
    for (row, col) in test_positions {
        let analytical = result.grad_w_k[(row, col)];
        check_matrix_grad_element(
            &config, seed, target, max_iters,
            "w_k", row, col, analytical, eps,
        );
    }
}

#[test]
fn numerical_gradient_check_w_v() {
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 3;
    let eps = 1e-3;

    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);
    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);

    let test_positions = [(0, 0), (2, 4), (5, 1)];
    for (row, col) in test_positions {
        let analytical = result.grad_w_v[(row, col)];
        check_matrix_grad_element(
            &config, seed, target, max_iters,
            "w_v", row, col, analytical, eps,
        );
    }
}

#[test]
fn numerical_gradient_check_w_o() {
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 3;
    let eps = 1e-3;

    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);
    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);

    let test_positions = [(0, 0), (3, 6), (7, 2)];
    for (row, col) in test_positions {
        let analytical = result.grad_w_o[(row, col)];
        check_matrix_grad_element(
            &config, seed, target, max_iters,
            "w_o", row, col, analytical, eps,
        );
    }
}

#[test]
fn step_backward_single_iteration_check() {
    // Verify step_backward for a single Picard step by comparing with finite differences.
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 1; // Single iteration makes the chain shorter and more precise
    let eps = 1e-3;

    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);
    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);

    // Check w_in with a single iteration (shorter chain = better numerical accuracy)
    let test_positions = [(0, 0), (3, 5), (7, 1)];
    for (row, col) in test_positions {
        let analytical = result.grad_w_in[(row, col)];
        check_matrix_grad_element(
            &config, seed, target, max_iters,
            "w_in", row, col, analytical, eps,
        );
    }
}

#[test]
fn loss_decreases_with_gradient_step() {
    // Verify that taking a small step in the negative gradient direction decreases the loss.
    let config = small_config();
    let seed = 42u64;
    let target = 7u32;
    let max_iters = 3;
    let lr = 1e-3;

    let reasoning = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    let lm_head = make_lm_head(&config);
    let s = make_s(&config);

    let result = unrolled_backward(&reasoning, &lm_head, &s, target, max_iters);
    let loss_before = result.loss;

    // Apply a gradient descent step to w_in
    let mut r_updated = MambaSlotReasoning::new_with_seed(config.clone(), seed);
    r_updated.w_in -= &result.grad_w_in * lr;
    r_updated.w_q -= &result.grad_w_q * lr;
    r_updated.w_k -= &result.grad_w_k * lr;
    r_updated.w_v -= &result.grad_w_v * lr;
    r_updated.w_o -= &result.grad_w_o * lr;

    let loss_after = compute_loss(&r_updated, &lm_head, &s, target, max_iters);

    assert!(
        loss_after < loss_before,
        "Loss should decrease after gradient step: before={loss_before:.6}, after={loss_after:.6}"
    );
}
