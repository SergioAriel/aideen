use aideen_backbone::lm_head::LmHead;
use aideen_core::state::{ArchitectureConfig, HSlots};
use nalgebra::DVector;

fn make_small_config() -> ArchitectureConfig {
    let mut config = ArchitectureConfig::default();
    config.d_r = 8;
    config.h_slots = 2;
    config.vocab_size = 10;
    config
}

/// Build an LmHead with non-trivial weights for testing.
/// Uses larger scale W and non-trivial g so that gradients are numerically
/// significant in f32 finite differences.
fn make_test_head(config: &ArchitectureConfig) -> LmHead {
    let mut head = LmHead::new(config.clone());
    // Set g to non-trivial values (away from 1.0)
    for d in 0..config.d_r {
        head.g[d] = 0.5 + 0.5 * ((d as f32 * 1.234).sin());
    }
    // Scale W up slightly for more pronounced gradients
    head.w *= 3.0;
    head
}

/// Build HSlots with non-trivial values so gradients are meaningful.
fn make_test_h_star(config: &ArchitectureConfig) -> HSlots {
    let mut h = HSlots::zeros(config);
    for k in 0..config.h_slots {
        let mut slot = DVector::zeros(config.d_r);
        for d in 0..config.d_r {
            slot[d] = ((k * config.d_r + d) as f32 * 0.7123 + 0.3).sin();
        }
        h.set_slot(k, &slot);
    }
    h
}

/// Check relative error with tolerance, allowing small absolute differences.
fn check_gradient(
    name: &str,
    idx: &str,
    numerical: f32,
    analytical: f32,
    rel_tol: f32,
    abs_tol: f32,
) {
    let abs_diff = (numerical - analytical).abs();
    let denom = numerical.abs().max(analytical.abs()).max(1e-7);
    let rel_err = abs_diff / denom;

    assert!(
        rel_err < rel_tol || abs_diff < abs_tol,
        "{} grad check failed at {}: numerical={:.8}, analytical={:.8}, rel_err={:.4}, abs_diff={:.2e}",
        name,
        idx,
        numerical,
        analytical,
        rel_err,
        abs_diff
    );
}

#[test]
fn lm_head_backward_gradient_check() {
    let config = make_small_config();
    let head = make_test_head(&config);
    let h_star = make_test_h_star(&config);

    // Pick a non-argmax target so gradients are non-trivial,
    // but not a very-low-probability token (to keep loss moderate).
    let target: u32 = 3;

    // Compute analytical gradients
    let (loss, grad_h_star, grads) = head.backward(&h_star, target);

    // Verify loss matches forward_loss
    let loss_fwd = head.forward_loss(&h_star, target);
    assert!(
        (loss - loss_fwd).abs() < 1e-6,
        "backward loss ({}) != forward_loss ({})",
        loss,
        loss_fwd
    );

    // eps=1e-3 gives good f32 precision for finite differences when loss
    // is moderate (< 5) and gradients are above ~1e-4.
    let eps = 1e-3_f32;

    // ── Check gradient w.r.t. h_star (all slots, all dimensions) ──
    for k in 0..config.h_slots {
        for d in 0..config.d_r {
            let mut h_plus = h_star.clone();
            let mut slot_p = h_plus.slot(k);
            slot_p[d] += eps;
            h_plus.set_slot(k, &slot_p);

            let mut h_minus = h_star.clone();
            let mut slot_m = h_minus.slot(k);
            slot_m[d] -= eps;
            h_minus.set_slot(k, &slot_m);

            let loss_plus = head.forward_loss(&h_plus, target);
            let loss_minus = head.forward_loss(&h_minus, target);
            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let analytical = grad_h_star.slot(k)[d];

            check_gradient(
                "h_star",
                &format!("slot={}, d={}", k, d),
                numerical,
                analytical,
                0.05,
                1e-4,
            );
        }
    }

    // ── Check gradient w.r.t. b (bias) ──
    for v in 0..config.vocab_size {
        let mut head_plus = LmHead::new(config.clone());
        head_plus.w = head.w.clone();
        head_plus.g = head.g.clone();
        head_plus.b = head.b.clone();
        head_plus.b[v] += eps;

        let mut head_minus = LmHead::new(config.clone());
        head_minus.w = head.w.clone();
        head_minus.g = head.g.clone();
        head_minus.b = head.b.clone();
        head_minus.b[v] -= eps;

        let loss_plus = head_plus.forward_loss(&h_star, target);
        let loss_minus = head_minus.forward_loss(&h_star, target);
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = grads.grad_b[v];

        check_gradient(
            "b",
            &format!("v={}", v),
            numerical,
            analytical,
            0.05,
            1e-4,
        );
    }

    // ── Check gradient w.r.t. g (RMSNorm scale) ──
    for d in 0..config.d_r {
        let mut head_plus = LmHead::new(config.clone());
        head_plus.w = head.w.clone();
        head_plus.b = head.b.clone();
        head_plus.g = head.g.clone();
        head_plus.g[d] += eps;

        let mut head_minus = LmHead::new(config.clone());
        head_minus.w = head.w.clone();
        head_minus.b = head.b.clone();
        head_minus.g = head.g.clone();
        head_minus.g[d] -= eps;

        let loss_plus = head_plus.forward_loss(&h_star, target);
        let loss_minus = head_minus.forward_loss(&h_star, target);
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = grads.grad_g[d];

        check_gradient(
            "g",
            &format!("d={}", d),
            numerical,
            analytical,
            0.05,
            1e-4,
        );
    }

    // ── Check gradient w.r.t. W (sample a subset of elements) ──
    let sample_indices: Vec<(usize, usize)> = vec![
        (0, 0),
        (1, 3),
        (target as usize, 0),
        (target as usize, config.d_r / 2),
        (target as usize, config.d_r - 1),
        (config.vocab_size - 1, 0),
        (config.vocab_size / 2, config.d_r / 2),
        (3, 5),
        (5, 7),
        (8, 2),
    ];

    for &(i, j) in &sample_indices {
        if i >= config.vocab_size || j >= config.d_r {
            continue;
        }

        let mut head_plus = LmHead::new(config.clone());
        head_plus.w = head.w.clone();
        head_plus.b = head.b.clone();
        head_plus.g = head.g.clone();
        head_plus.w[(i, j)] += eps;

        let mut head_minus = LmHead::new(config.clone());
        head_minus.w = head.w.clone();
        head_minus.b = head.b.clone();
        head_minus.g = head.g.clone();
        head_minus.w[(i, j)] -= eps;

        let loss_plus = head_plus.forward_loss(&h_star, target);
        let loss_minus = head_minus.forward_loss(&h_star, target);
        let numerical = (loss_plus - loss_minus) / (2.0 * eps);
        let analytical = grads.grad_w[(i, j)];

        check_gradient(
            "W",
            &format!("({}, {})", i, j),
            numerical,
            analytical,
            0.05,
            1e-4,
        );
    }
}

#[test]
fn lm_head_forward_loss_matches_manual() {
    let config = make_small_config();
    let head = make_test_head(&config);
    let h_star = make_test_h_star(&config);
    let target: u32 = 3;

    let logits = head.forward(&h_star);
    let probs = LmHead::softmax(&logits);
    let expected_loss = -probs[target as usize].max(1e-12).ln();
    let actual_loss = head.forward_loss(&h_star, target);

    assert!(
        (expected_loss - actual_loss).abs() < 1e-6,
        "forward_loss mismatch: expected={}, actual={}",
        expected_loss,
        actual_loss
    );
}
