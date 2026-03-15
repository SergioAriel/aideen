use aideen_training::optimizer::AdamW;

/// Minimize f(x) = x^2 using AdamW over 200 steps.
/// The gradient of f(x) = x^2 is 2x.
/// Starting from x = 5.0, AdamW should drive x close to 0.
#[test]
fn adamw_step_moves_params_toward_minimum() {
    let mut opt = AdamW::default_with_lr(0.1);
    let mut params = vec![5.0_f32];

    for _ in 0..200 {
        let grads: Vec<f32> = params.iter().map(|&x| 2.0 * x).collect();
        opt.step("x", &mut params, &grads);
    }

    assert!(
        params[0].abs() < 0.1,
        "After 200 AdamW steps on f(x)=x^2, |x| should be < 0.1 but got {}",
        params[0]
    );
}

/// With zero gradients, AdamW's decoupled weight decay should still shrink params.
#[test]
fn adamw_weight_decay_shrinks_params() {
    let wd = 0.05;
    let lr = 0.01;
    let mut opt = AdamW::new(lr, 0.9, 0.999, 1e-8, wd);
    let mut params = vec![1.0_f32, -2.0, 3.0];
    let original = params.clone();
    let grads = vec![0.0_f32; 3];

    // Run several steps with zero gradient.
    for _ in 0..10 {
        opt.step("wd_test", &mut params, &grads);
    }

    // Every parameter should have moved toward zero (shrunk in magnitude).
    for (i, (&orig, &current)) in original.iter().zip(params.iter()).enumerate() {
        assert!(
            current.abs() < orig.abs(),
            "Parameter {} should have shrunk: original={}, current={}",
            i,
            orig,
            current,
        );
    }
}

/// AdamW should maintain independent state for different named parameter groups.
#[test]
fn adamw_handles_multiple_params() {
    let mut opt = AdamW::default_with_lr(0.05);

    let mut params_a = vec![3.0_f32, -1.0];
    let mut params_b = vec![0.5_f32];

    for _ in 0..50 {
        // f(a) = a0^2 + a1^2
        let grads_a: Vec<f32> = params_a.iter().map(|&x| 2.0 * x).collect();
        opt.step("group_a", &mut params_a, &grads_a);

        // f(b) = b0^2
        let grads_b: Vec<f32> = params_b.iter().map(|&x| 2.0 * x).collect();
        opt.step("group_b", &mut params_b, &grads_b);
    }

    // Both groups should have moved toward zero.
    for (i, &p) in params_a.iter().enumerate() {
        assert!(
            p.abs() < 1.0,
            "group_a param {} should be < 1.0 but got {}",
            i,
            p
        );
    }
    assert!(
        params_b[0].abs() < 0.5,
        "group_b param 0 should be < 0.5 but got {}",
        params_b[0]
    );

    // Reset should clear state — after reset, moments start from scratch.
    opt.reset();

    // Running one more step should work fine (state re-initialised from zero).
    let grads_a: Vec<f32> = params_a.iter().map(|&x| 2.0 * x).collect();
    opt.step("group_a", &mut params_a, &grads_a);
}
