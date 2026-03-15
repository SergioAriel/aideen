use aideen_training::schedule::CosineSchedule;

#[test]
fn warmup_starts_near_zero() {
    let sched = CosineSchedule::new(100, 1000, 1e-3, 1e-5);
    let lr = sched.lr_at(0);
    assert!(lr < 1e-4, "lr at step 0 should be < 1e-4, got {lr}");
}

#[test]
fn warmup_reaches_base_lr() {
    let sched = CosineSchedule::new(100, 1000, 1e-3, 1e-5);
    let lr = sched.lr_at(100);
    let diff = (lr - 1e-3_f32).abs();
    assert!(
        diff < 1e-6,
        "lr at warmup_steps should be ~base_lr (1e-3), got {lr} (diff {diff})"
    );
}

#[test]
fn cosine_decays_to_min_lr() {
    let sched = CosineSchedule::new(100, 1000, 1e-3, 1e-5);
    let lr = sched.lr_at(1000);
    let diff = (lr - 1e-5_f32).abs();
    assert!(
        diff < 1e-6,
        "lr at total_steps should be ~min_lr (1e-5), got {lr} (diff {diff})"
    );
}

#[test]
fn cosine_midpoint_is_between() {
    let sched = CosineSchedule::new(100, 1000, 1e-3, 1e-5);
    // Midpoint of cosine decay region: step = warmup + (total - warmup) / 2 = 100 + 450 = 550
    let mid_step = 100 + (1000 - 100) / 2;
    let lr = sched.lr_at(mid_step);
    assert!(
        lr > 1e-5 && lr < 1e-3,
        "lr at midpoint should be between min_lr and base_lr, got {lr}"
    );
    // At the exact midpoint of a cosine decay, the value should be
    // min_lr + 0.5*(base_lr - min_lr)*(1 + cos(pi*0.5)) = min_lr + 0.5*(base_lr - min_lr)
    // which equals the arithmetic mean of base_lr and min_lr.
    let expected = (1e-3_f32 + 1e-5) / 2.0;
    let diff = (lr - expected).abs();
    assert!(
        diff < 1e-5,
        "lr at midpoint should be ~{expected}, got {lr} (diff {diff})"
    );
}
