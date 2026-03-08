#![cfg(not(target_arch = "wasm32"))]

use std::time::{Duration, Instant};

use aideen_node::peers::connector::FailureState;

/// Verifica que el backoff con jitter cae dentro de la banda ±30% del base esperado.
/// Banda ±30% (más amplia que el ±20% de jitter real) para absorber overhead de test.
#[test]
fn test_breaker_backoff_increases_and_within_jitter_band() {
    let node_id = [0xAAu8; 32];
    // base_secs esperados para fail_count 1..=7: 1, 2, 4, 8, 16, 32, 60 (cap)
    let expected_bases = [1u64, 2, 4, 8, 16, 32, 60];

    for (i, &base) in expected_bases.iter().enumerate() {
        let mut s = FailureState::default();
        // Llevar al fail_count correcto simulando breakers expirados
        for _ in 0..i {
            s.record_failure(&node_id);
            s.open_until = Some(Instant::now() - Duration::from_secs(1));
        }

        let before = Instant::now();
        s.record_failure(&node_id);

        let deadline = s.open_until.expect("open_until must be set after failure");
        let delta_ms = deadline.saturating_duration_since(before).as_millis() as i64;
        let base_ms = base as i64 * 1000;
        let lo = base_ms * 70 / 100;
        let hi = base_ms * 130 / 100;

        assert!(
            delta_ms >= lo && delta_ms <= hi,
            "fail_count={}: delta_ms={} fuera de [{}, {}] (base={}s)",
            i + 1,
            delta_ms,
            lo,
            hi,
            base
        );
    }
}
