#![cfg(not(target_arch = "wasm32"))]

use aideen_core::protocol::NetMsg;
use aideen_node::runner::RouterStatsAccumulator;

const NODE_ID: [u8; 32] = [1u8; 32];

// ── Test 1: flush devuelve None antes de alcanzar flush_every ─────────────────

#[test]
fn test_accumulator_no_flush_before_n() {
    let mut acc = RouterStatsAccumulator::new(5);
    for _ in 0..4 {
        acc.record(0.9, Some("expert_a"));
    }
    assert!(
        acc.flush(NODE_ID).is_none(),
        "no debe flush antes de 5 ticks"
    );
}

// ── Test 2: flush devuelve Some al llegar a N y resetea ──────────────────────

#[test]
fn test_accumulator_flush_at_n_and_reset() {
    let mut acc = RouterStatsAccumulator::new(3);
    acc.record(0.8, Some("expert_a"));
    acc.record(0.9, Some("expert_b"));
    acc.record(1.0, Some("expert_a"));

    let msg = acc.flush(NODE_ID);
    assert!(msg.is_some(), "debe flush al llegar a 3 ticks");

    match msg.unwrap() {
        NetMsg::RouterStats { window_ticks, .. } => {
            assert_eq!(window_ticks, 3);
        }
        _ => panic!("expected RouterStats"),
    }

    // Después del flush, vuelve a None
    assert!(acc.flush(NODE_ID).is_none(), "debe resetear tras flush");
}

// ── Test 3: q_mean, q_min, q_max correctos ───────────────────────────────────

#[test]
fn test_accumulator_q_stats() {
    let mut acc = RouterStatsAccumulator::new(3);
    acc.record(0.2, None);
    acc.record(0.6, None);
    acc.record(1.0, None);

    match acc.flush(NODE_ID).unwrap() {
        NetMsg::RouterStats {
            q_mean,
            q_min,
            q_max,
            ..
        } => {
            assert!((q_mean - 0.6).abs() < 1e-5, "q_mean={q_mean}");
            assert!((q_min - 0.2).abs() < 1e-5, "q_min={q_min}");
            assert!((q_max - 1.0).abs() < 1e-5, "q_max={q_max}");
        }
        _ => panic!("expected RouterStats"),
    }
}

// ── Test 4: expert_hits ordenados determinísticamente ────────────────────────

#[test]
fn test_accumulator_expert_hits_sorted() {
    let mut acc = RouterStatsAccumulator::new(4);
    acc.record(0.5, Some("z_expert"));
    acc.record(0.5, Some("a_expert"));
    acc.record(0.5, Some("z_expert"));
    acc.record(0.5, Some("m_expert"));

    match acc.flush(NODE_ID).unwrap() {
        NetMsg::RouterStats { expert_hits, .. } => {
            // Verificar orden lexicográfico
            let keys: Vec<&str> = expert_hits.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(keys, vec!["a_expert", "m_expert", "z_expert"]);
            // Verificar conteos
            let z_count = expert_hits.iter().find(|(k, _)| k == "z_expert").unwrap().1;
            assert_eq!(z_count, 2);
        }
        _ => panic!("expected RouterStats"),
    }
}
