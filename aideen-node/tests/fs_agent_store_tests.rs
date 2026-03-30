#![cfg(not(target_arch = "wasm32"))]

use aideen_core::agent::{AgentEvent, AgentStore};
use aideen_node::agent::FsAgentStore;

// ── Helper ────────────────────────────────────────────────────────────────

/// Unique temporary directory per test (based on nanos to avoid collisions).
fn tmp_dir() -> std::path::PathBuf {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("aideen_test_{}", n))
}

fn sample_event(unix_ts: u64) -> AgentEvent {
    AgentEvent::TickAttractor {
        q_total: 0.72,
        iters: 5,
        stop: 0,
        h_star_hash: [0xab; 32],
        unix_ts,
    }
}

// ── Test 1 ────────────────────────────────────────────────────────────────

/// set_pref → get_pref round-trip (incluye persistencia a disco).
#[test]
fn test_fs_prefs_roundtrip() {
    let base = tmp_dir().to_string_lossy().to_string();
    let mut store = FsAgentStore::open(&base, "agent_1").unwrap();

    store.set_pref("language", "es".into()).unwrap();
    store.set_pref("timezone", "UTC-3".into()).unwrap();

    assert_eq!(store.get_pref("language").as_deref(), Some("es"));
    assert_eq!(store.get_pref("timezone").as_deref(), Some("UTC-3"));
    assert!(store.get_pref("nonexistent").is_none());
}

// ── Test 2 ────────────────────────────────────────────────────────────────

/// append 5 events → recent(3) returns 3 in reverse chronological order.
#[test]
fn test_fs_event_append_and_recent() {
    let base = tmp_dir().to_string_lossy().to_string();
    let mut store = FsAgentStore::open(&base, "agent_2").unwrap();

    for ts in 0u64..5 {
        store.append_event(sample_event(ts)).unwrap();
    }

    let recent = store.recent_events(3);
    assert_eq!(recent.len(), 3, "must return exactly 3 events");

    // Reverse order: most recent (ts=4) must be first
    if let AgentEvent::TickAttractor { unix_ts, .. } = &recent[0] {
        assert_eq!(*unix_ts, 4, "first event must be the most recent (ts=4)");
    } else {
        panic!("expected TickAttractor");
    }
    if let AgentEvent::TickAttractor { unix_ts, .. } = &recent[2] {
        assert_eq!(*unix_ts, 2, "el tercer evento debe tener ts=2");
    } else {
        panic!("expected TickAttractor");
    }
}

// ── Test 3 ────────────────────────────────────────────────────────────────

/// open → append → drop → open → recent_events devuelve el mismo evento.
#[test]
fn test_fs_store_survives_restart() {
    let base = tmp_dir().to_string_lossy().to_string();

    {
        let mut store = FsAgentStore::open(&base, "agent_3").unwrap();
        store.set_pref("key", "value_orig".into()).unwrap();
        store.append_event(sample_event(42)).unwrap();
        // drop al salir del bloque
    }

    // Reabrir — simula restart del proceso
    let store2 = FsAgentStore::open(&base, "agent_3").unwrap();
    assert_eq!(store2.get_pref("key").as_deref(), Some("value_orig"));

    let events = store2.recent_events(5);
    assert_eq!(
        events.len(),
        1,
        "debe haber exactamente 1 evento persistido"
    );
    if let AgentEvent::TickAttractor { unix_ts, .. } = &events[0] {
        assert_eq!(*unix_ts, 42);
    } else {
        panic!("se esperaba TickAttractor con ts=42");
    }
}
