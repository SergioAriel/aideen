use aideen_core::agent::{AgentEvent, AgentStore, InMemoryAgentStore, NullAgentStore};

// ── Test 1 ────────────────────────────────────────────────────────────────

/// NullAgentStore.get_pref siempre devuelve None.
#[test]
fn test_null_store_get_returns_none() {
    let store = NullAgentStore;
    assert!(store.get_pref("language").is_none());
    assert!(store.get_pref("any_key").is_none());
}

// ── Test 2 ────────────────────────────────────────────────────────────────

/// AgentEvent::TickAttractor sobrevive un round-trip bincode.
#[test]
fn test_agent_event_serializes_bincode() {
    let event = AgentEvent::TickAttractor {
        q_total: 0.72,
        iters: 7,
        stop: 0,
        h_star_hash: [0xab; 32],
        unix_ts: 1_700_000_000,
    };

    let bytes = bincode::serialize(&event).expect("serialización debe funcionar");
    let decoded: AgentEvent = bincode::deserialize(&bytes).expect("deserialización debe funcionar");
    assert_eq!(event, decoded);
}

// ── Test 3 ────────────────────────────────────────────────────────────────

/// NullAgentStore.recent_events siempre devuelve [].
#[test]
fn test_null_store_recent_empty() {
    let mut store = NullAgentStore;
    let ev = AgentEvent::PreferenceSet {
        key: "lang".into(),
        value: "es".into(),
        unix_ts: 0,
    };
    store.append_event(ev).unwrap();
    assert!(store.recent_events(10).is_empty());
}

// ── Test 4 ────────────────────────────────────────────────────────────────

/// InMemoryAgentStore: set_pref + get_pref round-trip.
#[test]
fn test_in_memory_pref_round_trip() {
    let mut store = InMemoryAgentStore::new();
    store.set_pref("language", "es".into()).unwrap();
    assert_eq!(store.get_pref("language").as_deref(), Some("es"));
    assert!(store.get_pref("nonexistent").is_none());
}

// ── Test 5 ────────────────────────────────────────────────────────────────

/// InMemoryAgentStore: append_event + recent_events en orden inverso.
#[test]
fn test_in_memory_events_recent_order() {
    let mut store = InMemoryAgentStore::new();

    for i in 0u64..5 {
        store
            .append_event(AgentEvent::TickAttractor {
                q_total: 0.6 + i as f32 * 0.01,
                iters: i as u32,
                stop: 0,
                h_star_hash: [0u8; 32],
                unix_ts: i,
            })
            .unwrap();
    }

    let recent = store.recent_events(3);
    assert_eq!(recent.len(), 3);

    // Reverse order: most recent (unix_ts=4) must be first
    if let AgentEvent::TickAttractor { unix_ts, .. } = &recent[0] {
        assert_eq!(*unix_ts, 4, "first result must be the most recent event");
    } else {
        panic!("expected TickAttractor");
    }
}
