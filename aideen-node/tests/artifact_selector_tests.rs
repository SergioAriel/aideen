#![cfg(not(target_arch = "wasm32"))]

use aideen_core::artifacts::{ArtifactId, ArtifactKind, ArtifactMeta};
use aideen_core::capabilities::{Accel, DeviceClass, NodeCapabilities, QuantLevel};
use aideen_node::artifacts::policy::{CompatPolicy, SelectionPolicy};
use aideen_node::artifacts::store::ArtifactStore;

fn make_caps(quant: Vec<QuantLevel>) -> NodeCapabilities {
    NodeCapabilities {
        device_class: DeviceClass::Unknown,
        cpu_threads: 4,
        ram_mb: 8192,
        accel: Accel::Unknown,
        quant_support: quant,
    }
}

fn make_artifact(target: &str, quant: QuantLevel) -> ArtifactMeta {
    ArtifactMeta {
        id: ArtifactId {
            target_id: target.to_string(),
            domain: "math".to_string(),
            version: 1,
        },
        kind: ArtifactKind::Expert,
        quant,
        size_bytes: 1024,
    }
}

// ── Test 1: CompatPolicy filtra por QuantLevel ────────────────────────────────

#[test]
fn test_compat_policy_filters_by_quant() {
    let policy = CompatPolicy;
    let caps = make_caps(vec![QuantLevel::F32]);

    let available = vec![
        make_artifact("expert_a", QuantLevel::F32),
        make_artifact("expert_b", QuantLevel::Int8), // no soportado
        make_artifact("expert_c", QuantLevel::F32),
    ];

    let selected = policy.select(&caps, &available);
    assert_eq!(selected.len(), 2, "solo F32 deben quedar");
    assert!(selected.iter().all(|a| a.quant == QuantLevel::F32));
}

#[test]
fn test_compat_policy_empty_when_no_match() {
    let policy = CompatPolicy;
    let caps = make_caps(vec![QuantLevel::F32]);
    let available = vec![make_artifact("expert_x", QuantLevel::Int8)];
    let selected = policy.select(&caps, &available);
    assert!(selected.is_empty());
}

// ── Test 2: ArtifactStore insert / get / list ─────────────────────────────────

#[test]
fn test_artifact_store_insert_get() {
    let mut store = ArtifactStore::new();
    let id = ArtifactId {
        target_id: "expert_math".to_string(),
        domain: "math".to_string(),
        version: 1,
    };

    store.insert(id.clone(), vec![1, 2, 3, 4]);
    assert_eq!(store.get(&id), Some([1u8, 2, 3, 4].as_slice()));
}

#[test]
fn test_artifact_store_list() {
    let mut store = ArtifactStore::new();
    let id_a = ArtifactId {
        target_id: "a".to_string(),
        domain: "x".to_string(),
        version: 1,
    };
    let id_b = ArtifactId {
        target_id: "b".to_string(),
        domain: "x".to_string(),
        version: 1,
    };
    store.insert(id_a.clone(), vec![]);
    store.insert(id_b.clone(), vec![]);
    let mut list: Vec<_> = store.list().into_iter().cloned().collect();
    list.sort_by(|a, b| a.target_id.cmp(&b.target_id));
    assert_eq!(list, vec![id_a, id_b]);
}

#[test]
fn test_artifact_store_missing_key() {
    let store = ArtifactStore::new();
    let id = ArtifactId {
        target_id: "ghost".to_string(),
        domain: "x".to_string(),
        version: 1,
    };
    assert!(store.get(&id).is_none());
}
