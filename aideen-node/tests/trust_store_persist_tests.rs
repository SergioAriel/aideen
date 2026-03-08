#![cfg(not(target_arch = "wasm32"))]

use aideen_node::security::trust_store::{TrustDecision, TrustStore};

#[test]
fn test_trust_store_persist_roundtrip() {
    use rand::Rng;

    let suffix: u64 = rand::thread_rng().gen();
    let path = std::env::temp_dir().join(format!("aideen_trust_test_{suffix}.bin"));

    let id = [7u8; 32];
    let fp = [0xBBu8; 32];

    // Crear, TOFU, flush
    let mut store = TrustStore::new();
    let r = store.verify_or_tofu(id, fp, None).unwrap();
    assert!(matches!(r, TrustDecision::TofuStored));
    store.flush(&path).expect("flush must succeed");

    // Load desde disco — peer debe ser reconocido
    let mut store2 = TrustStore::load(&path).expect("load must succeed");
    let r2 = store2.verify_or_tofu(id, fp, None).unwrap();
    assert!(
        matches!(r2, TrustDecision::Trusted),
        "loaded store must recognize known peer"
    );

    // Verificar que load() con path inexistente devuelve store vacío (no error)
    let store3 = TrustStore::load(std::path::Path::new("/tmp/aideen_nonexistent_path_xyz.bin"))
        .expect("nonexistent path must return empty store");
    assert!(
        store3.get(&id).is_none(),
        "fresh store must have no entries"
    );

    let _ = std::fs::remove_file(&path);
}
