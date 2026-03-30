use aideen_backbone::ffn_reasoning::FfnReasoning;
use aideen_core::protocol::{
    encode_payload_zstd, KeyDelegation, ModelHash, ParamId, QuantizedDelta, SignedUpdate,
};
use aideen_core::state::ArchitectureConfig;
use aideen_node::update::UpdateManager;
use ed25519_dalek::{Signer, SigningKey};
use rand::rngs::OsRng;

fn generate_keys() -> (SigningKey, [u8; 32]) {
    let sk = SigningKey::generate(&mut OsRng);
    let pk = sk.verifying_key().to_bytes();
    (sk, pk)
}

fn create_valid_delegation(root_sk: &SigningKey, critic_pk: [u8; 32], epoch: u64) -> KeyDelegation {
    let mut del = KeyDelegation {
        epoch,
        critic_pk,
        valid_from_unix: 0,
        valid_to_unix: u64::MAX,
        signature_by_root: vec![],
    };
    let sig = root_sk.sign(&del.signing_bytes());
    del.signature_by_root = sig.to_bytes().to_vec();
    del
}

fn create_signed_update(
    critic_sk: &SigningKey,
    version: u64,
    target_id: &str,
    base_model_hash: ModelHash,
    payload: Vec<u8>,
) -> SignedUpdate {
    let mut upd = SignedUpdate {
        version,
        target_id: target_id.to_string(),
        bundle_version: 0,
        bundle_hash: [0u8; 32],
        base_model_hash,
        prev_update_hash: [0u8; 32],
        payload,
        signature: vec![],
    };
    let sig = critic_sk.sign(&upd.signing_bytes());
    upd.signature = sig.to_bytes().to_vec();
    upd
}

#[test]
fn update_flow_accepts_valid_update_and_rejects_replay() {
    // 0) Key setup (Simulates Private Laboratory vs Public Node)
    let (root_sk, root_pk) = generate_keys();
    let (critic_sk, critic_pk) = generate_keys();

    // 1) Model and backbone configuration
    let config = ArchitectureConfig::default();
    let mut ffn = FfnReasoning::new(64, config);
    let base_hash = ffn.current_model_hash();

    // 2) Public UpdateManager configuration (knows ROOT_PK)
    let mut mgr = UpdateManager::new(root_pk);

    // Create a valid but deterministic delta
    let delta = QuantizedDelta {
        param: ParamId::W1,
        scale: 0.1,
        idx: vec![0],
        q: vec![1],
    };
    let payload = encode_payload_zstd(&vec![delta]).expect("zstd encode");

    // Build the official mutation signed by Critic
    let upd = create_signed_update(&critic_sk, 1, "test_target", base_hash, payload.clone());

    // --- TEST A: Update without Delegation should fail ---
    let err = mgr.apply_update_bytes(&mut ffn, &bincode::serialize(&upd).unwrap());
    assert!(
        err.is_err(),
        "Must fail if there is no Critic key validation by default"
    );

    // --- TEST B: Delegate the Critic key correctly ---
    let delegation = create_valid_delegation(&root_sk, critic_pk, 1);
    mgr.apply_delegation(&delegation).expect("Valid delegation");

    assert_eq!(mgr.critic_pk().unwrap(), critic_pk);

    // --- TEST C: Applying update with delegation should succeed ---
    let upd_bytes = bincode::serialize(&upd).unwrap();
    mgr.apply_update_bytes(&mut ffn, &upd_bytes)
        .expect("Successful Update application");

    // --- TEST D: Re-applying the same update must be blocked (ReplayGuard) ---
    let err_replay = mgr.apply_update_bytes(&mut ffn, &upd_bytes);
    assert!(
        err_replay.unwrap_err().contains("replay detected"),
        "Must not allow updates with the same version"
    );

    // --- TEST E: Tampered update (Manipulated payload, Invalid signature) ---
    // Clone the successful update (use modified base_hash simulating correct advance but corrupting payload)
    let mut bad_upd = upd.clone();
    bad_upd.version = 2;
    bad_upd.payload[0] ^= 0x01; // corrupt the zstd file
    let bad_bytes = bincode::serialize(&bad_upd).unwrap();
    let err_bad = mgr.apply_update_bytes(&mut ffn, &bad_bytes);
    assert!(
        err_bad.is_err(),
        "Must block if payload is modified without re-signing / or if zstd is corrupted"
    );

    // --- TEST F: Update wrong model (Base hash match) ---
    let mut wrong_base =
        create_signed_update(&critic_sk, 3, "test_target", [9u8; 32], payload.clone());
    let wrong_bytes = bincode::serialize(&wrong_base).unwrap();
    let err_wrong = mgr.apply_update_bytes(&mut ffn, &wrong_bytes);
    assert!(
        err_wrong.unwrap_err().contains("base_model_hash mismatch"),
        "Must fail if the client model diverged from the update's base topology."
    );
}

#[test]
fn delegation_rollback_prevention() {
    let (root_sk, root_pk) = generate_keys();
    let (_, critic_pk1) = generate_keys();
    let (_, critic_pk2) = generate_keys();

    let mut mgr = UpdateManager::new(root_pk);

    let del1 = create_valid_delegation(&root_sk, critic_pk1, 10);
    mgr.apply_delegation(&del1).expect("Delegation epoch 10 OK");

    let del2 = create_valid_delegation(&root_sk, critic_pk2, 9);
    let err = mgr.apply_delegation(&del2);
    assert!(err.is_err(), "Must block epoch < current");

    let del3 = create_valid_delegation(&root_sk, critic_pk2, 11);
    mgr.apply_delegation(&del3).expect("Delegation epoch 11 OK");
}
