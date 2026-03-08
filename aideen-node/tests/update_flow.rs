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
    // 0) Configuración de llaves (Simula Laboratorio Privado vs Nodo Público)
    let (root_sk, root_pk) = generate_keys();
    let (critic_sk, critic_pk) = generate_keys();

    // 1) Configuración del modelo y backbone
    let config = ArchitectureConfig::default();
    let mut ffn = FfnReasoning::new(64, config);
    let base_hash = ffn.current_model_hash();

    // 2) Configuración del UpdateManager público (conoce ROOT_PK)
    let mut mgr = UpdateManager::new(root_pk);

    // Creamos un delta válido pero determinista
    let delta = QuantizedDelta {
        param: ParamId::W1,
        scale: 0.1,
        idx: vec![0],
        q: vec![1],
    };
    let payload = encode_payload_zstd(&vec![delta]).expect("zstd encode");

    // Construir la mutación oficial firmada por Critic
    let upd = create_signed_update(&critic_sk, 1, "test_target", base_hash, payload.clone());

    // --- TEST A: Actualizar sin Delegación debería fallar ---
    let err = mgr.apply_update_bytes(&mut ffn, &bincode::serialize(&upd).unwrap());
    assert!(
        err.is_err(),
        "Debe fallar si no hay validación de clave Critic por defecto"
    );

    // --- TEST B: Delegar la llave Critic correctamente ---
    let delegation = create_valid_delegation(&root_sk, critic_pk, 1);
    mgr.apply_delegation(&delegation)
        .expect("Delegation válida");

    assert_eq!(mgr.critic_pk().unwrap(), critic_pk);

    // --- TEST C: Aplicar el update con delegación debería tener éxito ---
    let upd_bytes = bincode::serialize(&upd).unwrap();
    mgr.apply_update_bytes(&mut ffn, &upd_bytes)
        .expect("Aplicación de Update Exitoso");

    // --- TEST D: Re-aplicar el mismo update debe bloquearse (ReplayGuard) ---
    let err_replay = mgr.apply_update_bytes(&mut ffn, &upd_bytes);
    assert!(
        err_replay.unwrap_err().contains("replay detected"),
        "No debe permitir updates con la misma versión"
    );

    // --- TEST E: Actualización alterada (Payload manipulado, Firma inválida) ---
    // Clonamos update exitoso (usamos base_hash modificado simulando un avance correcto pero corrompiendo payload)
    let mut bad_upd = upd.clone();
    bad_upd.version = 2;
    bad_upd.payload[0] ^= 0x01; // corrompemos archivo zstd
    let bad_bytes = bincode::serialize(&bad_upd).unwrap();
    let err_bad = mgr.apply_update_bytes(&mut ffn, &bad_bytes);
    assert!(
        err_bad.is_err(),
        "Debe bloquear si se modifica el payload sin re-firmar / o si el zstd se corrompe"
    );

    // --- TEST F: Actualizar modelo equivocado (Base hash match) ---
    let mut wrong_base =
        create_signed_update(&critic_sk, 3, "test_target", [9u8; 32], payload.clone());
    let wrong_bytes = bincode::serialize(&wrong_base).unwrap();
    let err_wrong = mgr.apply_update_bytes(&mut ffn, &wrong_bytes);
    assert!(
        err_wrong.unwrap_err().contains("base_model_hash mismatch"),
        "Debe fallar si el modelo en el cliente divergió de la topología base del update."
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
    assert!(err.is_err(), "Debe bloquear epoch < current");

    let del3 = create_valid_delegation(&root_sk, critic_pk2, 11);
    mgr.apply_delegation(&del3).expect("Delegation epoch 11 OK");
}
