use aideen_core::protocol::{encode_payload_zstd, ModelHash, QuantizedDelta, SignedUpdate};
use ed25519_dalek::{Signer, SigningKey};
use rand::rngs::OsRng;

/// Convenience (and secret) function to generate the key pair.
/// In production, we store this SigningKey cold, off GitHub and the internet.
pub fn generate_master_keys() -> (SigningKey, [u8; 32]) {
    let sk = SigningKey::generate(&mut OsRng);
    let pk = sk.verifying_key().to_bytes();
    (sk, pk)
}

/// Creates an authorised mutation for the Mainnet
pub fn sign_update(
    sk: &SigningKey,
    version: u64,
    target_id: String,
    bundle_version: u64,
    bundle_hash: [u8; 32],
    base_model_hash: ModelHash,
    prev_update_hash: [u8; 32],
    deltas: Vec<QuantizedDelta>,
) -> Result<SignedUpdate, String> {
    // 1. Convert deltas into a ZSTD-compressed payload.
    let payload = encode_payload_zstd(&deltas)?;

    // 2. Build the mutation shell
    let mut update = SignedUpdate {
        version,
        target_id,
        bundle_version,
        bundle_hash,
        base_model_hash,
        prev_update_hash,
        payload,
        signature: vec![0; 64], // Temporary placeholder
    };

    // 3. Sign the canonical byte block
    let sig = sk.sign(&update.signing_bytes());
    update.signature = sig.to_bytes().to_vec();

    Ok(update)
}

/// Signs a key delegation using the Master Root Key.
pub fn sign_key_delegation(
    root_sk: &SigningKey,
    epoch: u64,
    critic_pk: [u8; 32],
    valid_from_unix: u64,
    valid_to_unix: u64,
) -> Result<aideen_core::protocol::KeyDelegation, String> {
    let mut del = aideen_core::protocol::KeyDelegation {
        epoch,
        critic_pk,
        valid_from_unix,
        valid_to_unix,
        signature_by_root: vec![],
    };

    let sig = root_sk.sign(&del.signing_bytes());
    del.signature_by_root = sig.to_bytes().to_vec();

    Ok(del)
}
