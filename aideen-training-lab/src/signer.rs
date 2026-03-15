use aideen_core::protocol::{
    encode_payload_zstd, KeyDelegation, QuantizedDelta, SignedUpdate,
};
use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::rngs::OsRng;

/// Generate a new ed25519 keypair (signing key, verifying key bytes).
pub fn generate_master_keys() -> (SigningKey, [u8; 32]) {
    let sk = SigningKey::generate(&mut OsRng);
    let pk = VerifyingKey::from(&sk);
    (sk, pk.to_bytes())
}

/// Sign a key delegation message.
pub fn sign_key_delegation(
    root_sk: &SigningKey,
    epoch: u64,
    critic_pk: [u8; 32],
    valid_from: u64,
    valid_to: u64,
) -> Result<KeyDelegation, String> {
    use ed25519_dalek::Signer;
    let deleg = KeyDelegation {
        epoch,
        critic_pk,
        valid_from_unix: valid_from,
        valid_to_unix: valid_to,
        signature_by_root: Vec::new(),
    };
    let sig = root_sk.sign(&deleg.signing_bytes());
    Ok(KeyDelegation {
        signature_by_root: sig.to_bytes().to_vec(),
        ..deleg
    })
}

/// Sign an update message.
#[allow(clippy::too_many_arguments)]
pub fn sign_update(
    critic_sk: &SigningKey,
    bundle_version: u64,
    target_id: String,
    version: u64,
    base_model_hash: [u8; 32],
    prev_update_hash: [u8; 32],
    bundle_hash: [u8; 32],
    deltas: Vec<QuantizedDelta>,
) -> Result<SignedUpdate, String> {
    use ed25519_dalek::Signer;
    let payload = encode_payload_zstd(&deltas)?;
    let mut update = SignedUpdate {
        version,
        target_id,
        bundle_version,
        bundle_hash,
        base_model_hash,
        prev_update_hash,
        payload,
        signature: Vec::new(),
    };
    let sig = critic_sk.sign(&update.signing_bytes());
    update.signature = sig.to_bytes().to_vec();
    Ok(update)
}
