use sha2::{Digest, Sha256};

/// Deterministically assign a node to a cohort percentile bucket (0..100).
///
/// Uses sha256(node_id ‖ target_id ‖ update_id) so an attacker controlling
/// their node_id cannot choose which cohort they land in without changing identity.
pub fn assign_pct(node_id: &[u8; 32], target_id: &str, update_id: &[u8; 32]) -> u8 {
    let mut h = Sha256::new();
    h.update(node_id);
    h.update(target_id.as_bytes());
    h.update(update_id);
    let digest = h.finalize();
    digest[0] % 100
}

/// Returns true if this node is in the canary cohort for this update.
///
/// A node is in the canary cohort if `assign_pct(...)` < `canary_pct`.
pub fn is_canary(
    node_id: &[u8; 32],
    target_id: &str,
    update_id: &[u8; 32],
    canary_pct: u8,
) -> bool {
    assign_pct(node_id, target_id, update_id) < canary_pct
}
