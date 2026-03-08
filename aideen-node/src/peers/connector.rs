use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::peers::types::NodeId;

const BACKOFF_BASE_SECS: u64 = 1;
const BACKOFF_CAP_SECS: u64 = 60;

/// Estado de fallos de un peer — implementa circuit breaker con backoff exponencial.
///
/// Backoff: 1s, 2s, 4s, 8s, 16s, 32s, 60s (cap).
/// Scope 5M: jitter, persistencia entre reinicios.
#[derive(Default)]
pub struct FailureState {
    pub fail_count: u32,
    pub open_until: Option<Instant>,
}

impl FailureState {
    /// `true` si podemos intentar conectar (breaker cerrado o TTL expirado).
    pub fn can_try(&self) -> bool {
        match self.open_until {
            None => true,
            Some(t) => Instant::now() >= t,
        }
    }

    /// Registra un fallo y abre el breaker con backoff exponencial + jitter ±20% per-peer.
    /// Secuencia base: 1s, 2s, 4s, 8s, 16s, 32s, 60s (cap). Jitter: sha2(node_id ‖ fail_count ‖ ts/10).
    pub fn record_failure(&mut self, node_id: &NodeId) {
        self.fail_count += 1;
        let base = BACKOFF_BASE_SECS
            .saturating_mul(1u64 << self.fail_count.saturating_sub(1).min(6))
            .min(BACKOFF_CAP_SECS);
        let jitter = jitter_millis(node_id, self.fail_count, base);
        let total_ms = (base as i64 * 1000 + jitter).max(100) as u64;
        self.open_until = Some(Instant::now() + Duration::from_millis(total_ms));
    }

    /// Registra un éxito y cierra el breaker.
    pub fn record_success(&mut self) {
        self.fail_count = 0;
        self.open_until = None;
    }
}

/// Mapa de breakers por NodeId — vive en NodeRunner.
pub type PeerFailures = HashMap<NodeId, FailureState>;

/// Jitter ±20% determinista por (node_id, fail_count, time_bucket de 10s).
/// Devuelve milisegundos a sumar al base (puede ser negativo).
fn jitter_millis(node_id: &[u8; 32], fail_count: u32, base_secs: u64) -> i64 {
    use sha2::{Digest, Sha256};
    let ts_bucket = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        / 10; // bucket de 10 segundos — varía entre ventanas, estable dentro de cada una
    let mut buf = [0u8; 44]; // 32 (node_id) + 4 (fail_count) + 8 (ts_bucket)
    buf[..32].copy_from_slice(node_id);
    buf[32..36].copy_from_slice(&fail_count.to_le_bytes());
    buf[36..44].copy_from_slice(&ts_bucket.to_le_bytes());
    let h = Sha256::digest(&buf);
    let pct = (h[0] % 41) as i64 - 20; // rango -20..+20
    base_secs as i64 * 1000 * pct / 100
}
