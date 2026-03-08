/// Result of a consistency heuristic check on a Discovery message.
#[derive(Debug, Clone, PartialEq)]
pub enum ConsistencyVerdict {
    Ok,
    Anomaly { reason: String },
}

const Q_SPIKE_THRESHOLD: f32 = 3.0;
const Q_MAX: f32 = 1.0;

/// Lightweight consistency checks applied to each Discovery message.
///
/// Tracks per-node Q EMA and flags:
/// - Out-of-range Q values
/// - Zero-iteration reports
/// - Q spikes vs running EMA (> 3x)
#[derive(Default)]
pub struct ConsistencyChecker {
    q_ema: f32,
}

impl ConsistencyChecker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check a Discovery report. Updates internal EMA. Returns verdict.
    pub fn check_discovery(
        &mut self,
        q_total: f32,
        iters: u32,
        _context_hash: [u8; 32],
        _h_star_hash: [u8; 32],
    ) -> ConsistencyVerdict {
        if q_total < 0.0 || q_total > Q_MAX {
            return ConsistencyVerdict::Anomaly {
                reason: format!("q_total={q_total:.3} out of range [0.0, {Q_MAX}]"),
            };
        }

        if iters == 0 {
            return ConsistencyVerdict::Anomaly {
                reason: "iters=0 in discovery (no DEQ iterations reported)".into(),
            };
        }

        // Spike: q is more than Q_SPIKE_THRESHOLD × EMA
        if self.q_ema > 0.01 && q_total > self.q_ema * Q_SPIKE_THRESHOLD {
            let reason = format!(
                "q spike: q={q_total:.3} > {Q_SPIKE_THRESHOLD}x ema={:.3}",
                self.q_ema
            );
            self.q_ema = 0.9 * self.q_ema + 0.1 * q_total;
            return ConsistencyVerdict::Anomaly { reason };
        }

        self.q_ema = if self.q_ema == 0.0 {
            q_total
        } else {
            0.9 * self.q_ema + 0.1 * q_total
        };

        ConsistencyVerdict::Ok
    }
}
