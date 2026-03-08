/// Decides when and how many replay samples to request from nodes.
///
/// Triggers when the observed Q value spikes above `q_spike_threshold × EMA`.
pub struct ReplaySampler {
    q_ema: f32,
    q_spike_threshold: f32,
    replay_n: usize,
    next_id: u64,
}

impl ReplaySampler {
    pub fn new(q_spike_threshold: f32, replay_n: usize) -> Self {
        ReplaySampler {
            q_ema: 0.0,
            q_spike_threshold,
            replay_n,
            next_id: 1,
        }
    }

    /// Update EMA with a new Q observation. Returns how many replay samples to request (0 = none).
    pub fn update_and_sample_count(&mut self, q: f32) -> usize {
        let trigger = self.q_ema > 0.01 && q > self.q_ema * self.q_spike_threshold;
        self.q_ema = if self.q_ema == 0.0 {
            q
        } else {
            0.9 * self.q_ema + 0.1 * q
        };
        if trigger {
            self.replay_n
        } else {
            0
        }
    }

    /// Allocate the next unique sample_id for a ReplayRequest.
    pub fn next_sample_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

impl Default for ReplaySampler {
    fn default() -> Self {
        Self::new(3.0, 3)
    }
}
