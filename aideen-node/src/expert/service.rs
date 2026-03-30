use aideen_core::compute::ComputeBackend;
use aideen_core::protocol::NetMsg;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::HSlots;
use nalgebra::DVector;

/// Local expert inference service.
///
/// Receives an `ExpertTask` (flattened s_r = K×D_R floats),
/// reconstructs HSlots, runs its mini-DEQ and returns the flattened delta.
pub struct ExpertService<R: Reasoning> {
    pub reasoning: R,
    /// Maximum DEQ steps (formerly k_steps).
    pub k_max: usize,
    /// Early-stop: flat difference norm < eps_step → cut.
    pub eps_step: f32,
    /// Clamp: ||Δ_total|| > delta_cap → reescalar a delta_cap.
    pub delta_cap: f32,
}

/// stop codes (ExpertResult.stop):
/// 0 = eps_step gate (converged before k_max)
/// 1 = k_max gate (exhausted steps without converging)
/// 2 = delta_cap gate (delta clamped for being too large)

impl<R: Reasoning> ExpertService<R> {
    /// Processes a NetMsg::ExpertTask.
    /// Preserves Δ = f_i^k(H_k) − H_k starting from H0, not from init().
    pub fn process(&self, msg: &NetMsg) -> Result<NetMsg, String> {
        let (task_id, target_id, s_r) = match msg {
            NetMsg::ExpertTask {
                task_id,
                target_id,
                s_r,
                ..
            } => (*task_id, target_id.clone(), s_r),
            _ => return Err("ExpertService: expected ExpertTask".into()),
        };

        // Reconstruct H0 from flattened s_r
        let config = self.reasoning.config();
        let h0 = if s_r.len() == config.h_slots * config.d_r {
            HSlots::from_flat(s_r, config)
        } else {
            // Fallback: broadcast the first D_R if the size doesn't match
            let s = DVector::from_vec(s_r.clone());
            let d_r = config.d_r;
            HSlots::from_broadcast(&s.rows(0, s.len().min(d_r)).into_owned(), config)
        };

        // s is the projection of slot 0 as context for the step
        let s_ctx = DVector::from_vec(h0.slot(0).as_slice().to_vec());

        // Start from h0 (preserves Δ = f^k(H_k) - H_k)
        let mut h = h0.clone();
        let mut iters = 0u32;
        let mut stop: u8 = 1; // default: k_max gate

        for _ in 0..self.k_max {
            let h_next = self
                .reasoning
                .step(&h, &s_ctx, None::<&mut dyn ComputeBackend>);

            // Convergence over flats
            let flat_next = h_next.to_flat();
            let flat_curr = h.to_flat();
            let step_norm = flat_next
                .iter()
                .zip(flat_curr.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();

            h = h_next;
            iters += 1;
            if step_norm < self.eps_step {
                stop = 0;
                break;
            }
        }

        // Compute total delta: H_final - H0
        let flat_final = h.to_flat();
        let flat_init = h0.to_flat();
        let mut diff: Vec<f32> = flat_final
            .iter()
            .zip(flat_init.iter())
            .map(|(a, b)| a - b)
            .collect();

        // Delta norm
        let norm = diff.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Total clamp: if the delta is too large, rescale
        if norm > self.delta_cap {
            let scale = self.delta_cap / norm;
            diff.iter_mut().for_each(|x| *x *= scale);
            stop = 2;
        }

        let norm_capped = norm.min(self.delta_cap);
        let q_total = (-norm_capped / (1.0 + norm_capped)).exp();

        Ok(NetMsg::ExpertResult {
            task_id,
            target_id,
            delta: diff,
            q_total,
            iters,
            stop,
        })
    }
}
