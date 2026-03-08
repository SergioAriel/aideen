use aideen_core::compute::ComputeBackend;
use aideen_core::protocol::NetMsg;
use aideen_core::reasoning::Reasoning;
use aideen_core::state::HSlots;
use nalgebra::DVector;

/// Servicio de inferencia de experto local.
///
/// Recibe un `ExpertTask` (s_r aplanado = K×D_R floats),
/// reconstruye HSlots, ejecuta su mini-DEQ y devuelve el delta aplanado.
pub struct ExpertService<R: Reasoning> {
    pub reasoning: R,
    /// Pasos máximos de DEQ (antiguo k_steps).
    pub k_max: usize,
    /// Early-stop: norma de diferencia entre flats < eps_step → cortar.
    pub eps_step: f32,
    /// Clamp: ||Δ_total|| > delta_cap → reescalar a delta_cap.
    pub delta_cap: f32,
}

/// stop codes (ExpertResult.stop):
/// 0 = eps_step gate (convergió antes de k_max)
/// 1 = k_max gate (agotó pasos sin converger)
/// 2 = delta_cap gate (delta clampeado por ser demasiado grande)

impl<R: Reasoning> ExpertService<R> {
    /// Procesa un NetMsg::ExpertTask.
    /// Preserva Δ = f_i^k(H_k) − H_k arrancando desde H0, no desde init().
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

        // Reconstruir H0 desde s_r aplanado
        let config = self.reasoning.config();
        let h0 = if s_r.len() == config.h_slots * config.d_r {
            HSlots::from_flat(s_r, config)
        } else {
            // Fallback: broadcast del primer D_R si el tamaño no coincide
            let s = DVector::from_vec(s_r.clone());
            let d_r = config.d_r;
            HSlots::from_broadcast(&s.rows(0, s.len().min(d_r)).into_owned(), config)
        };

        // s es la proyección del slot 0 como contexto para el step
        let s_ctx = DVector::from_vec(h0.slot(0).as_slice().to_vec());

        // Arrancar desde h0 (preserva Δ = f^k(H_k) - H_k)
        let mut h = h0.clone();
        let mut iters = 0u32;
        let mut stop: u8 = 1; // default: k_max gate

        for _ in 0..self.k_max {
            let h_next = self
                .reasoning
                .step(&h, &s_ctx, None::<&mut dyn ComputeBackend>);

            // Convergencia sobre flats
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

        // Calcular delta total: H_final - H0
        let flat_final = h.to_flat();
        let flat_init = h0.to_flat();
        let mut diff: Vec<f32> = flat_final
            .iter()
            .zip(flat_init.iter())
            .map(|(a, b)| a - b)
            .collect();

        // Norma del delta
        let norm = diff.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Clamp total: si el delta es demasiado grande, reescalar
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
