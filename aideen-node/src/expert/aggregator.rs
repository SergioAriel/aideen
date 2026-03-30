use aideen_core::protocol::NetMsg;

pub struct Aggregator;

impl Aggregator {
    /// Combines weighted deltas from ExpertResults.
    /// `alphas` must already be normalised (Σα=1).
    /// `delta_cap_global`: final clamp on the combined delta (None = no limit).
    pub fn combine(
        alphas: &[f32],
        results: &[NetMsg],
        delta_cap_global: Option<f32>,
    ) -> Result<Vec<f32>, String> {
        if alphas.len() != results.len() || alphas.is_empty() {
            return Err("Aggregator: invalid input".into());
        }
        let mut combined: Option<Vec<f32>> = None;
        for (alpha, result) in alphas.iter().zip(results.iter()) {
            let delta = match result {
                NetMsg::ExpertResult { delta, .. } => delta,
                _ => return Err("Aggregator: expected ExpertResult".into()),
            };
            match combined.as_mut() {
                None => combined = Some(delta.iter().map(|x| alpha * x).collect()),
                Some(acc) => {
                    if acc.len() != delta.len() {
                        return Err("Aggregator: delta dim mismatch".into());
                    }
                    for (a, d) in acc.iter_mut().zip(delta.iter()) {
                        *a += alpha * d;
                    }
                }
            }
        }
        let mut out = combined.unwrap();

        // Clamp global: evita que peers ruidosos muevan demasiado el estado
        if let Some(cap) = delta_cap_global {
            let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > cap {
                let scale = cap / norm;
                for x in out.iter_mut() {
                    *x *= scale;
                }
            }
        }
        Ok(out)
    }
}
