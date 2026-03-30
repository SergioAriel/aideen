pub mod aggregator;
pub mod client;
pub mod router;
pub mod service;

pub use aggregator::Aggregator;
pub use client::ExpertClient;
pub use router::{Router, UniformRouter};
pub use service::ExpertService;

use aideen_core::protocol::NetMsg;

use crate::peers::types::NodeId;

/// Result of a query to the expert pipeline.
///
/// Contains the combined delta ready to inject with `apply_expert_result()`,
/// plus batch statistics for telemetry and damping.
pub struct RunResult {
    /// Combined delta (pre-β), ready for inject_delta_r_scaled.
    pub delta: Vec<f32>,
    /// Weighted mean quality of non-dropped peers.
    pub q_mean: f32,
    /// ||delta||₂ post-aggregation pre-β.
    pub delta_norm: f32,
    /// Peers dropped as outliers in the batch.
    pub drops_count: u32,
}

/// P2P expert query pipeline.
///
/// Orchestrates: Router → ExpertClient → outlier filter → Aggregator.
/// Call before tick() for warm-start of the local DEQ with external knowledge.
pub struct ExpertPipeline {
    pub router: Box<dyn Router>,
    pub client: ExpertClient,
    pub bundle_version: u64,
    /// Global clamp applied to the combined delta (None = no limit).
    pub delta_cap_global: Option<f32>,
    /// None = disabled.
    /// Some(f) = drops peers with ||Δ_i|| > f × batch median.
    /// Only acts if k_successes ≥ 2 AND f > 1.0.
    pub outlier_factor: Option<f32>,
}

impl ExpertPipeline {
    /// Queries selected peers in parallel (MVP: sequential) and aggregates deltas.
    /// `h_k_slice`: slice del subespacio S_R del estado actual (len = D_R = 1024).
    /// Returns RunResult ready for `apply_expert_result`.
    pub fn run(&mut self, h_k_slice: &[f32]) -> Result<RunResult, String> {
        // Deterministic order via BTreeMap natural ordering
        let peer_ids: Vec<NodeId> = self.client.sorted_peer_ids();
        let n = peer_ids.len();
        if n == 0 {
            return Err("ExpertPipeline: no peers".into());
        }

        // Router returns indices over peer_ids; we translate them to NodeIds
        let selected_indices = self.router.select(h_k_slice, n);
        if selected_indices.is_empty() {
            return Err("ExpertPipeline: no peers selected".into());
        }
        let selected: Vec<(NodeId, f32)> = selected_indices
            .iter()
            .map(|(idx, alpha)| (peer_ids[*idx], *alpha))
            .collect();

        let task = NetMsg::ExpertTask {
            task_id: task_id_now(),
            target_id: "global".to_string(),
            s_r: h_k_slice.to_vec(),
            bundle_version: self.bundle_version,
            round: 0,
            time_budget_ms: 5000,
        };

        let raw = self.client.query(task, &selected);

        // Filter successes — alphas and results stay aligned 1:1
        let (alphas, results): (Vec<f32>, Vec<NetMsg>) = raw
            .into_iter()
            .filter_map(|(a, r)| r.ok().map(|m| (a, m)))
            .unzip();

        if alphas.is_empty() {
            return Err("ExpertPipeline: all peers failed".into());
        }

        // Base renormalisation (Σ=1)
        let total: f32 = alphas.iter().sum();
        let mut norm_alphas: Vec<f32> = alphas.iter().map(|a| a / total).collect();

        // Outlier detection (hard drop)
        let mut drops_count = 0u32;
        if let Some(factor) = self.outlier_factor {
            // micro B: factor <= 1.0 es config peligrosa — skip
            if factor > 1.0 && results.len() >= 2 {
                // Normas: NaN → INFINITY para que sean dropeados (micro A)
                let norms: Vec<f32> = results
                    .iter()
                    .map(|msg| {
                        if let NetMsg::ExpertResult { delta, .. } = msg {
                            let n = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
                            if n.is_nan() {
                                f32::INFINITY
                            } else {
                                n
                            }
                        } else {
                            0.0
                        }
                    })
                    .collect();

                // Mediana: k=2 → promedio; k≥3 → mediana-low
                let mut sorted_norms = norms.clone();
                sorted_norms.sort_by(|a, b| a.total_cmp(b)); // micro A: total_cmp
                let median = if sorted_norms.len() == 2 {
                    (sorted_norms[0] + sorted_norms[1]) / 2.0
                } else {
                    sorted_norms[(sorted_norms.len() - 1) / 2]
                };

                // Guard median≈0: si todos los deltas son casi cero, skip
                if median > 1e-6 {
                    for (i, &n) in norms.iter().enumerate() {
                        if n > factor * median {
                            norm_alphas[i] = 0.0;
                            drops_count += 1;
                        }
                    }
                    let remaining: f32 = norm_alphas.iter().sum();
                    if remaining == 0.0 {
                        return Err("ExpertPipeline: all peers are outliers".into());
                    }
                    norm_alphas.iter_mut().for_each(|a| *a /= remaining);
                }
            }
        }

        let combined = Aggregator::combine(&norm_alphas, &results, self.delta_cap_global)?;

        let delta_norm = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        let q_mean: f32 = results
            .iter()
            .zip(&norm_alphas)
            .filter(|(_, a)| **a > 0.0)
            .map(|(msg, a)| match msg {
                NetMsg::ExpertResult { q_total, .. } => q_total * a,
                _ => 0.0,
            })
            .sum();

        Ok(RunResult {
            delta: combined,
            q_mean,
            delta_norm,
            drops_count,
        })
    }
}

fn task_id_now() -> [u8; 16] {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    let mut id = [0u8; 16];
    id[..4].copy_from_slice(&ns.to_le_bytes());
    id
}
