use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-parameter optimizer state (first moment, second moment, step count).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ParamState {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u64,
}

/// AdamW optimizer with decoupled weight decay.
///
/// Implements the algorithm from Loshchilov & Hutter (2019):
///   m_t  = beta1 * m_{t-1} + (1 - beta1) * g_t
///   v_t  = beta2 * v_{t-1} + (1 - beta2) * g_t^2
///   m_hat = m_t / (1 - beta1^t)
///   v_hat = v_t / (1 - beta2^t)
///   theta = theta - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * theta)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    state: HashMap<String, ParamState>,
}

impl AdamW {
    /// Create a new AdamW optimizer with all hyperparameters specified.
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            state: HashMap::new(),
        }
    }

    /// Create a new AdamW optimizer with default hyperparameters and the given
    /// learning rate. Defaults: beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01.
    pub fn default_with_lr(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Perform one AdamW update step for the named parameter group.
    ///
    /// `name`   — a unique identifier for this parameter group (used to look up
    ///            per-parameter moment buffers).
    /// `params` — the parameter slice to update in-place.
    /// `grads`  — the gradient slice (must have the same length as `params`).
    ///
    /// Panics if `params.len() != grads.len()`.
    pub fn step(&mut self, name: &str, params: &mut [f32], grads: &[f32]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "AdamW::step: params and grads must have the same length"
        );

        let n = params.len();

        // Get or create the state entry for this parameter group.
        let ps = self
            .state
            .entry(name.to_owned())
            .or_insert_with(|| ParamState {
                m: vec![0.0; n],
                v: vec![0.0; n],
                t: 0,
            });

        // If the parameter size changed (shouldn't happen, but be safe), resize.
        if ps.m.len() != n {
            ps.m.resize(n, 0.0);
            ps.v.resize(n, 0.0);
        }

        ps.t += 1;
        let t = ps.t;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let lr = self.lr;
        let eps = self.eps;
        let wd = self.weight_decay;

        // Bias correction denominators.
        let bc1 = 1.0 - beta1.powi(t as i32);
        let bc2 = 1.0 - beta2.powi(t as i32);

        for i in 0..n {
            let g = grads[i];

            // Update biased first moment estimate.
            ps.m[i] = beta1 * ps.m[i] + (1.0 - beta1) * g;
            // Update biased second raw moment estimate.
            ps.v[i] = beta2 * ps.v[i] + (1.0 - beta2) * g * g;

            // Bias-corrected estimates.
            let m_hat = ps.m[i] / bc1;
            let v_hat = ps.v[i] / bc2;

            // Decoupled weight decay + Adam update.
            params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * params[i]);
        }
    }

    /// Clear all optimizer state (moments and step counts).
    pub fn reset(&mut self) {
        self.state.clear();
    }
}
