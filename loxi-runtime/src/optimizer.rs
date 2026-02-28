// optimizer.rs
// ─────────────────────────────────────────────────────────────────────────────
// AdamW optimizer for distributed expert training.
//
// Architecture:
//   Optimizer holds a set of ParameterGroups.
//   Each group can have different learning rates (e.g., different lr for
//   backbone vs expert layers, or lr warmup schedule).
//
// GPU integration:
//   The actual parameter update is dispatched to the AdamW WGSL kernel.
//   This means: weights, gradients, and moments all live on GPU.
//   Only the step counter and bias corrections are managed on CPU.
//
// Learning rate schedule:
//   Linear warmup for first WARMUP_STEPS steps, then cosine decay.
//   This is standard practice for transformer training.
//
// Gradient clipping:
//   Global norm clipping applied before optimizer step.
//   Clip norm = 1.0 (standard for LLMs).
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use anyhow::Result;

use crate::tensor::{GpuContext, Shape, Tensor};
use crate::dispatch::{AdamWParams, Dispatcher};

// ─── Learning Rate Schedule ───────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LrSchedule {
    /// Peak learning rate (reached after warmup)
    pub peak_lr:       f32,
    /// Number of warmup steps
    pub warmup_steps:  u32,
    /// Total training steps (for cosine decay calculation)
    pub total_steps:   u32,
    /// Minimum LR at end of cosine decay
    pub min_lr:        f32,
}

impl LrSchedule {
    /// Standard schedule for backbone pretraining.
    pub fn backbone(total_steps: u32) -> Self {
        Self {
            peak_lr:      3e-4,
            warmup_steps: (total_steps as f32 * 0.02) as u32,
            total_steps,
            min_lr:       3e-5,
        }
    }

    /// Schedule for expert fine-tuning (lower LR, shorter warmup).
    pub fn expert(total_steps: u32) -> Self {
        Self {
            peak_lr:      1e-4,
            warmup_steps: (total_steps as f32 * 0.01) as u32,
            total_steps,
            min_lr:       1e-5,
        }
    }

    /// Compute LR for current step.
    pub fn lr_at(&self, step: u32) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.peak_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine decay from peak to min
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let cosine = (std::f32::consts::PI * progress).cos();
            self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + cosine)
        }
    }
}

// ─── Parameter ────────────────────────────────────────────────────────────────

/// A single learnable parameter with its optimizer state.
pub struct Parameter {
    pub name:    String,
    pub weight:  Tensor,   // θ: the actual parameter
    pub grad:    Tensor,   // ∂L/∂θ: gradient (written by backward pass)
    pub moment1: Tensor,   // m: first moment (initialized to 0)
    pub moment2: Tensor,   // v: second moment (initialized to 0)
    /// Whether this parameter is frozen (e.g., backbone in expert training)
    pub frozen:  bool,
}

impl Parameter {
    pub fn new(name: &str, weight: Tensor) -> Self {
        let shape = weight.shape.clone();
        let ctx   = weight.ctx.clone();
        Self {
            name:    name.to_string(),
            grad:    Tensor::zeros(shape.clone(), ctx.clone()),
            moment1: Tensor::zeros(shape.clone(), ctx.clone()),
            moment2: Tensor::zeros(shape,         ctx),
            weight,
            frozen: false,
        }
    }

    pub fn frozen(mut self) -> Self {
        self.frozen = true;
        self
    }
}

// ─── Optimizer ────────────────────────────────────────────────────────────────

/// AdamW optimizer with gradient clipping and LR schedule.
pub struct AdamW {
    params:       Vec<Parameter>,
    dispatch:     Dispatcher,
    schedule:     LrSchedule,
    /// Current training step
    pub step:     u32,
    /// Beta1 (momentum decay)
    pub beta1:    f32,
    /// Beta2 (RMS decay)
    pub beta2:    f32,
    /// Epsilon (numerical stability)
    pub eps:      f32,
    /// Weight decay λ
    pub wd:       f32,
    /// Gradient clip norm
    pub clip_norm: f32,
}

impl AdamW {
    pub fn new(
        params:   Vec<Parameter>,
        schedule: LrSchedule,
        ctx:      GpuContext,
    ) -> Self {
        let dispatch = Dispatcher::new(ctx);
        Self {
            params,
            dispatch,
            schedule,
            step:      0,
            beta1:     0.9,
            beta2:     0.999,
            eps:       1e-8,
            wd:        0.01,
            clip_norm: 1.0,
        }
    }

    /// Perform one optimizer step.
    ///
    /// Assumes gradients have been written to param.grad by the backward pass.
    /// After this call, step counter is incremented.
    pub async fn step_params(&mut self) -> Result<()> {
        self.step += 1;

        let lr  = self.schedule.lr_at(self.step);
        let bc1 = 1.0 - self.beta1.powi(self.step as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step as i32);

        // Compute global gradient norm for clipping
        let global_norm = self.compute_global_grad_norm().await?;
        let clip_scale  = if global_norm > self.clip_norm {
            self.clip_norm / global_norm
        } else {
            1.0
        };

        // If clipping needed, scale all gradients
        if clip_scale < 1.0 - 1e-6 {
            self.apply_grad_scaling(clip_scale).await?;
        }

        // Apply AdamW to each non-frozen parameter
        for param in self.params.iter_mut() {
            if param.frozen { continue; }

            let adamw_params = AdamWParams {
                n:            param.weight.numel() as u32,
                lr,
                beta1:        self.beta1,
                beta2:        self.beta2,
                eps:          self.eps,
                weight_decay: self.wd,
                bc1,
                bc2,
            };

            self.dispatch.adamw_step(
                &param.weight,
                &param.grad,
                &param.moment1,
                &param.moment2,
                adamw_params,
            )?;
        }

        tracing::debug!("Step {} | lr={:.2e} | grad_norm={:.4}", self.step, lr, global_norm);
        Ok(())
    }

    /// Zero out all gradients. Call before each backward pass.
    pub fn zero_grad(&self) {
        for param in &self.params {
            if !param.frozen {
                // Write zeros to grad buffer
                let zeros = vec![0.0f32; param.grad.numel()];
                let _ = param.grad.write_from_cpu(&zeros);
            }
        }
    }

    /// Compute L2 norm across all gradients (for gradient clipping).
    async fn compute_global_grad_norm(&self) -> Result<f32> {
        let mut sq_sum = 0.0f32;
        for param in &self.params {
            if param.frozen { continue; }
            let grad_cpu = param.grad.read_to_cpu().await?;
            sq_sum += grad_cpu.iter().map(|x| x * x).sum::<f32>();
        }
        Ok(sq_sum.sqrt())
    }

    /// Scale all gradients by a scalar.
    async fn apply_grad_scaling(&self, scale: f32) -> Result<()> {
        for param in &self.params {
            if param.frozen { continue; }
            let mut g = param.grad.read_to_cpu().await?;
            for x in g.iter_mut() { *x *= scale; }
            param.grad.write_from_cpu(&g)?;
        }
        Ok(())
    }

    /// Get parameter by name.
    pub fn get_param(&self, name: &str) -> Option<&Parameter> {
        self.params.iter().find(|p| p.name == name)
    }

    /// Get mutable parameter by name.
    pub fn get_param_mut(&mut self, name: &str) -> Option<&mut Parameter> {
        self.params.iter_mut().find(|p| p.name == name)
    }

    pub fn current_lr(&self) -> f32 {
        self.schedule.lr_at(self.step)
    }

    pub fn param_count(&self) -> usize {
        self.params.iter()
            .filter(|p| !p.frozen)
            .map(|p| p.weight.numel())
            .sum()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_schedule_warmup() {
        let sched = LrSchedule::backbone(1000);
        // At step 0: lr should be ~0
        assert!(sched.lr_at(0) < 1e-6);
        // At warmup end: lr should be at peak
        let warmup = sched.warmup_steps;
        let peak = sched.lr_at(warmup);
        assert!((peak - sched.peak_lr).abs() < 1e-6, "peak={}", peak);
        // At total steps: lr should be at min
        let end = sched.lr_at(sched.total_steps);
        assert!((end - sched.min_lr).abs() < 1e-5, "end={}", end);
    }

    #[test]
    fn test_lr_schedule_monotone_decay() {
        let sched = LrSchedule::expert(500);
        let w = sched.warmup_steps;
        // After warmup, LR should monotonically decrease
        let lrs: Vec<f32> = (w..sched.total_steps).step_by(10)
            .map(|s| sched.lr_at(s)).collect();
        for i in 1..lrs.len() {
            assert!(lrs[i] <= lrs[i-1] + 1e-9, "LR increased at step {}", w + i as u32 * 10);
        }
    }
}
