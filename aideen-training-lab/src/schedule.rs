/// Cosine learning rate schedule with linear warmup.
pub struct CosineSchedule {
    pub warmup_steps: usize,
    pub total_steps: usize,
    pub base_lr: f32,
    pub min_lr: f32,
}

impl CosineSchedule {
    pub fn new(warmup_steps: usize, total_steps: usize, base_lr: f32, min_lr: f32) -> Self {
        Self {
            warmup_steps,
            total_steps,
            base_lr,
            min_lr,
        }
    }

    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear ramp from 0 to base_lr during warmup.
            if self.warmup_steps == 0 {
                return self.base_lr;
            }
            self.base_lr * (step as f32 / self.warmup_steps as f32)
        } else if step >= self.total_steps {
            self.min_lr
        } else {
            // Cosine decay from base_lr to min_lr.
            let decay_steps = self.total_steps - self.warmup_steps;
            let progress = (step - self.warmup_steps) as f32 / decay_steps as f32;
            self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
        }
    }
}
