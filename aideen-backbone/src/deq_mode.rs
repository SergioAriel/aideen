#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeqComparisonMode {
    None,
    DeqOnly,
    InitFpm,
    FixedFpm,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeqRuntimeConfig {
    pub comparison_mode: DeqComparisonMode,
    pub residual_alpha_uniform: f32,
}

impl DeqRuntimeConfig {
    pub fn from_env() -> Self {
        if std::env::var("AIDEEN_DEQ_ONLY").ok().as_deref() == Some("1") {
            return Self {
                comparison_mode: DeqComparisonMode::DeqOnly,
                residual_alpha_uniform: -2.0,
            };
        }
        if std::env::var("AIDEEN_DEQ_INIT_FPM").ok().as_deref() == Some("1") {
            return Self {
                comparison_mode: DeqComparisonMode::InitFpm,
                residual_alpha_uniform: -0.75,
            };
        }
        if std::env::var("AIDEEN_DEQ_FIXED_FPM").ok().as_deref() == Some("1") {
            return Self {
                comparison_mode: DeqComparisonMode::FixedFpm,
                residual_alpha_uniform: -0.25,
            };
        }
        if let Ok(raw) = std::env::var("AIDEEN_DEQ_RESIDUAL_ALPHA") {
            if let Ok(alpha) = raw.trim().parse::<f32>() {
                return Self {
                    comparison_mode: DeqComparisonMode::None,
                    residual_alpha_uniform: alpha.clamp(0.0, 1.0),
                };
            }
        }
        // Runtime baseline keeps the unified slot branch enabled in forward, but without entering
        // any explicit comparison mode that should disable FPM or reroute the backward stack.
        Self {
            comparison_mode: DeqComparisonMode::None,
            residual_alpha_uniform: -1.0,
        }
    }

    pub fn is_deq_only(self) -> bool {
        matches!(self.comparison_mode, DeqComparisonMode::DeqOnly)
    }

    pub fn has_explicit_slot_comparison(self) -> bool {
        matches!(
            self.comparison_mode,
            DeqComparisonMode::InitFpm | DeqComparisonMode::FixedFpm
        )
    }
}
